#!/usr/bin/env python3
"""Trace-driven structural pilot for the direction-1 placement hypothesis.

This is a CPU-only discrete-event simulator. It is deliberately not presented as
a model-serving benchmark: it has no measured GPU service times, KV transfer
bandwidth, cache capacity, or inference accuracy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import heapq
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


POLICIES = (
    "local-only",
    "least-loaded",
    "KV-cost",
    "tree-aware-greedy-oracle",
    "sibling-aware-oracle",
)


@dataclass
class Job:
    tree_id: int
    node_id: int
    depth: int
    prompt_tokens: int
    decode_tokens: int
    children: list[int] = field(default_factory=list)
    parent_id: int | None = None
    service_s: float = 0.0
    prefix_tokens: int = 0
    subtree_work_s: float = 0.0


@dataclass
class Tree:
    tree_id: int
    jobs: dict[int, Job]
    root_id: int


def make_tree(tree_id: int, shape: str, rng: random.Random) -> Tree:
    prompt_tokens = rng.randint(2048, 6144)
    jobs: dict[int, Job] = {}
    next_id = 0

    def new_job(depth: int, parent_id: int | None) -> int:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        decode_tokens = rng.randint(32, 96)
        job = Job(
            tree_id=tree_id,
            node_id=node_id,
            depth=depth,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            parent_id=parent_id,
            prefix_tokens=prompt_tokens if parent_id is None else 0,
            service_s=decode_tokens / 60.0 + (prompt_tokens / 25_000.0 if parent_id is None else 0.0),
        )
        jobs[node_id] = job
        return node_id

    root = new_job(0, None)

    if shape == "balanced":
        def grow(parent: int, depth: int) -> None:
            if depth >= 4:
                return
            for _ in range(2):
                child = new_job(depth + 1, parent)
                jobs[parent].children.append(child)
                grow(child, depth + 1)

        grow(root, 0)
    elif shape == "uneven":
        # Same 31 jobs as the balanced tree, but distribute them unevenly:
        # one 23-node broad-then-deep subtree and one 7-node chain.
        if rng.random() < 0.5:
            wide_root = new_job(1, root)
            chain_root = new_job(1, root)
        else:
            chain_root = new_job(1, root)
            wide_root = new_job(1, root)
        jobs[root].children.extend((wide_root, chain_root))
        wide_leaves: list[int] = []

        def grow_wide(parent: int, depth: int) -> None:
            if depth >= 4:
                wide_leaves.append(parent)
                return
            for _ in range(2):
                child = new_job(depth + 1, parent)
                jobs[parent].children.append(child)
                grow_wide(child, depth + 1)

        grow_wide(wide_root, 1)
        cursor = wide_leaves[0]
        for depth in range(5, 13):
            child = new_job(depth, cursor)
            jobs[cursor].children.append(child)
            cursor = child

        cursor = chain_root
        for depth in range(2, 8):
            child = new_job(depth, cursor)
            jobs[cursor].children.append(child)
            cursor = child
    elif shape == "mixed":
        # A workload with per-tree pruning/continuation variation. Every root
        # starts two branches; later nodes independently stop, continue once,
        # or split, up to depth five.
        for _ in range(2):
            child = new_job(1, root)
            jobs[root].children.append(child)

        def grow_mixed(parent: int, depth: int) -> None:
            if depth >= 5:
                return
            draw = rng.random()
            child_count = 0 if draw < 0.25 else 1 if draw < 0.50 else 2
            for _ in range(child_count):
                child = new_job(depth + 1, parent)
                jobs[parent].children.append(child)
                grow_mixed(child, depth + 1)

        for child_id in jobs[root].children:
            grow_mixed(child_id, 1)
    else:
        raise ValueError(f"unknown tree shape: {shape}")

    # A node's KV state is the prompt plus all generated tokens on its path.
    def set_prefix(node_id: int, prefix_tokens: int) -> None:
        job = jobs[node_id]
        job.prefix_tokens = prefix_tokens
        child_prefix = prefix_tokens + job.decode_tokens
        for child_id in job.children:
            set_prefix(child_id, child_prefix)

    set_prefix(root, prompt_tokens)

    def set_subtree_work(node_id: int) -> float:
        job = jobs[node_id]
        job.subtree_work_s = job.service_s + sum(set_subtree_work(child) for child in job.children)
        return job.subtree_work_s

    set_subtree_work(root)
    return Tree(tree_id=tree_id, jobs=jobs, root_id=root)


def build_trace(seed: int, shape: str, tree_count: int) -> list[Tree]:
    rng = random.Random(seed)
    return [make_tree(tree_id, shape, rng) for tree_id in range(tree_count)]


def transfer_and_recompute(prefix_tokens: int, bandwidth_gbps: float) -> tuple[float, float, float]:
    # Assumptions for sensitivity analysis only: 0.5 MiB/token KV and 25k
    # tokens/s prompt recompute. Neither value is calibrated to this host.
    kv_gb = prefix_tokens * 0.5 / 1024.0
    transfer_s = 0.001 + kv_gb / bandwidth_gbps
    recompute_s = prefix_tokens / 25_000.0
    return transfer_s, recompute_s, kv_gb


def choose_action(
    job: Job,
    parent_gpu: int | None,
    gpu: int,
    bandwidth_gbps: float,
    gpu_free: list[float],
    ready_at: float,
) -> tuple[str, float, float]:
    if parent_gpu is None or parent_gpu == gpu:
        finish = max(ready_at, gpu_free[gpu]) + job.service_s
        return "local", finish, 0.0

    transfer_s, recompute_s, kv_gb = transfer_and_recompute(job.prefix_tokens, bandwidth_gbps)
    copy_finish = max(ready_at + transfer_s, gpu_free[gpu]) + job.service_s
    recompute_finish = max(ready_at, gpu_free[gpu]) + recompute_s + job.service_s
    if copy_finish <= recompute_finish:
        return "transfer", copy_finish, kv_gb
    return "recompute", recompute_finish, 0.0


def simulate(trees: list[Tree], policy: str, gpu_count: int, bandwidth_gbps: float) -> dict[str, float | int | str]:
    gpu_free = [0.0] * gpu_count
    tree_home: dict[int, int] = {}
    tree_done: dict[int, float] = {}
    completion_events: list[tuple[float, int, int, int]] = []
    ready_events: list[tuple[float, int, int, int | None]] = []
    transferred_gb = 0.0
    recomputed_prefix_tokens = 0

    for tree in trees:
        root = tree.jobs[tree.root_id]
        heapq.heappush(ready_events, (0.0, tree.tree_id, root.node_id, None))

    while ready_events or completion_events:
        next_ready = ready_events[0][0] if ready_events else math.inf
        next_done = completion_events[0][0] if completion_events else math.inf
        now = min(next_ready, next_done)

        # Complete work at this event time and release dependent child jobs.
        while completion_events and completion_events[0][0] <= now + 1e-12:
            finish, tree_id, node_id, gpu = heapq.heappop(completion_events)
            tree_done[tree_id] = max(tree_done.get(tree_id, 0.0), finish)
            tree = trees[tree_id]
            for child_id in tree.jobs[node_id].children:
                heapq.heappush(ready_events, (finish, tree_id, child_id, gpu))

        batch: list[tuple[float, int, int, int | None]] = []
        while ready_events and ready_events[0][0] <= now + 1e-12:
            batch.append(heapq.heappop(ready_events))
        if not batch:
            continue

        if policy in ("tree-aware-greedy-oracle", "sibling-aware-oracle"):
            batch.sort(
                key=lambda item: (
                    -trees[item[1]].jobs[item[2]].subtree_work_s,
                    item[1],
                    item[2],
                )
            )
        else:
            batch.sort(key=lambda item: (item[1], item[2]))

        if policy == "sibling-aware-oracle":
            grouped: dict[tuple[int, int | None, int | None], list[tuple[float, int, int, int | None]]] = {}
            for item in batch:
                _, tree_id, node_id, parent_gpu = item
                parent_id = trees[tree_id].jobs[node_id].parent_id
                grouped.setdefault((tree_id, parent_id, parent_gpu), []).append(item)
            groups = list(grouped.values())
            groups.sort(
                key=lambda group: (
                    -sum(trees[item[1]].jobs[item[2]].subtree_work_s for item in group),
                    group[0][1],
                )
            )
            for group in groups:
                group.sort(key=lambda item: (-trees[item[1]].jobs[item[2]].subtree_work_s, item[2]))
                ready_at = group[0][0]
                best = None
                for assignment in itertools.product(range(gpu_count), repeat=len(group)):
                    temp_free = list(gpu_free)
                    projected = []
                    transfer_total = 0.0
                    for item, gpu in zip(group, assignment):
                        _, tree_id, node_id, item_parent_gpu = item
                        job = trees[tree_id].jobs[node_id]
                        action, finish, kv_gb = choose_action(
                            job, item_parent_gpu, gpu, bandwidth_gbps, temp_free, item[0]
                        )
                        descendants = max(0.0, job.subtree_work_s - job.service_s)
                        projected.append(finish + descendants)
                        temp_free[gpu] = finish
                        transfer_total += kv_gb if action == "transfer" else 0.0
                    objective = (max(projected), sum(projected), transfer_total, assignment)
                    if best is None or objective < best[0]:
                        best = (objective, assignment)

                assert best is not None
                # Commit immediately so the next independent tree group sees
                # the projected queue and reservations from this group.
                for item, gpu in zip(group, best[1]):
                    ready_at, tree_id, node_id, parent_gpu = item
                    job = trees[tree_id].jobs[node_id]
                    action, finish, kv_gb = choose_action(
                        job, parent_gpu, gpu, bandwidth_gbps, gpu_free, ready_at
                    )
                    gpu_free[gpu] = finish
                    descendants = max(0.0, job.subtree_work_s - job.service_s)
                    if node_id == trees[tree_id].root_id:
                        tree_home[tree_id] = gpu
                    if action == "transfer":
                        transferred_gb += kv_gb
                    elif action == "recompute":
                        recomputed_prefix_tokens += job.prefix_tokens
                    heapq.heappush(completion_events, (finish, tree_id, node_id, gpu))
            continue

        for ready_at, tree_id, node_id, parent_gpu in batch:
            job = trees[tree_id].jobs[node_id]
            candidates: list[tuple[float, int, str, float]] = []
            gpu_range = range(gpu_count)

            if policy == "local-only" and tree_id in tree_home:
                gpu_range = (tree_home[tree_id],)

            for gpu in gpu_range:
                action, finish, kv_gb = choose_action(job, parent_gpu, gpu, bandwidth_gbps, gpu_free, ready_at)
                if policy == "least-loaded":
                    score = gpu_free[gpu]
                elif policy == "tree-aware-greedy-oracle":
                    descendants = max(0.0, job.subtree_work_s - job.service_s)
                    score = finish + descendants
                else:
                    score = finish
                candidates.append((score, gpu, action, kv_gb))

            _, gpu, action, kv_gb = min(candidates, key=lambda item: (item[0], item[1]))
            _, finish, _ = choose_action(job, parent_gpu, gpu, bandwidth_gbps, gpu_free, ready_at)
            gpu_free[gpu] = finish
            if node_id == trees[tree_id].root_id:
                tree_home[tree_id] = gpu

            descendants = max(0.0, job.subtree_work_s - job.service_s)
            if action == "transfer":
                transferred_gb += kv_gb
            elif action == "recompute":
                recomputed_prefix_tokens += job.prefix_tokens

            heapq.heappush(completion_events, (finish, tree_id, node_id, gpu))

    latencies = list(tree_done.values())
    ordered = sorted(latencies)

    def percentile(values: list[float], q: float) -> float:
        index = max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))
        return values[index]

    return {
        "policy": policy,
        "tree_count": len(trees),
        "p50_s": percentile(ordered, 0.50),
        "p95_s": percentile(ordered, 0.95),
        "p99_s": percentile(ordered, 0.99),
        "transferred_kv_gb": transferred_gb,
        "recomputed_prefix_tokens": recomputed_prefix_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--tree-counts", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--bandwidths-gbps", type=float, nargs="+", default=[5, 20, 80])
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for shape in ("balanced", "uneven", "mixed"):
        for tree_count in args.tree_counts:
            for bandwidth in args.bandwidths_gbps:
                for seed in range(args.seeds):
                    trees = build_trace(seed, shape, tree_count)
                    for policy in POLICIES:
                        result = simulate(trees, policy, args.gpus, bandwidth)
                        rows.append(
                            {
                                "shape": shape,
                                "tree_count": tree_count,
                                "gpus": args.gpus,
                                "bandwidth_gbps_assumed": bandwidth,
                                "seed": seed,
                                **result,
                            }
                        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
