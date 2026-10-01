#!/usr/bin/env python3
"""Replay small reasoning trees through independent vLLM replicas.

This is a real inference placement pilot, not a KV-transfer implementation.
The KV-cost baseline estimates local prefix reuse versus remote recomputation;
vLLM prefix caching is enabled on each replica, while cross-replica KV transfer
is unavailable in this prototype.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


POLICIES = (
    "local-only",
    "least-loaded",
    "kv-cost",
    "sibling-lookahead",
    "kv-cost-group-flat",
    "kv-cost-group-tree",
)
SYSTEM_PROMPT = (
    "You are a concise arithmetic reasoning assistant. Work carefully and keep "
    "each search step short."
)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass
class NodePlan:
    node_id: int
    parent_id: int | None
    depth: int
    edge_label: str
    children: list[int] = field(default_factory=list)


@dataclass
class TreePlan:
    tree_id: int
    dataset_index: int
    question: str
    expected_answer: str
    shape: str
    nodes: dict[int, NodePlan]


@dataclass
class RuntimeNode:
    tree: TreePlan
    plan: NodePlan
    messages: list[dict[str, str]]
    instruction: str
    max_tokens_inner: int
    max_tokens_leaf: int

    @property
    def is_leaf(self) -> bool:
        return not self.plan.children

    @property
    def max_tokens(self) -> int:
        return self.max_tokens_leaf if self.is_leaf else self.max_tokens_inner


@dataclass
class RouteChoice:
    node: RuntimeNode
    worker: int
    cached_tokens: int
    estimate_s: float
    queue_estimate_s: float = 0.0
    remaining_work_estimate_s: float = 0.0
    projected_peak_s: float | None = None
    group_size: int = 1
    candidate_assignment_count: int = 1


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().replace(",", "").replace("$", "")
    value = value.rstrip(".% ")
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def expected_from_gsm8k(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError("GSM8K answer has no #### target marker")
    target = answer.rsplit(marker, 1)[1].strip().splitlines()[0]
    normalized = normalize_number(target)
    if normalized is None:
        raise ValueError(f"Could not parse GSM8K target: {target!r}")
    return normalized


def extract_answer(text: str) -> tuple[str | None, str]:
    """Parse only an explicit final marker after a closed reasoning section."""
    # vLLM's prompt template opens the DeepSeek reasoning section before the
    # generated content, so the returned content need not contain <think>.
    if "</think>" not in text:
        return None, "unclosed_reasoning"
    final_region = text.rsplit("</think>", 1)[1]
    if "####" not in final_region:
        return None, "missing_final_marker"
    matches = NUMBER_RE.findall(final_region.rsplit("####", 1)[1])
    if not matches:
        return None, "marker_without_number"
    return normalize_number(matches[-1]), "parsed_final"


def build_tree(tree_id: int, dataset_index: int, row: dict[str, str], shape: str) -> TreePlan:
    nodes: dict[int, NodePlan] = {}

    def add(parent_id: int | None, depth: int, edge_label: str) -> int:
        node_id = len(nodes)
        nodes[node_id] = NodePlan(node_id, parent_id, depth, edge_label)
        if parent_id is not None:
            nodes[parent_id].children.append(node_id)
        return node_id

    root = add(None, 0, "root")
    if shape == "balanced":
        def grow(parent_id: int, depth: int) -> None:
            if depth >= 2:
                return
            for label in ("A", "B"):
                child = add(parent_id, depth + 1, label)
                grow(child, depth + 1)

        grow(root, 0)
    elif shape == "broad":
        for label in ("A", "B", "C"):
            child = add(root, 1, label)
            add(child, 2, f"{label}1")
            add(child, 2, f"{label}2")
    elif shape == "chain":
        parent = root
        for depth in range(1, 6):
            parent = add(parent, depth, f"step-{depth}")
    elif shape == "skewed":
        branch_a = add(root, 1, "A")
        branch_b = add(root, 1, "B")
        branch_a1 = add(branch_a, 2, "A1")
        add(branch_a, 2, "A2")
        branch_a1a = add(branch_a1, 3, "A1a")
        add(branch_a1, 3, "A1b")
        add(branch_a1a, 4, "A1a-i")
        add(branch_b, 2, "B1")
    else:
        raise ValueError(f"Unknown tree shape: {shape}")

    return TreePlan(
        tree_id=tree_id,
        dataset_index=dataset_index,
        question=row["question"],
        expected_answer=expected_from_gsm8k(row["answer"]),
        shape=shape,
        nodes=nodes,
    )


def load_trees(dataset_path: Path, question_count: int, seed: int) -> list[TreePlan]:
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    if question_count > len(rows):
        raise ValueError(f"Requested {question_count} questions; dataset has {len(rows)}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), question_count))
    shapes = ("balanced", "broad", "chain", "skewed")
    return [
        build_tree(i, dataset_index, rows[dataset_index], shapes[i % len(shapes)])
        for i, dataset_index in enumerate(indices)
    ]


def instruction_for(tree: TreePlan, plan: NodePlan) -> str:
    if plan.depth == 0:
        action = "Give one concise opening reasoning step. Do not finish the problem yet."
    elif not plan.children:
        action = "Solve this terminal branch and end with exactly one line: #### <number>."
    else:
        action = "Continue this branch with one concise reasoning step. Do not finish yet."
    return (
        f"Problem: {tree.question}\n"
        f"Search tree: {tree.shape}; path branch: {plan.edge_label}.\n"
        f"{action}"
    )


def runtime_node(
    tree: TreePlan,
    node_id: int,
    messages: list[dict[str, str]],
    max_tokens_inner: int,
    max_tokens_leaf: int,
    *,
    full_prompt: bool = False,
) -> RuntimeNode:
    plan = tree.nodes[node_id]
    instruction = instruction_for(tree, plan)
    prompt_messages = (
        messages
        if full_prompt
        else messages + [{"role": "user", "content": instruction}]
    )
    return RuntimeNode(
        tree,
        plan,
        prompt_messages,
        instruction,
        max_tokens_inner,
        max_tokens_leaf,
    )


def chat_messages_for_root(tree: TreePlan) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def lcp_len(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    i = 0
    while i < limit and left[i] == right[i]:
        i += 1
    return i


def scrape_metric_lines(text: str) -> list[str]:
    tracked = (
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:request_inference_time_seconds",
        "vllm:request_prefill_kv_computed_tokens",
        "vllm:time_to_first_token_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:e2e_request_latency_seconds",
    )
    return [
        line
        for line in text.splitlines()
        if line and not line.startswith("#") and line.startswith(tracked)
    ]


def metric_value(lines: list[str], metric_name: str) -> float | None:
    for line in lines:
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            try:
                return float(line.rsplit(maxsplit=1)[1])
            except (IndexError, ValueError):
                return None
    return None


def metric_total(lines: list[str], metric_name: str) -> float | None:
    values = []
    for line in lines:
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            try:
                values.append(float(line.rsplit(maxsplit=1)[1]))
            except (IndexError, ValueError):
                continue
    return sum(values) if values else None


def metric_delta(before: list[str], after: list[str], metric_name: str) -> float | None:
    start = metric_total(before, metric_name)
    end = metric_total(after, metric_name)
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def metric_observation_delta(before: list[str], after: list[str], metric_name: str) -> dict[str, float | None]:
    total = metric_delta(before, after, metric_name + "_sum")
    count = metric_delta(before, after, metric_name + "_count")
    return {
        "sum_s": total,
        "count": count,
        "mean_s": total / count if total is not None and count else None,
    }


async def reset_prefix_cache(client: httpx.AsyncClient, endpoint: str) -> None:
    response = await client.post(endpoint.rstrip("/") + "/reset_prefix_cache", timeout=30.0)
    response.raise_for_status()
    result = response.json()
    if result.get("success") is not True:
        raise RuntimeError(f"vLLM prefix cache reset failed at {endpoint}: {result!r}")


def load_service_rates(args: argparse.Namespace, endpoints: list[str]) -> tuple[list[float], list[float]]:
    if args.service_profile_json:
        profile = json.loads(Path(args.service_profile_json).read_text(encoding="utf-8"))
        workers = profile.get("workers", [])
        if len(workers) != len(endpoints):
            raise ValueError("Service profile worker count does not match --workers")
        for index, (record, endpoint) in enumerate(zip(workers, endpoints)):
            if record.get("endpoint") != endpoint:
                raise ValueError(
                    f"Service profile endpoint {index} mismatch: "
                    f"{record.get('endpoint')!r} != {endpoint!r}"
                )
        return (
            [float(record["prefill_tps"]) for record in workers],
            [float(record["decode_tps"]) for record in workers],
        )
    return (
        [float(args.prefill_tps)] * len(endpoints),
        [float(args.decode_tps)] * len(endpoints),
    )


def load_output_profile(path: str) -> dict[str, float] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    profile = data.get("expected_output_tokens")
    if not isinstance(profile, dict) or not {"inner", "leaf"}.issubset(profile):
        raise ValueError("Output profile must define expected_output_tokens for inner and leaf nodes")
    return {role: float(profile[role]) for role in ("inner", "leaf")}


async def metrics_snapshot(client: httpx.AsyncClient, endpoint: str) -> list[str]:
    try:
        response = await client.get(endpoint.rstrip("/") + "/metrics", timeout=10.0)
        response.raise_for_status()
        return scrape_metric_lines(response.text)
    except Exception as exc:  # Metrics are supplementary; inference is primary.
        return [f"metrics_unavailable: {type(exc).__name__}: {exc}"]


async def metrics_snapshot_when_request_counts_complete(
    client: httpx.AsyncClient,
    endpoint: str,
    before: list[str],
    expected_requests: int,
    timeout_s: float = 15.0,
) -> list[str]:
    required = (
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
    )
    deadline = time.monotonic() + timeout_s
    while True:
        after = await metrics_snapshot(client, endpoint)
        counts = {
            name: metric_observation_delta(before, after, name)["count"]
            for name in required
        }
        if all(count == expected_requests for count in counts.values()):
            return after
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Incomplete vLLM phase metrics at {endpoint}: "
                f"expected {expected_requests} observations, got {counts}"
            )
        await asyncio.sleep(0.25)


class Router:
    def __init__(
        self,
        policy: str,
        endpoints: list[str],
        tokenizer: Any,
        router_capacity: int,
        prefill_tps: list[float],
        decode_tps: list[float],
        block_size: int,
        max_tokens_inner: int,
        max_tokens_leaf: int,
        output_profile: dict[str, float] | None = None,
    ) -> None:
        self.policy = policy
        self.endpoints = endpoints
        self.tokenizer = tokenizer
        self.capacity = router_capacity
        self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps
        self.block_size = block_size
        self.max_tokens_inner = max_tokens_inner
        self.max_tokens_leaf = max_tokens_leaf
        self.output_profile = output_profile or {}
        self.pending_work = [0.0 for _ in endpoints]
        self.cache_sequences: list[list[list[int]]] = [[] for _ in endpoints]
        self.round_robin = 0
        self.tree_home: dict[int, int] = {}
        self.prompt_ids_by_node: dict[tuple[int, int], list[int]] = {}

    def token_ids(self, node: RuntimeNode) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            node.messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        return [int(token) for token in encoded]

    def cached_prefix_tokens(self, worker: int, prompt_ids: list[int]) -> int:
        best = max(
            (lcp_len(prompt_ids, cached) for cached in self.cache_sequences[worker]),
            default=0,
        )
        return (best // self.block_size) * self.block_size

    def expected_output_tokens(self, node: RuntimeNode) -> float:
        role = "leaf" if node.is_leaf else "inner"
        expected = self.output_profile.get(role, float(node.max_tokens))
        return min(float(node.max_tokens), max(1.0, expected))

    def service_estimate(
        self,
        node: RuntimeNode,
        prompt_ids: list[int],
        worker: int,
        use_output_profile: bool = False,
    ) -> tuple[float, int]:
        cached = self.cached_prefix_tokens(worker, prompt_ids)
        uncached = max(0, len(prompt_ids) - cached)
        output_tokens = self.expected_output_tokens(node) if use_output_profile else node.max_tokens
        seconds = (
            0.02
            + uncached / self.prefill_tps[worker]
            + output_tokens / self.decode_tps[worker]
        )
        return seconds, cached

    def remaining_tree_work_estimate(self, node: RuntimeNode, worker: int) -> float:
        """Estimate future descendant work if this branch stays on one worker."""
        total = 0.0
        stack = list(node.plan.children)
        while stack:
            node_id = stack.pop()
            plan = node.tree.nodes[node_id]
            prompt_ids = self.prompt_ids_by_node[(node.tree.tree_id, node_id)]
            cached = self.cached_prefix_tokens(worker, prompt_ids)
            uncached = max(0, len(prompt_ids) - cached)
            role = "leaf" if not plan.children else "inner"
            output_tokens = max(1.0, self.output_profile[role])
            output_cap = self.max_tokens_leaf if not plan.children else self.max_tokens_inner
            output_tokens = min(float(output_cap), output_tokens)
            total += (
                0.02
                + uncached / self.prefill_tps[worker]
                + output_tokens / self.decode_tps[worker]
            )
            stack.extend(plan.children)
        return total

    def subtree_work_estimate(self, node: RuntimeNode, prompt_len: int, worker: int) -> float:
        """Predict branch work from the known tree topology and token budgets."""
        tree = node.tree
        total = 0.0
        stack = [(node.plan.node_id, prompt_len)]
        while stack:
            node_id, estimated_prompt_len = stack.pop()
            plan = tree.nodes[node_id]
            output_budget = self.max_tokens_leaf if not plan.children else self.max_tokens_inner
            total += (
                0.02
                + estimated_prompt_len / self.prefill_tps[worker]
                + output_budget / self.decode_tps[worker]
            )
            child_prompt_len = estimated_prompt_len + output_budget + 24
            stack.extend((child_id, child_prompt_len) for child_id in plan.children)
        return total

    def choose_group(
        self, group: list[RuntimeNode], prompt_ids: dict[int, list[int]]
    ) -> list[RouteChoice]:
        if self.policy == "local-only":
            tree_id = group[0].tree.tree_id
            worker = self.tree_home.setdefault(tree_id, tree_id % len(self.endpoints))
            choices: list[RouteChoice] = []
            for node in group:
                estimate, cached = self.service_estimate(node, prompt_ids[node.plan.node_id], worker)
                queue = self.pending_work[worker] / self.capacity
                self.pending_work[worker] += estimate
                choices.append(
                    RouteChoice(
                        node=node,
                        worker=worker,
                        cached_tokens=cached,
                        estimate_s=estimate,
                        queue_estimate_s=queue,
                    )
                )
            return choices

        if self.policy in ("least-loaded", "kv-cost"):
            choices: list[RouteChoice] = []
            for node in group:
                ids = prompt_ids[node.plan.node_id]
                options = []
                for worker in range(len(self.endpoints)):
                    estimate, cached = self.service_estimate(node, ids, worker)
                    queue = self.pending_work[worker] / self.capacity
                    if self.policy == "least-loaded":
                        score = (queue, self.pending_work[worker], (worker - self.round_robin) % len(self.endpoints))
                    else:
                        score = (queue + estimate, self.pending_work[worker], (worker - self.round_robin) % len(self.endpoints))
                    options.append((score, worker, cached, estimate))
                score, worker, cached, estimate = min(options, key=lambda option: option[0])
                queue = self.pending_work[worker] / self.capacity
                self.round_robin = (worker + 1) % len(self.endpoints)
                self.pending_work[worker] += estimate
                choices.append(
                    RouteChoice(
                        node=node,
                        worker=worker,
                        cached_tokens=cached,
                        estimate_s=estimate,
                        queue_estimate_s=queue,
                    )
                )
            return choices

        if self.policy in ("kv-cost-group-flat", "kv-cost-group-tree"):
            use_tree_term = self.policy == "kv-cost-group-tree"
            assignments = itertools.product(range(len(self.endpoints)), repeat=len(group))
            best: tuple[
                tuple[float, float, tuple[int, ...]],
                tuple[int, ...],
                list[tuple[float, int, float, float]],
            ] | None = None
            for assignment in assignments:
                projected = [load / self.capacity for load in self.pending_work]
                current_work = 0.0
                node_costs = []
                for node, worker in zip(group, assignment):
                    ids = prompt_ids[node.plan.node_id]
                    estimate, cached = self.service_estimate(
                        node, ids, worker, use_output_profile=True
                    )
                    remaining = self.remaining_tree_work_estimate(node, worker)
                    charged = estimate + (remaining if use_tree_term else 0.0)
                    projected[worker] += charged / self.capacity
                    current_work += estimate
                    node_costs.append((estimate, cached, remaining, self.pending_work[worker] / self.capacity))
                key = (max(projected, default=0.0), current_work, tuple(assignment))
                if best is None or key < best[0]:
                    best = (key, tuple(assignment), node_costs)
            assert best is not None
            choices = []
            assignment_count = len(self.endpoints) ** len(group)
            for node, worker, cost in zip(group, best[1], best[2]):
                estimate, cached, remaining, queue = cost
                self.pending_work[worker] += estimate
                choices.append(
                    RouteChoice(
                        node=node,
                        worker=worker,
                        cached_tokens=cached,
                        estimate_s=estimate,
                        queue_estimate_s=queue,
                        remaining_work_estimate_s=remaining,
                        projected_peak_s=best[0][0],
                        group_size=len(group),
                        candidate_assignment_count=assignment_count,
                    )
                )
            return choices

        # For sibling-lookahead, assign all currently ready siblings jointly.
        # Each branch's known remaining topology is charged to its chosen worker
        # as a locality-preserving lookahead approximation.
        assignments = itertools.product(range(len(self.endpoints)), repeat=len(group))
        best: tuple[tuple[float, float, tuple[int, ...]], tuple[int, ...]] | None = None
        for assignment in assignments:
            projected = [load / self.capacity for load in self.pending_work]
            current_work = 0.0
            for node, worker in zip(group, assignment):
                ids = prompt_ids[node.plan.node_id]
                estimate, cached = self.service_estimate(node, ids, worker)
                subtree_work = self.subtree_work_estimate(node, len(ids), worker)
                subtree_work = max(0.0, subtree_work - cached / self.prefill_tps[worker])
                projected[worker] += subtree_work / self.capacity
                current_work += estimate
            key = (max(projected, default=0.0), current_work, tuple(assignment))
            if best is None or key < best[0]:
                best = (key, tuple(assignment))
        assert best is not None
        choices = []
        for node, worker in zip(group, best[1]):
            estimate, cached = self.service_estimate(node, prompt_ids[node.plan.node_id], worker)
            queue = self.pending_work[worker] / self.capacity
            subtree_work = self.subtree_work_estimate(
                node, len(prompt_ids[node.plan.node_id]), worker
            )
            subtree_work = max(0.0, subtree_work - cached / self.prefill_tps[worker])
            self.pending_work[worker] += estimate
            choices.append(
                RouteChoice(
                    node=node,
                    worker=worker,
                    cached_tokens=cached,
                    estimate_s=estimate,
                    queue_estimate_s=queue,
                    remaining_work_estimate_s=max(0.0, subtree_work - estimate),
                    projected_peak_s=best[0][0],
                    group_size=len(group),
                    candidate_assignment_count=len(self.endpoints) ** len(group),
                )
            )
        return choices

    def complete(self, worker: int, estimate: float, prompt_ids: list[int], output_text: str) -> None:
        self.pending_work[worker] = max(0.0, self.pending_work[worker] - estimate)
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        if hasattr(output_ids, "ids"):
            output_ids = output_ids.ids
        self.cache_sequences[worker].append(prompt_ids + [int(token) for token in output_ids])


async def run_policy(
    args: argparse.Namespace,
    trees: list[TreePlan],
    tokenizer: Any,
    prompt_trace: dict[tuple[int, int], list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    endpoints = args.workers
    prefill_tps_by_worker, decode_tps_by_worker = load_service_rates(args, endpoints)
    output_profile = load_output_profile(args.output_length_profile_json)
    if args.strategy in ("kv-cost-group-flat", "kv-cost-group-tree"):
        if prompt_trace is None:
            raise ValueError(f"{args.strategy} requires a frozen --trace-json")
        if output_profile is None:
            raise ValueError(f"{args.strategy} requires --output-length-profile-json")
    clients = [
        httpx.AsyncClient(
            timeout=httpx.Timeout(args.request_timeout_s),
            limits=httpx.Limits(max_connections=args.max_concurrency * 2, max_keepalive_connections=args.max_concurrency * 2),
        )
        for _ in endpoints
    ]
    router = Router(
        args.strategy,
        endpoints,
        tokenizer,
        args.router_capacity or args.max_concurrency,
        prefill_tps_by_worker,
        decode_tps_by_worker,
        args.block_size,
        args.max_tokens_inner,
        args.max_tokens_leaf,
        output_profile,
    )
    # Keep the local-only control balanced across tree shapes. Cycling a
    # fixed shape order against tree_id % worker_count confounds placement
    # with systematically different node and decode-token workloads.
    trees_by_shape: dict[str, list[TreePlan]] = {}
    for tree in trees:
        trees_by_shape.setdefault(tree.shape, []).append(tree)
    for shape_trees in trees_by_shape.values():
        for index, tree in enumerate(shape_trees):
            router.tree_home[tree.tree_id] = index % len(endpoints)
    if args.reset_prefix_cache:
        await asyncio.gather(
            *(reset_prefix_cache(client, endpoint) for client, endpoint in zip(clients, endpoints))
        )
    metrics_before = await asyncio.gather(
        *(metrics_snapshot(client, endpoint) for client, endpoint in zip(clients, endpoints))
    )
    semaphores = [asyncio.Semaphore(args.max_concurrency) for _ in endpoints]
    tree_start: dict[int, float] = {}
    tree_completed: dict[int, int] = {tree.tree_id: 0 for tree in trees}
    tree_latencies: dict[int, float] = {}
    request_rows: list[dict[str, Any]] = []
    def make_node(tree: TreePlan, node_id: int, history: list[dict[str, str]] | None = None) -> RuntimeNode:
        trace_messages = None if prompt_trace is None else prompt_trace[(tree.tree_id, node_id)]
        if trace_messages is not None:
            return runtime_node(
                tree,
                node_id,
                trace_messages,
                args.max_tokens_inner,
                args.max_tokens_leaf,
                full_prompt=True,
            )
        if history is None:
            raise ValueError("A chat history is required when no frozen prompt trace is loaded")
        return runtime_node(
            tree,
            node_id,
            history,
            args.max_tokens_inner,
            args.max_tokens_leaf,
        )

    if args.strategy in ("kv-cost-group-flat", "kv-cost-group-tree"):
        for tree in trees:
            for node_id in tree.nodes:
                node = make_node(tree, node_id)
                router.prompt_ids_by_node[(tree.tree_id, node_id)] = router.token_ids(node)

    ready_groups: list[list[RuntimeNode]] = [
        [make_node(tree, 0, chat_messages_for_root(tree))] for tree in trees
    ]
    pending: set[asyncio.Task[dict[str, Any]]] = set()
    policy_start = time.perf_counter()
    inference_wall_time_s: float | None = None
    node_ready_offset_s: dict[tuple[int, int], float] = {
        (tree.tree_id, 0): 0.0 for tree in trees
    }

    async def send_request(
        route: RouteChoice,
        prompt_ids: list[int],
        ready_offset_s: float,
        decision_offset_s: float,
    ) -> dict[str, Any]:
        worker = route.worker
        node = route.node
        start = time.perf_counter()
        payload = {
            "model": args.model_name,
            "messages": node.messages,
            "max_tokens": node.max_tokens,
            "temperature": 0,
        }
        async with semaphores[worker]:
            http_start = time.perf_counter()
            response = await clients[worker].post(
                endpoints[worker].rstrip("/") + "/v1/chat/completions", json=payload
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"worker {worker} returned HTTP {response.status_code}: {response.text[:1500]}"
                )
            result = response.json()
        end = time.perf_counter()
        choice = result["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        usage = result.get("usage") or {}
        return {
            "worker": worker,
            "node": node,
            "prompt_ids": prompt_ids,
            "route": route,
            "cached_tokens_estimate": route.cached_tokens,
            "latency_s": end - start,
            "http_latency_s": end - http_start,
            "client_semaphore_wait_s": http_start - start,
            "node_ready_offset_s": ready_offset_s,
            "decision_offset_s": decision_offset_s,
            "request_start_offset_s": start - policy_start,
            "http_start_offset_s": http_start - policy_start,
            "request_end_offset_s": end - policy_start,
            "ready_to_decision_s": decision_offset_s - ready_offset_s,
            "ready_to_http_start_s": http_start - policy_start - ready_offset_s,
            "content": content,
            "prompt_tokens": int(usage.get("prompt_tokens", len(prompt_ids))),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "finish_reason": choice.get("finish_reason", "unknown"),
        }

    try:
        while ready_groups or pending:
            while ready_groups:
                group = ready_groups.pop(0)
                prompt_ids = {node.plan.node_id: router.token_ids(node) for node in group}
                now = time.perf_counter()
                for node in group:
                    if node.plan.parent_id is None:
                        tree_start.setdefault(node.tree.tree_id, now)
                choices = router.choose_group(group, prompt_ids)
                decision_offset_s = time.perf_counter() - policy_start
                for route in choices:
                    node = route.node
                    worker = route.worker
                    task = asyncio.create_task(
                        send_request(
                            route,
                            prompt_ids[node.plan.node_id],
                            node_ready_offset_s[(node.tree.tree_id, node.plan.node_id)],
                            decision_offset_s,
                        ),
                        name=f"{node.tree.tree_id:06d}:{node.plan.node_id:06d}",
                    )
                    pending.add(task)

            if not pending:
                continue
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=lambda completed: completed.get_name()):
                result = task.result()
                route: RouteChoice = result["route"]
                worker = route.worker
                node = route.node
                prompt_ids: list[int] = result["prompt_ids"]
                router.complete(worker, route.estimate_s, prompt_ids, result["content"])
                tree = node.tree
                plan = node.plan
                prediction, parse_status = (
                    extract_answer(result["content"])
                    if node.is_leaf
                    else (None, "not_a_leaf")
                )
                request_row = {
                        "strategy": args.strategy,
                        "tree_id": tree.tree_id,
                        "dataset_index": tree.dataset_index,
                        "shape": tree.shape,
                        "node_id": plan.node_id,
                        "parent_id": "" if plan.parent_id is None else plan.parent_id,
                        "depth": plan.depth,
                        "worker": worker,
                        "endpoint": endpoints[worker],
                        "estimated_cached_prefix_tokens": route.cached_tokens,
                        "ready_offset_s": round(result["node_ready_offset_s"], 6),
                        "decision_offset_s": round(result["decision_offset_s"], 6),
                        "request_start_offset_s": round(result["request_start_offset_s"], 6),
                        "http_start_offset_s": round(result["http_start_offset_s"], 6),
                        "request_end_offset_s": round(result["request_end_offset_s"], 6),
                        "ready_to_decision_s": round(result["ready_to_decision_s"], 6),
                        "client_semaphore_wait_s": round(result["client_semaphore_wait_s"], 6),
                        "ready_to_http_start_s": round(result["ready_to_http_start_s"], 6),
                        "http_latency_s": round(result["http_latency_s"], 6),
                        "predicted_queue_work_s": round(route.queue_estimate_s, 6),
                        "predicted_current_service_s": round(route.estimate_s, 6),
                        "predicted_remaining_tree_work_s": round(route.remaining_work_estimate_s, 6),
                        "projected_group_peak_s": (
                            round(route.projected_peak_s, 6)
                            if route.projected_peak_s is not None
                            else ""
                        ),
                        "ready_group_size": route.group_size,
                        "candidate_assignment_count": route.candidate_assignment_count,
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "finish_reason": result["finish_reason"],
                        "latency_s": round(result["latency_s"], 6),
                        "is_leaf": node.is_leaf,
                        "expected_answer": tree.expected_answer if node.is_leaf else "",
                        "predicted_answer": prediction or "",
                        "answer_parse_status": parse_status,
                        "correct": bool(prediction == tree.expected_answer) if node.is_leaf else "",
                    }
                if not args.omit_output_text:
                    request_row["content"] = result["content"]
                request_rows.append(request_row)
                checkpoint_path = getattr(args, "_checkpoint_path", None)
                if checkpoint_path is not None:
                    append_checkpoint_row(Path(checkpoint_path), request_row)
                tree_completed[tree.tree_id] += 1
                if tree_completed[tree.tree_id] == len(tree.nodes):
                    tree_latencies[tree.tree_id] = time.perf_counter() - tree_start[tree.tree_id]

                if plan.children:
                    child_ready_offset_s = result["request_end_offset_s"]
                    if prompt_trace is None:
                        assistant_content = result["content"]
                        history = node.messages + [
                            {"role": "assistant", "content": assistant_content}
                        ]
                        child_group = [
                            make_node(tree, child_id, history)
                            for child_id in plan.children
                        ]
                    else:
                        child_group = [
                            make_node(tree, child_id) for child_id in plan.children
                        ]
                    for child in child_group:
                        node_ready_offset_s[(tree.tree_id, child.plan.node_id)] = child_ready_offset_s
                    ready_groups.append(child_group)
        inference_wall_time_s = time.perf_counter() - policy_start
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))

    if len(tree_latencies) != len(trees):
        raise RuntimeError(f"Only {len(tree_latencies)}/{len(trees)} trees completed")

    # Aggregate majority-vote and any-correct accuracy across terminal leaves.
    tree_rows: list[dict[str, Any]] = []
    for tree in trees:
        leaves = [
            row
            for row in request_rows
            if row["tree_id"] == tree.tree_id and row["is_leaf"]
        ]
        predictions = [row["predicted_answer"] for row in leaves if row["predicted_answer"]]
        votes: dict[str, int] = {}
        for prediction in predictions:
            votes[prediction] = votes.get(prediction, 0) + 1
        majority = min(votes, key=lambda answer: (-votes[answer], answer)) if votes else ""
        tree_requests = [row for row in request_rows if row["tree_id"] == tree.tree_id]
        requests_by_node = {int(row["node_id"]): row for row in tree_requests}
        critical_path: tuple[float, list[int]] = (-1.0, [])
        for leaf in leaves:
            path_ids = []
            node_id: int | None = int(leaf["node_id"])
            while node_id is not None:
                path_ids.append(node_id)
                parent_id = tree.nodes[node_id].parent_id
                node_id = parent_id
            path_ids.reverse()
            path_s = sum(
                float(requests_by_node[node_id]["request_end_offset_s"])
                - float(requests_by_node[node_id]["ready_offset_s"])
                for node_id in path_ids
            )
            if path_s > critical_path[0]:
                critical_path = (path_s, path_ids)
        tree_rows.append(
            {
                "strategy": args.strategy,
                "tree_id": tree.tree_id,
                "dataset_index": tree.dataset_index,
                "shape": tree.shape,
                "node_count": len(tree.nodes),
                "leaf_count": len(leaves),
                "completion_latency_s": round(tree_latencies[tree.tree_id], 6),
                "critical_path_latency_s": round(max(0.0, critical_path[0]), 6),
                "critical_path_node_ids": json.dumps(critical_path[1]),
                "mean_request_latency_s": round(statistics.mean(row["latency_s"] for row in tree_requests), 6),
                "mean_client_semaphore_wait_s": round(
                    statistics.mean(row["client_semaphore_wait_s"] for row in tree_requests), 6
                ),
                "prompt_tokens": sum(row["prompt_tokens"] for row in tree_requests),
                "completion_tokens": sum(row["completion_tokens"] for row in tree_requests),
                "majority_prediction": majority,
                "expected_answer": tree.expected_answer,
                "majority_correct": bool(majority == tree.expected_answer),
                "any_leaf_correct": any(row["correct"] for row in leaves),
                "parsed_leaf_count": sum(bool(row["predicted_answer"]) for row in leaves),
                "inner_node_count": sum(not row["is_leaf"] for row in tree_requests),
                "unclosed_reasoning_leaf_count": sum(
                    row["answer_parse_status"] == "unclosed_reasoning" for row in leaves
                ),
                "length_finished_leaf_count": sum(
                    row["finish_reason"] == "length" for row in leaves
                ),
                "length_finished_inner_count": sum(
                    not row["is_leaf"] and row["finish_reason"] == "length"
                    for row in tree_requests
                ),
                "leaf_exact_match_rate": (
                    sum(bool(row["correct"]) for row in leaves) / len(leaves)
                    if leaves
                    else 0.0
                ),
            }
        )

    async with httpx.AsyncClient(timeout=10.0) as metrics_client:
        request_counts_by_endpoint = [
            sum(row["worker"] == worker for row in request_rows)
            for worker in range(len(endpoints))
        ]
        metrics_after = await asyncio.gather(
            *(
                metrics_snapshot_when_request_counts_complete(
                    metrics_client, endpoint, before, expected
                )
                for endpoint, before, expected in zip(
                    endpoints, metrics_before, request_counts_by_endpoint
                )
            )
        )
    prefix_cache_deltas = []
    phase_metric_deltas = []
    for endpoint, before, after in zip(endpoints, metrics_before, metrics_after):
        queries_before = metric_value(before, "vllm:prefix_cache_queries_total")
        queries_after = metric_value(after, "vllm:prefix_cache_queries_total")
        hits_before = metric_value(before, "vllm:prefix_cache_hits_total")
        hits_after = metric_value(after, "vllm:prefix_cache_hits_total")
        query_delta = (
            queries_after - queries_before
            if queries_before is not None and queries_after is not None
            else None
        )
        hit_delta = (
            hits_after - hits_before
            if hits_before is not None and hits_after is not None
            else None
        )
        prefix_cache_deltas.append(
            {
                "endpoint": endpoint,
                "query_delta": query_delta,
                "hit_delta": hit_delta,
                "token_hit_rate": hit_delta / query_delta if query_delta else None,
            }
        )
        phase_metric_deltas.append(
            {
                "endpoint": endpoint,
                "request_queue_time": metric_observation_delta(
                    before, after, "vllm:request_queue_time_seconds"
                ),
                "request_prefill_time": metric_observation_delta(
                    before, after, "vllm:request_prefill_time_seconds"
                ),
                "request_decode_time": metric_observation_delta(
                    before, after, "vllm:request_decode_time_seconds"
                ),
                "request_inference_time": metric_observation_delta(
                    before, after, "vllm:request_inference_time_seconds"
                ),
                "time_to_first_token": metric_observation_delta(
                    before, after, "vllm:time_to_first_token_seconds"
                ),
                "inter_token_latency": metric_observation_delta(
                    before, after, "vllm:inter_token_latency_seconds"
                ),
                "e2e_request_latency": metric_observation_delta(
                    before, after, "vllm:e2e_request_latency_seconds"
                ),
                "prefill_kv_computed_tokens": metric_delta(
                    before,
                    after,
                    "vllm:request_prefill_kv_computed_tokens_sum",
                ),
            }
        )

    latencies = [row["completion_latency_s"] for row in tree_rows]
    request_latencies = [row["latency_s"] for row in request_rows]
    queue_metric_sums = [
        item["request_queue_time"]["sum_s"] for item in phase_metric_deltas
    ]
    queue_metric_counts = [
        item["request_queue_time"]["count"] for item in phase_metric_deltas
    ]
    total_queue_metric_s = sum(value for value in queue_metric_sums if value is not None)
    total_queue_metric_count = sum(value for value in queue_metric_counts if value is not None)
    summary = {
        "strategy": args.strategy,
        "run_label": args.run_label,
        "prompt_trace_mode": "frozen" if prompt_trace is not None else "online_generated_history",
        "model": args.model_name,
        "workers": endpoints,
        "question_count": len(trees),
        "tree_count": len(trees),
        "selected_tree_ids": [tree.tree_id for tree in trees],
        "request_count": len(request_rows),
        "node_count_per_tree": {row["tree_id"]: row["node_count"] for row in tree_rows},
        "local_only_tree_homes": dict(sorted(router.tree_home.items())),
        "worker_request_counts": {
            worker: sum(row["worker"] == worker for row in request_rows)
            for worker in range(len(endpoints))
        },
        "estimated_cached_prefix_tokens_total": sum(
            row["estimated_cached_prefix_tokens"] for row in request_rows
        ),
        "requests_with_estimated_prefix_reuse": sum(
            row["estimated_cached_prefix_tokens"] > 0 for row in request_rows
        ),
        "max_tokens_inner": args.max_tokens_inner,
        "max_tokens_leaf": args.max_tokens_leaf,
        "temperature": 0,
        "max_concurrency_per_worker": args.max_concurrency,
        "router_capacity_per_worker": router.capacity,
        "router_prefill_tps_assumption": args.prefill_tps,
        "router_decode_tps_assumption": args.decode_tps,
        "router_prefill_tps_by_worker": prefill_tps_by_worker,
        "router_decode_tps_by_worker": decode_tps_by_worker,
        "service_profile_json": args.service_profile_json,
        "output_length_profile_json": args.output_length_profile_json,
        "output_text_recorded": not args.omit_output_text,
        "kv_block_size_assumption": args.block_size,
        "inference_wall_time_s": round(inference_wall_time_s or 0.0, 6),
        "tree_completion_p50_s": round(statistics.median(latencies), 6),
        "tree_completion_median_s": round(statistics.median(latencies), 6),
        "tree_completion_p95_s": round(percentile(latencies, 0.95), 6),
        "request_latency_p50_s": round(statistics.median(request_latencies), 6),
        "request_latency_p95_s": round(percentile(request_latencies, 0.95), 6),
        "mean_client_semaphore_wait_s": round(
            statistics.mean(row["client_semaphore_wait_s"] for row in request_rows), 6
        ),
        "majority_exact_match_rate": round(
            sum(row["majority_correct"] for row in tree_rows) / len(tree_rows), 6
        ),
        "any_leaf_exact_match_rate": round(
            sum(row["any_leaf_correct"] for row in tree_rows) / len(tree_rows), 6
        ),
        "leaf_answer_parse_rate": round(
            sum(row["parsed_leaf_count"] for row in tree_rows)
            / sum(row["leaf_count"] for row in tree_rows),
            6,
        ),
        "unclosed_reasoning_leaf_count": sum(
            row["unclosed_reasoning_leaf_count"] for row in tree_rows
        ),
        "length_finished_leaf_count": sum(
            row["length_finished_leaf_count"] for row in tree_rows
        ),
        "inner_node_count": sum(row["inner_node_count"] for row in tree_rows),
        "length_finished_inner_count": sum(
            row["length_finished_inner_count"] for row in tree_rows
        ),
        "length_finished_request_count": sum(
            row["finish_reason"] == "length" for row in request_rows
        ),
        "prefix_cache_metrics_before": metrics_before,
        "prefix_cache_metrics_after": metrics_after,
        "prefix_cache_metric_deltas": prefix_cache_deltas,
        "vllm_phase_metric_deltas": phase_metric_deltas,
        "vllm_mean_request_queue_time_s": (
            total_queue_metric_s / total_queue_metric_count
            if total_queue_metric_count
            else None
        ),
        "vllm_request_queue_observation_count": total_queue_metric_count,
    }
    return {"summary": summary, "requests": request_rows, "trees": tree_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_checkpoint_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


async def async_main(args: argparse.Namespace) -> None:
    if args.strategy not in POLICIES:
        raise ValueError(f"Unknown strategy {args.strategy!r}; choose from {', '.join(POLICIES)}")
    if len(args.workers) < 2:
        raise ValueError("The direction-1 placement pilot needs at least two serving replicas")
    if args.router_capacity is not None and args.router_capacity < 1:
        raise ValueError("--router-capacity must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    all_trees = load_trees(Path(args.dataset), args.questions, args.seed)
    trees = all_trees
    if args.tree_ids:
        requested_tree_ids = set(args.tree_ids)
        available_tree_ids = {tree.tree_id for tree in all_trees}
        missing_tree_ids = requested_tree_ids - available_tree_ids
        if missing_tree_ids:
            raise ValueError(f"Unknown --tree-ids: {sorted(missing_tree_ids)}")
        trees = [tree for tree in all_trees if tree.tree_id in requested_tree_ids]
        if not trees:
            raise ValueError("--tree-ids selected no trees")
    prompt_trace = None
    trace_digest = None
    if args.trace_json:
        trace_path = Path(args.trace_json)
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        if trace_data.get("format") != "prismserve-placement-prompt-trace-v1":
            raise ValueError("Unsupported prompt trace format")
        if (
            int(trace_data.get("dataset_seed", -1)) != args.seed
            or int(trace_data.get("question_count", -1)) != args.questions
            or trace_data.get("selected_dataset_indices")
            != [tree.dataset_index for tree in all_trees]
        ):
            raise ValueError("Prompt trace dataset selection does not match this run")
        prompt_trace = {
            (int(item["tree_id"]), int(item["node_id"])): item["messages"]
            for item in trace_data["nodes"]
        }
        expected_keys = {
            (tree.tree_id, node_id)
            for tree in all_trees
            for node_id in tree.nodes
        }
        if set(prompt_trace) != expected_keys:
            raise ValueError("Prompt trace node ids do not match the selected dataset trees")
        selected_ids = {tree.tree_id for tree in trees}
        prompt_trace = {
            key: messages for key, messages in prompt_trace.items() if key[0] in selected_ids
        }
        trace_digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_label = re.sub(r"[^A-Za-z0-9_-]+", "-", args.run_label).strip("-")
    label = f"_{run_label}" if run_label else ""
    prefix = f"REAL_PLACEMENT_{args.strategy}{label}_{stamp}"
    args._checkpoint_path = out / f"{prefix}_REQUESTS.partial.csv"
    result = await run_policy(args, trees, tokenizer, prompt_trace)
    write_csv(out / f"{prefix}_REQUESTS.csv", result["requests"])
    args._checkpoint_path.unlink(missing_ok=True)
    write_csv(out / f"{prefix}_TREES.csv", result["trees"])
    summary = result["summary"]
    summary["dataset_path"] = args.dataset
    summary["dataset_source"] = "https://github.com/openai/grade-school-math"
    summary["dataset_seed"] = args.seed
    summary["prompt_trace_path"] = args.trace_json or ""
    summary["prompt_trace_sha256"] = trace_digest or ""
    summary["selected_dataset_indices"] = [tree.dataset_index for tree in trees]
    summary["tree_shapes"] = {tree.tree_id: tree.shape for tree in trees}
    summary["note"] = (
        "Real vLLM inference on a synthetic chat-history branching workload over GSM8K questions. "
        "DeepSeek chat templating can remove or reformat think-marked assistant history, so this "
        "does not claim parent-output reasoning KV reuse. There is no cross-replica KV transfer; "
        "remote prefixes are recomputed. Prefix estimates use completed prompts only and do not "
        "model active sibling sharing, cache eviction, or exact generated token IDs. Reported "
        "vLLM prefix-cache metric deltas are token-level counters. "
        "Sibling-lookahead is a topology-aware heuristic, not an oracle. Descendant workloads "
        "can differ across placements; inspect recorded outputs and token counts. Scheduler "
        "service estimates use the explicit assumptions or calibration profile recorded above. "
        "With a small number of trees, p95 is the maximum observed latency and is exploratory."
    )
    (out / f"{prefix}_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=POLICIES)
    parser.add_argument("--workers", nargs="+", required=True, help="OpenAI API base URLs, one per replica")
    parser.add_argument("--model-name", default="deepseek-r1-distill-llama-70b")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--dataset", required=True, help="GSM8K test.jsonl path")
    parser.add_argument(
        "--trace-json",
        default="",
        help="Frozen node-prompt trace; use identical saved histories for policy replay",
    )
    parser.add_argument("--output-dir", default="idea-stage/pilots/direction1")
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-tokens-inner", type=int, default=1024)
    parser.add_argument("--max-tokens-leaf", type=int, default=1024)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--prefill-tps", type=float, default=25000.0)
    parser.add_argument("--decode-tps", type=float, default=60.0)
    parser.add_argument("--service-profile-json", default="")
    parser.add_argument("--output-length-profile-json", default="")
    parser.add_argument("--tree-ids", nargs="+", type=int, default=[])
    parser.add_argument("--router-capacity", type=int)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--omit-output-text", action="store_true")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
