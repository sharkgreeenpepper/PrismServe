#!/usr/bin/env python3
"""Compare paired QKV and unseen-work live-tree mechanism-smoke outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def semantic_routes(events: list[dict[str, Any]]) -> tuple[dict[str, int], set[str]]:
    path_by_id: dict[int, str] = {}
    for event in events:
        if event["event_type"] != "node_released":
            continue
        node_id = int(event["node_id"])
        parent_id = event.get("parent_id")
        if parent_id is None:
            path_by_id[node_id] = "root"
        else:
            parent_path = path_by_id[int(parent_id)]
            path_by_id[node_id] = f"{parent_path}/{event['action']}"
    routes = {
        path_by_id[int(event["node_id"])]: int(event["worker"])
        for event in events
        if event["event_type"] == "dispatch_decision"
    }
    return routes, set(path_by_id.values())


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def compare(
    qkv_events_dir: Path,
    proposed_events_dir: Path,
    qkv_run_id: str,
    proposed_run_id: str,
    qkv_trees_file: Path,
    proposed_trees_file: Path,
    indices: list[int],
    output: Path,
) -> dict[str, object]:
    qkv_rows = {int(row["dataset_index"]): row for row in read_jsonl(qkv_trees_file)}
    proposed_rows = {
        int(row["dataset_index"]): row for row in read_jsonl(proposed_trees_file)
    }
    pairs: list[dict[str, object]] = []
    qkv_latencies: list[float] = []
    proposed_latencies: list[float] = []
    counterfactual_changed_questions = 0
    cross_run_route_changed_questions = 0
    topology_equal_questions = 0
    missing_event_files: list[str] = []
    missing_summary_indices: list[int] = []
    for index in indices:
        if index not in qkv_rows or index not in proposed_rows:
            missing_summary_indices.append(index)
            continue
        qkv_path = qkv_events_dir / f"{qkv_run_id}-{index}_EVENTS.jsonl"
        proposed_path = proposed_events_dir / f"{proposed_run_id}-{index}_EVENTS.jsonl"
        if not qkv_path.is_file() or not proposed_path.is_file():
            missing_event_files.extend(str(path) for path in (qkv_path, proposed_path) if not path.is_file())
            continue
        qkv_events = read_jsonl(qkv_path)
        proposed_events = read_jsonl(proposed_path)
        qkv_routes, qkv_topology = semantic_routes(qkv_events)
        proposed_routes, proposed_topology = semantic_routes(proposed_events)
        shared = sorted(set(qkv_routes) & set(proposed_routes))
        changed_paths = [
            path for path in shared if qkv_routes[path] != proposed_routes[path]
        ]
        topology_equal = qkv_topology == proposed_topology
        if topology_equal:
            topology_equal_questions += 1
        if changed_paths and topology_equal:
            cross_run_route_changed_questions += 1
        changed_counterfactual_batches: dict[int, list[int]] = {}
        for event in proposed_events:
            if event["event_type"] != "dispatch_decision":
                continue
            baseline_worker = event.get("counterfactual_worker_same_snapshot")
            if baseline_worker is None:
                raise ValueError("proposed arm lacks same-snapshot routing counterfactual")
            if int(event["worker"]) != int(baseline_worker):
                batch_id = int(event["dispatch_batch_id"])
                changed_counterfactual_batches.setdefault(batch_id, []).append(
                    int(event["node_id"])
                )
        if changed_counterfactual_batches:
            counterfactual_changed_questions += 1
        qkv = qkv_rows[index]
        proposed = proposed_rows[index]
        qkv_latency = (
            float(qkv["latency_s"])
            if qkv["tree_status"] == "complete" and qkv.get("latency_s") is not None
            else None
        )
        proposed_latency = (
            float(proposed["latency_s"])
            if proposed["tree_status"] == "complete" and proposed.get("latency_s") is not None
            else None
        )
        if qkv_latency is not None:
            qkv_latencies.append(qkv_latency)
        if proposed_latency is not None:
            proposed_latencies.append(proposed_latency)
        pairs.append(
            {
                "dataset_index": index,
                "qkv_status": qkv["tree_status"],
                "proposed_status": proposed["tree_status"],
                "qkv_latency_s": qkv_latency,
                "proposed_latency_s": proposed_latency,
                "latency_delta_s": (
                    proposed_latency - qkv_latency
                    if qkv_latency is not None and proposed_latency is not None
                    else None
                ),
                "qkv_exact_match": qkv["exact_match"],
                "proposed_exact_match": proposed["exact_match"],
                "qkv_released_node_count": qkv["released_node_count"],
                "proposed_released_node_count": proposed["released_node_count"],
                "qkv_completed_node_count": qkv["completed_node_count"],
                "proposed_completed_node_count": proposed["completed_node_count"],
                "qkv_completion_tokens": qkv["completion_tokens"],
                "proposed_completion_tokens": proposed["completion_tokens"],
                "shared_semantic_paths": len(shared),
                "topology_equal": topology_equal,
                "changed_route_paths": changed_paths,
                "same_snapshot_changed_batches": [
                    {"batch_id": batch_id, "node_ids": node_ids}
                    for batch_id, node_ids in sorted(changed_counterfactual_batches.items())
                ],
                "qkv_only_paths": sorted(qkv_topology - proposed_topology),
                "proposed_only_paths": sorted(proposed_topology - qkv_topology),
                "qkv_metrics_complete": qkv["required_metrics_complete"],
                "proposed_metrics_complete": proposed["required_metrics_complete"],
            }
        )

    all_required_metrics_complete = all(
        bool(pair["qkv_metrics_complete"]) and bool(pair["proposed_metrics_complete"])
        for pair in pairs
    ) and len(pairs) == len(indices)
    all_trees_complete = len(pairs) == len(indices) and all(
        pair["qkv_status"] == "complete" and pair["proposed_status"] == "complete"
        for pair in pairs
    )
    route_change_gate_pass = counterfactual_changed_questions >= 4 and len(pairs) == len(indices)
    result: dict[str, object] = {
        "format": "prismserve-live-tree-arm-comparison-v1",
        "scope": "B1 mechanism screen only; direct gate uses same-snapshot counterfactuals, not cross-run routes",
        "paired_question_count": len(pairs),
        "requested_question_count": len(indices),
        "same_snapshot_route_changed_question_count": counterfactual_changed_questions,
        "cross_run_route_changed_question_count": cross_run_route_changed_questions,
        "topology_equal_question_count": topology_equal_questions,
        "route_change_gate_pass": route_change_gate_pass,
        "all_trees_complete": all_trees_complete,
        "qkv_tree_p50_s": statistics.median(qkv_latencies) if qkv_latencies else None,
        "qkv_tree_p95_s": nearest_rank(qkv_latencies, 0.95),
        "qkv_completed_tree_count": len(qkv_latencies),
        "proposed_tree_p50_s": statistics.median(proposed_latencies) if proposed_latencies else None,
        "proposed_tree_p95_s": nearest_rank(proposed_latencies, 0.95),
        "proposed_completed_tree_count": len(proposed_latencies),
        "qkv_exact_match_count": sum(bool(pair["qkv_exact_match"]) for pair in pairs),
        "proposed_exact_match_count": sum(bool(pair["proposed_exact_match"]) for pair in pairs),
        "all_required_metrics_complete": all_required_metrics_complete,
        "b1_screen_gate_pass": (
            route_change_gate_pass and all_trees_complete and all_required_metrics_complete
        ),
        "missing_event_files": missing_event_files,
        "missing_summary_indices": missing_summary_indices,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_indices(raw: str) -> list[int]:
    try:
        result = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("indices must be nonempty and unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv-events-dir", type=Path, required=True)
    parser.add_argument("--proposed-events-dir", type=Path, required=True)
    parser.add_argument("--qkv-run-id", required=True)
    parser.add_argument("--proposed-run-id", required=True)
    parser.add_argument("--qkv-trees-file", type=Path, required=True)
    parser.add_argument("--proposed-trees-file", type=Path, required=True)
    parser.add_argument("--indices", type=parse_indices, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.qkv_events_dir,
                args.proposed_events_dir,
                args.qkv_run_id,
                args.proposed_run_id,
                args.qkv_trees_file,
                args.proposed_trees_file,
                args.indices,
                args.output,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
