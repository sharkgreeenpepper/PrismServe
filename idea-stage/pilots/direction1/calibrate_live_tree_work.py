#!/usr/bin/env python3
"""Fit simple held-out scaling factors for live-tree self-estimated work."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_events(paths: list[Path]) -> tuple[list[list[dict[str, Any]]], list[int]]:
    trees = []
    dataset_indices = []
    for path in paths:
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not events or events[-1]["event_type"] != "tree_completed":
            raise ValueError(f"calibration source is not a complete tree: {path}")
        indices = {int(event["dataset_index"]) for event in events if "dataset_index" in event}
        if len(indices) != 1:
            raise ValueError(f"calibration events must contain one dataset_index: {path}")
        trees.append(events)
        dataset_indices.extend(indices)
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("calibration event files contain duplicate dataset indices")
    return trees, sorted(dataset_indices)


def fit_scale(predictions: list[float], actuals: list[float], name: str) -> float:
    denominator = sum(value * value for value in predictions)
    if denominator <= 0:
        raise ValueError(f"calibration predictions for {name} are all zero")
    scale = sum(predicted * actual for predicted, actual in zip(predictions, actuals)) / denominator
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"invalid fitted scale for {name}: {scale}")
    return scale


def rmse(predictions: list[float], actuals: list[float]) -> float:
    if not predictions:
        return 0.0
    return math.sqrt(
        sum((prediction - actual) ** 2 for prediction, actual in zip(predictions, actuals))
        / len(predictions)
    )


def calibrate(paths: list[Path], run_config_path: Path, output: Path) -> dict[str, object]:
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    trees, event_indices = load_events(paths)
    if not trees:
        raise ValueError("no complete calibration trees found")
    configured_indices = sorted(int(value) for value in run_config.get("dataset_indices", []))
    if event_indices != configured_indices:
        raise ValueError(
            f"calibration config/event index mismatch: {configured_indices} vs {event_indices}"
        )
    predicted_nodes: list[float] = []
    actual_nodes: list[float] = []
    predicted_tokens: list[float] = []
    actual_tokens: list[float] = []
    observations = 0
    for events in trees:
        parent_by_node = {
            int(event["node_id"]): (
                None if event.get("parent_id") is None else int(event["parent_id"])
            )
            for event in events
            if event["event_type"] == "node_released"
        }
        predicted_by_node = {
            int(event["node_id"]): (
                float(event["predicted_remaining_nodes"]),
                float(event["predicted_remaining_tokens"]),
            )
            for event in events
            if event["event_type"] == "dispatch_decision"
        }
        tokens_by_node = {
            int(event["node_id"]): int(event.get("completion_tokens") or 0)
            for event in events
            if event["event_type"] == "node_completed"
        }
        for node_id, (node_prediction, token_prediction) in predicted_by_node.items():
            if node_id not in parent_by_node or parent_by_node[node_id] is None:
                # The root has no parent-produced estimate available before dispatch.
                continue
            descendants = []
            for candidate in parent_by_node:
                parent = parent_by_node[candidate]
                visited: set[int] = set()
                while parent is not None:
                    if parent not in parent_by_node or parent in visited:
                        raise ValueError(f"invalid parent chain in {events[0]['tree_id']}")
                    visited.add(parent)
                    if parent == node_id:
                        descendants.append(candidate)
                        break
                    parent = parent_by_node[parent]
            predicted_nodes.append(node_prediction)
            actual_nodes.append(float(len(descendants)))
            predicted_tokens.append(token_prediction)
            actual_tokens.append(float(sum(tokens_by_node.get(node, 0) for node in descendants)))
            observations += 1

    node_scale = fit_scale(predicted_nodes, actual_nodes, "descendant node count")
    token_scale = fit_scale(predicted_tokens, actual_tokens, "descendant tokens")
    calibrated_nodes = [value * node_scale for value in predicted_nodes]
    calibrated_tokens = [value * token_scale for value in predicted_tokens]
    result: dict[str, object] = {
        "format": "prismserve-live-tree-work-calibration-v1",
        "source_event_files": [path.name for path in paths],
        "source_run_config": run_config_path.name,
        "dataset_id": run_config["dataset_id"],
        "dataset_file_sha256": run_config["dataset_file_sha256"],
        "dataset_indices": event_indices,
        "complete_tree_count": len(trees),
        "node_observation_count": observations,
        "node_scale": node_scale,
        "token_scale": token_scale,
        "node_rmse_before": rmse(predicted_nodes, actual_nodes),
        "node_rmse_after": rmse(calibrated_nodes, actual_nodes),
        "token_rmse_before": rmse(predicted_tokens, actual_tokens),
        "token_rmse_after": rmse(calibrated_tokens, actual_tokens),
        "mean_actual_descendants": statistics.mean(actual_nodes),
        "mean_actual_descendant_tokens": statistics.mean(actual_tokens),
        "scope": "scaling fit on separate calibration questions; not a test-set result",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.events, args.run_config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
