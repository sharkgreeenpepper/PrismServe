#!/usr/bin/env python3
"""Zero-GPU fixture showing that calibrated branch estimates can change a batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from live_tree_contract import ReleasedNode, RouteObservation
from run_live_tree_smoke import QueueKvRouter


def run_audit() -> dict[str, object]:
    # Same prompt and parent placement; only predicted descendant work differs.
    nodes = tuple(
        ReleasedNode(
            tree_id="routing-fixture",
            node_id=index + 1,
            parent_id=0,
            depth=1,
            prompt="same synthetic branch prompt",
            predicted_remaining_nodes=remaining_nodes,
            predicted_remaining_tokens=remaining_nodes * 40,
        )
        for index, remaining_nodes in enumerate((0.0, 10.0, 0.0))
    )
    observations = tuple(RouteObservation(node, (0, 0)) for node in nodes)
    assignments: dict[str, list[int]] = {}
    same_snapshot_baselines: dict[str, list[int]] = {}
    peaks: dict[str, float] = {}
    for strategy in ("qkv", "unseen-work"):
        router = QueueKvRouter(
            strategy=strategy,
            prefill_tps=[2000.0, 2000.0],
            decode_tps=[40.0, 40.0],
            max_tokens=256,
            mean_node_service_s=8.0,
            prefix_fraction=0.5,
            descendant_node_scale=1.0,
            descendant_token_scale=1.0,
        )
        router.worker_by_node[("routing-fixture", 0)] = 0
        decisions = router.choose_batch(observations)
        assignments[strategy] = [decision.worker for decision in decisions]
        same_snapshot_baselines[strategy] = [
            decision.counterfactual_worker_same_snapshot for decision in decisions
        ]
        peaks[strategy] = decisions[0].projected_group_peak_s
    changed = assignments["qkv"] != assignments["unseen-work"]
    if not changed:
        raise AssertionError("unseen-work fixture failed to change the joint assignment")
    return {
        "status": "B0_ROUTING_FIXTURE_PASS",
        "evidence_scope": "synthetic mechanism fixture only; no model or performance evidence",
        "predicted_remaining_nodes": [0, 10, 0],
        "qkv_assignment": assignments["qkv"],
        "unseen_work_assignment": assignments["unseen-work"],
        "unseen_work_same_snapshot_qkv_counterfactual": same_snapshot_baselines[
            "unseen-work"
        ],
        "projected_group_peak_s": peaks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
