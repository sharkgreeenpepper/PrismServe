#!/usr/bin/env python3
"""Derive a role-level output-length prior from calibration-tree request rows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--tree-ids", required=True, nargs="+", type=int)
    args = parser.parse_args()

    selected = set(args.tree_ids)
    values: dict[str, list[int]] = {"inner": [], "leaf": []}
    with args.requests_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["tree_id"]) not in selected:
                continue
            if row["finish_reason"] == "length":
                continue
            role = "leaf" if row["is_leaf"].lower() == "true" else "inner"
            values[role].append(int(row["completion_tokens"]))

    missing = [role for role, observations in values.items() if not observations]
    if missing:
        raise SystemExit(f"No uncapped calibration observations for roles: {missing}")

    profile = {
        "source_requests_csv": str(args.requests_csv),
        "calibration_tree_ids": sorted(selected),
        "excluded_finish_reason": "length",
        "expected_output_tokens": {
            role: statistics.median(observations)
            for role, observations in values.items()
        },
        "observations": {
            role: {
                "count": len(observations),
                "median_tokens": statistics.median(observations),
                "mean_tokens": round(statistics.mean(observations), 3),
                "min_tokens": min(observations),
                "max_tokens": max(observations),
            }
            for role, observations in values.items()
        },
        "note": (
            "Role medians from held-out tree IDs in the earlier local-only frozen replay. "
            "This is an exploratory workload prior, not an independently validated predictor."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
