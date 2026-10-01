#!/usr/bin/env python3
"""Freeze a completed placement run's exact chat prompts for policy replay."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vllm_placement_pilot import (  # noqa: E402
    TreePlan,
    chat_messages_for_root,
    load_trees,
    runtime_node,
)


def build_trace(
    requests_csv: Path,
    dataset: Path,
    tokenizer_dir: Path,
    output_path: Path,
    question_count: int,
    seed: int,
    max_tokens_inner: int,
    max_tokens_leaf: int,
    strict_prompt_token_counts: bool = False,
    expected_dataset_indices: list[int] | None = None,
) -> None:
    with requests_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    responses = {
        (int(row["tree_id"]), int(row["node_id"])): row
        for row in rows
    }
    trees = load_trees(dataset, question_count, seed)
    selected_indices = [tree.dataset_index for tree in trees]
    if expected_dataset_indices is not None and selected_indices != expected_dataset_indices:
        raise ValueError(
            "Sample selection differs from the frozen cohort: "
            f"expected={expected_dataset_indices}, actual={selected_indices}"
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    trace_nodes: list[dict[str, Any]] = []
    token_count_mismatches: list[dict[str, int]] = []

    for tree in trees:
        def visit(node_id: int, history: list[dict[str, str]]) -> None:
            row = responses.get((tree.tree_id, node_id))
            if row is None:
                raise ValueError(f"Missing source response for tree {tree.tree_id}, node {node_id}")
            node = runtime_node(
                tree,
                node_id,
                history,
                max_tokens_inner,
                max_tokens_leaf,
            )
            encoded = tokenizer.apply_chat_template(
                node.messages, tokenize=True, add_generation_prompt=True
            )
            token_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            local_prompt_tokens = len(token_ids)
            server_prompt_tokens = int(row["prompt_tokens"])
            if local_prompt_tokens != server_prompt_tokens:
                token_count_mismatches.append(
                    {
                        "tree_id": tree.tree_id,
                        "node_id": node_id,
                        "local_prompt_tokens": local_prompt_tokens,
                        "server_prompt_tokens": server_prompt_tokens,
                    }
                )
            trace_nodes.append(
                {
                    "tree_id": tree.tree_id,
                    "node_id": node_id,
                    "messages": node.messages,
                    "source_prompt_tokens": server_prompt_tokens,
                    "trace_prompt_tokens": local_prompt_tokens,
                }
            )
            for child_id in tree.nodes[node_id].children:
                child_history = node.messages + [
                    {"role": "assistant", "content": row["content"]}
                ]
                visit(child_id, child_history)

        visit(0, chat_messages_for_root(tree))

    expected_count = sum(len(tree.nodes) for tree in trees)
    if len(trace_nodes) != expected_count:
        raise ValueError(f"Trace has {len(trace_nodes)} nodes; expected {expected_count}")
    if strict_prompt_token_counts and token_count_mismatches:
        raise ValueError(
            "Frozen prompt token counts differ from the source run: "
            f"{token_count_mismatches[:5]}"
        )
    payload = {
        "format": "prismserve-placement-prompt-trace-v1",
        "source_requests_csv": str(requests_csv),
        "dataset_path": str(dataset),
        "dataset_seed": seed,
        "question_count": question_count,
        "selected_dataset_indices": [tree.dataset_index for tree in trees],
        "tree_shapes": {str(tree.tree_id): tree.shape for tree in trees},
        "max_tokens_inner": max_tokens_inner,
        "max_tokens_leaf": max_tokens_leaf,
        "node_count": expected_count,
        "prompt_token_count_mismatches": token_count_mismatches,
        "nodes": trace_nodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "trace_path": str(output_path),
                "node_count": expected_count,
                "prompt_token_count_mismatches": len(token_count_mismatches),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-csv", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-tokens-inner", type=int, default=1024)
    parser.add_argument("--max-tokens-leaf", type=int, default=1024)
    parser.add_argument("--strict-prompt-token-counts", action="store_true")
    parser.add_argument("--expected-dataset-indices", nargs="+", type=int)
    args = parser.parse_args()
    build_trace(
        args.requests_csv,
        args.dataset,
        args.tokenizer_dir,
        args.output,
        args.questions,
        args.seed,
        args.max_tokens_inner,
        args.max_tokens_leaf,
        args.strict_prompt_token_counts,
        args.expected_dataset_indices,
    )


if __name__ == "__main__":
    main()
