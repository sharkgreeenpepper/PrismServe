#!/usr/bin/env python3
"""Freeze a completed placement run's exact chat prompts for policy replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_tree(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    temporary.replace(path)


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
    dataset_id: str = "gsm8k:test",
    cohort_id: str = "",
    source_run_id: str = "",
    manifest_output: Path | None = None,
) -> None:
    requests_bytes = requests_csv.read_bytes()
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
    dataset_bytes = dataset.read_bytes()
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
        "dataset_id": dataset_id,
        "dataset_sha256": sha256_bytes(dataset_bytes),
        "dataset_record_count": sum(1 for line in dataset_bytes.splitlines() if line.strip()),
        "source_run_id": source_run_id,
        "cohort_id": cohort_id,
        "source_requests_sha256": sha256_bytes(requests_bytes),
        "tokenizer_sha256": sha256_tree(tokenizer_dir),
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
    trace_bytes = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_atomic(output_path, trace_bytes)
    manifest = {
        "format": "prismserve-direction1-cohort-manifest-v1",
        "status": "TRACE_FROZEN",
        "cohort_id": cohort_id,
        "parent_run_id": cohort_id,
        "dataset_id": dataset_id,
        "dataset_sha256": payload["dataset_sha256"],
        "dataset_record_count": payload["dataset_record_count"],
        "seed": seed,
        "question_count": question_count,
        "selected_dataset_indices": selected_indices,
        "tree_shapes": payload["tree_shapes"],
        "node_count": expected_count,
        "source_run_id": source_run_id,
        "source_requests_sha256": payload["source_requests_sha256"],
        "tokenizer_sha256": payload["tokenizer_sha256"],
        "max_tokens_inner": max_tokens_inner,
        "max_tokens_leaf": max_tokens_leaf,
        "prompt_token_count_mismatches": token_count_mismatches,
        "trace_file_sha256": sha256_bytes(trace_bytes),
        "trace_canonical_sha256": canonical_sha256(payload),
    }
    manifest["trace_config_sha256"] = canonical_sha256(
        {
            key: manifest[key]
            for key in (
                "cohort_id",
                "dataset_id",
                "dataset_sha256",
                "dataset_record_count",
                "seed",
                "question_count",
                "selected_dataset_indices",
                "tree_shapes",
                "tokenizer_sha256",
                "max_tokens_inner",
                "max_tokens_leaf",
            )
        }
    )
    manifest_path = manifest_output or output_path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {**existing_manifest, **manifest}
    write_atomic(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "trace_file_sha256": manifest["trace_file_sha256"],
                "trace_canonical_sha256": manifest["trace_canonical_sha256"],
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
    parser.add_argument("--dataset-id", default="gsm8k:test")
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--manifest-output", type=Path)
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
        args.dataset_id,
        args.cohort_id,
        args.source_run_id,
        args.manifest_output,
    )


if __name__ == "__main__":
    main()
