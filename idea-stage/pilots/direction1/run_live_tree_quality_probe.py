#!/usr/bin/env python3
"""Run a direct-answer quality probe against the same vLLM model endpoints.

This diagnostic checks whether low live-tree answer quality is specific to the
tree protocol. It is not a latency comparison or a replacement for the official
GSM8K evaluation pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any


def normalize_answer(raw: str) -> str:
    value = raw.strip().replace(",", "").replace("$", "")
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def expected_answer(answer_text: str) -> str:
    if "####" not in answer_text:
        raise ValueError("dataset answer is missing GSM8K #### marker")
    return normalize_answer(answer_text.rsplit("####", 1)[1].strip().splitlines()[0])


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of empty input")
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("dataset rows must be JSON objects")
                rows.append(row)
    return rows


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    endpoints = [value.strip().rstrip("/") for value in args.endpoints.split(",") if value.strip()]
    indices = [int(value.strip()) for value in args.indices.split(",") if value.strip()]
    if not endpoints or not indices or len(indices) != len(set(indices)):
        raise ValueError("provide endpoints and unique dataset indices")
    rows = load_rows(args.dataset_jsonl)
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError("dataset index is out of range")
    dataset_sha256 = hashlib.sha256(args.dataset_jsonl.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hosts = []
    for endpoint in endpoints:
        parsed = urlsplit(endpoint)
        hosts.append(f"{parsed.scheme}://{parsed.hostname or ''}" + (f":{parsed.port}" if parsed.port else ""))

    config = {
        "run_id": args.run_id,
        "evaluation_type": "real_gt_direct_answer_quality_probe",
        "dataset_id": args.dataset_id,
        "dataset_file_name": args.dataset_jsonl.name,
        "dataset_file_size_bytes": args.dataset_jsonl.stat().st_size,
        "dataset_file_sha256": dataset_sha256,
        "dataset_indices": indices,
        "endpoint_hosts": hosts,
        "model_name": args.model_name,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "response_format": "json_schema",
        "batch_concurrency": args.concurrency,
        "latency_claim": False,
    }
    (args.output_dir / f"{args.run_id}_CONFIG.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    system_prompt = (
        "Solve the math word problem and return exactly one JSON object with "
        '"status":"final" and "answer" set to the final numeric answer. '
        "Do not return a plan or child branches. Do not use outside information."
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    clients = [httpx.AsyncClient(timeout=args.timeout_s) for _ in endpoints]

    async def one(position: int, dataset_index: int) -> dict[str, Any]:
        row = rows[dataset_index]
        prompt = f"Question: {row['question']}\nSolve this problem and provide the final numeric answer."
        worker = position % len(endpoints)
        started = time.perf_counter()
        async with semaphore:
            try:
                response = await clients[worker].post(
                    endpoints[worker] + "/v1/chat/completions",
                    json={
                        "model": args.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "gsm8k_final_answer",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "enum": ["final"]},
                                        "answer": {"type": "number"},
                                    },
                                    "required": ["status", "answer"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                    },
                )
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                text = choice.get("message", {}).get("content") or ""
                decoded = json.loads(text)
                prediction = None
                if isinstance(decoded, dict) and decoded.get("status") == "final" and decoded.get("answer") is not None:
                    prediction = normalize_answer(str(decoded["answer"]))
                usage = payload.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                metrics = payload.get("metrics") or {}
                return {
                    "dataset_index": dataset_index,
                    "worker": worker,
                    "request_id": payload.get("id"),
                    "question": row["question"],
                    "completion_text": text,
                    "finish_reason": choice.get("finish_reason"),
                    "latency_s": time.perf_counter() - started,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_prompt_tokens": details.get("cached_tokens"),
                    "queue_time_ms": metrics.get("queue_time_ms"),
                    "time_to_first_token_ms": metrics.get("time_to_first_token_ms"),
                    "generation_time_ms": metrics.get("generation_time_ms"),
                    "predicted_answer": prediction,
                    "expected_answer": expected_answer(str(row["answer"])),
                    "exact_match": prediction == expected_answer(str(row["answer"])),
                    "error": None,
                }
            except Exception as exc:
                return {
                    "dataset_index": dataset_index,
                    "worker": worker,
                    "latency_s": time.perf_counter() - started,
                    "predicted_answer": None,
                    "expected_answer": expected_answer(str(row["answer"])),
                    "exact_match": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }

    try:
        records = await asyncio.gather(*(one(position, index) for position, index in enumerate(indices)))
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))

    records.sort(key=lambda item: indices.index(int(item["dataset_index"])))
    records_path = args.output_dir / f"{args.run_id}_RECORDS.jsonl"
    records_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8"
    )
    latencies = [float(item["latency_s"]) for item in records if item.get("error") is None]
    summary = {
        "run_id": args.run_id,
        "dataset_id": args.dataset_id,
        "dataset_indices": indices,
        "question_count": len(records),
        "successful_request_count": len(latencies),
        "failed_request_count": sum(item.get("error") is not None for item in records),
        "exact_match_count": sum(bool(item["exact_match"]) for item in records),
        "tree_or_search_quality_reference_only": True,
        "latency_claim": False,
        "request_latency_p50_s": statistics.median(latencies) if latencies else None,
        "request_latency_p95_s": percentile(latencies, 0.95) if latencies else None,
        "records_file": records_path.name,
    }
    summary_path = args.output_dir / f"{args.run_id}_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=180)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
