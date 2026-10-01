#!/usr/bin/env python3
"""Measure per-replica prefill/decode rates using short, isolated trace prompts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx


METRICS = (
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
)


async def scrape(client: httpx.AsyncClient, endpoint: str) -> list[str]:
    response = await client.get(endpoint.rstrip("/") + "/metrics")
    response.raise_for_status()
    return [
        line
        for line in response.text.splitlines()
        if line and not line.startswith("#") and line.startswith(METRICS)
    ]


def total(lines: list[str], name: str) -> float | None:
    values = []
    for line in lines:
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                values.append(float(line.rsplit(maxsplit=1)[1]))
            except (IndexError, ValueError):
                continue
    return sum(values) if values else None


def observation(before: list[str], after: list[str], name: str) -> dict[str, float | None]:
    sum_before = total(before, name + "_sum")
    sum_after = total(after, name + "_sum")
    count_before = total(before, name + "_count")
    count_after = total(after, name + "_count")
    elapsed = None if sum_before is None or sum_after is None else max(0.0, sum_after - sum_before)
    count = None if count_before is None or count_after is None else max(0.0, count_after - count_before)
    return {"sum": elapsed, "count": count, "mean": elapsed / count if elapsed is not None and count else None}


async def reset_cache(client: httpx.AsyncClient, endpoint: str) -> None:
    response = await client.post(endpoint.rstrip("/") + "/reset_prefix_cache", timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("success") is not True:
        raise RuntimeError(f"prefix cache reset failed at {endpoint}: {payload!r}")


async def scrape_when_request_counts_complete(
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
        after = await scrape(client, endpoint)
        counts = {
            name: observation(before, after, name)["count"]
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


def select_prompts(trace: dict[str, Any], tree_ids: set[int]) -> list[dict[str, Any]]:
    nodes_by_tree: dict[int, list[dict[str, Any]]] = {}
    for node in trace["nodes"]:
        tree_id = int(node["tree_id"])
        if tree_id in tree_ids:
            nodes_by_tree.setdefault(tree_id, []).append(node)
    selected = []
    for tree_id in sorted(tree_ids):
        nodes = nodes_by_tree.get(tree_id, [])
        if not nodes:
            raise ValueError(f"Calibration tree {tree_id} is absent from the prompt trace")
        shortest_prompt = min(nodes, key=lambda node: int(node["source_prompt_tokens"]))
        longest_prompt = max(nodes, key=lambda node: int(node["source_prompt_tokens"]))
        selected.extend([shortest_prompt, longest_prompt])
    return selected


async def calibrate_worker(
    endpoint: str,
    prompts: list[dict[str, Any]],
    model_name: str,
    max_tokens: int,
) -> dict[str, Any]:
    requests = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        before = await scrape(client, endpoint)
        for node in prompts:
            await reset_cache(client, endpoint)
            started = time.perf_counter()
            response = await client.post(
                endpoint.rstrip("/") + "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": node["messages"],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - started
            result = response.json()
            choice = result["choices"][0]
            usage = result.get("usage") or {}
            requests.append(
                {
                    "tree_id": int(node["tree_id"]),
                    "node_id": int(node["node_id"]),
                    "trace_prompt_tokens": int(node["source_prompt_tokens"]),
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "finish_reason": choice.get("finish_reason", "unknown"),
                    "http_latency_s": round(elapsed, 6),
                }
            )
        after = await scrape_when_request_counts_complete(
            client, endpoint, before, len(requests)
        )

    queue = observation(before, after, "vllm:request_queue_time_seconds")
    prefill = observation(before, after, "vllm:request_prefill_time_seconds")
    decode = observation(before, after, "vllm:request_decode_time_seconds")
    prefill_tokens_before = total(
        before, "vllm:request_prefill_kv_computed_tokens_sum"
    )
    prefill_tokens_after = total(
        after, "vllm:request_prefill_kv_computed_tokens_sum"
    )
    computed_tokens = (
        None
        if prefill_tokens_before is None or prefill_tokens_after is None
        else max(0.0, prefill_tokens_after - prefill_tokens_before)
    )
    completion_tokens = sum(row["completion_tokens"] for row in requests)
    prefill_seconds = prefill["sum"]
    decode_seconds = decode["sum"]
    if not prefill_seconds or not decode_seconds or not computed_tokens:
        raise RuntimeError(
            f"Missing calibration counters at {endpoint}: "
            f"prefill_tokens={computed_tokens}, prefill_s={prefill_seconds}, decode_s={decode_seconds}"
        )
    prefill_tps = computed_tokens / prefill_seconds
    decode_tps = completion_tokens / decode_seconds
    if not all(math.isfinite(value) and value > 0 for value in (prefill_tps, decode_tps)):
        raise RuntimeError(f"Invalid service rates at {endpoint}: {prefill_tps}, {decode_tps}")
    return {
        "endpoint": endpoint,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "calibration_request_count": len(requests),
        "calibration_completion_tokens": completion_tokens,
        "calibration_prefill_kv_tokens": computed_tokens,
        "queue_time": queue,
        "prefill_time": prefill,
        "decode_time": decode,
        "request_inference_time": observation(
            before, after, "vllm:request_inference_time_seconds"
        ),
        "time_to_first_token": observation(
            before, after, "vllm:time_to_first_token_seconds"
        ),
        "inter_token_latency": observation(
            before, after, "vllm:inter_token_latency_seconds"
        ),
        "requests": requests,
        "metrics_before": before,
        "metrics_after": after,
    }


async def async_main(args: argparse.Namespace) -> None:
    trace_path = Path(args.trace_json)
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes)
    prompts = select_prompts(trace, set(args.tree_ids))
    workers = []
    workers = await asyncio.gather(
        *(
            calibrate_worker(endpoint, prompts, args.model_name, args.max_tokens)
            for endpoint in args.workers
        )
    )
    result = {
        "model": args.model_name,
        "trace_json": str(trace_path),
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "calibration_tree_ids": sorted(set(args.tree_ids)),
        "calibration_max_tokens": args.max_tokens,
        "service_rates_source": "vLLM request prefill/decode phase metrics from isolated requests",
        "workers": workers,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--tree-ids", nargs="+", type=int, required=True)
    parser.add_argument("--model-name", default="deepseek-r1-distill-llama-70b")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
