#!/usr/bin/env python3
"""Run one live, model-expanded reasoning tree per input question.

The queue/KV arm ignores model work estimates. The proposed arm adds only the
unexpanded-work estimate emitted with a completed parent's child proposals.
Gold answers are loaded by the offline evaluator and are never passed to the
driver, adapter, expansion policy, or router.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from live_tree_contract import (
    JsonExpansionPolicy,
    LiveTreeDriver,
    OpenAICompatibleVllmAdapter,
    RouteDecision,
    RouteObservation,
)


SYSTEM_PROMPT = """You are solving a math problem by exploring a small reasoning tree.
Return exactly one JSON object and no Markdown. If the current path has a
complete answer, return {\"status\":\"final\",\"answer\":\"number\",\"children\":[]}.
Otherwise return {\"status\":\"expand\",\"answer\":\"\",\"children\":[...]}
with at most the configured fanout. Every child object must contain a distinct short
\"action\", \"predicted_remaining_nodes\" (expected number of future nodes
beyond that child's first request), and \"predicted_remaining_tokens\"
(expected generated tokens in those future descendants). These are estimates,
not commitments. Use only the question and reasoning shown in the current
prompt. Do not invent or use a reference answer."""


def parse_csv_values(raw: str, cast: type, field: str) -> list[Any]:
    try:
        values = [cast(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {field}: {raw}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{field} must contain at least one value")
    return values


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


def parse_final_answer(completion_text: str) -> str | None:
    raw = completion_text.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("status") == "final":
        value = payload.get("answer")
        return normalize_answer(str(value)) if value is not None else None
    return None


class QueueKvRouter:
    """Greedy queue/service estimate with parent-prefix locality heuristic."""

    def __init__(
        self,
        strategy: str,
        prefill_tps: list[float],
        decode_tps: list[float],
        max_tokens: int,
        mean_node_service_s: float,
        prefix_fraction: float,
        descendant_node_scale: float = 1.0,
        descendant_token_scale: float = 1.0,
    ) -> None:
        if strategy not in {"qkv", "unseen-work"}:
            raise ValueError(f"unknown routing strategy: {strategy}")
        if len(prefill_tps) != len(decode_tps):
            raise ValueError("prefill and decode rates must match worker count")
        if any(not math.isfinite(rate) or rate <= 0 for rate in [*prefill_tps, *decode_tps]):
            raise ValueError("all service rates must be positive")
        if not 0 <= prefix_fraction <= 1:
            raise ValueError("prefix fraction must be between 0 and 1")
        self.strategy = strategy
        self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps
        self.max_tokens = max_tokens
        self.mean_node_service_s = mean_node_service_s
        self.prefix_fraction = prefix_fraction
        self.descendant_node_scale = descendant_node_scale
        self.descendant_token_scale = descendant_token_scale
        if max_tokens < 1 or not math.isfinite(mean_node_service_s) or mean_node_service_s <= 0:
            raise ValueError("max_tokens and mean_node_service_s must be positive")
        if any(
            not math.isfinite(value) or value < 0
            for value in (descendant_node_scale, descendant_token_scale)
        ):
            raise ValueError("descendant calibration scales must be finite and nonnegative")
        self.worker_by_node: dict[tuple[str, int], int] = {}

    def _costs(
        self,
        observation: RouteObservation,
        worker: int,
        include_descendant_work: bool,
    ) -> tuple[float, float, float, int, int]:
        node = observation.node
        prompt_tokens = max(1, len(node.prompt.encode("utf-8")) // 4)
        parent_worker = (
            self.worker_by_node.get((node.tree_id, node.parent_id))
            if node.parent_id is not None
            else None
        )
        cached = int(prompt_tokens * self.prefix_fraction) if parent_worker == worker else 0
        inflight = observation.replica_inflight[worker]
        queue_cost = inflight * self.mean_node_service_s
        current_cost = (
            max(0, prompt_tokens - cached) / self.prefill_tps[worker]
            + self.max_tokens / self.decode_tps[worker]
        )
        descendant_cost = 0.0
        if include_descendant_work:
            descendant_cost = (
                node.predicted_remaining_nodes
                * self.descendant_node_scale
                * self.mean_node_service_s
                + node.predicted_remaining_tokens
                * self.descendant_token_scale
                / self.decode_tps[worker]
            )
        return queue_cost, current_cost, descendant_cost, cached, prompt_tokens

    def choose_batch(self, observations: tuple[RouteObservation, ...]) -> tuple[RouteDecision, ...]:
        if not observations:
            return ()
        workers = len(observations[0].replica_inflight)
        base_inflight = observations[0].replica_inflight
        if any(item.replica_inflight != base_inflight for item in observations):
            raise ValueError("all nodes in a ready batch must share one queue snapshot")

        def optimize(include_descendant_work: bool) -> tuple[tuple[int, ...], float]:
            best_key: tuple[float, float, tuple[int, ...]] | None = None
            best_assignment: tuple[int, ...] | None = None
            for assignment in itertools.product(range(workers), repeat=len(observations)):
                projected = [inflight * self.mean_node_service_s for inflight in base_inflight]
                for observation, worker in zip(observations, assignment):
                    _, current, descendant, _, _ = self._costs(
                        observation, worker, include_descendant_work
                    )
                    projected[worker] += current + descendant
                key = (max(projected, default=0.0), sum(projected), assignment)
                if best_key is None or key < best_key:
                    best_key = key
                    best_assignment = assignment
            assert best_assignment is not None and best_key is not None
            return best_assignment, best_key[0]

        baseline_assignment, baseline_peak = optimize(include_descendant_work=False)
        if self.strategy == "unseen-work":
            best_assignment, best_peak = optimize(include_descendant_work=True)
        else:
            best_assignment, best_peak = baseline_assignment, baseline_peak

        decisions: list[RouteDecision] = []
        for position, (observation, worker) in enumerate(zip(observations, best_assignment)):
            costs = [
                self._costs(
                    observation,
                    candidate,
                    include_descendant_work=self.strategy == "unseen-work",
                )
                for candidate in range(workers)
            ]
            queue_cost, current_cost, descendant_cost, cached, prompt_tokens = costs[worker]
            score_by_worker = tuple(
                queue + current + descendant
                for queue, current, descendant, _, _ in costs
            )
            self.worker_by_node[(observation.node.tree_id, observation.node.node_id)] = worker
            decisions.append(
                RouteDecision(
                    worker=worker,
                    counterfactual_worker_same_snapshot=baseline_assignment[position],
                    score_by_worker_s=score_by_worker,
                    queue_cost_s=queue_cost,
                    current_service_cost_s=current_cost,
                    predicted_descendant_cost_s=descendant_cost,
                    estimated_cached_prefix_tokens=cached,
                    estimated_prompt_tokens=prompt_tokens,
                    group_size=len(observations),
                    projected_group_peak_s=best_peak,
                )
            )
        return tuple(decisions)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise ValueError(f"invalid GSM8K JSONL row at line {line_number}")
            if not isinstance(row.get("answer"), str):
                raise ValueError(f"missing answer at line {line_number}")
            rows.append(row)
    return rows


async def run(args: argparse.Namespace) -> dict[str, object]:
    endpoints = [value.strip() for value in args.endpoints.split(",") if value.strip()]
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    predictor_node_scale = 0.0
    predictor_token_scale = 0.0
    calibration: dict[str, Any] | None = None
    calibration_sha256: str | None = None
    if args.strategy == "unseen-work":
        if args.predictor_calibration_json is None:
            raise ValueError("unseen-work strategy requires --predictor-calibration-json")
        calibration = json.loads(args.predictor_calibration_json.read_text(encoding="utf-8"))
        calibration_sha256 = hashlib.sha256(
            args.predictor_calibration_json.read_bytes()
        ).hexdigest()
        if calibration.get("format") != "prismserve-live-tree-work-calibration-v1":
            raise ValueError("unsupported predictor calibration format")
        predictor_node_scale = float(calibration["node_scale"])
        predictor_token_scale = float(calibration["token_scale"])
        if any(
            not math.isfinite(value) or value < 0
            for value in (predictor_node_scale, predictor_token_scale)
        ):
            raise ValueError("predictor calibration scales must be finite and nonnegative")
    prefill_tps = parse_csv_values(args.prefill_tps, float, "prefill-tps")
    decode_tps = parse_csv_values(args.decode_tps, float, "decode-tps")
    if len(prefill_tps) == 1:
        prefill_tps *= len(endpoints)
    if len(decode_tps) == 1:
        decode_tps *= len(endpoints)
    if len(prefill_tps) != len(endpoints) or len(decode_tps) != len(endpoints):
        raise ValueError("provide one service rate or one rate per endpoint")

    rows = load_rows(args.dataset_jsonl)
    if args.indices:
        indices = parse_csv_values(args.indices, int, "indices")
    else:
        if args.count < 1 or len(rows) < args.count:
            raise ValueError("dataset has fewer rows than requested count")
        indices = list(range(min(args.count, len(rows))))
    if len(indices) != len(set(indices)):
        raise ValueError("selected dataset indices must be unique")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError("selected dataset index is out of range")
    if not indices:
        raise ValueError("select at least one dataset row")
    dataset_sha256 = hashlib.sha256(args.dataset_jsonl.read_bytes()).hexdigest()
    calibration_indices: list[int] = []
    if calibration is not None:
        if calibration.get("dataset_id") != args.dataset_id:
            raise ValueError("calibration and test dataset IDs differ")
        if calibration.get("dataset_file_sha256") != dataset_sha256:
            raise ValueError("calibration and test dataset file hashes differ")
        calibration_indices = [int(value) for value in calibration.get("dataset_indices", [])]
        if not calibration_indices or len(calibration_indices) != len(set(calibration_indices)):
            raise ValueError("calibration must record unique source dataset indices")
        overlap = sorted(set(calibration_indices) & set(indices))
        if overlap:
            raise ValueError(f"calibration/test index leakage: {overlap}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.run_id}_TREES.jsonl").write_text("", encoding="utf-8")
    endpoint_hosts = []
    for endpoint in endpoints:
        parsed = urlsplit(endpoint)
        endpoint_hosts.append(
            f"{parsed.scheme}://{parsed.hostname or ''}"
            + (f":{parsed.port}" if parsed.port else "")
        )
    run_config = {
        "run_id": args.run_id,
        "strategy": args.strategy,
        "dataset_id": args.dataset_id,
        "dataset_file_path": str(args.dataset_jsonl.resolve()),
        "dataset_file_name": args.dataset_jsonl.name,
        "dataset_file_size_bytes": args.dataset_jsonl.stat().st_size,
        "dataset_file_sha256": dataset_sha256,
        "dataset_indices": indices,
        "endpoint_hosts": endpoint_hosts,
        "model_name": args.model_name,
        "system_prompt_version": "live-tree-json-schema-estimates-v2",
        "response_format": "json_schema_strict",
        "max_nodes": args.max_nodes,
        "max_depth": args.max_depth,
        "max_fanout": args.max_fanout,
        "max_inflight": args.max_inflight,
        "max_tokens": args.max_tokens,
        "max_predicted_nodes": args.max_nodes,
        "max_predicted_tokens": args.max_nodes * args.max_tokens,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "mean_node_service_s": args.mean_node_service_s,
        "prefix_fraction": args.prefix_fraction,
        "predictor_calibration_file": (
            args.predictor_calibration_json.name if args.predictor_calibration_json else None
        ),
        "predictor_calibration_sha256": calibration_sha256,
        "predictor_node_scale": predictor_node_scale,
        "predictor_token_scale": predictor_token_scale,
        "predictor_calibration_dataset_id": (
            calibration.get("dataset_id") if calibration is not None else None
        ),
        "predictor_calibration_dataset_sha256": (
            calibration.get("dataset_file_sha256") if calibration is not None else None
        ),
        "predictor_calibration_indices": calibration_indices,
        "prompt_token_estimator": "utf8_bytes_div_4",
        "batch_objective": "minimize_max_projected_worker_load",
    }
    (output_dir / f"{args.run_id}_CONFIG.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    adapter = OpenAICompatibleVllmAdapter(
        endpoints=endpoints,
        model=args.model_name,
        system_prompt=SYSTEM_PROMPT.replace("configured fanout", str(args.max_fanout)),
        max_tokens=args.max_tokens,
        max_fanout=args.max_fanout,
        timeout_s=args.timeout_s,
    )
    summaries: list[dict[str, object]] = []
    all_latencies: list[float] = []
    try:
        for dataset_index in indices:
            row = rows[dataset_index]
            tree_id = f"{args.run_id}-{dataset_index}"
            router = QueueKvRouter(
                strategy=args.strategy,
                prefill_tps=prefill_tps,
                decode_tps=decode_tps,
                max_tokens=args.max_tokens,
                mean_node_service_s=args.mean_node_service_s,
                prefix_fraction=args.prefix_fraction,
                descendant_node_scale=predictor_node_scale,
                descendant_token_scale=predictor_token_scale,
            )
            driver = LiveTreeDriver(
                tree_id=tree_id,
                adapter=adapter,
                expansion_policy=JsonExpansionPolicy(
                    args.max_fanout,
                    max_predicted_nodes=float(args.max_nodes),
                    max_predicted_tokens=float(args.max_nodes * args.max_tokens),
                ),
                router=router,
                worker_count=len(endpoints),
                max_inflight=args.max_inflight,
                max_nodes=args.max_nodes,
                max_depth=args.max_depth,
            )
            root_prompt = (
                f"Question: {row['question']}\n\n"
                "Solve this problem. At this point, produce either a final answer "
                "or the next distinct reasoning actions using the required JSON format."
            )
            events = await driver.run(root_prompt)
            for event in events:
                event["dataset_index"] = dataset_index
            event_path = output_dir / f"{tree_id}_EVENTS.jsonl"
            event_path.write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
                encoding="utf-8",
            )
            node_outputs = {
                int(event["node_id"]): str(event["completion_text"])
                for event in events
                if event["event_type"] == "node_completed"
            }
            answers = [
                answer
                for answer in (parse_final_answer(text) for text in node_outputs.values())
                if answer is not None
            ]
            counts: dict[str, int] = {}
            for answer in answers:
                counts[answer] = counts.get(answer, 0) + 1
            majority = (
                min(counts, key=lambda value: (-counts[value], value))
                if counts
                else None
            )
            terminal_elapsed_s = float(events[-1]["offset_s"])
            tree_status = "complete" if events[-1]["event_type"] == "tree_completed" else "failed"
            latency_s = terminal_elapsed_s if tree_status == "complete" else None
            if latency_s is not None:
                all_latencies.append(latency_s)
            released_nodes = sum(event["event_type"] == "node_released" for event in events)
            completed_nodes = sum(event["event_type"] == "node_completed" for event in events)
            failed_nodes = sum(event["event_type"] == "node_failed" for event in events)
            cancelled_nodes = sum(event["event_type"] == "node_cancelled" for event in events)
            aborted_nodes = sum(
                event["event_type"] == "node_aborted_before_dispatch" for event in events
            )
            token_count = sum(
                int(event["completion_tokens"] or 0)
                for event in events
                if event["event_type"] == "node_completed"
            )
            required_fields = (
                "queue_time_ms",
                "time_to_first_token_ms",
                "generation_time_ms",
                "cached_prompt_tokens",
            )
            completion_events = [
                event for event in events if event["event_type"] == "node_completed"
            ]
            missing_metrics = sorted(
                {
                    field
                    for event in completion_events
                    for field in required_fields
                    if event.get(field) is None
                }
            )
            exact_match = majority == expected_answer(row["answer"]) and tree_status == "complete"
            summary = {
                "run_id": args.run_id,
                "strategy": args.strategy,
                "dataset_id": args.dataset_id,
                "tree_id": tree_id,
                "dataset_index": dataset_index,
                "tree_status": tree_status,
                "latency_s": latency_s,
                "terminal_elapsed_s": terminal_elapsed_s,
                "released_node_count": released_nodes,
                "completed_node_count": completed_nodes,
                "failed_node_count": failed_nodes,
                "cancelled_node_count": cancelled_nodes,
                "aborted_before_dispatch_node_count": aborted_nodes,
                "terminal_answer_count": len(answers),
                "predicted_answer": majority,
                "expected_answer": expected_answer(row["answer"]),
                "exact_match": exact_match,
                "required_metrics_complete": not missing_metrics and bool(completion_events),
                "missing_metric_fields": missing_metrics,
                "completion_tokens": token_count,
                "event_file": event_path.name,
            }
            summaries.append(summary)
            with (output_dir / f"{args.run_id}_TREES.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
                stream.flush()
            print(json.dumps(summary, ensure_ascii=False))
            if tree_status == "failed":
                break
    finally:
        await adapter.close()

    summaries_path = output_dir / f"{args.run_id}_TREES.jsonl"
    ordered = sorted(all_latencies)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    result = {
        "run_id": args.run_id,
        "strategy": args.strategy,
        "dataset_id": args.dataset_id,
        "dataset_file_name": args.dataset_jsonl.name,
        "dataset_indices": indices,
        "run_config_file": f"{args.run_id}_CONFIG.json",
        "question_count": len(summaries),
        "tree_p50_s": statistics.median(all_latencies) if all_latencies else None,
        "tree_p95_s": p95,
        "completed_tree_count": len(all_latencies),
        "failed_tree_count": sum(row["tree_status"] == "failed" for row in summaries),
        "exact_match_count": sum(bool(row["exact_match"]) for row in summaries),
        "released_node_count": sum(int(row["released_node_count"]) for row in summaries),
        "completed_node_count": sum(int(row["completed_node_count"]) for row in summaries),
        "failed_node_count": sum(int(row["failed_node_count"]) for row in summaries),
        "cancelled_node_count": sum(int(row["cancelled_node_count"]) for row in summaries),
        "aborted_before_dispatch_node_count": sum(
            int(row["aborted_before_dispatch_node_count"]) for row in summaries
        ),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in summaries),
        "tree_results_file": summaries_path.name,
    }
    summary_path = output_dir / f"{args.run_id}_SUMMARY.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True, help="versioned dataset/split identifier")
    parser.add_argument("--indices", default="")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--endpoints", required=True, help="comma-separated vLLM base URLs")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--strategy", choices=("qkv", "unseen-work"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=9)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-fanout", type=int, default=3)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prefill-tps", required=True)
    parser.add_argument("--decode-tps", required=True)
    parser.add_argument("--mean-node-service-s", type=float, required=True)
    parser.add_argument("--prefix-fraction", type=float, required=True)
    parser.add_argument("--predictor-calibration-json", type=Path)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    print(json.dumps(asyncio.run(run(parsed_args)), ensure_ascii=False))
