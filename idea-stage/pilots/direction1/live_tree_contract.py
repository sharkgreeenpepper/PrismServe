#!/usr/bin/env python3
"""Minimal parent-first live-tree driver and zero-GPU contract fixture.

The router receives only a released node and currently observed replica queue
depths. Child proposals are parsed from a model completion after that node has
completed; they are never part of the router-facing objects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ReleasedNode:
    tree_id: str
    node_id: int
    parent_id: int | None
    depth: int
    prompt: str
    action: str = "root"
    predicted_remaining_nodes: float = 0.0
    predicted_remaining_tokens: float = 0.0
    force_final: bool = False


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None
    queue_time_ms: float | None = None
    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    cached_prompt_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class RouteObservation:
    node: ReleasedNode
    replica_inflight: tuple[int, ...]


@dataclass(frozen=True)
class RouteDecision:
    worker: int
    counterfactual_worker_same_snapshot: int
    score_by_worker_s: tuple[float, ...]
    queue_cost_s: float
    current_service_cost_s: float
    predicted_descendant_cost_s: float
    estimated_cached_prefix_tokens: int
    estimated_prompt_tokens: int
    group_size: int
    projected_group_peak_s: float


@dataclass(frozen=True)
class Expansion:
    children: tuple["ChildProposal", ...]
    stop_reason: str


@dataclass(frozen=True)
class ChildProposal:
    action: str
    predicted_remaining_nodes: float
    predicted_remaining_tokens: float


class ModelAdapter(Protocol):
    async def generate(
        self, worker: int, prompt: str, *, force_final: bool = False
    ) -> GenerationResult: ...


class ExpansionPolicy(Protocol):
    def expand(self, node: ReleasedNode, completion: GenerationResult) -> Expansion: ...


class Router(Protocol):
    def choose_batch(self, observations: tuple[RouteObservation, ...]) -> tuple[RouteDecision, ...]: ...


class EventLog:
    def __init__(self, tree_id: str) -> None:
        self.tree_id = tree_id
        self.started = time.perf_counter()
        self.events: list[dict[str, object]] = []

    def emit(self, event_type: str, **payload: object) -> None:
        self.events.append(
            {
                "event_seq": len(self.events),
                "tree_id": self.tree_id,
                "event_type": event_type,
                "offset_s": round(time.perf_counter() - self.started, 9),
                **payload,
            }
        )


class LiveTreeDriver:
    """Run one tree while keeping future topology outside the router API."""

    def __init__(
        self,
        tree_id: str,
        adapter: ModelAdapter,
        expansion_policy: ExpansionPolicy,
        router: Router,
        worker_count: int,
        max_inflight: int,
        max_nodes: int,
        max_depth: int = 32,
        terminal_reserve_nodes: int = 0,
    ) -> None:
        if (
            worker_count < 1
            or max_inflight < 1
            or max_nodes < 1
            or max_depth < 1
            or terminal_reserve_nodes < 0
            or terminal_reserve_nodes >= max_nodes
        ):
            raise ValueError("worker_count, max_inflight, max_nodes, and max_depth must be positive")
        self.tree_id = tree_id
        self.adapter = adapter
        self.expansion_policy = expansion_policy
        self.router = router
        self.worker_count = worker_count
        self.max_inflight = max_inflight
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.terminal_reserve_nodes = terminal_reserve_nodes
        self.log = EventLog(tree_id)

    async def run(self, root_prompt: str) -> list[dict[str, object]]:
        root = ReleasedNode(self.tree_id, 0, None, 0, root_prompt, "root")
        self.log.emit(
            "node_released",
            node_id=0,
            parent_id=None,
            depth=0,
            action="root",
            reason="root",
            force_final=False,
        )
        ready = [root]
        replica_inflight = [0] * self.worker_count
        pending: dict[asyncio.Task[GenerationResult], tuple[ReleasedNode, int]] = {}
        next_node_id = 1
        completed: set[int] = set()
        failed_nodes: list[int] = []
        cancelled_nodes: list[int] = []
        aborted_before_dispatch_nodes: list[int] = []
        batch_id = 0

        while ready or pending:
            while ready and len(pending) < self.max_inflight:
                batch_size = min(len(ready), self.max_inflight - len(pending))
                batch = ready[:batch_size]
                del ready[:batch_size]
                observations = tuple(
                    RouteObservation(node, tuple(replica_inflight)) for node in batch
                )
                decisions = self.router.choose_batch(observations)
                batch_id += 1
                if len(decisions) != len(batch):
                    raise ValueError("router must return one route decision per ready node")
                for node, observation, decision in zip(batch, observations, decisions):
                    worker = decision.worker
                    if not 0 <= worker < self.worker_count:
                        raise ValueError(f"router selected invalid worker {worker}")
                    self.log.emit(
                        "dispatch_decision",
                        node_id=node.node_id,
                        dispatch_batch_id=batch_id,
                        parent_id=node.parent_id,
                        depth=node.depth,
                        action=node.action,
                        worker=worker,
                        counterfactual_worker_same_snapshot=(
                            decision.counterfactual_worker_same_snapshot
                        ),
                        predicted_remaining_nodes=node.predicted_remaining_nodes,
                        predicted_remaining_tokens=node.predicted_remaining_tokens,
                        replica_inflight=list(observation.replica_inflight),
                        score_by_worker_s=list(decision.score_by_worker_s),
                        predicted_queue_cost_s=decision.queue_cost_s,
                        predicted_current_service_cost_s=decision.current_service_cost_s,
                        predicted_descendant_cost_s=decision.predicted_descendant_cost_s,
                        estimated_cached_prefix_tokens=decision.estimated_cached_prefix_tokens,
                        estimated_prompt_tokens=decision.estimated_prompt_tokens,
                        group_size=decision.group_size,
                        projected_group_peak_s=decision.projected_group_peak_s,
                        force_final=node.force_final,
                    )
                    replica_inflight[worker] += 1
                    mode_directive = (
                        "\n\nMODE=FINALIZE. Solve the original problem using the full visible "
                        "context. Return a nonempty final answer and no children. Do not continue search."
                        if node.force_final
                        else "\n\nMODE=SEARCH. Return a final answer if the complete problem is solved; "
                        "otherwise propose only useful next reasoning actions."
                    )
                    task = asyncio.create_task(
                        self.adapter.generate(
                            worker,
                            node.prompt + mode_directive,
                            force_final=node.force_final,
                        )
                    )
                    pending[task] = (node, worker)

            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # Sort simultaneous completions to make fixture event logs stable.
            for task in sorted(done, key=lambda item: pending[item][0].node_id):
                node, worker = pending.pop(task)
                replica_inflight[worker] -= 1
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    cancelled_nodes.append(node.node_id)
                    self.log.emit(
                        "node_cancelled",
                        node_id=node.node_id,
                        parent_id=node.parent_id,
                        worker=worker,
                        action=node.action,
                        reason="sibling_request_failed",
                    )
                    continue
                except Exception as exc:
                    failed_nodes.append(node.node_id)
                    self.log.emit(
                        "node_failed",
                        node_id=node.node_id,
                        parent_id=node.parent_id,
                        depth=node.depth,
                        action=node.action,
                        worker=worker,
                        error_type=type(exc).__name__,
                        error=str(exc)[:1000],
                    )
                    continue
                completed.add(node.node_id)
                self.log.emit(
                    "node_completed",
                    node_id=node.node_id,
                    parent_id=node.parent_id,
                    depth=node.depth,
                    action=node.action,
                    worker=worker,
                    completion_text=result.content,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    force_final=node.force_final,
                    request_id=result.request_id,
                    queue_time_ms=result.queue_time_ms,
                    time_to_first_token_ms=result.time_to_first_token_ms,
                    generation_time_ms=result.generation_time_ms,
                    cached_prompt_tokens=result.cached_prompt_tokens,
                    request_latency_s=result.latency_s,
                    finish_reason=result.finish_reason,
                )

                # This is the first point at which the search policy may reveal
                # children. All resulting release events occur after completion.
                expansion = self.expansion_policy.expand(node, result)
                self.log.emit(
                    "expansion_decision",
                    node_id=node.node_id,
                    offered_children=[
                        {
                            "action": child.action,
                            "predicted_remaining_nodes": child.predicted_remaining_nodes,
                            "predicted_remaining_tokens": child.predicted_remaining_tokens,
                        }
                        for child in expansion.children
                    ],
                    stop_reason=expansion.stop_reason,
                )
                if not expansion.children:
                    self.log.emit(
                        "branch_stopped",
                        node_id=node.node_id,
                        reason=expansion.stop_reason,
                    )
                for proposal in expansion.children:
                    child_label = proposal.action
                    if node.depth >= self.max_depth:
                        self.log.emit(
                            "branch_pruned",
                            parent_id=node.node_id,
                            child_label=child_label,
                            reason="max_depth",
                        )
                        continue
                    if next_node_id >= self.max_nodes:
                        self.log.emit(
                            "branch_pruned",
                            parent_id=node.node_id,
                            child_label=child_label,
                            reason="max_nodes",
                        )
                        continue
                    child_force_final = (
                        node.depth + 1 >= self.max_depth
                        or next_node_id >= self.max_nodes - self.terminal_reserve_nodes
                    )
                    child = ReleasedNode(
                        tree_id=self.tree_id,
                        node_id=next_node_id,
                        parent_id=node.node_id,
                        depth=node.depth + 1,
                        prompt=(
                            f"{node.prompt}\n\nCompleted parent output:\n{result.content}"
                            f"\n\nContinue using proposal: {child_label}"
                        ),
                        action=child_label,
                        predicted_remaining_nodes=(
                            0.0 if child_force_final else proposal.predicted_remaining_nodes
                        ),
                        predicted_remaining_tokens=(
                            0.0 if child_force_final else proposal.predicted_remaining_tokens
                        ),
                        force_final=child_force_final,
                    )
                    next_node_id += 1
                    self.log.emit(
                        "node_released",
                        node_id=child.node_id,
                        parent_id=child.parent_id,
                        depth=child.depth,
                        action=child.action,
                        reason="parent_completed",
                        force_final=child.force_final,
                    )
                    ready.append(child)

            if failed_nodes:
                for node in ready:
                    aborted_before_dispatch_nodes.append(node.node_id)
                    self.log.emit(
                        "node_aborted_before_dispatch",
                        node_id=node.node_id,
                        parent_id=node.parent_id,
                        depth=node.depth,
                        action=node.action,
                        reason="sibling_request_failed",
                    )
                ready.clear()
                remaining = list(pending.items())
                for task, _ in remaining:
                    task.cancel()
                results = await asyncio.gather(
                    *(task for task, _ in remaining), return_exceptions=True
                )
                for (task, (node, worker)), result in zip(remaining, results):
                    pending.pop(task, None)
                    replica_inflight[worker] -= 1
                    if isinstance(result, GenerationResult):
                        completed.add(node.node_id)
                        self.log.emit(
                            "node_completed",
                            node_id=node.node_id,
                            parent_id=node.parent_id,
                            depth=node.depth,
                            action=node.action,
                            worker=worker,
                            completion_text=result.content,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                            force_final=node.force_final,
                            request_id=result.request_id,
                            queue_time_ms=result.queue_time_ms,
                            time_to_first_token_ms=result.time_to_first_token_ms,
                            generation_time_ms=result.generation_time_ms,
                            cached_prompt_tokens=result.cached_prompt_tokens,
                            request_latency_s=result.latency_s,
                            finish_reason=result.finish_reason,
                            expansion_skipped=True,
                        )
                    elif isinstance(result, asyncio.CancelledError):
                        cancelled_nodes.append(node.node_id)
                        self.log.emit(
                            "node_cancelled",
                            node_id=node.node_id,
                            parent_id=node.parent_id,
                            worker=worker,
                            action=node.action,
                            reason="sibling_request_failed",
                        )
                    else:
                        failed_nodes.append(node.node_id)
                        self.log.emit(
                            "node_failed",
                            node_id=node.node_id,
                            parent_id=node.parent_id,
                            depth=node.depth,
                            action=node.action,
                            worker=worker,
                            error_type=type(result).__name__,
                            error=str(result)[:1000],
                            during_abort=True,
                        )
                self.log.emit(
                    "tree_failed",
                    released_node_count=next_node_id,
                    completed_node_count=len(completed),
                    failed_node_count=len(set(failed_nodes)),
                    cancelled_node_count=len(set(cancelled_nodes)),
                    aborted_before_dispatch_node_count=len(
                        set(aborted_before_dispatch_nodes)
                    ),
                    completed_node_ids=sorted(completed),
                    failed_node_ids=sorted(set(failed_nodes)),
                    cancelled_node_ids=sorted(set(cancelled_nodes)),
                    aborted_before_dispatch_node_ids=sorted(
                        set(aborted_before_dispatch_nodes)
                    ),
                )
                return self.log.events

        self.log.emit(
            "tree_completed",
            released_node_count=next_node_id,
            completed_node_count=len(completed),
            failed_node_count=0,
            cancelled_node_count=0,
            aborted_before_dispatch_node_count=0,
            completed_node_ids=sorted(completed),
            status="complete",
        )
        return self.log.events


class FixtureAdapter:
    """Deterministic stand-in for the model; never supplies topology up front."""

    async def generate(
        self, worker: int, prompt: str, *, force_final: bool = False
    ) -> GenerationResult:
        del worker
        await asyncio.sleep(0)
        if force_final:
            payload = {"status": "final", "answer": "42", "children": []}
            return GenerationResult(json.dumps(payload), prompt_tokens=20, completion_tokens=8)
        if "Continue using proposal: left-deep" in prompt:
            payload = {
                "status": "expand",
                "children": [
                    {
                        "action": "over-budget",
                        "predicted_remaining_nodes": 1,
                        "predicted_remaining_tokens": 32,
                    }
                ],
            }
        elif "Continue using proposal: left" in prompt:
            payload = {
                "status": "expand",
                "children": [
                    {
                        "action": "left-deep",
                        "predicted_remaining_nodes": 1.5,
                        "predicted_remaining_tokens": 48,
                    }
                ],
            }
        elif "Continue using proposal: right" in prompt:
            payload = {"status": "final", "answer": "42", "children": []}
        else:
            payload = {
                "status": "expand",
                "children": [
                    {
                        "action": "left",
                        "predicted_remaining_nodes": 2,
                        "predicted_remaining_tokens": 64,
                    },
                    {
                        "action": "right",
                        "predicted_remaining_nodes": 0,
                        "predicted_remaining_tokens": 0,
                    },
                ],
            }
        return GenerationResult(json.dumps(payload), prompt_tokens=20, completion_tokens=8)


class FixtureExpansionPolicy:
    def expand(self, node: ReleasedNode, completion: GenerationResult) -> Expansion:
        return JsonExpansionPolicy(max_fanout=2).expand(node, completion)


class FailureFixtureAdapter:
    async def generate(
        self, worker: int, prompt: str, *, force_final: bool = False
    ) -> GenerationResult:
        del worker
        if force_final:
            return GenerationResult('{"status":"final","answer":"0","children":[]}')
        if "Continue using proposal: failing" in prompt:
            raise RuntimeError("fixture request failure")
        if "Continue using proposal: sibling" in prompt:
            await asyncio.sleep(1.0)
            return GenerationResult('{"status":"final","answer":"0"}')
        payload = {
            "status": "expand",
            "children": [
                {
                    "action": "failing",
                    "predicted_remaining_nodes": 0,
                    "predicted_remaining_tokens": 0,
                },
                {
                    "action": "sibling",
                    "predicted_remaining_nodes": 0,
                    "predicted_remaining_tokens": 0,
                },
            ],
        }
        return GenerationResult(json.dumps(payload))


class JsonExpansionPolicy:
    """Parse model-produced branch proposals only after node completion."""

    def __init__(
        self,
        max_fanout: int,
        max_predicted_nodes: float = 10000.0,
        max_predicted_tokens: float = 10000000.0,
    ) -> None:
        if max_fanout < 1 or max_predicted_nodes < 0 or max_predicted_tokens < 0:
            raise ValueError("fanout must be positive and prediction caps nonnegative")
        self.max_fanout = max_fanout
        self.max_predicted_nodes = max_predicted_nodes
        self.max_predicted_tokens = max_predicted_tokens

    def expand(self, node: ReleasedNode, completion: GenerationResult) -> Expansion:
        raw = completion.content.strip()
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return Expansion((), "malformed_json")
        if not isinstance(payload, dict):
            return Expansion((), "invalid_payload")
        status = payload.get("status")
        if status == "final":
            answer = payload.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                return Expansion((), "invalid_final_answer")
            return Expansion((), "model_terminal")
        if node.force_final:
            return Expansion((), "finalization_required")
        if status != "expand":
            return Expansion((), "invalid_status")
        values = payload.get("children")
        if not isinstance(values, list) or len(values) > self.max_fanout:
            return Expansion((), "invalid_fanout")
        children: list[ChildProposal] = []
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("action"), str):
                return Expansion((), "invalid_child")
            try:
                nodes = float(value.get("predicted_remaining_nodes", 0))
                tokens = float(value.get("predicted_remaining_tokens", 0))
            except (TypeError, ValueError):
                return Expansion((), "invalid_prediction")
            if not math.isfinite(nodes) or not math.isfinite(tokens) or nodes < 0 or tokens < 0:
                return Expansion((), "invalid_prediction")
            action = value["action"].strip()
            if not action or any(child.action == action for child in children):
                return Expansion((), "invalid_or_duplicate_action")
            children.append(
                ChildProposal(
                    action,
                    min(nodes, self.max_predicted_nodes),
                    min(tokens, self.max_predicted_tokens),
                )
            )
        if not children:
            return Expansion((), "empty_expand")
        return Expansion(tuple(children), "model_expand")


class OpenAICompatibleVllmAdapter:
    """Async adapter for vLLM OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        endpoints: list[str],
        model: str,
        system_prompt: str,
        max_tokens: int,
        max_fanout: int = 3,
        timeout_s: float = 180.0,
    ) -> None:
        import httpx

        self.endpoints = endpoints
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        if max_fanout < 1:
            raise ValueError("max_fanout must be positive")
        self.max_fanout = max_fanout
        self.clients = [httpx.AsyncClient(timeout=timeout_s) for _ in endpoints]

    def _response_format(self, force_final: bool = False) -> dict[str, Any]:
        child_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "predicted_remaining_nodes": {"type": "number"},
                "predicted_remaining_tokens": {"type": "number"},
            },
            "required": [
                "action",
                "predicted_remaining_nodes",
                "predicted_remaining_tokens",
            ],
            "additionalProperties": False,
        }
        common = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["final", "expand"]},
                "answer": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": child_schema,
                    "maxItems": 0 if force_final else self.max_fanout,
                },
            },
            "required": ["status", "answer", "children"],
            "additionalProperties": False,
        }
        if force_final:
            common["properties"]["status"]["enum"] = ["final"]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "live_reasoning_tree_node",
                "strict": True,
                "schema": common,
            },
        }

    async def generate(
        self, worker: int, prompt: str, *, force_final: bool = False
    ) -> GenerationResult:
        if not 0 <= worker < len(self.endpoints):
            raise ValueError(f"invalid worker index {worker}")
        started = time.perf_counter()
        response = await self.clients[worker].post(
            self.endpoints[worker].rstrip("/") + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": self._response_format(force_final=force_final),
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"worker {worker} returned HTTP {response.status_code}: {response.text[:1500]}"
            )
        payload: dict[str, Any] = response.json()
        choice = payload["choices"][0]
        content: Any = choice.get("message", {}).get("content") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        metrics = payload.get("metrics") or {}

        def number(value: Any) -> float | None:
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        def integer(value: Any) -> int | None:
            try:
                return None if value is None else int(value)
            except (TypeError, ValueError):
                return None

        return GenerationResult(
            content=str(content),
            prompt_tokens=integer(usage.get("prompt_tokens")),
            completion_tokens=integer(usage.get("completion_tokens")),
            request_id=payload.get("id"),
            queue_time_ms=number(metrics.get("queue_time_ms")),
            time_to_first_token_ms=number(metrics.get("time_to_first_token_ms")),
            generation_time_ms=number(metrics.get("generation_time_ms")),
            cached_prompt_tokens=integer(details.get("cached_tokens")),
            latency_s=time.perf_counter() - started,
            finish_reason=choice.get("finish_reason"),
        )

    async def close(self) -> None:
        await asyncio.gather(*(client.aclose() for client in self.clients))


class RoundRobinFixtureRouter:
    def __init__(self) -> None:
        self.observations: list[RouteObservation] = []

    def choose_batch(self, observations: tuple[RouteObservation, ...]) -> tuple[RouteDecision, ...]:
        self.observations.extend(observations)
        if not observations:
            return ()
        inflight = list(observations[0].replica_inflight)
        decisions = []
        for observation in observations:
            worker = (observation.node.node_id + observation.node.depth) % len(inflight)
            decisions.append(
                RouteDecision(
                    worker=worker,
                    counterfactual_worker_same_snapshot=worker,
                    score_by_worker_s=tuple(float(value) for value in inflight),
                    queue_cost_s=float(inflight[worker]),
                    current_service_cost_s=0.0,
                    predicted_descendant_cost_s=0.0,
                    estimated_cached_prefix_tokens=0,
                    estimated_prompt_tokens=max(1, len(observation.node.prompt) // 4),
                    group_size=len(observations),
                    projected_group_peak_s=float(max(inflight) + 1),
                )
            )
            inflight[worker] += 1
        return tuple(decisions)


def audit_fixture(events: list[dict[str, object]], router: RoundRobinFixtureRouter) -> dict[str, int]:
    sequences = [int(event["event_seq"]) for event in events]
    assert sequences == list(range(len(events))), "event sequence is not contiguous"
    released_ids = [
        int(event["node_id"])
        for event in events
        if event["event_type"] == "node_released"
    ]
    completed_ids = [
        int(event["node_id"])
        for event in events
        if event["event_type"] == "node_completed"
    ]
    release_seq = {
        int(event["node_id"]): int(event["event_seq"])
        for event in events
        if event["event_type"] == "node_released"
    }
    completion_seq = {
        int(event["node_id"]): int(event["event_seq"])
        for event in events
        if event["event_type"] == "node_completed"
    }
    assert len(released_ids) == len(set(released_ids)), "duplicate node IDs"
    assert len(completed_ids) == len(set(completed_ids)), "duplicate completion IDs"
    for event in events:
        if event["event_type"] == "node_released" and event["parent_id"] is not None:
            parent = int(event["parent_id"])
            assert parent in completion_seq, "child was released before parent completed"
            assert completion_seq[parent] < int(event["event_seq"])
        if event["event_type"] == "dispatch_decision":
            node_id = int(event["node_id"])
            assert release_seq[node_id] < int(event["event_seq"])
            assert node_id not in completion_seq or completion_seq[node_id] > int(event["event_seq"])
        if event["event_type"] == "expansion_decision":
            node_id = int(event["node_id"])
            assert completion_seq[node_id] < int(event["event_seq"])
    assert set(release_seq) == set(completion_seq), "released node lacks completion"
    assert events[-1]["event_type"] == "tree_completed", "missing terminal tree event"

    fields = set(RouteObservation.__dataclass_fields__)
    assert fields == {"node", "replica_inflight"}
    node_fields = set(ReleasedNode.__dataclass_fields__)
    assert node_fields == {
        "tree_id",
        "node_id",
        "parent_id",
        "depth",
        "prompt",
        "action",
        "predicted_remaining_nodes",
        "predicted_remaining_tokens",
    }
    assert not ({"children", "expected_answer", "gold_answer", "future_topology"} & node_fields)
    assert all(
        not hasattr(observation.node, "children")
        and not hasattr(observation.node, "expected_answer")
        for observation in router.observations
    )
    assert all(
        math.isfinite(observation.node.predicted_remaining_nodes)
        and math.isfinite(observation.node.predicted_remaining_tokens)
        for observation in router.observations
    )

    events_by_type: dict[str, int] = {}
    for event in events:
        kind = str(event["event_type"])
        events_by_type[kind] = events_by_type.get(kind, 0) + 1
    assert events_by_type.get("branch_pruned", 0) >= 1, "fixture did not cover budget pruning"
    assert events_by_type.get("branch_stopped", 0) >= 1, "fixture did not cover terminal stopping"
    assert events_by_type.get("node_released", 0) >= 3, "fixture did not expand dynamically"
    return events_by_type


async def run_fixture() -> tuple[list[dict[str, object]], dict[str, int]]:
    router = RoundRobinFixtureRouter()
    driver = LiveTreeDriver(
        tree_id="b0-fixture-001",
        adapter=FixtureAdapter(),
        expansion_policy=FixtureExpansionPolicy(),
        router=router,
        worker_count=2,
        max_inflight=2,
        max_nodes=4,
    )
    events = await driver.run("Solve the fixture problem.")
    return events, audit_fixture(events, router)


async def run_failure_fixture() -> list[dict[str, object]]:
    driver = LiveTreeDriver(
        tree_id="b0-failure-fixture-001",
        adapter=FailureFixtureAdapter(),
        expansion_policy=FixtureExpansionPolicy(),
        router=RoundRobinFixtureRouter(),
        worker_count=2,
        max_inflight=2,
        max_nodes=4,
    )
    return await driver.run("Run the synthetic failure fixture.")


def audit_failure_fixture(events: list[dict[str, object]]) -> dict[str, int]:
    assert events[-1]["event_type"] == "tree_failed", "request failure was not reflected at tree level"
    kinds = [str(event["event_type"]) for event in events]
    assert "node_failed" in kinds, "failed request event is missing"
    assert "node_cancelled" in kinds, "sibling cancellation event is missing"
    released = {
        int(event["node_id"])
        for event in events
        if event["event_type"] == "node_released"
    }
    closed = {
        int(event["node_id"])
        for event in events
        if event["event_type"]
        in {"node_completed", "node_failed", "node_cancelled", "node_aborted_before_dispatch"}
    }
    assert released <= closed, "a released node has no completion/failure/abort record"
    return {kind: kinds.count(kind) for kind in sorted(set(kinds))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--failure-output-jsonl", type=Path)
    args = parser.parse_args()
    events, counts = asyncio.run(run_fixture())
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
            encoding="utf-8",
        )
    failure_events = asyncio.run(run_failure_fixture())
    failure_counts = audit_failure_fixture(failure_events)
    if args.failure_output_jsonl:
        args.failure_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.failure_output_jsonl.write_text(
            "".join(
                json.dumps(event, ensure_ascii=False) + "\n" for event in failure_events
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "B0_FIXTURE_PASS",
                "event_counts": counts,
                "failure_fixture_status": "B0_FAILURE_AUDIT_PASS",
                "failure_event_counts": failure_counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
