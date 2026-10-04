#!/usr/bin/env python3
"""Run the frozen, budget-capped Direction 1 lookahead-weight screen."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path("/home/bumi/infra/cache/prismserve-direction1-lookahead-weight-20261004")
SOURCE_ROOT = Path("/home/bumi/infra/cache/prismserve-direction1-replication-20261004/attempt2")
TRACE = SOURCE_ROOT / "c4/cohort_trace.json"
TRACE_MANIFEST = TRACE.with_suffix(".manifest.json")
DATASET = Path("/home/bumi/infra/assets/gsm8k/test.jsonl")
SERVICE_PROFILE = Path(
    "/home/bumi/infra/cache/prismserve-direction1-request-telemetry-20261004/service_rate_profile.json"
)
OUTPUT_PROFILE = Path(
    "/home/bumi/.codex/worktrees/direction1-replication-claim/PrismServe/idea-stage/pilots/direction1/mechanism-ablation/output_length_profile.json"
)
MODEL = Path("/home/bumi/infra/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
TOKENIZER = Path("/home/bumi/infra/runtime/vllm-0.30.0-cu129/tokenizer-generic-fast")
RUNTIME = Path("/home/bumi/infra/runtime/vllm-0.30.0-cu129")
PYTHON = RUNTIME / "bin/python"
VLLM = RUNTIME / "bin/vllm"
CUDA = Path("/home/bumi/infra/runtime/cuda-toolkit-12-9/usr/local/cuda-12.9")
PILOT = Path(
    "/home/bumi/.codex/worktrees/direction1-replication-claim/PrismServe/idea-stage/pilots/direction1/run_vllm_placement_pilot.py"
)
GPU_IDS = [4, 5, 6, 7]
MAX_GPU_MIN = 200.0
CLEANUP_RESERVE_GPU_MIN = 20.0
EXPECTED_RUN_SECONDS = 400.0
MODEL_NAME = "deepseek-r1-distill-llama-70b"
METRICS = (
    "vllm:request_queue_time_seconds_count",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_count",
)
CONDITIONS = [
    ("A", "tree", 1.0),
    ("A", "flat", 0.0),
    ("A", "tree", 0.5),
    ("B", "tree", 0.5),
    ("B", "flat", 0.0),
    ("B", "tree", 1.0),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def http_text(url: str, method: str = "GET", timeout: float = 3.0) -> str:
    with urlopen(Request(url, data=None, method=method), timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def stop_group(process: subprocess.Popen[str], wait: float = 8.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        pass
    else:
        os.killpg(process.pid, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
    if process.poll() is None:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    raise RuntimeError(f"Owned process group {process.pid} survived SIGKILL")


class Campaign:
    def __init__(self) -> None:
        self.started: float | None = None
        self.services: list[subprocess.Popen[str]] = []
        self.active: subprocess.Popen[str] | None = None
        self.status = "PREFLIGHT"
        self.error = ""
        self.stages: list[dict[str, object]] = []
        self.results: dict[str, dict[str, dict[str, object]]] = {"A": {}, "B": {}}
        self.manifest: dict[str, object] = {}
        self.trace_sha = ""
        self.environment_spec_sha = "2b1cf04c53a841305bda3c8841cbe0841c778bf81d3ba49096f5cf12e7667bfc"
        self.last_block_a_gate: dict[str, object] | None = None
        self.block_gates: dict[str, dict[str, object]] = {}

    @property
    def elapsed(self) -> float:
        return 0.0 if self.started is None else time.monotonic() - self.started

    @property
    def work_left_s(self) -> float:
        return max(0.0, (MAX_GPU_MIN - CLEANUP_RESERVE_GPU_MIN) * 60 / len(GPU_IDS) - self.elapsed)

    @property
    def total_left_s(self) -> float:
        return max(0.0, MAX_GPU_MIN * 60 / len(GPU_IDS) - self.elapsed)

    def save(self) -> None:
        write_json(
            ROOT / "BUDGET_LOG.json",
            {
                "status": self.status,
                "updated_at_utc": utc(),
                "experiment_id": "direction1-lookahead-weight-screen-20261004",
                "parent_cohort_id": self.manifest.get("cohort_id"),
                "parent_trace_sha256": self.trace_sha,
                "dataset_id": self.manifest.get("dataset_id"),
                "dataset_sha256": self.manifest.get("dataset_sha256"),
                "dataset_seed": self.manifest.get("seed"),
                "selected_dataset_indices": self.manifest.get("selected_dataset_indices", []),
                "environment_spec_sha256": self.environment_spec_sha,
                "gpu_ids": GPU_IDS,
                "gpu_count": len(GPU_IDS),
                "max_gpu_min": MAX_GPU_MIN,
                "cleanup_reserve_gpu_min": CLEANUP_RESERVE_GPU_MIN,
                "elapsed_wall_s": round(self.elapsed, 3),
                "attempt_gpu_min": round(self.elapsed * len(GPU_IDS) / 60, 3),
                "remaining_gpu_min": round(max(0, MAX_GPU_MIN - self.elapsed * len(GPU_IDS) / 60), 3),
                "work_remaining_s": round(self.work_left_s, 3),
                "trace_path": str(TRACE),
                "trace_manifest_path": str(TRACE_MANIFEST),
                "service_profile_path": str(SERVICE_PROFILE),
                "output_length_profile_path": str(OUTPUT_PROFILE),
                "last_block_a_gate": self.last_block_a_gate,
                "block_gates": self.block_gates,
                "last_error": self.error,
                "stages": self.stages,
                "logs_dir": str(ROOT / "logs"),
            },
        )

    def preflight(self) -> None:
        required = [
            TRACE,
            TRACE_MANIFEST,
            DATASET,
            SERVICE_PROFILE,
            OUTPUT_PROFILE,
            MODEL / "config.json",
            MODEL / "model.safetensors.index.json",
            TOKENIZER / "tokenizer_config.json",
            PYTHON,
            VLLM,
            CUDA / "bin/nvcc",
            PILOT,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Missing required input: {path}")
        self.manifest = json.loads(TRACE_MANIFEST.read_text())
        trace = json.loads(TRACE.read_text())
        if self.manifest.get("status") != "TRACE_FROZEN":
            raise ValueError("Input trace manifest is not TRACE_FROZEN")
        self.trace_sha = sha_file(TRACE)
        if self.manifest.get("trace_file_sha256") != self.trace_sha:
            raise ValueError("Frozen trace file hash does not match its manifest")
        if self.manifest.get("trace_canonical_sha256") != canonical_sha(trace):
            raise ValueError("Frozen trace canonical hash does not match its manifest")
        if self.manifest.get("dataset_sha256") != sha_file(DATASET):
            raise ValueError("Frozen trace dataset hash does not match the local dataset")
        if self.manifest.get("tokenizer_sha256") != sha_tree(TOKENIZER):
            raise ValueError("Frozen trace tokenizer hash does not match the local tokenizer")
        if self.manifest.get("source_run_id") != trace.get("source_run_id"):
            raise ValueError("Frozen trace source run ID does not match its manifest")
        if self.manifest.get("question_count") != len(self.manifest.get("selected_dataset_indices", [])):
            raise ValueError("Frozen trace manifest has an inconsistent question count")
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "run-output").mkdir(exist_ok=True)
        if any((ROOT / "run-output").iterdir()):
            raise RuntimeError("Refusing to reuse a non-empty run-output directory")
        write_json(
            ROOT / "RUN_PLAN.json",
            {
                "experiment_id": "direction1-lookahead-weight-screen-20261004",
                "parent_cohort_id": self.manifest["cohort_id"],
                "parent_trace_sha256": self.trace_sha,
                "parent_manifest_sha256": sha_file(TRACE_MANIFEST),
                "environment_spec_sha256": self.environment_spec_sha,
                "model": MODEL_NAME,
                "gpu_ids": GPU_IDS,
                "max_gpu_min": MAX_GPU_MIN,
                "cleanup_reserve_gpu_min": CLEANUP_RESERVE_GPU_MIN,
                "tree_ids": list(range(int(self.manifest["question_count"]))),
                "concurrency_per_worker": 4,
                "router_capacity_per_worker": 2,
                "conditions_in_order": [
                    {"block": block, "policy": policy, "tree_work_weight": weight}
                    for block, policy, weight in CONDITIONS
                ],
                "success_screen": "weight 0.5 at least 5% faster than flat in both blocks, complete telemetry, and no lower majority-correct count",
                "early_stop": "skip block B if block A does not meet the 5% latency and correctness screen",
                "expected_run_seconds": EXPECTED_RUN_SECONDS,
            },
        )
        self.save()

    def check_free(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        state: dict[int, tuple[int, int]] = {}
        for line in result.stdout.splitlines():
            index, memory, utilization = (int(part.strip()) for part in line.split(","))
            state[index] = (memory, utilization)
        busy = {
            index: state.get(index)
            for index in GPU_IDS
            if index not in state or state[index][0] > 1024 or state[index][1] > 0
        }
        if busy:
            raise RuntimeError(f"Selected GPUs are not idle: {busy}")
        for port in (8004, 8005):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    raise RuntimeError(f"Required port {port} is already in use")
            except OSError:
                pass

    def launch(self) -> None:
        self.check_free()
        self.started = time.monotonic()
        self.status = "SERVERS_STARTING"
        self.save()
        groups = (GPU_IDS[:2], GPU_IDS[2:])
        for replica, (port, gpus) in enumerate(zip((8004, 8005), groups)):
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_HOME": str(CUDA),
                    "CUDA_PATH": str(CUDA),
                    "PATH": f"{CUDA / 'bin'}{os.pathsep}{environment.get('PATH', '')}",
                    "LD_LIBRARY_PATH": os.pathsep.join(
                        path
                        for path in (str(CUDA / "lib64"), environment.get("LD_LIBRARY_PATH", ""))
                        if path
                    ),
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
                    "VLLM_SERVER_DEV_MODE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [
                str(VLLM),
                "serve",
                str(MODEL),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tensor-parallel-size",
                "2",
                "--dtype",
                "bfloat16",
                "--max-num-seqs",
                "2",
                "--enable-prefix-caching",
                "--served-model-name",
                MODEL_NAME,
                "--max-model-len",
                "8192",
                "--tokenizer",
                str(TOKENIZER),
                "--enable-per-request-metrics",
                "--enable-prompt-tokens-details",
            ]
            log_path = ROOT / "logs" / f"vllm-replica-{replica}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            self.services.append(process)
        self.wait_ready()

    def wait_ready(self) -> None:
        last_error = "not ready"
        while self.work_left_s > 30:
            if any(process.poll() is not None for process in self.services):
                raise RuntimeError("A vLLM process exited during startup")
            try:
                for port in (8004, 8005):
                    url = f"http://127.0.0.1:{port}"
                    http_text(url + "/health")
                    metrics = http_text(url + "/metrics", timeout=5)
                    missing = [metric for metric in METRICS if metric not in metrics]
                    if missing:
                        raise RuntimeError(f"Missing required request metrics at {port}: {missing}")
                    response = json.loads(http_text(url + "/reset_prefix_cache", method="POST"))
                    if response.get("success") is not True:
                        raise RuntimeError(f"Prefix cache reset failed at {port}")
                self.stages.append({"label": "service-startup-and-preflight", "status": "READY", "time_utc": utc()})
                self.status = "SERVERS_READY"
                self.save()
                return
            except Exception as error:  # startup errors are preserved in the budget record
                last_error = f"{type(error).__name__}: {error}"
                time.sleep(min(2.0, max(0.1, self.work_left_s - 30)))
        raise TimeoutError(f"vLLM startup ran out of work budget: {last_error}")

    @staticmethod
    def variant_name(policy: str, weight: float) -> str:
        return "flat" if policy == "flat" else f"tree-w{weight:.1f}"

    def run_condition(self, index: int, block: str, policy: str, weight: float) -> None:
        remaining_in_block = sum(1 for condition in CONDITIONS[index:] if condition[0] == block)
        if self.work_left_s < remaining_in_block * EXPECTED_RUN_SECONDS:
            raise TimeoutError("Insufficient GPU budget to finish the current block at the frozen run estimate")
        strategy = "kv-cost-group-flat" if policy == "flat" else "kv-cost-group-tree"
        variant = self.variant_name(policy, weight)
        run_label = f"block{block}-{variant}"
        run_id = f"direction1-lookahead-weight-screen-20261004:{run_label}"
        experiment_id = str(self.manifest["cohort_id"])
        source_run_id = str(self.manifest["source_run_id"])
        output_dir = ROOT / "run-output" / f"block{block}"
        command = [
            str(PYTHON),
            str(PILOT),
            "--strategy",
            strategy,
            "--workers",
            "http://127.0.0.1:8004",
            "http://127.0.0.1:8005",
            "--model-name",
            MODEL_NAME,
            "--dataset-id",
            str(self.manifest["dataset_id"]),
            "--tokenizer-dir",
            str(TOKENIZER),
            "--dataset",
            str(DATASET),
            "--trace-json",
            str(TRACE),
            "--output-dir",
            str(output_dir),
            "--questions",
            str(self.manifest["question_count"]),
            "--seed",
            str(self.manifest["seed"]),
            "--max-concurrency",
            "4",
            "--router-capacity",
            "2",
            "--service-profile-json",
            str(SERVICE_PROFILE),
            "--output-length-profile-json",
            str(OUTPUT_PROFILE),
            "--request-timeout-s",
            "300",
            "--reset-prefix-cache",
            "--require-vllm-request-metrics",
            "--tree-ids",
            *map(str, range(int(self.manifest["question_count"]))),
            "--max-tokens-inner",
            "1024",
            "--max-tokens-leaf",
            "1024",
            "--run-label",
            run_label,
            "--experiment-id",
            experiment_id,
            "--run-id",
            run_id,
            "--parent-run-id",
            source_run_id,
        ]
        if policy == "tree":
            command.extend(["--tree-work-weight", str(weight)])
        remaining_after = remaining_in_block - 1
        timeout = min(520.0, self.work_left_s - remaining_after * EXPECTED_RUN_SECONDS)
        if timeout < 60:
            raise TimeoutError("The remaining GPU budget cannot safely start this run")
        log_path = ROOT / "logs" / f"{run_label}.log"
        started = time.monotonic()
        record: dict[str, object] = {
            "block": block,
            "variant": variant,
            "weight": weight,
            "strategy": strategy,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "parent_run_id": source_run_id,
            "command": command,
            "started_at_utc": utc(),
            "timeout_s": round(timeout, 3),
            "log": str(log_path),
            "status": "RUNNING",
        }
        self.stages.append(record)
        self.status = f"RUNNING_{run_label}"
        self.save()
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.active = process
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_group(process)
                self.active = None
                record.update(
                    {
                        "status": "TIMED_OUT",
                        "ended_at_utc": utc(),
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
                self.save()
                raise TimeoutError(f"{run_label} exceeded {timeout:.1f}s")
            except BaseException:
                if process.poll() is not None:
                    self.active = None
                raise
            else:
                self.active = None
        record.update(
            {
                "status": "COMPLETE" if return_code == 0 else "FAILED",
                "return_code": return_code,
                "ended_at_utc": utc(),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )
        self.save()
        if return_code != 0:
            raise RuntimeError(f"{run_label} failed with exit code {return_code}; see {log_path}")
        self.collect_result(block, variant, run_id, output_dir)

    def collect_result(self, block: str, variant: str, run_id: str, output_dir: Path) -> None:
        matched: tuple[Path, dict[str, object]] | None = None
        for summary_path in output_dir.glob("*_SUMMARY.json"):
            summary = json.loads(summary_path.read_text())
            if summary.get("run_id") == run_id:
                matched = (summary_path, summary)
                break
        if matched is None:
            raise RuntimeError(f"No completed summary found for {run_id}")
        summary_path, summary = matched
        stem = summary_path.name.removesuffix("_SUMMARY.json")
        tree_path = output_dir / f"{stem}_TREES.csv"
        request_path = output_dir / f"{stem}_REQUESTS.csv"
        with tree_path.open(newline="") as stream:
            trees = list(csv.DictReader(stream))
        with request_path.open(newline="") as stream:
            requests = list(csv.DictReader(stream))
        expected_trees = int(self.manifest["question_count"])
        expected_ids = set(range(expected_trees))
        actual_ids = {int(row["tree_id"]) for row in trees}
        if len(trees) != expected_trees or actual_ids != expected_ids:
            raise ValueError(f"{run_id} has incomplete tree output: {len(trees)} rows, ids={sorted(actual_ids)}")
        expected_requests = sum(int(row["node_count"]) for row in trees)
        if len(requests) != expected_requests:
            raise ValueError(f"{run_id} has {len(requests)} request rows; expected {expected_requests}")
        latencies = [float(row["completion_latency_s"]) for row in trees]
        correct = sum(row["majority_correct"].strip().lower() == "true" for row in trees)
        if summary.get("tree_work_weight") != (0.0 if variant == "flat" else float(variant[-3:])):
            raise ValueError(f"{run_id} summary does not record its effective tree-work weight")
        self.results[block][variant] = {
            "run_id": run_id,
            "summary": str(summary_path),
            "tree_csv": str(tree_path),
            "request_csv": str(request_path),
            "tree_count": len(trees),
            "request_count": len(requests),
            "median_tree_completion_latency_s": statistics.median(latencies),
            "majority_correct_trees": correct,
            "tree_latencies_s": {int(row["tree_id"]): float(row["completion_latency_s"]) for row in trees},
        }
        self.save()

    def evaluate_block(self, block: str) -> bool:
        flat = self.results[block].get("flat")
        discounted = self.results[block].get("tree-w0.5")
        if flat is None or discounted is None:
            raise RuntimeError(f"Block {block} is missing flat or discounted-tree results")
        flat_latency = float(flat["median_tree_completion_latency_s"])
        discounted_latency = float(discounted["median_tree_completion_latency_s"])
        speedup = (flat_latency - discounted_latency) / flat_latency
        correctness_ok = int(discounted["majority_correct_trees"]) >= int(flat["majority_correct_trees"])
        full_weight = self.results[block].get("tree-w1.0")
        full_weight_quality_ok = (
            None
            if full_weight is None
            else int(full_weight["majority_correct_trees"])
            >= int(flat["majority_correct_trees"])
        )
        gate = {
            "flat_median_s": flat_latency,
            "weight_0_5_median_s": discounted_latency,
            "weight_0_5_speedup_vs_flat": speedup,
            "weight_0_5_correctness_gate_passed": correctness_ok,
            "weight_1_0_correctness_gate_passed": full_weight_quality_ok,
            "latency_gate_passed": speedup >= 0.05,
            "gate_passed": speedup >= 0.05 and correctness_ok,
        }
        self.block_gates[block] = gate
        if block == "A":
            self.last_block_a_gate = gate
        self.save()
        return speedup >= 0.05 and correctness_ok

    def run(self) -> int:
        self.preflight()
        self.launch()
        for index, (block, policy, weight) in enumerate(CONDITIONS):
            self.run_condition(index, block, policy, weight)
            if index == 1:
                flat = self.results["A"]["flat"]
                full_weight = self.results["A"]["tree-w1.0"]
                if int(full_weight["majority_correct_trees"]) < int(flat["majority_correct_trees"]):
                    self.status = "STOPPED_AFTER_BLOCK_A_QUALITY_GATE"
                    self.stages.append(
                        {"label": "block-A-full-weight-quality-gate", "status": "STOPPED", "time_utc": utc()}
                    )
                    self.save()
                    return 0
            if index == 2 and not self.evaluate_block("A"):
                self.status = "STOPPED_AFTER_BLOCK_A_SCREEN"
                self.stages.append(
                    {"label": "block-A-decision-gate", "status": "STOPPED", "time_utc": utc()}
                )
                self.save()
                return 0
            if index == 4 and not self.evaluate_block("B"):
                self.status = "STOPPED_AFTER_BLOCK_B_SCREEN"
                self.stages.append(
                    {"label": "block-B-decision-gate", "status": "STOPPED", "time_utc": utc()}
                )
                self.save()
                return 0
        self.evaluate_block("B")
        passed = all(
            self.results[block]["tree-w0.5"]["median_tree_completion_latency_s"]
            <= 0.95 * self.results[block]["flat"]["median_tree_completion_latency_s"]
            and self.results[block]["tree-w0.5"]["majority_correct_trees"]
            >= self.results[block]["flat"]["majority_correct_trees"]
            and self.results[block]["tree-w1.0"]["majority_correct_trees"]
            >= self.results[block]["flat"]["majority_correct_trees"]
            for block in ("A", "B")
        )
        self.status = "SCREEN_PASS" if passed else "SCREEN_FAIL"
        self.stages.append({"label": "final-decision-gate", "status": self.status, "time_utc": utc()})
        self.save()
        return 0

    def cleanup(self) -> None:
        owned = ([self.active] if self.active is not None else []) + self.services
        for process in owned:
            try:
                stop_group(process)
            except RuntimeError as error:
                self.error = "; ".join(filter(None, [self.error, str(error)]))
                self.status = "CLEANUP_INCOMPLETE"
        self.save()


def main() -> int:
    campaign = Campaign()
    return_code = 0
    try:
        return_code = campaign.run()
    except KeyboardInterrupt as error:
        campaign.error = f"Interrupted: {error}"
        campaign.status = "INTERRUPTED"
        campaign.save()
        return_code = 130
    except BaseException as error:
        campaign.error = f"{type(error).__name__}: {error}"
        campaign.status = "INCOMPLETE_OR_FAILED"
        campaign.save()
        print(campaign.error, file=sys.stderr)
        return_code = 2
    finally:
        campaign.cleanup()
    if campaign.status == "CLEANUP_INCOMPLETE" and return_code == 0:
        return 2
    return return_code


if __name__ == "__main__":
    def handle(signal_number: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signal_number}")

    for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signal_number, handle)
    raise SystemExit(main())
