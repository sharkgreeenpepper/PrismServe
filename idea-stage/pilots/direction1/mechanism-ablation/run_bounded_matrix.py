#!/usr/bin/env python3
"""Run the direction-1 mechanism screen under a wall-clock GPU budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


MODEL_NAME = "deepseek-r1-distill-llama-70b"
WORKERS = ("http://127.0.0.1:8004", "http://127.0.0.1:8005")
PRIMARY_RUNS = (
    ("B1-flat", "kv-cost-group-flat"),
    ("B1-tree", "kv-cost-group-tree"),
    ("B2-tree", "kv-cost-group-tree"),
    ("B2-flat", "kv-cost-group-flat"),
)
OPTIONAL_RUNS = (
    ("reference-local", "local-only"),
    ("reference-least-loaded", "least-loaded"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_text(url: str, method: str = "GET", timeout_s: float = 2.0) -> str:
    request = Request(url, data=None, method=method)
    with urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8", errors="replace")


def check_ports_free(ports: tuple[int, ...]) -> None:
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                raise RuntimeError(f"Port {port} is already serving; refusing to reuse it")
        except OSError:
            pass


def check_gpus_free(gpu_ids: tuple[int, ...]) -> None:
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
    observed: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        index, memory, utilization = (part.strip() for part in line.split(","))
        observed[int(index)] = (int(memory), int(utilization))
    busy = {
        gpu: observed.get(gpu)
        for gpu in gpu_ids
        if gpu not in observed or observed[gpu][0] > 1024 or observed[gpu][1] > 0
    }
    if busy:
        raise RuntimeError(f"Selected GPUs are not idle (MiB, utilization percent): {busy}")


def terminate_group(process: subprocess.Popen[str], wait_s: float = 10.0) -> None:
    pgid = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(0.0, wait_s)
    while group_exists() and time.monotonic() < deadline:
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if group_exists():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + 2.0
        while group_exists() and time.monotonic() < kill_deadline:
            time.sleep(0.1)
    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    if group_exists():
        raise RuntimeError(f"Owned process group {pgid} survived SIGKILL")


class BudgetController:
    def __init__(self, args: argparse.Namespace, log_dir: Path) -> None:
        self.args = args
        self.log_dir = log_dir
        self.service_processes: list[subprocess.Popen[str]] = []
        self.active_stage_process: subprocess.Popen[str] | None = None
        self.started_at = 0.0
        self.stage_records: list[dict[str, Any]] = []
        self.state = "NOT_STARTED"
        self.last_error = ""
        self.full_budget_s = (args.max_gpu_min - args.prior_gpu_min) * 60 / len(args.gpu_ids)
        self.cleanup_reserve_s = args.cleanup_reserve_gpu_min * 60 / len(args.gpu_ids)
        self.work_budget_s = (
            args.max_gpu_min - args.prior_gpu_min - args.cleanup_reserve_gpu_min
        ) * 60 / len(args.gpu_ids)
        if self.full_budget_s <= 0 or self.work_budget_s <= 0:
            raise ValueError("Prior usage and cleanup reserve leave no GPU budget")

    @property
    def elapsed_s(self) -> float:
        return 0.0 if not self.started_at else time.monotonic() - self.started_at

    @property
    def work_remaining_s(self) -> float:
        return self.work_budget_s - self.elapsed_s

    @property
    def total_remaining_s(self) -> float:
        return self.full_budget_s - self.elapsed_s

    def save_state(self) -> None:
        write_json(
            self.args.budget_log,
            {
                "status": self.state,
                "updated_at_utc": utc_now(),
                "experiment_id": self.args.experiment_id,
                "model": MODEL_NAME,
                "dataset_id": "gsm8k:test",
                "dataset_sha256": file_sha256(Path(self.args.dataset)),
                "dataset_seed": self.args.seed,
                "selected_dataset_indices": self.args.expected_dataset_indices or [],
                "excluded_dataset_indices": self.args.excluded_dataset_indices,
                "environment_spec_sha256": self.args.environment_spec_sha256,
                "cohort_manifest_sha256": file_sha256(
                    Path(self.args.trace_json).with_suffix(".manifest.json")
                ) if Path(self.args.trace_json).with_suffix(".manifest.json").is_file() else None,
                "gpu_ids": self.args.gpu_ids,
                "gpu_count": len(self.args.gpu_ids),
                "max_concurrency_per_worker": self.args.max_concurrency,
                "primary_runs": self.args.primary_runs or (
                    [self.args.only_primary_run]
                    if self.args.only_primary_run
                    else [label for label, _ in PRIMARY_RUNS]
                ),
                "max_gpu_min": self.args.max_gpu_min,
                "prior_gpu_min": self.args.prior_gpu_min,
                "cleanup_reserve_gpu_min": self.args.cleanup_reserve_gpu_min,
                "elapsed_wall_s": round(self.elapsed_s, 3),
                "attempt_gpu_min": round(self.elapsed_s * len(self.args.gpu_ids) / 60, 3),
                "cumulative_gpu_min": round(
                    self.args.prior_gpu_min + self.elapsed_s * len(self.args.gpu_ids) / 60,
                    3,
                ),
                "remaining_gpu_min": round(
                    max(
                        0.0,
                        self.args.max_gpu_min
                        - self.args.prior_gpu_min
                        - self.elapsed_s * len(self.args.gpu_ids) / 60,
                    ),
                    3,
                ),
                "work_remaining_s": round(max(0.0, self.work_remaining_s), 3),
                "cuda_home": self.args.cuda_home,
                "sampling_backend": "vllm-native (VLLM_USE_FLASHINFER_SAMPLER=0)",
                "request_telemetry_enabled": self.args.enable_request_telemetry,
                "last_error": self.last_error,
                "stages": self.stage_records,
                "logs_dir": str(self.log_dir),
            },
        )

    def launch_services(self) -> None:
        model_dir = Path(self.args.model_dir)
        for path in (model_dir / "config.json", model_dir / "model.safetensors.index.json"):
            if not path.is_file():
                raise FileNotFoundError(f"Model file is missing: {path}")
        check_gpus_free(tuple(self.args.gpu_ids))
        check_ports_free((8004, 8005))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.monotonic()
        self.state = "SERVERS_STARTING"
        self.save_state()
        gpu_groups = (self.args.gpu_ids[:2], self.args.gpu_ids[2:])
        cuda_home = Path(self.args.cuda_home)
        for replica, (port, gpus) in enumerate(zip((8004, 8005), gpu_groups)):
            env = os.environ.copy()
            env["CUDA_HOME"] = str(cuda_home)
            env["CUDA_PATH"] = str(cuda_home)
            env["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
            cuda_lib = str(cuda_home / "lib64")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                part for part in (cuda_lib, env.get("LD_LIBRARY_PATH", "")) if part
            )
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
                    "VLLM_SERVER_DEV_MODE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [
                self.args.vllm_bin,
                "serve",
                str(model_dir),
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
                self.args.tokenizer_dir,
            ]
            if self.args.enable_request_telemetry:
                command.extend(
                    ["--enable-per-request-metrics", "--enable-prompt-tokens-details"]
                )
            with (self.log_dir / f"vllm-replica-{replica}.log").open(
                "w", encoding="utf-8"
            ) as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            self.service_processes.append(process)
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        required_metrics = (
            "vllm:request_queue_time_seconds_count",
            "vllm:request_prefill_time_seconds_count",
            "vllm:request_decode_time_seconds_count",
        )
        last_error = "not ready"
        while self.work_remaining_s > 0:
            all_ready = True
            for process, endpoint in zip(self.service_processes, WORKERS):
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
                try:
                    http_text(endpoint + "/health")
                except (OSError, URLError, TimeoutError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    all_ready = False
            if all_ready:
                for endpoint in WORKERS:
                    metrics = http_text(endpoint + "/metrics", timeout_s=5.0)
                    missing = [name for name in required_metrics if name not in metrics]
                    if missing:
                        raise RuntimeError(f"Required metrics are absent at {endpoint}: {missing}")
                    payload = json.loads(
                        http_text(endpoint + "/reset_prefix_cache", method="POST")
                    )
                    if payload.get("success") is not True:
                        raise RuntimeError(f"Prefix cache reset rejected at {endpoint}: {payload!r}")
                self.state = "SERVERS_READY"
                self.save_state()
                return
            time.sleep(min(2.0, max(0.1, self.work_remaining_s)))
        raise TimeoutError(f"vLLM did not become ready before the work cutoff: {last_error}")

    def run_stage(
        self, label: str, command: list[str], max_stage_s: float | None = None
    ) -> bool:
        remaining = self.work_remaining_s
        if remaining <= 0:
            self.stage_records.append({"label": label, "status": "SKIPPED_BUDGET", "time_utc": utc_now()})
            self.save_state()
            return False
        timeout_s = remaining if max_stage_s is None else min(max_stage_s, remaining)
        log_path = self.log_dir / f"{label}.log"
        record: dict[str, Any] = {
            "label": label,
            "experiment_id": self.args.experiment_id,
            "run_id": f"{self.args.experiment_id}:{label}",
            "parent_run_id": (
                self.args.experiment_id
                if label == "cohort-source-local"
                else f"{self.args.experiment_id}:cohort-source-local"
            ),
            "command": command,
            "started_at_utc": utc_now(),
            "timeout_s": round(timeout_s, 3),
            "log": str(log_path),
        }
        self.state = f"RUNNING_{label}"
        self.save_state()
        stage_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.active_stage_process = process
            try:
                return_code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    terminate_group(process, wait_s=min(10.0, max(1.0, self.total_remaining_s)))
                except RuntimeError as exc:
                    self.last_error = f"Stage process cleanup failed: {exc}"
                record.update(
                    {
                        "status": "TIMED_OUT",
                        "ended_at_utc": utc_now(),
                        "elapsed_s": round(time.monotonic() - stage_started, 3),
                    }
                )
                self.stage_records.append(record)
                self.save_state()
                raise TimeoutError(f"Stage {label} exceeded its {timeout_s:.1f}s budget")
            except BaseException as interrupt_exc:
                try:
                    terminate_group(process, wait_s=min(10.0, max(1.0, self.total_remaining_s)))
                except RuntimeError as cleanup_exc:
                    self.last_error = f"Stage process cleanup failed: {cleanup_exc}"
                record.update(
                    {
                        "status": "INTERRUPTED",
                        "ended_at_utc": utc_now(),
                        "elapsed_s": round(time.monotonic() - stage_started, 3),
                        "interrupt": f"{type(interrupt_exc).__name__}: {interrupt_exc}",
                    }
                )
                self.stage_records.append(record)
                self.save_state()
                raise
            finally:
                self.active_stage_process = None
        record.update(
            {
                "status": "COMPLETE" if return_code == 0 else "FAILED",
                "return_code": return_code,
                "ended_at_utc": utc_now(),
                "elapsed_s": round(time.monotonic() - stage_started, 3),
            }
        )
        self.stage_records.append(record)
        self.save_state()
        if return_code != 0:
            raise RuntimeError(f"Stage {label} failed with exit code {return_code}; see {log_path}")
        return True

    def python_command(self, script: Path, *arguments: str) -> list[str]:
        return [self.args.python_bin, str(script), *arguments]

    def run_experiments(self) -> None:
        script_dir = Path(__file__).resolve().parent
        project_dir = Path(__file__).resolve().parents[4]
        profile_path = (
            Path(self.args.reuse_service_profile)
            if self.args.reuse_service_profile
            else Path(self.args.output_dir) / "service_rate_profile.json"
        )
        output_profile = Path(self.args.output_profile)
        tree_ids = [int(tree_id) for tree_id in self.args.tree_ids]
        tree_id_args = [str(tree_id) for tree_id in tree_ids]
        base_common = [
            "--workers",
            *WORKERS,
            "--model-name",
            MODEL_NAME,
            "--dataset-id",
            "gsm8k:test",
            "--tokenizer-dir",
            self.args.tokenizer_dir,
            "--dataset",
            self.args.dataset,
            "--trace-json",
            self.args.trace_json,
            "--output-dir",
            self.args.output_dir,
            "--questions",
            str(self.args.questions),
            "--seed",
            str(self.args.seed),
            "--max-concurrency",
            str(self.args.max_concurrency),
            "--router-capacity",
            "2",
            "--service-profile-json",
            str(profile_path),
            "--output-length-profile-json",
            str(output_profile),
            "--request-timeout-s",
            "300",
            "--reset-prefix-cache",
        ]
        if self.args.fixed_output_tokens is not None:
            base_common.extend(["--fixed-output-tokens", str(self.args.fixed_output_tokens)])
        if self.args.enable_request_telemetry:
            base_common.append("--require-vllm-request-metrics")
        calibrator = script_dir / "calibrate_vllm_service_rates.py"
        runner = project_dir / "idea-stage/pilots/direction1/run_vllm_placement_pilot.py"
        if self.args.prepare_independent_trace:
            source_common = [
                *base_common[: base_common.index("--trace-json")],
                *base_common[base_common.index("--trace-json") + 2 :],
            ]
            source_command = [
                "--strategy",
                # Placement strategies need the frozen trace they are meant
                # to help construct. Generate the natural source history
                # with local-only, which accepts live parent outputs.
                "local-only",
                *source_common,
                "--tree-ids",
                *tree_id_args,
                "--max-tokens-inner",
                "1024",
                "--max-tokens-leaf",
                "1024",
                "--run-label",
                "cohort-source-local",
                "--experiment-id",
                self.args.experiment_id,
                "--run-id",
                f"{self.args.experiment_id}:cohort-source-local",
                "--parent-run-id",
                self.args.experiment_id,
            ]
            service_profile_index = source_command.index("--service-profile-json")
            source_command[service_profile_index + 1] = self.args.bootstrap_service_profile
            if not self.run_stage(
                "cohort-source-local", self.python_command(runner, *source_command)
            ):
                self.state = "INCOMPLETE_COHORT_SOURCE_BUDGET"
                self.save_state()
                return
            source_candidates = sorted(
                Path(self.args.output_dir).glob(
                    "REAL_PLACEMENT_local-only_cohort-source-local_*_REQUESTS.csv"
                )
            )
            if len(source_candidates) != 1:
                raise RuntimeError(
                    "Expected one source-pass request CSV, found "
                    f"{[str(path) for path in source_candidates]}"
                )
            trace_builder = project_dir / "idea-stage/pilots/direction1/build_prompt_trace.py"
            build_trace_command = self.python_command(
                trace_builder,
                "--requests-csv",
                str(source_candidates[0]),
                "--dataset",
                self.args.dataset,
                "--tokenizer-dir",
                self.args.tokenizer_dir,
                "--output",
                self.args.trace_json,
                "--questions",
                str(self.args.questions),
                "--seed",
                str(self.args.seed),
                "--max-tokens-inner",
                "1024",
                "--max-tokens-leaf",
                "1024",
                "--strict-prompt-token-counts",
                "--dataset-id",
                "gsm8k:test",
                "--cohort-id",
                self.args.experiment_id,
                "--source-run-id",
                f"{self.args.experiment_id}:cohort-source-local",
                "--manifest-output",
                str(Path(self.args.trace_json).with_suffix(".manifest.json")),
                "--expected-dataset-indices",
                *[str(index) for index in self.args.expected_dataset_indices],
            )
            if not self.run_stage("build-frozen-trace", build_trace_command, max_stage_s=120.0):
                self.state = "INCOMPLETE_TRACE_BUILD_BUDGET"
                self.save_state()
                return
            if not Path(self.args.trace_json).is_file():
                raise RuntimeError(f"Trace builder did not create {self.args.trace_json}")

        common = [*base_common, "--tree-ids", *tree_id_args]
        calibration_tree_ids = self.args.calibration_tree_ids
        if calibration_tree_ids is None:
            calibration_tree_ids = (
                tree_ids
                if self.args.prepare_independent_trace
                else [tree_id for tree_id in (0, 2, 4, 6) if tree_id < self.args.questions]
            )
        if self.args.reuse_service_profile:
            self.stage_records.append(
                {
                    "label": "service-calibration-reused",
                    "status": "REUSED",
                    "profile": str(profile_path),
                    "time_utc": utc_now(),
                }
            )
            self.save_state()
        else:
            self.run_stage(
                # Both replicas run the identical calibration set concurrently.
                "service-calibration",
                self.python_command(
                    calibrator,
                    "--workers",
                    *WORKERS,
                    "--trace-json",
                    self.args.trace_json,
                    "--tree-ids",
                    *[str(tree_id) for tree_id in calibration_tree_ids],
                    "--model-name",
                    MODEL_NAME,
                    "--max-tokens",
                    "128",
                    "--output-json",
                    str(profile_path),
                ),
            )
            if profile_path.is_file():
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                profile.update(
                    {
                        "experiment_id": self.args.experiment_id,
                        "run_id": f"{self.args.experiment_id}:service-calibration",
                        "parent_run_id": f"{self.args.experiment_id}:cohort-source-local",
                    }
                )
                write_json(profile_path, profile)
        if not profile_path.is_file():
            self.state = "INCOMPLETE_SERVICE_CALIBRATION"
            self.save_state()
            return
        if not self.args.prepare_independent_trace and not self.args.skip_sanity:
            sanity_label = (
                f"sanity-tree{tree_ids[0]}-fixed{self.args.fixed_output_tokens}"
                if self.args.fixed_output_tokens is not None
                else f"sanity-tree{tree_ids[0]}-64tok"
            )
            sanity = [
                *base_common,
                "--tree-ids",
                str(tree_ids[0]),
                "--max-tokens-inner",
                "64",
                "--max-tokens-leaf",
                "64",
                "--run-label",
                sanity_label,
            ]
            if not self.run_stage(
                "sanity-tree1",
                self.python_command(runner, "--strategy", "kv-cost-group-tree", *sanity),
            ):
                self.state = "INCOMPLETE_SANITY_BUDGET"
                self.save_state()
                return

        primary_common = [*base_common, "--tree-ids", *tree_id_args]
        selected_primary_labels = (
            [self.args.only_primary_run]
            if self.args.only_primary_run
            else self.args.primary_runs or [label for label, _ in PRIMARY_RUNS]
        )
        primary_runs = [
            next(run for run in PRIMARY_RUNS if run[0] == label)
            for label in selected_primary_labels
        ]
        for label, strategy in primary_runs:
            completed = self.run_stage(
                label,
                self.python_command(
                    runner,
                    "--strategy",
                    strategy,
                    *primary_common,
                    "--max-tokens-inner",
                    "1024",
                    "--max-tokens-leaf",
                    "1024",
                    "--run-label",
                    (
                        f"{label}-fixed{self.args.fixed_output_tokens}"
                        if self.args.fixed_output_tokens is not None
                        else label
                    ),
                    "--experiment-id",
                    self.args.experiment_id,
                    "--run-id",
                    f"{self.args.experiment_id}:{label}",
                    "--parent-run-id",
                    f"{self.args.experiment_id}:cohort-source-local",
                ),
            )
            if not completed:
                self.state = "INCOMPLETE_PRIMARY_BUDGET"
                return

        self.state = "PRIMARY_COMPLETE"
        self.save_state()
        if self.args.skip_references:
            self.state = "PRIMARY_COMPLETE_REFERENCES_SKIPPED"
            self.stage_records.extend(
                {
                    "label": label,
                    "status": "SKIPPED_BY_FROZEN_PLAN",
                    "experiment_id": self.args.experiment_id,
                    "run_id": f"{self.args.experiment_id}:{label}",
                    "parent_run_id": f"{self.args.experiment_id}:cohort-source-local",
                    "time_utc": utc_now(),
                }
                for label, _ in OPTIONAL_RUNS
            )
            self.save_state()
            return
        all_references_complete = True
        for label, strategy in OPTIONAL_RUNS:
            if self.work_remaining_s < self.args.optional_run_reserve_s:
                all_references_complete = False
                self.stage_records.append(
                    {
                        "label": label,
                        "status": "SKIPPED_OPTIONAL_BUDGET",
                        "work_remaining_s": round(self.work_remaining_s, 3),
                        "time_utc": utc_now(),
                    }
                )
                self.save_state()
                continue
            try:
                reference_completed = self.run_stage(
                    label,
                    self.python_command(
                        runner,
                        "--strategy",
                        strategy,
                        *primary_common,
                        "--max-tokens-inner",
                        "1024",
                        "--max-tokens-leaf",
                        "1024",
                        "--run-label",
                        (
                            f"{label}-fixed{self.args.fixed_output_tokens}"
                            if self.args.fixed_output_tokens is not None
                            else label
                        ),
                    ),
                )
                if not reference_completed:
                    all_references_complete = False
            except (TimeoutError, RuntimeError) as exc:
                all_references_complete = False
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.state = "PRIMARY_COMPLETE_REFERENCE_RUN_INCOMPLETE"
                self.save_state()
                return
        self.state = (
            "COMPLETE_WITH_REFERENCES"
            if all_references_complete
            else "PRIMARY_COMPLETE_REFERENCES_SKIPPED"
        )
        self.save_state()

    def cleanup(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        owned_processes = list(self.service_processes)
        if self.active_stage_process is not None:
            owned_processes.insert(0, self.active_stage_process)
        if not owned_processes:
            return
        for process in owned_processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        cleanup_deadline = time.monotonic() + max(0.0, self.total_remaining_s)
        failures = []
        for process in owned_processes:
            try:
                terminate_group(
                    process,
                    wait_s=max(0.0, min(8.0, cleanup_deadline - time.monotonic())),
                )
            except RuntimeError as exc:
                failures.append(str(exc))
        if failures:
            self.last_error = "; ".join(filter(None, [self.last_error, *failures]))
            self.state = "CLEANUP_INCOMPLETE"
        try:
            self.save_state()
        except OSError:
            print("Could not write final budget log after process cleanup", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-bin", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--cuda-home", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--environment-spec-json", required=True)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--tree-ids", nargs="+", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--calibration-tree-ids", nargs="+", type=int)
    parser.add_argument(
        "--reuse-service-profile",
        help="Reuse a calibrated service profile when continuing the same cohort",
    )
    parser.add_argument(
        "--only-primary-run",
        choices=[label for label, _ in PRIMARY_RUNS],
        help="Run only one frozen primary condition during a bounded continuation",
    )
    parser.add_argument(
        "--primary-runs",
        nargs="+",
        choices=[label for label, _ in PRIMARY_RUNS],
        help="Run this ordered subset of frozen primary conditions",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum in-flight client requests per worker (default: 4)",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip the short sanity replay when continuing a previously verified cohort",
    )
    parser.add_argument("--prepare-independent-trace", action="store_true")
    parser.add_argument("--bootstrap-service-profile")
    parser.add_argument("--expected-dataset-indices", nargs="+", type=int)
    parser.add_argument("--excluded-dataset-indices", nargs="+", type=int, default=[])
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-log", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--gpu-ids", nargs=4, type=int, default=[4, 5, 6, 7])
    parser.add_argument("--max-gpu-min", type=float, default=70.0)
    parser.add_argument("--prior-gpu-min", type=float, default=0.0)
    parser.add_argument("--cleanup-reserve-gpu-min", type=float, default=5.0)
    parser.add_argument("--optional-run-reserve-s", type=float, default=360.0)
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fixed-output-tokens", type=int)
    parser.add_argument(
        "--enable-request-telemetry",
        action="store_true",
        help=(
            "Enable per-request vLLM timing/prompt-cache metrics and fail runs "
            "that do not return all required fields"
        ),
    )
    args = parser.parse_args()
    if len(set(args.gpu_ids)) != 4 or any(gpu < 0 for gpu in args.gpu_ids):
        parser.error("--gpu-ids must contain four distinct non-negative indices")
    if args.prior_gpu_min < 0 or args.prior_gpu_min >= args.max_gpu_min:
        parser.error("--prior-gpu-min must be non-negative and less than --max-gpu-min")
    if args.cleanup_reserve_gpu_min < 0 or (
        args.prior_gpu_min + args.cleanup_reserve_gpu_min >= args.max_gpu_min
    ):
        parser.error("prior usage and cleanup reserve must leave positive work budget")
    if args.fixed_output_tokens is not None and args.fixed_output_tokens < 1:
        parser.error("--fixed-output-tokens must be positive")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if args.only_primary_run and args.primary_runs:
        parser.error("use only one of --only-primary-run and --primary-runs")
    if args.primary_runs and len(set(args.primary_runs)) != len(args.primary_runs):
        parser.error("--primary-runs must not contain duplicate labels")
    if args.questions < 1:
        parser.error("--questions must be positive")
    if not args.tree_ids or len(set(args.tree_ids)) != len(args.tree_ids):
        parser.error("--tree-ids must contain one or more unique ids")
    if any(tree_id < 0 or tree_id >= args.questions for tree_id in args.tree_ids):
        parser.error("every --tree-ids value must be between 0 and questions-1")
    if args.prepare_independent_trace:
        if args.fixed_output_tokens is not None:
            parser.error("independent cohort preparation requires natural generation")
        if set(args.tree_ids) != set(range(args.questions)):
            parser.error(
                "independent trace preparation requires every tree id from 0 to questions-1"
            )
        if not args.bootstrap_service_profile:
            parser.error("--bootstrap-service-profile is required with --prepare-independent-trace")
        if not args.expected_dataset_indices or len(args.expected_dataset_indices) != args.questions:
            parser.error(
                "--expected-dataset-indices must provide one frozen index per question"
            )
        if len(set(args.expected_dataset_indices)) != len(args.expected_dataset_indices):
            parser.error("--expected-dataset-indices must be unique")
        if set(args.expected_dataset_indices) & set(args.excluded_dataset_indices):
            parser.error("frozen cohort indices overlap the excluded prior cohort")
        if len(set(args.excluded_dataset_indices)) != len(args.excluded_dataset_indices):
            parser.error("--excluded-dataset-indices must be unique")
        if not args.excluded_dataset_indices:
            parser.error("--excluded-dataset-indices must list prior used samples")
        if not args.skip_references:
            parser.error("independent cohort plan requires --skip-references")
        if args.reuse_service_profile:
            parser.error("--reuse-service-profile cannot be used while preparing a new trace")
        if args.only_primary_run:
            parser.error("--only-primary-run cannot be used while preparing a new trace")
        if args.primary_runs:
            parser.error("--primary-runs cannot be used while preparing a new trace")
        if args.skip_sanity:
            parser.error("--skip-sanity cannot be used while preparing a new trace")
    elif args.calibration_tree_ids is not None and not args.calibration_tree_ids:
        parser.error("--calibration-tree-ids must contain one or more ids")
    if args.calibration_tree_ids is not None and any(
        tree_id < 0 or tree_id >= args.questions for tree_id in args.calibration_tree_ids
    ):
        parser.error("every --calibration-tree-ids value must be between 0 and questions-1")
    for path_arg in (
        "vllm_bin",
        "python_bin",
        "cuda_home",
        "model_dir",
        "tokenizer_dir",
        "dataset",
        "environment_spec_json",
        "output_profile",
    ):
        path = Path(getattr(args, path_arg))
        if not path.exists():
            parser.error(f"{path_arg.replace('_', '-')} does not exist: {path}")
        if path_arg == "cuda_home" and not (path / "bin/nvcc").is_file():
            parser.error(f"CUDA toolkit has no bin/nvcc: {path}")
    dataset_records = sum(
        1 for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if args.questions > dataset_records:
        parser.error(f"requested {args.questions} questions; dataset has {dataset_records}")
    if any(index < 0 or index >= dataset_records for index in args.excluded_dataset_indices):
        parser.error("every excluded dataset index must exist in the selected dataset")
    sampled_indices = sorted(random.Random(args.seed).sample(range(dataset_records), args.questions))
    if args.prepare_independent_trace and sampled_indices != args.expected_dataset_indices:
        parser.error(
            "seeded sample does not match --expected-dataset-indices: "
            f"expected={args.expected_dataset_indices}, sampled={sampled_indices}"
        )
    args.sampled_indices = sampled_indices
    args.dataset_record_count = dataset_records
    environment_spec = json.loads(
        Path(args.environment_spec_json).read_text(encoding="utf-8")
    )
    canonical_spec = json.dumps(
        environment_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    args.environment_spec_sha256 = hashlib.sha256(canonical_spec).hexdigest()
    if not args.prepare_independent_trace and not Path(args.trace_json).exists():
        parser.error(f"trace-json does not exist: {args.trace_json}")
    if args.prepare_independent_trace:
        bootstrap = Path(args.bootstrap_service_profile)
        if not bootstrap.is_file():
            parser.error(f"bootstrap-service-profile does not exist: {bootstrap}")
    if args.reuse_service_profile:
        profile = Path(args.reuse_service_profile)
        if not profile.is_file():
            parser.error(f"reuse-service-profile does not exist: {profile}")
        args.reuse_service_profile = str(profile.resolve())
    if args.skip_sanity and not args.reuse_service_profile:
        parser.error("--skip-sanity requires --reuse-service-profile")
    if args.preflight_only and not args.prepare_independent_trace:
        parser.error("--preflight-only requires --prepare-independent-trace")
    return args


def main() -> int:
    def request_stop(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)
    args = parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.budget_log = Path(args.budget_log).resolve()
    args.logs_dir = str(Path(args.logs_dir).resolve())
    for path_arg in (
        "vllm_bin",
        "python_bin",
        "cuda_home",
        "model_dir",
        "tokenizer_dir",
        "dataset",
        "trace_json",
        "output_profile",
    ):
        path = Path(getattr(args, path_arg))
        # Resolving a venv's `bin/python` symlink discards its adjacent
        # pyvenv.cfg, so child stages lose the runtime's site-packages.
        normalized = Path(os.path.abspath(path)) if path_arg == "python_bin" else path.resolve()
        setattr(args, path_arg, str(normalized))
    if args.bootstrap_service_profile:
        args.bootstrap_service_profile = str(Path(args.bootstrap_service_profile).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.trace_json)
    if args.prepare_independent_trace and trace_path.parent.resolve() != output_dir.resolve():
        raise SystemExit("trace-json must be stored inside output-dir for the cohort manifest")
    shapes = ("balanced", "broad", "chain", "skewed")
    preflight = {
        "format": "prismserve-direction1-cohort-manifest-v1",
        "status": "SAMPLE_FROZEN_TRACE_PENDING",
        "cohort_id": args.experiment_id,
        "dataset_id": "gsm8k:test",
        "dataset_sha256": file_sha256(Path(args.dataset)),
        "dataset_record_count": args.dataset_record_count,
        "seed": args.seed,
        "question_count": args.questions,
        "selected_dataset_indices": args.sampled_indices,
        "excluded_dataset_indices": args.excluded_dataset_indices,
        "tree_shapes": {
            str(tree_id): shapes[tree_id % len(shapes)]
            for tree_id in range(args.questions)
        },
        "model": MODEL_NAME,
        "environment_spec_sha256": args.environment_spec_sha256,
        "source_run_config": {
            "strategy": "local-only",
            "temperature": 0,
            "max_tokens_inner": 1024,
            "max_tokens_leaf": 1024,
            "natural_eos": True,
            "max_concurrency_per_worker": args.max_concurrency,
            "router_capacity_per_worker": 2,
            "request_timeout_s": 300,
        },
        "calibration_config": {
            "max_tokens": 128,
            "tree_ids": list(range(args.questions)),
        },
        "paired_run_config": {
            "strategies": ["kv-cost-group-flat", "kv-cost-group-tree"],
            "block_order": ["flat", "tree", "tree", "flat"],
            "temperature": 0,
            "max_tokens_inner": 1024,
            "max_tokens_leaf": 1024,
            "natural_eos": True,
            "max_concurrency_per_worker": args.max_concurrency,
            "router_capacity_per_worker": 2,
            "request_timeout_s": 300,
            "reset_prefix_cache_before_each_arm": True,
            "require_vllm_request_metrics": args.enable_request_telemetry,
        },
        "request_telemetry_enabled": args.enable_request_telemetry,
        "gpu_ids": args.gpu_ids,
        "max_concurrency_per_worker": args.max_concurrency,
        "primary_runs": args.primary_runs or (
            [args.only_primary_run]
            if args.only_primary_run
            else [label for label, _ in PRIMARY_RUNS]
        ),
        "max_gpu_min": args.max_gpu_min,
        "prior_gpu_min": args.prior_gpu_min,
        "cleanup_reserve_gpu_min": args.cleanup_reserve_gpu_min,
        "planned_runs": [
            f"{args.experiment_id}:cohort-source-local",
            f"{args.experiment_id}:service-calibration",
            *[f"{args.experiment_id}:{label}" for label, _ in PRIMARY_RUNS],
        ],
        "no_reference_runs": args.skip_references,
        "output_length_profile_sha256": file_sha256(Path(args.output_profile)),
        "bootstrap_service_profile_sha256": (
            file_sha256(Path(args.bootstrap_service_profile))
            if args.bootstrap_service_profile
            else None
        ),
    }
    canonical_preflight = json.dumps(
        preflight, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    preflight["cohort_config_sha256"] = hashlib.sha256(canonical_preflight).hexdigest()
    manifest_path = trace_path.with_suffix(".manifest.json")
    existing_files = list(output_dir.iterdir())
    if args.preflight_only:
        if existing_files:
            raise SystemExit(f"Refusing to overwrite a non-empty output directory: {output_dir}")
        write_json(manifest_path, preflight)
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0
    if existing_files:
        if (
            len(existing_files) != 1
            or existing_files[0].resolve() != manifest_path.resolve()
            or not manifest_path.is_file()
        ):
            raise SystemExit(f"Refusing to reuse a non-empty output directory: {output_dir}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != preflight:
            raise SystemExit("Preflight manifest differs from the requested run configuration")
    else:
        write_json(manifest_path, preflight)
    args.budget_log.parent.mkdir(parents=True, exist_ok=True)
    controller = BudgetController(args, Path(args.logs_dir))
    exit_code = 2
    try:
        controller.launch_services()
        controller.run_experiments()
        exit_code = 0 if controller.state in (
            "PRIMARY_COMPLETE",
            "COMPLETE_WITH_REFERENCES",
            "PRIMARY_COMPLETE_REFERENCES_SKIPPED",
            "PRIMARY_COMPLETE_REFERENCE_RUN_INCOMPLETE",
        ) else 2
    except BaseException as exc:
        controller.last_error = f"{type(exc).__name__}: {exc}"
        primary_labels = {label for label, _ in PRIMARY_RUNS}
        primary_records = [
            stage for stage in controller.stage_records if stage.get("label") in primary_labels
        ]
        primary_complete = len(primary_records) == len(PRIMARY_RUNS) and all(
            stage.get("status") == "COMPLETE" for stage in primary_records
        )
        optional_run_started = controller.state.startswith("RUNNING_reference-") or any(
            stage.get("label", "").startswith("reference-") for stage in controller.stage_records
        )
        if primary_complete and optional_run_started:
            controller.state = "PRIMARY_COMPLETE_REFERENCE_RUN_INCOMPLETE"
        elif any(
            stage.get("status") in ("TIMED_OUT", "INTERRUPTED")
            and stage.get("label") in primary_labels
            for stage in controller.stage_records
        ) or any(stage.get("label") in primary_labels and stage.get("status") != "COMPLETE" for stage in primary_records):
            controller.state = "INCOMPLETE_BUDGET_EVIDENCE"
        elif controller.state != "INCOMPLETE_PRIMARY_BUDGET":
            controller.state = "INCOMPLETE_OR_FAILED"
        controller.save_state()
        print(controller.last_error, file=sys.stderr)
        return 2
    finally:
        controller.cleanup()
        if controller.state == "CLEANUP_INCOMPLETE":
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
