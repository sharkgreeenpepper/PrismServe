# B1 live-tree mechanism screen

**Date:** 2026-10-04  
**Scope:** mechanism and execution screen; not a performance evaluation  
**Raw run:** `/home/bumi/infra/cache/direction1-live-tree-b1-20261004_2029`  
**Integrity audit:** [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)

## Setup

- Model: DeepSeek-R1-Distill-Llama-70B, vLLM 0.30.0+cu129.
- Two replicas, tensor parallel size 2 each, GPUs 4–7.
- Dataset: GSM8K test, SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- Predictor calibration: 4 questions `[18, 56, 300, 340]`; held-out mechanism screen: 8 disjoint questions `[420, 743, 976, 996, 1049, 1058, 1166, 1210]`.
- Search caps: fanout 3, depth 3, 9 nodes per tree, 256 generated tokens per node, max in-flight 4, temperature 0.
- Arms: QKV-only versus QKV plus calibrated unseen-descendant work. Each arm ran once; test questions were processed sequentially.

The first calibration attempt used unconstrained JSON output. All four responses reached the token limit as ordinary text and stopped at the root as `malformed_json`. Those files are retained under `calibration/` and excluded from the corrected calibration. The corrected run uses JSON-object response format and is under `calibration-json/`.

## Results

| Measure | QKV | Unseen-work | Reading |
|---|---:|---:|---|
| Same-snapshot questions with at least one changed routing batch | — | 8/8 | The added term changes assignments on all observed test questions. |
| Cross-run route changed | — | 5/8 | Actual route logs differ on five paired questions. |
| Equal observed topology across arms | — | 5/8 | Three paired questions released different paths. |
| Completed trees / requests | 8/8; 72/72 | 8/8; 72/72 | All required request telemetry is present. |
| Trees reaching the 9-node cap | 8/8 | 8/8 | Every test tree used its full node budget. |
| Questions with any terminal answer | 3/8 | 3/8 | Five questions per arm had no final answer. |
| Exact-match questions | 1/8 | 1/8 | Equal observed counts; too small for a quality claim. |
| Tree latency p50 | 6.363 s | 6.223 s | Descriptive; one run per arm. |
| Tree latency p95 | 9.804 s | 9.577 s | With 8 samples, nearest-rank p95 is the maximum. |
| Mean actual cached prompt tokens/request | 296.9 | 357.6 | Prefix-cache telemetry, not a randomized cache comparison. |
| Total cached prompt tokens | 21,376 | 25,744 | Proposed arm logged 20.4% more cached prompt tokens. |

For additional request telemetry, mean queue time was 0.041 ms for QKV and 0.353 ms for unseen-work; mean TTFT was 85.4 ms and 53.4 ms; mean generation time was 1.484 s and 1.508 s. Total prompt tokens were almost identical (27,874 vs 27,876). The small tree-latency difference is confounded by retained prefix cache and fixed arm order; it does not isolate routing as the cause.

The B1 comparison JSON reports `route_change_gate_pass=true`, `all_trees_complete=true`, `all_required_metrics_complete=true`, and `b1_screen_gate_pass=true`. This is a mechanism-screen pass only. It says that a same-state dispatch can change; it does not check answer quality, answer availability, natural stopping, predictor accuracy, or causal latency.

The calibration fit used 32 node observations from four capped trees. Its in-sample node RMSE changed from 1.414 to 1.400, and token RMSE from 100.824 to 95.931. These are fit values, not held-out prediction accuracy. All four corrected calibration trees also reached the 9-node cap and produced no terminal answer.

## SLO and scheduler overhead

No tree-latency SLO was specified before this screen, so no formal SLO-goodput number is reported. As a descriptive threshold count only, 6/8 QKV trees and 7/8 unseen-work trees finished under 8 seconds; both arms were 8/8 under 10 seconds. These cutoffs were not preregistered.

The runner recorded model queue, TTFT, generation, and cache telemetry, but did not time the router's CPU decision path. Scheduler overhead is therefore unavailable for this run and must be instrumented before the next performance experiment.

## Outcome

The B1 mechanism condition is met: same-snapshot route changes occurred for 8/8 test questions. The broader experiment is not ready to support a useful tree-search or performance claim. Only 3/8 questions per arm had any terminal answer, exact match was 1/8 in both arms, and every tree saturated the node cap. The independent audit is **WARN**; see [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md).

The experiment used 67.335 GPU-min under its 80 GPU-min B1 cap. At B1 close, the parent 400 GPU-min ledger had 190.010 GPU-min remaining, including the 20 GPU-min cleanup reserve. A separate direct-answer quality diagnostic later used 20.292 GPU-min and exceeded its own 20 GPU-min stage cap by 0.292; the parent reserve remained intact. Its results are documented separately and are not part of B1.

## Reproduction artifacts

- Raw events, configs, logs, and summaries: `/home/bumi/infra/cache/direction1-live-tree-b1-20261004_2029/`
- Machine-readable comparison: `B1_COMPARISON.json` in that run directory.
- Budget and launch plan: `BUDGET_LOG.json` and `RUN_PLAN.json` in that run directory.
- Artifact checksums: `ARTIFACT_MANIFEST.json` in that run directory.
