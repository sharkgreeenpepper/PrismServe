# Live-tree terminal-budget quality validation

**Date:** 2026-10-04
**Run:** /home/bumi/infra/cache/direction1-terminal-budget-quality-20261004
**Purpose:** fresh-cohort quality stop/continue screen after adding forced terminal-answer budget. No scheduler, latency, cache, or SLO claim.

## Frozen setup

- Model/runtime: DeepSeek-R1-Distill-Llama-70B, vLLM 0.30.0+cu129, two TP=2 replicas on GPUs 4–7.
- Dataset: GSM8K test, SHA-256 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14.
- Direct and tree arms used the same new cohort: [1036,350,1112,1102,142,741,791,85,755,678,1140,347,1208,434,488,1261]. The cohort was checked against 88 prior used/reserved indices.
- Tree policy: unseen-work; temperature 0; max nodes 9; max depth 3; max fanout 3; max in-flight 4; per-node max 256 tokens. At max depth and for the last two allocatable nodes, the model is required to return a final numeric answer under strict schema; forced-final nodes cannot expand and receive predicted future work zero.
- Frozen gates: direct valid responses at least 16/16; valid trees at least 15/16; numeric answer coverage at least 12/16; tree exact-match drop versus direct no more than 2; forced-final schema violations zero; max-depth pruning zero. Stage hard cap 40 GPU-min.

## Results

| Measure | Direct | Live tree |
|---|---:|---:|
| Valid direct responses / complete trees | 16/16 | 16/16 |
| Numeric final-answer coverage | 16/16 | 16/16 |
| Exact match | 8/16 | 5/16 |
| Forced-final nodes | — | 23 |
| Natural terminal branches | — | 7 |
| Trees reaching 9-node cap | — | 9/16 |
| Completed nodes / parseable node JSON | — | 100 / 99 |
| Forced-final schema violations | — | 0 |
| Max-depth child pruning | — | 0 |

The tree exact-match drop was 3, greater than the allowed 2, so the accuracy gate failed. Paired correctness was: both correct 5; direct-only correct 3; tree-only correct 0; both wrong 8. One node output was malformed after finish_reason=length (question index 678, node 3); that tree still had a numeric answer and complete metrics. A gold-aware offline candidate oracle is not a runnable policy and is not evidence of online accuracy.

Per-question outcomes:

| Index | Gold | Direct answer | Tree answer | Nodes | Forced-final nodes |
|---:|---:|---|---|---:|---:|
| 1036 | 10 | 10 (correct) | 10 (correct) | 1 | 0 |
| 350 | 8 | 6.5 (wrong) | 6 (wrong) | 1 | 0 |
| 1112 | 5 | 5.25 (wrong) | 5.5 (wrong) | 9 | 2 |
| 1102 | 113 | 84.5 (wrong) | 70 (wrong) | 9 | 2 |
| 142 | 140 | 140 (correct) | 140 (correct) | 9 | 3 |
| 741 | 28 | 32 (wrong) | 32 (wrong) | 9 | 2 |
| 791 | 100 | 100 (correct) | 100 (correct) | 1 | 0 |
| 85 | 44 | 44 (correct) | 44 (correct) | 9 | 2 |
| 755 | 7 | 7 (correct) | 5 (wrong) | 1 | 0 |
| 678 | 120 | 120 (correct) | 120 (correct) | 7 | 1 |
| 1140 | 27 | 30 (wrong) | 15 (wrong) | 7 | 2 |
| 347 | 36 | 36 (correct) | 24 (wrong) | 1 | 0 |
| 1208 | 27 | 9 (wrong) | 9 (wrong) | 9 | 2 |
| 434 | 5 | 5 (correct) | 3 (wrong) | 9 | 2 |
| 488 | 6 | 3 (wrong) | 9 (wrong) | 9 | 3 |
| 1261 | 235 | 345 (wrong) | 350 (wrong) | 9 | 2 |

## Interpretation and decision

The terminal budget repaired numeric-answer coverage in this fresh cohort, but did not meet the frozen accuracy screen. The earlier B1.5 cohort had 3/16 answers and 2/16 exact matches; this follow-up used different questions, so the coverage difference is descriptive only and is not a paired improvement estimate. There is no tree-only-correct question in this cohort.

**Decision: stop before B2.** Do not run scheduler baselines or claim lower latency, SLO improvement, cache benefit, or general accuracy. The 16-question screen is not a formal non-inferiority test and cannot establish model-wide GSM8K accuracy. Reopening requires a materially different, frozen search/answer-selection policy and another untouched cohort.

Tree p50/p95 were saved as descriptive-only values and are not interpreted as performance results.

## Compute and cleanup

The stage used 13.471 GPU-min against its 40 GPU-min cap. The authorized parent ledger is 257.086/400 used, 142.914 remaining, with the 20 GPU-min cleanup reserve preserved (122.914 usable). Both serving processes were stopped; ports 8004/8005 were closed; all eight GPUs returned to 0 MiB.

Reproduction artifacts, raw outputs, logs, and checksums are in the run directory. See RESULTS_SUMMARY.json, RUN_PLAN.json, corrected BUDGET_LOG.json, ARTIFACT_MANIFEST.json, and stage_status.jsonl.
