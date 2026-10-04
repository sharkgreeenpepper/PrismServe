# Live-tree strict-output quality validation

**Date:** 2026-10-04  
**Run:** `/home/bumi/infra/cache/direction1-live-tree-quality-validation-20261004`  
**Purpose:** protocol and quality stop/continue gate only; no scheduler performance comparison.

## Setup

- DeepSeek-R1-Distill-Llama-70B served by vLLM 0.30.0+cu129 on two TP=2 replicas, GPUs 4–7.
- GSM8K test SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- Same 16 new questions for direct answer and online tree: `[844,856,376,186,709,498,103,281,838,1242,991,652,927,713,712,661]`. The cohort is disjoint from the 44 prior used/reserved indices, the 12 B1 indices, and the previous 16-question direct diagnostic.
- Direct answers used strict JSON Schema, temperature 0, and up to 512 generated tokens. The tree used the strict node schema, temperature 0, `fanout=3`, `depth=3`, `nodes=9`, `max_tokens=256`, `max_inflight=4`, and the B1 unseen-work calibration.
- Direct requests ran first. Prefix caching remained enabled and was not reset between blocks. The plan excluded latency and cache comparisons.

## Results

| Measure | Direct | Strict-schema live tree |
|---|---:|---:|
| Successful requests / completed trees | 16/16 | 16/16 |
| Invalid structured outputs | 0/16 | 0/136 completed nodes |
| Questions with an answer | 16/16 | 3/16 |
| Exact matches | 6/16 | 2/16 |
| Trees at 9-node cap | — | 13/16 |
| Natural model-terminal branches | — | 4 |

All required tree request metrics were present. The tree arm generated 136 completed nodes and 8,095 completion tokens. Tree p50/p95 were saved for reproducibility but are not compared or interpreted as a scheduler result.

## Frozen gate and decision

The direct schema/request gate passed (16/16). The tree schema-validity gate passed (16/16 trees; zero invalid node outputs). The answer-coverage gate failed (3/16 versus the 12/16 threshold), and the exact-match screen failed (2/16 versus direct 6/16, a four-answer gap; the allowed gap was two). Thirteen trees still used the full node budget.

**Decision: stop before B2.** The JSON protocol is now structurally reliable, but under the fixed B1 search policy the trees usually exhaust their node budget without returning an answer. Do not spend the remaining allocation on a four-arm scheduler matrix for this protocol. A renewed direction-1 attempt would first need a revised search and stopping policy, then a newly held-out protocol check. These 16 questions are retired from tuning and confirmation.

This small diagnostic does not establish a statistical quality difference, model-wide GSM8K accuracy, scheduler effect, latency benefit, SLO goodput, or cache benefit.

## Compute and cleanup

The stage used 13.333 GPU-min against its 100 GPU-min cap. Both servers stopped; ports 8004/8005 were closed and all eight GPUs returned to 0 MiB used. The parent authorization now records 243.615/400 GPU-min used and 156.385 remaining, including the preserved 20 GPU-min cleanup reserve (136.385 usable).

Reproduction artifacts, raw outputs, logs, and checksums are in the run directory. The machine summary is `RESULTS_SUMMARY.json`.
