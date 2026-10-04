# Paired replication: B2 routing at concurrency 4 and 8

**Scope:** Two runs per cell, each on the same frozen 16-tree trace. Results are descriptive; the second run estimates whether the first pattern repeats but does not support strong run-to-run inference.
**Trace SHA-256:** `4e6eff34c48d7d8d25761578aaa9f03618ce70a7c9fc7121a9edc99179eaf13d`
**Integrity:** 128 unique requests, 16 trees, and all 8 required vLLM telemetry fields on every request in each of 8 runs; dataset and tree identities match across runs.

## Per-run outcomes

| Run | Policy | Tree p50 (s) | Tree p95 (s) | Mean request latency (s) | Semaphore wait (s) | vLLM queue (s) | Decode (s) | Exact majority |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| first / c4 | B2-flat | 337.69 | 402.83 | 91.21 | 67.81 | 11.45 | 11.76 | 100% |
| first / c4 | B2-tree | 338.23 | 392.27 | 93.13 | 69.76 | 11.39 | 11.78 | 100% |
| first / c8 | B2-flat | 326.88 | 401.08 | 92.08 | 48.59 | 31.50 | 11.79 | 100% |
| first / c8 | B2-tree | 334.03 | 397.43 | 91.33 | 47.82 | 31.59 | 11.72 | 100% |
| repeat / c4 | B2-flat | 346.54 | 399.97 | 93.18 | 69.52 | 11.52 | 11.95 | 100% |
| repeat / c4 | B2-tree | 336.57 | 395.77 | 92.89 | 69.58 | 11.32 | 11.80 | 100% |
| repeat / c8 | B2-flat | 335.53 | 396.39 | 92.22 | 48.71 | 31.53 | 11.78 | 100% |
| repeat / c8 | B2-tree | 336.36 | 395.51 | 92.84 | 49.17 | 31.67 | 11.79 | 100% |

## Paired tree effect: B2-tree versus B2-flat

Positive means B2-tree completed the same tree faster. Bootstrap intervals resample four trees within each shape (50,000 resamples); they are within-cohort only.

| Run / concurrency | Median speedup | 95% interval | Mean speedup | Faster trees |
|---|---:|---:|---:|---:|
| first run-c4 | +0.10% | [-1.53%, +1.88%] | -0.11% | 9/16 |
| repeat run-c4 | +1.78% | [+0.33%, +2.40%] | +1.48% | 12/16 |
| first run-c8 | -0.78% | [-1.86%, +0.05%] | -0.77% | 6/16 |
| repeat run-c8 | +0.63% | [-1.16%, +1.20%] | +0.13% | 9/16 |

## Run-to-run movement

| Arm | Tree p50 first (s) | Tree p50 repeat (s) | Change |
|---|---:|---:|---:|
| c4-flat | 337.69 | 346.54 | +8.84s (+2.62%) |
| c4-tree | 338.23 | 336.57 | -1.66s (-0.49%) |
| c8-flat | 326.88 | 335.53 | +8.64s (+2.64%) |
| c8-tree | 334.03 | 336.36 | +2.33s (+0.70%) |

## Reading the repeat

The possible routing interaction from the first screen repeats only partially: at c4 the repeat favors B2-tree more than the first run, while at c8 the repeat is near neutral and the first run favored flat. The sign and size therefore are not stable enough to claim a general B2-tree benefit. All four run-level p95 maxima are numerically lower for B2-tree (0.22% to 2.62%), but each is only the maximum of 16 trees. In addition, the repeat p50s shifted by several seconds for the flat arms, making a single-run comparison vulnerable to run variation.

Concurrency 8 again lowers client semaphore waiting and raises vLLM queue waiting substantially relative to c4, while decode time stays similar. This is consistent with moving the queue boundary inside the engine rather than increasing effective model throughput under `max_num_seqs=2`.

Factual request-trace replay previously matched measured timing closely with 1–10 ms grouping windows and showed semaphore waiting dominates the observed request critical paths; at a 20 ms grouping window, B2-flat replay p50 error grew to 8.75 s and the maximum tree error to 18.01 s. This repeat tests the resulting routing/concurrency screen on the frozen trace. It does not show that B2-tree improves end-to-end serving in general.

## Limits and next step

There are two runs per cell and only 16 trees. The bootstrap intervals quantify tree-to-tree variation within this chosen cohort; they do not measure run-to-run, time-of-day, or hardware variability. Majority accuracy is 100% for these sampled trees, which is not an accuracy claim for GSM8K overall.

Result-to-claim verdict: no (same-family GPT-6-Astra review; provisional), with high confidence that these results do not support a stable latency claim. The evidence does not justify expanding the current heuristic unchanged or launching a larger GPU matrix. First diagnose placement, generated-token differences, and client/engine admission accounting from existing records; resume only with a distinct mechanism and preregistered practical effect/stopping rule. No independent experiment-integrity audit is available.

## Reproducibility

- Repeat output and logs: `/home/bumi/infra/cache/prismserve-direction1-replication-20261004/attempt2`
- First c4 source: `/home/bumi/infra/cache/prismserve-direction1-request-telemetry-20261004/`
- First c8 source: `/home/bumi/infra/cache/prismserve-direction1-concurrency8-20261004/attempt2/`
- Analysis script SHA-256: `ec544845c6c90aabe7439c28abc3e79879bb656030aff0aed06b5673d4a4bc0a`

## Result-to-claim judgment

**Verdict:** `claim_supported: no`. A same-family GPT-6-Astra reviewer using the user's requested light reasoning level judged, with high confidence, that this evidence does not support a stable latency claim. Acceptance remains provisional. The deterministic evidence precheck found 61/61 cited values; this verifies their presence, not their interpretation. No independent experiment-integrity audit is available.

The reviewer observed that all four run-level p95 maxima numerically favored B2-tree, but each maximum is based on only 16 trees. This does not support a reliable tail-latency claim. The current heuristic should not be expanded unchanged. Any new experiment should start from a concrete revised mechanism and a preregistered practical effect and stopping rule.

Full raw request, tree, service logs, manifests, and GPU ledger are stored at `/home/bumi/infra/cache/prismserve-direction1-replication-20261004/attempt2/`. The full reviewer trace is local at `.aris/traces/result-to-claim/2026-10-04_run01/` and is excluded from Git.
