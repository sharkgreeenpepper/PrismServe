# Weighted lookahead weight screen — 2026-10-04

## Status

**Stopped after block A under the frozen early-stop rule.** Weight 0.5 improved the median tree-completion latency over flat by 0.94%, below the required 5% in this first block. Block B was not run. The result does not support continuing the scalar weight-0.5 change as specified.

## Raw comparison

The primary metric is the median of 16 individual tree-completion latencies in each run. It is not the elapsed time for the whole 16-tree run. The separate `inference_wall_time_s` values below are the full-run wall times.

| Condition | Tree p50 (s) | Tree p95 (s) | p50 speedup vs flat | Majority-correct trees | Requests | Full-run wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| Flat (weight 0) | 333.744 | 389.148 | — | 16/16 | 128/128 | 389.153 |
| Tree (weight 1.0) | 327.680 | 383.004 | +1.82% | 16/16 | 128/128 | 383.005 |
| Tree (weight 0.5) | 330.590 | 383.970 | +0.94% | 16/16 | 128/128 | 383.971 |

The weight-0.5 arm was 2.91 seconds, or 0.89%, slower at p50 than weight 1.0 in this single block. Neither tree arm reached the frozen 5% screen. The 16-tree p95 is descriptive and is not a stable tail-latency estimate.

Per-tree latency, delta, and correctness values are in [`WEIGHTED_LOOKAHEAD_PER_TREE_20261004.csv`](../../WEIGHTED_LOOKAHEAD_PER_TREE_20261004.csv). Complete run summaries, tree rows, request rows, run plan, and budget ledger are retained under [`attempt1/`](attempt1/); the artifact hashes and frozen input identities are in [`ARTIFACT_MANIFEST.json`](attempt1/ARTIFACT_MANIFEST.json).

The follow-up, no-GPU estimate check found a likely calibration issue: the held-out leaf output profile was 532.5 tokens, while the flat arm's 60 leaves had 391.7 mean and 355 median output tokens. Per-tree root work estimates had 13–18% median absolute relative error and 0.57–0.63 Spearman rank association with realized descendant TTFT-plus-decode work. See [`the exploratory forecast diagnostic`](../../WEIGHTED_LOOKAHEAD_FORECAST_DIAGNOSTIC_20261004.md) and its [per-tree CSV](../../WEIGHTED_LOOKAHEAD_FORECAST_DIAGNOSTIC_20261004.csv). This analysis is post hoc and does not change the failed 5% gate. The output-length prior is a plausible calibration error, not an established cause of the routing or latency outcome.

## Findings

1. **The latency screen failed.** The 0.5 weight delivered a 0.94% p50 reduction versus flat, short of the required 5%. The preregistered runner therefore did not launch block B; there is no second-block replication or inferential test.
2. **The parameter changed routing but did not produce the required aggregate benefit.** Weight 0.5 changed the assigned worker for 40 of 128 matched nodes relative to weight 1.0. It changed 57/128 assignments relative to flat; weight 1.0 changed 47/128 relative to flat. These are descriptive routing counts from one cohort.
3. **Quality and telemetry completeness gates passed.** Each arm completed all 16 trees and 128 requests. All three arms had 16/16 majority-correct trees, with request ID, server queue, time-to-first-token, and generation-time observations present for 128/128 requests.
4. **The measured queue summaries do not explain a weight-0.5 win.** Mean server request-queue time was 11.081 s for flat, 11.018 s for weight 1.0, and 11.244 s for weight 0.5. Mean TTFT was 162, 172, and 171 ms respectively. These are run-level summaries, not causal decompositions.

| Tree ID | Flat (s) | Weight 1.0 (s) | Weight 0.5 (s) | Weight 0.5 vs flat |
|---:|---:|---:|---:|---:|
| 1 | 190.262 | 181.146 | 179.451 | +5.68% |
| 13 | 319.452 | 285.674 | 316.985 | +0.77% |

Tree IDs 1 and 13 were selected as diagnostics from earlier results, so these individual comparisons are post hoc. They do not override the aggregate screen.

## Experimental scope and limits

- All three arms reused the same frozen 16-tree GSM8K trace, dataset, service-rate profile, and output-length profile. The source trace SHA-256 was `4e6eff34c48d7d8d25761578aaa9f03618ce70a7c9fc7121a9edc99179eaf13d`.
- The single executed order was weight 1.0 → flat → weight 0.5. Prefix caches were reset before each condition. The planned reversed block was skipped by the prespecified gate.
- Completion-token totals differed across arms (58,312 flat; 58,609 at weight 1.0; 58,908 at weight 0.5) despite identical prompts and decoding configuration. Natural generation and run order therefore remain possible sources of variation.
- This is a one-block screening result on one reused cohort, not an independent-cohort confirmation, significance test, shape-general result, correctness non-inferiority result, or SLO claim.
- The campaign used 83.915 GPU-min of the approved 200 GPU-min cap. All owned vLLM services were stopped, selected GPUs returned to idle, and ports 8004/8005 were free after cleanup.

## Next step

Stop the current weight-0.5 sweep. Use the retained per-request records for offline calibration analysis of predicted remaining-tree work versus realized request service and queue time. Only propose another GPU run if that analysis yields a new, falsifiable routing hypothesis and a new preregistered screen; do not spend the remaining budget on block B after the frozen stop condition.

## Provisional result-to-claim verdict

**Not supported** for the claim that weight 0.5 beats flat by at least 5% in both blocks. The result supports only the narrower observation that weight 0.5 changed placements and produced a 0.94% p50 reduction in block A, below the decision threshold. A same-family GPT-6-Astra review at the lowest supported reasoning effort (`low`) returned `no` with high confidence; acceptance remains provisional and external review is pending. The deterministic precheck found 5/5 cited values, which verifies presence only. No integrity audit specifically covers this screen. See the local trace at `.aris/traces/result-to-claim/2026-10-04_run02/`.
