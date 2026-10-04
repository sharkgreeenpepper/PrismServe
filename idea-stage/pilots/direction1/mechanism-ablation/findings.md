# Research Findings: Direction 1 Mechanism Ablation

## Claim tested

Adding the tested calibrated descendant-work term to the flat queue/KV-cost grouping scheduler would reduce median tree completion latency by at least 5% in both counterbalanced blocks, while not reducing majority exact-match correctness.

## Result

The claim was **not supported**. Tree-aware median tree-completion latency was 10.28% higher than flat in B1 and 10.29% higher in B2, failing the prespecified latency criterion in both blocks. The queue-pressure gate passed in all four primary runs. Observed majority-correct counts were 4/4 for tree-aware and 3/4 for flat, but four trees do not support a correctness-improvement or non-inferiority claim. The same-family `gpt-6-astra/ultra` result-to-claim review also returned `no` with provisional acceptance status.

## What was tested

- Real vLLM inference on two TP=2 replicas with the same 38 frozen requests per run and four evaluation trees (IDs 1, 3, 5, 7).
- The flat and tree policies shared the calibrated queue/KV-cost model and sibling-ready assignment procedure; the tested difference was the predicted remaining-descendant-work term.
- Two reversed-order blocks were completed. All primary CSVs are complete. Calibration and the separate short-token sanity run completed. Local-only and least-loaded references were skipped because the remaining work window was below the planned 360-second minimum.
- Every arm used vLLM's native sampler after FlashInfer sampler startup incompatibility. Results should not be compared in absolute latency to earlier runs using a different sampling backend.

## Interpretation and unresolved explanations

The result is evidence against continuing the **current heuristic unchanged** on this workload, not against the broader idea of tree-aware scheduling. Trees 1, 3, and 5 regressed in both blocks; tree 7 improved in both. The tree arm generated 3.65–3.86% more completion tokens and had 0.43–0.50 seconds more weighted mean server queue time per block. These differences could explain part of the latency result, but the experiment does not establish causality. Output lengths were predicted from an earlier small pilot, and the request outputs were omitted from the saved records. Leaf parsing was low: 70% for flat and 75% for tree, with only 1/6 leaves parsed for tree 5 in the tree arm.

## Constraints for future work

- Do not claim a latency win, reliable p95 behavior, correctness non-inferiority, SLO-goodput improvement, or broad ineffectiveness from this run.
- Keep the 5% two-block screening threshold frozen for any direct replication; evaluate a revised heuristic as a new experiment rather than changing this result's threshold.
- Do not compare absolute latency with the earlier FlashInfer-sampler pilot.
- Before another GPU sweep, inspect the per-node placement and timing records, especially the repeated regressions on trees 1, 3, and 5. Diagnose generated work separately from scheduler placement effects.
- If continuing, use additional independent trees and more randomized/counterbalanced blocks; keep natural end-to-end generation as the primary endpoint and add a controlled-output-length diagnostic.

## Audit status

The deterministic precheck verifies the cited metric values exist. The experiment-integrity audit is same-family and provisional with **WARN** status; see `EXPERIMENT_AUDIT.md`. The semantic reviewer was same-family and provisional, with external review pending. See `RESULTS_ANALYSIS.md` and `.aris/traces/result-to-claim/2026-10-01_run02/`.

## Fixed-output diagnostic (2026-10-01)

- Four main runs and the one-tree sanity run returned exactly 256 completion tokens per request; both paired blocks passed the server-queue pressure gate. Flat/tree median tree completion was 59.086/59.516 s in B1 and 59.250/59.292 s in B2, so the aggregate difference was within 1%, not the prior natural-generation gap of about 10.3%.
- Tree-level effects were mixed but repeated: tree 3 was about 11% slower under tree-aware placement, while tree 5 was 9.8–10.9% faster. Tree routing changed 9/38 and 11/38 node placements, and its estimated cached-prefix tokens fell from 9,712 to 8,672/7,968 while aggregate queue mean rose 0.32–0.41 s. The queue and cache measurements are run-level; they do not identify per-tree causes.
- **Interpretation:** equalizing generated work removes the aggregate latency gap in this four-tree diagnostic, consistent with output-work differences contributing to the earlier result. This does not establish causality because the router's output-length profile also changed. The result is inconclusive under the frozen ±5% screen and does not support a general benefit or ineffectiveness claim. Do not spend more compute repeating the same four trees; any follow-up should use an independent tree cohort, natural generation as the primary endpoint, and fixed-output runs only as a diagnostic.
- Detailed results: `FIXED_OUTPUT_DIAGNOSTIC_RESULTS.md`; raw summaries and logs: `results/20261001_113516_fixed_output_256/`.

## Independent natural-generation cohort (2026-10-01)

- **Claim judgment:** the intended ≥5% tree-aware median-latency improvement is **not supported** (`claim_supported: no`). The preregistered screen outcome is **inconclusive**: B1 favored tree-aware placement by 1.25%, while B2 was 6.81% slower under tree-aware placement. The two-block positive screen and the consistent reverse screen both fail. The queue-pressure gate passed in all four completed arms.
- **Evidence:** four disjoint GSM8K questions, four tree shapes, 32 matched trace nodes per arm; all arms have 12,105 prompt tokens and complete queue/prefill/decode observations. All recorded 4/4 majority-exact trees and 15/15 parsed exact-match leaves in each arm, but the sample is descriptive only and full generated text was not retained for independent output review.
- **Execution caveat:** the budget ledgers report B2-flat timeouts at 30/32 and 31/32 before a 32/32 completion in attempt4. The two partial CSVs are absent from the retained result tree, so their row counts, missing-node identities, and 13,002-token total cannot be independently recomputed from per-request records. Only one B2-flat run completed. It ran in a separate restarted server session, so B2 preserved tree→flat order but was not an uninterrupted temporal crossover; do not treat completion as proof of session comparability.
- **Integrity/review:** deterministic evidence precheck passed 14/14 cited values; this confirms value presence only. A same-family, provisional experiment-integrity audit returned **WARN**: supplied summary arithmetic reproduced, but the redacted trace hash lacks a transformation manifest and partial-run details lack retained CSVs. The audit found no evidence of fabrication. The semantic review also found high confidence in the arithmetic and inconclusive screen classification, not in causal attribution or generalization.
- **Next step:** preserve this cohort and its frozen thresholds; do not rerun these four questions. Before another GPU sweep, add a trace redaction/hash manifest and either recover the excluded partial CSVs or leave their detailed counts explicitly marked as ledger-reported and not independently verified. Any follow-up should add independent questions per shape, randomized/counterbalanced order, explicit restart/censoring rules, and retention of generated outputs. Keep natural generation as the primary endpoint and treat fixed-output runs only as diagnostics.
- Detailed report and raw artifacts: `results/20261001_225325_independent_natural_generation/INDEPENDENT_COHORT_RESULTS.md`.

## Independent natural-generation cohort (2026-10-04)

- **Claim judgment:** the ≥5% tree-aware latency benefit is **not supported** by this screen (`claim_supported: no`). Under the frozen rule, the result is **inconclusive**: median per-question paired speedup −0.45% (shape-stratified bootstrap 95% CI −2.56% to +1.25%); B1 and B2 medians were −0.42% and −0.48%. The queue gate passed in all four policy arms.
- **Evidence:** 16 new GSM8K questions, four tree shapes, 128 frozen trace nodes, and four complete policy arms with 128 request rows and 16 tree rows each. Prompt-token mismatches were zero, and completion text was retained. Tree arms used 0.53% fewer completion tokens than flat arms; majority exact match was 15/16 for tree and 16/16 for flat, with the same tree-arm miss at dataset index 57 in both blocks. These quality and token differences are descriptive and do not establish a causal explanation.
- **Execution:** 142.655 of the approved 400 GPU-min were used; all source, calibration, and primary stages completed, the optional references and fixed-output diagnostic were not run, and the 20 GPU-min cleanup reserve remained. vLLM processes stopped and all GPUs were idle after cleanup.
- **Integrity/review:** deterministic evidence precheck verified textual presence for 5/5 cited values only. The same-family GPT-6-Astra light integrity audit returned **WARN**; no fabrication or self-normalization was found. The plan and tracker are now updated; sample-selection implementation and the absence of explicit session-ID fields are documented. The same-family result-to-claim review returned `no`, with provisional status pending external review.
- **Scope:** this is a practical frozen-history synthetic branching replay on one model and one 16-question cohort. It does not establish live-search performance, causal effects independent of generated work, correctness non-inferiority, tail-SLO gains, or broad generalization. No fixed-output diagnostic or powered confirmation cohort was run.
- Detailed summary and key artifact hashes: `refine-logs/INDEPENDENT_COHORT_RESULTS_20261003.md`; row-level outputs remain in the run artifact cache.

## Weighted lookahead weight screen (2026-10-04)

- **Claim judgment:** `claim_supported: no` for the preregistered claim that weight 0.5 is at least 5% faster than flat in both blocks. The same-family GPT-6-Astra review at its lowest supported reasoning effort (`low`) returned `no` with high confidence; semantic acceptance is provisional, with external review pending.
- **Evidence:** in block A, flat p50 was 333.744 s, weight 1.0 was 327.680 s (+1.82%), and weight 0.5 was 330.590 s (+0.94%). Weight 0.5 was 0.89% slower than weight 1.0. All three arms completed 16/16 trees and 128/128 requests, with the required request-ID, queue, TTFT, and generation telemetry present for every request; majority-correct count was 16/16 in each arm.
- **Mechanism observation:** weight 0.5 changed worker assignment for 40/128 matched nodes relative to weight 1.0, but did not meet the latency threshold. Mean vLLM queue time was 11.244 s at weight 0.5 versus 11.081 s flat and 11.018 s at weight 1.0. This one-block comparison does not identify a causal mechanism.
- **Execution and scope:** the frozen early-stop gate was applied, so block B was skipped. The campaign consumed 83.915/200 GPU-min and all services were stopped. This is one sequential block on a reused 16-tree GSM8K cohort; completion-token totals differed across arms, and no uncertainty estimate or independent cohort exists. The deterministic evidence precheck verified 5/5 values for presence only; no integrity audit specifically covers this screen.
- **Offline follow-up:** a post-hoc no-GPU diagnostic found 12.97–18.49% median absolute relative error between predicted remaining descendant work and realized descendant TTFT-plus-decode. The held-out leaf output-length mean was 391.7 tokens versus a 532.5-token prior (26.4% lower), with no leaf reaching the token limit. This flags a likely calibration issue, but does not establish that it caused the routing or latency result. Summed descendant queue waits are cumulative request-level values, not tree wall time. Full method and caveats are in `WEIGHTED_LOOKAHEAD_FORECAST_DIAGNOSTIC_20261004.md`.
- **Decision:** stop the current weight-0.5 sweep and do not spend the remaining budget on block B. Before any new GPU allocation, build a larger shape-stratified calibration set, separate output-length error from per-worker service-rate error, and evaluate a revised falsifiable hypothesis on a new independent cohort under a new frozen plan.
- **Artifacts:** detailed report at `results/20261004_124417_weighted_lookahead_screen/WEIGHTED_LOOKAHEAD_RESULTS.md`; retained summaries, trees, requests, plan, and budget ledger under `results/20261004_124417_weighted_lookahead_screen/attempt1/`. Same-family semantic review trace is local at `.aris/traces/result-to-claim/2026-10-04_run02/` and excluded from Git.
