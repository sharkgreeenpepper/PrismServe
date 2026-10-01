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

The deterministic precheck verifies the cited metric values exist. No experiment integrity audit artifact was found; integrity status is unavailable. The semantic reviewer was same-family and provisional, with external review pending. See `RESULTS_ANALYSIS.md` and `.aris/traces/result-to-claim/2026-10-01_run02/`.

## Fixed-output diagnostic (2026-10-01)

- Four main runs and the one-tree sanity run returned exactly 256 completion tokens per request; both paired blocks passed the server-queue pressure gate. Flat/tree median tree completion was 59.086/59.516 s in B1 and 59.250/59.292 s in B2, so the aggregate difference was within 1%, not the prior natural-generation gap of about 10.3%.
- Tree-level effects were mixed but repeated: tree 3 was about 11% slower under tree-aware placement, while tree 5 was 9.8–10.9% faster. Tree routing changed 9/38 and 11/38 node placements, and its estimated cached-prefix tokens fell from 9,712 to 8,672/7,968 while aggregate queue mean rose 0.32–0.41 s. The queue and cache measurements are run-level; they do not identify per-tree causes.
- **Interpretation:** equalizing generated work removes the aggregate latency gap in this four-tree diagnostic, consistent with output-work differences contributing to the earlier result. This does not establish causality because the router's output-length profile also changed. The result is inconclusive under the frozen ±5% screen and does not support a general benefit or ineffectiveness claim. Do not spend more compute repeating the same four trees; any follow-up should use an independent tree cohort, natural generation as the primary endpoint, and fixed-output runs only as a diagnostic.
- Detailed results: `FIXED_OUTPUT_DIAGNOSTIC_RESULTS.md`; raw summaries and logs: `results/20261001_113516_fixed_output_256/`.
