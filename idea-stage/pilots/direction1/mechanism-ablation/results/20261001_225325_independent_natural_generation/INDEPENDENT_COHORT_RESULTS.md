# Independent natural-generation cohort results

Status: all four primary runs are complete. The two timed-out B2-flat attempts are excluded. The same-family result-to-claim review returned `claim_supported: no`; the frozen screen outcome is **inconclusive**. A same-family, provisional experiment-integrity audit returned **WARN**; see `../../EXPERIMENT_AUDIT.md`.

## Design and evidence integrity

- Dataset: GSM8K test, seed `20261002`, selected indices `[289, 329, 758, 1108]`. These are disjoint from the eight questions used in the earlier direction-1 pilot.
- Trees 0–3 are balanced, broad, chain, and skewed, with 7, 10, 6, and 9 nodes (32 requests per arm).
- Each primary arm replayed the same frozen natural-history prompt trace. The runtime trace SHA-256 recorded in the run summaries is `51e7ea09d3d050c96718a1cef784f5c91500836de278e11b6fd7cdf003c5b483`; the committed redacted `cohort_trace.json` has SHA-256 `3b4d99431e488e5b7a0328b5108e5c3238072221e7b09e893b085b51736b4dec`. Redacted path fields change the file bytes. This package does not include a reversible path-redaction manifest, so the runtime digest cannot be reproduced from the exported file alone. Strict tokenizer prompt-count verification passed for all nodes.
- Inference used two vLLM replicas (tensor parallel size 2 each), temperature 0, natural EOS, and a 1,024-token request cap. There was no forced output length. Prefix caches were reset before every arm.
- B1 ran flat then tree; B2 ran tree then flat. Each primary CSV contains 32 complete request rows, and queue, prefill, and decode metrics each have 32 observations per arm.
- Each arm has the same 12,105 prompt tokens across its 32 requests. All request prompt counts match the frozen trace; all summaries report the same runtime trace hash.
- The source pass used `local-only` only to generate natural parent outputs for the trace. It is not included in the comparisons.

## Primary results

The primary metric is each arm's median of its four tree-completion latencies. The “tree speedup” column is `(flat median - tree median) / flat median`; positive values favor tree-aware placement.

| Block | Flat median (s) | Tree median (s) | Tree speedup | Flat round wall (s) | Tree round wall (s) | Flat mean server queue (s) | Tree mean server queue (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1, flat → tree | 75.078032 | 74.139680 | +1.25% | 107.566080 | 94.857090 | 6.937381 | 6.918331 |
| B2, tree → flat | 69.356858 | 74.082937 | −6.81% | 94.809192 | 94.835268 | 6.696297 | 6.915637 |

All four server-queue means exceed the frozen 0.1-second pressure threshold, with 32 queue observations per arm. Weighted means use summed replica seconds divided by summed replica observations; raw before/after counter deltas reproduce the reported aggregates. Queue, prefill, decode, TTFT, inference, and end-to-end metrics each have 32 observations per arm. Weighted prefill/decode means (seconds per request) are B1-flat 0.118654/9.911588, B1-tree 0.131299/9.501500, B2-tree 0.130586/9.494230, and B2-flat 0.133433/9.310891. The queue-pressure gate passed.

The “round wall” values cover completing all four trees in that arm. They are distinct from the primary median of four tree latencies and from request-level p50 latency.

## Tree-level variation

Percent change is `(tree-aware latency - flat latency) / flat latency`; positive values mean tree-aware placement was slower.

| Tree | Shape | B1 change | B2 change |
|---:|---|---:|---:|
| 0 | balanced | +4.07% | +7.69% |
| 1 | broad | +5.94% | +0.96% |
| 2 | chain | −11.82% | +0.03% |
| 3 | skewed | −6.13% | +5.94% |

## Correctness and generated work

All arms recorded 4/4 majority-exact trees, 4/4 trees with at least one exact leaf, and 15/15 parsed exact-match leaves. No request hit the length cap. The leaves are correlated within four questions, not 15 independent quality examples. These descriptive values do not establish correctness non-inferiority or improvement. The saved summaries mark full output text as unrecorded, so the reviewer could not independently inspect the generated reasoning or re-evaluate parsing decisions.

Generated work varied despite the shared prompt trace: B1 flat and tree produced 12671 and 12119 completion tokens; B2 tree and flat produced 12119 and 11938. These totals match sums from request and tree CSVs.

| Tree | Shape | B1-flat tokens | B1-tree tokens | B2-tree tokens | B2-flat tokens |
|---:|---|---:|---:|---:|---:|
| 0 | balanced | 3,419 | 3,415 | 3,415 | 3,439 |
| 1 | broad | 3,360 | 3,415 | 3,415 | 3,263 |
| 2 | chain | 2,977 | 2,642 | 2,642 | 2,605 |
| 3 | skewed | 2,915 | 2,647 | 2,647 | 2,631 |
| **Total** | | **12,671** | **12,119** | **12,119** | **11,938** |

B1-tree produced 4.36% fewer tokens than B1-flat; 16/32 matched requests changed length. The chain lost 335 tokens, including one inner request changing from 963 to 632 tokens. B2-tree produced 1.52% more tokens than B2-flat; 17/32 matched requests changed length. Natural-generation timing therefore includes differences in generated work and cannot isolate the placement heuristic's causal contribution. Queue measurements are server-side run aggregates, not per-tree queue waits, predicted router queue work, or client semaphore waits.

## Screen interpretation

The frozen screen required at least a 5% tree-aware speedup in both blocks for a positive mechanism signal, or at least a 5% slowdown in both blocks for consistent reverse evidence. B1 favored tree-aware placement by only 1.25%; B2 was 6.81% slower under tree-aware placement. Neither screen passes. The result is **inconclusive** for the latency claim and supports no general claim about tree-aware scheduling. Do not infer p95 behavior from four trees.

The reviewer returned `claim_supported: no` for the intended ≥5% improvement claim, with high confidence in the arithmetic and frozen-screen classification. This means the claim is unsupported by this evidence; it does not turn the prespecified cohort outcome into consistent reverse evidence. The review is same-family and provisional.

## Interrupted runs and session caveat

- The budget ledgers report that the first B2-flat attempt in `runs_attempt2/` timed out at 30/32 requests, missing chain nodes 4 and 5, and the retry in `runs_attempt3/` timed out at 31/32, missing chain node 5. Both partial attempts are excluded from every latency, queue, token, and correctness comparison. Their partial CSVs are absent from the retained result tree, so the row counts and missing-node identities cannot be independently recomputed from per-request records.
- The budget ledger reports 13,002 completion tokens for the 31-request partial attempt, versus 11,938 tokens in the included 32-request B2-flat run. The partial token total cannot be independently recomputed without its CSV; it is ledger-reported evidence only.
- B2-flat completed in `runs_attempt4/` after two timeout/restart cycles. The tree→flat order was retained, but this was not an uninterrupted temporal crossover: its server counters begin in a separate session from the first three primary arms. The completed arm therefore does not rule out server-session variation as an explanation.
- The experiment-integrity audit returned **WARN**: supplied summary arithmetic reproduced, but the trace redaction/hash mapping and excluded partial-run CSVs are not retained in this package. The audit found no evidence of fabrication. The deterministic evidence precheck verified 14/14 cited values exist; that check does not validate their interpretation or establish run integrity.

## Execution accounting

- Source generation, strict trace validation, and calibration completed. The source run is excluded from the outcome table.
- Three primary conditions completed in the second run directory. Its B2-flat timeout ledgers report 30/32 requests, and a continuation ledger reports 31/32; the associated partial CSVs are absent, so these row counts are not independently verifiable from raw requests. Neither partial is used.
- The complete B2-flat result is in `runs_attempt4/`. The bounded controller finished with cumulative use of 79.508 GPU-min under the revised 87 GPU-min ceiling (7.492 GPU-min remained, including the 5 GPU-min cleanup reserve); optional local-only and least-loaded reference runs were skipped.
- Experiment-integrity audit: **WARN**. See `../../EXPERIMENT_AUDIT.md`; the machine-readable report is `../../EXPERIMENT_AUDIT.json`. See the per-run summary JSON, request CSV, tree CSV, frozen trace, and budget logs alongside this report.
