# Direction 1: Result-to-Claim Judgment

**Verdict:** `no` — the completed pilot does not support the intended performance claim.

**Reviewer:** GPT-6-Astra, same-family review; acceptance is provisional.

**Confidence:** High that the supplied evidence does not support the claim; low about the true effect or whether a substantially different tree-aware system could succeed.

**Integrity audit:** unavailable (`EXPERIMENT_AUDIT.json` was not present).

**Evidence pre-check:** 14/14 referenced values were found in their cited files. This verifies evidence existence only, not the truth or interpretation of the claim.

## Intended claim

For multi-GPU inference over branching reasoning/search trees, using tree topology, expected remaining branch work, and KV/cache locality to place branches can lower per-tree p95 solve latency (or improve SLO goodput) versus local-only, least-loaded, and queue/prefix-cost routing while maintaining answer quality, particularly under uneven branches or GPU queue pressure. The differentiating claim is that tree/search completion information adds benefit beyond general per-request scheduling and KV-cost routing.

## Evidence reviewed

The real-runtime pilot used vLLM 0.30.0 with DeepSeek-R1-Distill-Llama-70B bf16 on four GPUs as two TP=2 replicas. It replayed the same 64 frozen node prompts from eight synthetic GSM8K trees (four shapes, two trees per shape). Each policy was run once after restarting services. The per-tree p95 is the nearest-rank percentile over eight observations, so it equals the maximum.

| Policy | Tree p50 (s) | Tree p95 (s) | p95 vs local-only | Majority exact match | Prefix-cache token hit rate |
|---|---:|---:|---:|---:|---:|
| local-only | 124.02 | 196.26 | baseline | 7/8 | 75.1% |
| least-loaded | 146.34 | 191.76 | -2.3% | 7/8 | 64.5% |
| kv-cost | 145.34 | 207.27 | +5.6% | 7/8 | 62.3% |
| sibling-lookahead | 147.35 | 204.39 | +4.1% | 7/8 | 64.9% |

All four policies recorded 7/8 majority exact-match accuracy. Least-loaded reduced the single-run observed maximum by about 4.50 seconds but increased the archived pilot's nearest-rank p50 by 18.0%. Neither tree-aware policy beat local-only p95. The structural simulator also showed no stable p95 gain from its sibling-aware oracle heuristic over KV-cost routing; that is evidence about the simplified simulator, not an optimality bound or real-system result.

## What the results support

- In this single frozen-prompt replay, local-only had the lowest observed p50 and the highest prefix-cache token hit rate.
- Least-loaded had a slightly lower observed maximum than local-only, alongside a materially slower nearest-rank p50 estimate. This archived pilot's p50 convention differs from the conventional median used in later runs. With eight trees and no repeats, this is not stable tail-latency evidence.
- The tested `kv-cost` and `sibling-lookahead` implementations did not demonstrate a p95 improvement over local-only.
- Equal 7/8 aggregate accuracy is descriptive for this small sample; it does not establish answer-quality equivalence or non-inferiority.

## What the results do not support

- A stable reduction in per-tree p95 from tree completion information, or an incremental benefit beyond queue and KV-cost routing.
- Better SLO goodput under queue pressure or uneven branches; no offered-load sweep or controlled pressure study was run.
- Live-search performance, parent-output reasoning KV reuse, cross-replica KV migration, or superiority to an independently reproduced MemServe baseline.
- A general conclusion that tree-aware scheduling cannot help. These negative results apply to the limited workload and policy implementations tested here.

## Scoped descriptive statement

> In a single frozen-prompt replay of eight synthetic GSM8K trees on two TP=2 vLLM replicas, sibling-lookahead did not improve observed tree completion latency over local-only or least-loaded routing. All four policies recorded 7/8 majority exact-match accuracy. This pilot demonstrates no advantage from adding tree completion information to the tested routing policies; its small sample and simplified workload leave broader effectiveness unresolved.

## Missing evidence and next experiments

1. Inspect the existing per-node records for generated-token differences, capped responses, finish reasons, queue waits, service times, and critical paths. Archive a direction-1 claim contract specifying the mechanism, primary endpoint, answer-quality criterion, practical improvement threshold, and stopping rule. The current `idea-stage/IDEA_REPORT.md` has since been updated to direction 3, so it is not a source of the direction-1 claim evaluated here.
2. Run a bounded mechanism experiment that compares one calibrated queue/KV scheduler with topology and remaining-work terms enabled versus disabled. Use matched traces with controlled branch imbalance and queue pressure, record placement decisions, randomize policy order across temporal repeats, and include local-only and least-loaded references.
3. If the enabled-versus-disabled comparison shows a reproducible incremental benefit, confirm it on independent cohorts of real search traces and then live searches. Set sample size from observed variability and the preregistered practical effect; estimate uncertainty across independent runs or cohorts, not dependent node requests.
4. For any retained SLO claim, sweep online offered load and deadlines and measure correctly solved trees completed within deadline alongside p50, p95, and answer quality.
5. For any retained KV-transfer claim, implement and measure reusable prefixes, transfer bytes and time, eviction, and recomputation, with a documented independent queue/KV baseline. If the bounded mechanism test remains negative, stop expanding the current heuristic and revise or pivot the hypothesis.

## Provenance

- Main real-runtime evidence: `idea-stage/pilots/direction1/REAL_PLACEMENT_PILOT_REPORT.md` and its four `REAL_PLACEMENT_*_SUMMARY.json` files.
- Structural simulation evidence: `idea-stage/pilots/direction1/STRUCTURAL_PILOT_REPORT.md`.
- Deterministic evidence-existence results: `.aris/evidence_precheck.json` (local trace data; not committed).
- The first semantic review attempt on 2026-09-30 was unavailable due to an account usage limit. The successful same-family retry on 2026-10-01 supersedes that unavailable status; full review trace remains in local `.aris/traces/` and is not committed.

## Independent natural-generation cohort (2026-10-01)

**Claim judgment:** `claim_supported: no`; frozen cohort screen: **inconclusive**. The same-family semantic review is provisional. The experiment-integrity audit is same-family and provisional with **WARN** status; see `idea-stage/pilots/direction1/mechanism-ablation/EXPERIMENT_AUDIT.md`.

The new four-question GSM8K cohort replayed 32 matched natural-history nodes per arm under the frozen queue-pressure gate. Tree-aware placement changed the median of four tree-completion latencies by +1.25% in B1 and −6.81% in B2 (positive values favor tree-aware placement). Thus neither the prespecified ≥5% speedup in both blocks nor consistent ≥5% slowdown in both blocks passed. The correct interpretation is an inconclusive pilot and an unsupported targeted improvement claim, not evidence of equivalence or a general scheduling effect.

All arms recorded 4/4 majority-exact trees and 15/15 parsed exact-match leaves; these four questions do not establish correctness non-inferiority. Completion-token totals differed across arms, and full generated outputs were not saved. The completed B2-flat run followed two excluded timeouts and ran in a separate server session; its tree→flat order does not make it an uninterrupted crossover. The deterministic evidence precheck found 14/14 cited values, which establishes evidence presence only.

See `idea-stage/pilots/direction1/mechanism-ablation/results/20261001_225325_independent_natural_generation/INDEPENDENT_COHORT_RESULTS.md` and the local provisional review trace under `.aris/traces/result-to-claim/2026-10-01_run03/`.
