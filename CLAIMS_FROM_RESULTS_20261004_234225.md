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

## B2-routing replication at concurrency 4 and 8 (2026-10-04)

**Verdict:** `claim_supported: no` for a stable latency advantage from the tested tree-aware B2 routing over flat queue/KV routing. A same-family GPT-6-Astra review at the user's requested light reasoning level returned `no` with high confidence that the current evidence does not support the claim; acceptance is provisional.

Two runs per cell were completed over the same frozen 16-question GSM8K cohort. The paired median speedup of B2-tree versus B2-flat was +0.10% and +1.78% at concurrency 4, then −0.78% and +0.63% at concurrency 8 (positive favors tree-aware). This effect changes across repetitions and load settings. Run-level p95 maxima were numerically lower under B2-tree in all four comparisons, but each is only the maximum of 16 observations. All eight runs had 16/16 majority-correct trees, which does not establish quality non-inferiority.

Concurrency 8 reduced client semaphore wait and increased engine queue wait without materially changing decode time. This suggests waiting moved inside vLLM but does not show a throughput gain. A factual event replay fit at 1–10 ms grouping windows, while B2-flat replay error increased at 20 ms (p50 error 8.75 s; maximum tree error 18.01 s); this validates only the tested factual replay settings, not counterfactual routing.

**Decision:** Stop expanding the current heuristic unchanged. The existing-record diagnosis is complete; it narrows attention to two repeatedly regressing broad trees but does not isolate causality. Only resume with a specific revised mechanism, a preregistered practical effect/stopping rule, and independent cohorts with counterbalanced runs. The deterministic evidence precheck found 61/61 primary-analysis values and 64/64 follow-up diagnostic values; these checks do not validate interpretation. No independent integrity audit was available. GPU use reached 476.487/500 GPU-min; 23.513 GPU-min remain including the 20 GPU-min cleanup reserve.

Follow-up diagnostics on the existing records found 54–60 changed assignments per comparison, aggregate token deltas from −1.22% to +0.24%, and consistent regressions for individual broad trees 1 and 13 alongside repeated critical-path queue increases. This narrows a future investigation but does not establish a shape effect or causality; see `idea-stage/pilots/direction1/B2_ROUTING_MECHANISM_DIAGNOSTICS_20261004.md`.

See `idea-stage/pilots/direction1/B2_ROUTING_REPLICATION_20261004.md` and the raw artifacts under `/home/bumi/infra/cache/prismserve-direction1-replication-20261004/attempt2/`. The full reviewer trace is local at `.aris/traces/result-to-claim/2026-10-04_run01/` and excluded from Git.

## Weighted lookahead weight screen (2026-10-04)

**Verdict:** `claim_supported: no` for the frozen claim that weight 0.5 would beat flat by at least 5% in both counterbalanced blocks without reducing majority-correct tree count. The same-family GPT-6-Astra review at the lowest supported reasoning effort (`low`) returned `no` with high confidence; acceptance is provisional and external review is pending.

In the sole executed block, median per-tree completion latency was 333.744 s for flat, 327.680 s for weight 1.0 (+1.82% vs flat), and 330.590 s for weight 0.5 (+0.94%). Weight 0.5 changed 40/128 node assignments versus weight 1.0 but was 0.89% slower at p50. All three arms completed 16/16 trees and 128/128 requests with complete request-ID, queue, TTFT, and generation-time telemetry; all had 16/16 majority-correct trees. Since weight 0.5 missed the frozen 5% gate, block B was not launched under the predeclared early-stop rule.

This supports only the descriptive observation that the weight changed placements and yielded a sub-threshold result on one reused cohort. It does not establish a causal latency benefit, superiority over weight 1.0, correctness non-inferiority, generalization, or SLO improvement. Completion-token totals differed across arms; flat was run in the middle and weight 0.5 last. No independent cohort or uncertainty estimate is available. Deterministic evidence precheck verified 5/5 values (presence only); no integrity audit specifically covers this screen.

**Offline follow-up:** A post-hoc, no-GPU comparison found median absolute relative error of 12.97–18.49% between the predicted remaining descendant work and realized descendant TTFT-plus-decode. The held-out leaf output-length mean was 391.7 tokens versus a 532.5-token prior (26.4% lower); no leaf request hit the token cap. This makes the small leaf prior a plausible calibration error, but does not show it caused routing or latency. Summed queue waits are request-level cumulative measures and cannot be interpreted as tree wall time.

**Decision:** Stop the current scalar weight-0.5 sweep; do not spend the remaining campaign budget on block B. Before allocating GPU again, build a larger shape-stratified calibration set, separate output-length and per-worker service-rate error, and test any revised hypothesis on a new independent cohort with a frozen plan. Detailed results and row-level data are in `idea-stage/pilots/direction1/mechanism-ablation/results/20261004_124417_weighted_lookahead_screen/WEIGHTED_LOOKAHEAD_RESULTS.md`; the exploratory forecast diagnostic is `idea-stage/pilots/direction1/mechanism-ablation/WEIGHTED_LOOKAHEAD_FORECAST_DIAGNOSTIC_20261004.md`; the local reviewer trace is `.aris/traces/result-to-claim/2026-10-04_run02/`.

## Live-tree terminal-budget quality screen (2026-10-04)

**Verdict:** claim_supported: no for the intended quality-preserving scheduler performance claim. B1.6 failed its frozen exact-match gate; the strong-baseline scheduler comparison B2 was not run.

**Reviewer:** GPT-6-Astra at the user-requested light reasoning effort (low). The review is same-family and provisional, not cross-family acceptance. Confidence is high that the supplied evidence does not support the performance claim; confidence about the general effectiveness of tree-aware scheduling remains low.

**Integrity:** B1.6 has no dedicated integrity audit, so its integrity status is unavailable. The earlier B1 audit was WARN and applies only to the B1 screen; it does not transfer to B1.6. Deterministic evidence precheck verified 6/6 cited values in source artifacts; this establishes presence only, not provenance or validity.

### Intended claim

For online, dynamically expanding reasoning/search trees with unknown future topology, predicting unseen-descendant work together with queue/KV locality can improve end-to-end tree p95 or SLO goodput over strong queue, known-topology, or workflow schedulers while maintaining answer quality.

### Evidence and judgment

B1 showed only that the unseen-work term changed same-snapshot assignments on 8/8 observed questions. It did not establish useful natural completion, held-out predictor accuracy, or causal latency benefit.

B1.5 had 3/16 trees with numeric answers and 2/16 exact matches, compared with 6/16 direct exact matches. The follow-up B1.6 used a different fresh cohort and reserved nodes for forced final answers. It achieved numeric answer coverage of 16/16, but exact match was 5/16 against direct's 8/16, a three-answer deficit when the frozen screen allowed at most two. Paired outcomes were both-correct 5, direct-only 3, tree-only 0, both-wrong 8. One of 100 node outputs was malformed after length truncation; all 16 trees still returned numeric answers.

The cross-cohort coverage difference from B1.5 to B1.6 is descriptive only, not a paired estimate. No scheduler baseline, latency, cache, SLO, or performance comparison was run. Direct answering is a quality reference, not a scheduler baseline.

### What the results support

- The live-tree driver exercised online expansion and the unseen-work signal changed dispatch decisions on the observed B1 states.
- On the B1.6 cohort, the terminal-budget policy returned numeric answers for all 16 questions.
- The B1.6 exact-match screen failed and appropriately stopped further scheduler comparison.

### What the results do not support

- A causal or general latency/SLO benefit, cache benefit, strong-baseline superiority, held-out predictor accuracy, or quality non-inferiority.
- A conclusion that all tree-aware scheduling is ineffective; the performance claim remains untested because the quality gate failed first.
- An improvement in answer coverage inferred from B1.5 to B1.6, because these were different cohorts.

### Decision

Keep B2/B3 stopped. Do not repeat the current terminal-budget screen or tune on either quality cohort. Reopen only with a materially different, preregistered search or answer-selection policy that passes a fresh direct-reference quality gate on an untouched cohort. Only then plan a counterbalanced comparison against strong queue/KV, exposed-DAG, and attained-service schedulers with overhead, cache controls, uncertainty estimates, and adequate power.

### Provenance

- Full result report: idea-stage/pilots/direction1/LIVE_TREE_TERMINAL_BUDGET_VALIDATION_REPORT.md
- Raw B1.6 results and run plan: /home/bumi/infra/cache/direction1-terminal-budget-quality-20261004/RESULTS_SUMMARY.json and RUN_PLAN.json
- B1 integrity audit: idea-stage/pilots/direction1/live-tree-b1/EXPERIMENT_AUDIT.json
- Evidence precheck: .aris/traces/result-to-claim/2026-10-04_run03/evidence_precheck.json
- Full same-family reviewer trace is local under .aris/traces/result-to-claim/2026-10-04_run03/ and is not intended for submission as cross-family acceptance.
