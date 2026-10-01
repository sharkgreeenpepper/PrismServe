# Direction 1 Mechanism Ablation: Results Analysis

Date: 2026-10-01  
Result batch: `results/20261001_023430/`  
Purpose: small mechanism screen; the four evaluation trees are descriptive units, not enough for significance or population-tail estimates.

## Raw run table

| Block / arm | Tree completion median (s) | Tree maximum (s; summary `p95`) | Weighted server queue mean (s) | Majority-correct trees | Parsed leaves | Completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| B1 flat | 81.507395 | 99.427513 | 6.917119 | 3/4 | 14/20 (70%) | 13,734 |
| B1 tree | 89.887384 | 98.577225 | 7.415737 | 4/4 | 15/20 (75%) | 14,235 |
| B2 tree | 90.297769 | 98.709379 | 7.379757 | 4/4 | 15/20 (75%) | 14,264 |
| B2 flat | 81.874364 | 99.955324 | 6.945286 | 3/4 | 14/20 (70%) | 13,734 |

Each arm processed the same 38 requests over trees 1, 3, 5, and 7; each pair used identical prompts and 17,580 prompt tokens. B1 order was flat then tree; B2 order was tree then flat. The queue-pressure threshold was 0.1 s and every main run exceeded it by a wide margin.

The reported tree-completion p50 is the conventional median across four per-tree completion times (mean of the middle two after sorting). It is **not** the time for a complete four-tree run. The two serving replicas process requests concurrently, so the complete run wall times were about 98.6–100.0 seconds; see `inference_wall_time_s` in each summary JSON. With only four trees, the summary `p95` is just the observed maximum and is descriptive.

## Per-tree completion times

| Tree | B1 flat (s) | B1 tree (s) | B1 change | B2 flat (s) | B2 tree (s) | B2 change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79.060839 | 87.545588 | +8.484749 (+10.73%) | 79.467206 | 87.666790 | +8.199584 (+10.32%) |
| 3 | 83.953950 | 98.577225 | +14.623275 (+17.42%) | 84.281522 | 98.709379 | +14.427857 (+17.12%) |
| 5 | 54.127763 | 56.145217 | +2.017454 (+3.73%) | 54.447990 | 56.181265 | +1.733275 (+3.18%) |
| 7 | 99.427513 | 92.229180 | −7.198333 (−7.24%) | 99.955324 | 92.928747 | −7.026577 (−7.03%) |

## Key findings

1. **The preregistered latency screen failed in both blocks.** Tree-aware median completion was 10.281% slower in B1 and 10.288% slower in B2 than flat `kv-cost-group`. It did not meet the required 5% improvement in either block. Trees 1, 3, and 5 regressed in both blocks; tree 7 improved in both.
2. **Queue pressure was present, but the tree arm did not reduce the observed queue mean.** Weighted mean server queue wait was 6.917/6.945 s for flat and 7.416/7.380 s for tree in B1/B2. This passes the pressure gate, but does not show a queueing benefit from the tested tree term.
3. **Observed answer counts favored the tree arm, but do not establish a quality improvement.** Majority exact match was 4/4 trees for tree and 3/4 for flat in both blocks. The sample has four questions; leaf answer parsing was only 70–75%, and tree 5 had just 1/6 parsed leaves under tree-aware versus 0/6 under flat.
4. **Generated work differed despite identical prompts.** Tree-aware generated 14,235 tokens in B1 and 14,264 in B2 versus 13,734 for flat (+3.65% and +3.86%). All main runs had zero length-finished requests. More generated tokens may contribute to the latency difference, but these results do not isolate the cause.
5. **The optional reference policies were skipped under the frozen budget rule.** The run ledger ended at 59.177/70 GPU-min, with 87.346 seconds of work budget remaining, below the 360-second requirement for either `local-only` or `least-loaded`. These references therefore have no result in this batch.

## Interpretation and next experiments

The data supports a narrow negative screening result: the tested descendant-work estimate did not meet its latency objective on these four frozen GSM8K trees under the measured queue pressure. It does not show that all tree-aware schedulers are ineffective. One plausible explanation is that predicted descendant work did not match actual generated work; the tree arm generated more completion tokens and showed higher mean queue wait. This is a hypothesis, not a causal finding. The opposing response on tree 7 also suggests workload-shape sensitivity.

Before spending more compute on this heuristic, inspect request-level placements and timing for trees 1, 3, 5, and 7, including the generated-token differences and the client/server wait records. A revised mechanism test should keep the same primary endpoint and thresholds, include independent trees and additional randomized/counterbalanced blocks, and separately diagnose scheduling with controlled output lengths while retaining natural generation as the end-to-end endpoint. Do not reinterpret the current result as a p95 result, a statistically established correctness result, or a general verdict on tree-aware scheduling.

## Provenance and review status

- Four main-run summaries and per-tree CSVs are in `results/20261001_023430/`; the final budget ledger is `BUDGET_LOG.json`.
- The direction-specific evidence precheck verifies the eight cited median and queue values. Verification confirms the values exist in their source files; it does not validate the causal interpretation.
- No `EXPERIMENT_AUDIT.json` was present, so integrity audit status is unavailable.
- The `result-to-claim` reviewer returned `claim_supported: no`, confidence `high` for failure of this narrow prespecified screen and low for effect-size generalization. Review independence is `same-family`; acceptance remains provisional and external review is pending. Full prompt, response, and verdict are in `.aris/traces/result-to-claim/2026-10-01_run02/`.
