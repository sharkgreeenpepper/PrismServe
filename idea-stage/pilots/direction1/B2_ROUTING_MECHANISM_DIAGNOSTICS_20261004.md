# Mechanism diagnostics for the B2 routing replication

**Trace SHA-256:** `4e6eff34c48d7d8d25761578aaa9f03618ce70a7c9fc7121a9edc99179eaf13d`
**Method:** Same tree/node request assignments are aligned across B2-flat and B2-tree. Critical-path component medians sum each run’s own observed path-node metrics, then take the median over 16 trees. Generated-token changes are observed outcomes, not a controlled output-length counterfactual.

## Routing and generated work

| Run / concurrency | Changed assignments | Flat→tree worker transitions | Flat tokens | Tree tokens | Token change | Trees faster with fewer tokens | Trees faster with more tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| first-c4 | 54/128 (42.2%) | 0->0: 37, 0->1: 29, 1->0: 25, 1->1: 37 | 59,460 | 59,507 | +0.08% | 3/16 | 6/16 |
| first-c8 | 57/128 (44.5%) | 0->0: 33, 0->1: 30, 1->0: 27, 1->1: 38 | 59,642 | 59,160 | -0.81% | 4/16 | 1/16 |
| repeat-c4 | 57/128 (44.5%) | 0->0: 34, 0->1: 31, 1->0: 26, 1->1: 37 | 60,272 | 59,537 | -1.22% | 5/16 | 7/16 |
| repeat-c8 | 60/128 (46.9%) | 0->0: 32, 0->1: 32, 1->0: 28, 1->1: 36 | 59,395 | 59,537 | +0.24% | 3/16 | 6/16 |

The policy changed 54–60 of 128 worker assignments per comparison. Total generated-token differences stayed within −1.22% to +0.24%, but per-tree speedups did not consistently coincide with fewer generated tokens. In the 16-tree paired breakdown, trees that improved despite more tokens and trees that regressed despite fewer tokens both occur. This weakens a simple aggregate token-volume explanation, but does not isolate a placement effect because per-tree output and queue interactions remain coupled.

## Admission and service accounting

| Run / concurrency / policy | Mean semaphore wait (s/request) | Mean vLLM queue (s/request) | Mean TTFT (s/request) | Mean decode (s/request) | Mean request latency (s) | Residual after these components (s) |
|---|---:|---:|---:|---:|---:|---:|
| first-c4 / B2-flat | 67.81 | 11.45 | 0.166 | 11.76 | 91.21 | 0.028 |
| first-c4 / B2-tree | 69.76 | 11.39 | 0.173 | 11.78 | 93.13 | 0.028 |
| first-c8 / B2-flat | 48.59 | 31.50 | 0.163 | 11.79 | 92.08 | 0.035 |
| first-c8 / B2-tree | 47.82 | 31.59 | 0.169 | 11.72 | 91.33 | 0.033 |
| repeat-c4 / B2-flat | 69.52 | 11.52 | 0.161 | 11.95 | 93.18 | 0.028 |
| repeat-c4 / B2-tree | 69.58 | 11.32 | 0.167 | 11.80 | 92.89 | 0.028 |
| repeat-c8 / B2-flat | 48.71 | 31.53 | 0.163 | 11.78 | 92.22 | 0.033 |
| repeat-c8 / B2-tree | 49.17 | 31.67 | 0.169 | 11.79 | 92.84 | 0.035 |

At concurrency 4, per-request client semaphore waiting is about 68–70 s, compared with 11.3–11.5 s in the engine queue. At concurrency 8, semaphore waiting is about 48–49 s while engine queue waiting rises to about 31.5–31.7 s. The four measured terms account for nearly all mean request latency (roughly 0.03 s residual). Increasing concurrency therefore mostly relocates waiting; decode stays near 11.7–12.0 s/request.

## Observed critical-path components

| Run / concurrency | Policy | Critical path (s) | Semaphore (s) | Engine queue (s) | TTFT (s) | Decode (s) | Path output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| first-c4 | B2-flat | 337.69 | 229.08 | 44.96 | 0.73 | 43.05 | 1697 |
| first-c4 | B2-tree | 338.24 | 229.03 | 42.13 | 0.53 | 43.83 | 1725 |
| first-c8 | B2-flat | 326.88 | 150.34 | 113.50 | 0.74 | 41.33 | 1638 |
| first-c8 | B2-tree | 334.03 | 143.06 | 117.19 | 0.53 | 44.71 | 1750 |
| repeat-c4 | B2-flat | 346.54 | 236.24 | 47.55 | 0.53 | 44.91 | 1774 |
| repeat-c4 | B2-tree | 336.57 | 237.92 | 42.52 | 0.59 | 45.78 | 1805 |
| repeat-c8 | B2-flat | 335.53 | 154.35 | 113.51 | 0.65 | 43.06 | 1701 |
| repeat-c8 | B2-tree | 336.36 | 150.93 | 114.44 | 0.59 | 45.75 | 1805 |

The concurrency-8 critical paths retain substantial queue wait inside vLLM (about 113–117 s across each path), while concurrency-4 paths show roughly 42–48 s of engine-queue wait and more semaphore wait. The tree-aware route redistributes which requests share the critical path and changes the path’s queue/decode components, but the median pattern differs by run. These path summaries are diagnostics, not a replay of alternate placements; each policy’s critical path may contain different nodes.

## Trees for focused diagnosis

Speedups are listed in this order: first c4, repeat c4, first c8, repeat c8. A positive number favors B2-tree. Queue Δ is the repeated run’s B2-tree minus flat engine-queue sum on each arm’s own observed critical path.

| Tree (shape) | Speedups (%) | Token deltas (tree−flat) | Assignment changes (out of nodes) | Repeat c4 path queue Δ (s) | Repeat c8 path queue Δ (s) |
|---|---|---|---|---:|---:|
| 1 (broad) | -5.15, -4.87, -3.67, -4.68 | +273, -125, +7, -125 | 5, 7, 4, 7 | +12.81 | +11.18 |
| 13 (broad) | -0.15, -1.45, -3.25, -3.90 | +263, -6, -31, -2 | 3, 4, 3, 4 | +10.50 | +15.94 |
| 0 (balanced) | +0.14, +1.03, +5.33, +1.25 | +108, +41, +43, +41 | 1, 2, 3, 2 | +0.05 | -2.88 |
| 14 (chain) | +2.62, +1.05, +0.91, +0.22 | +36, +15, -309, +4 | 3, 3, 5, 3 | -14.83 | +1.62 |

Trees 1 and 13 (both broad) are slower under B2-tree in all four comparisons. In the repeated runs they remain slower even when tree-token totals are lower or nearly equal, while their own critical-path engine queue sums rise by about 10–16 s. Trees 0 (balanced) and 14 (chain) improve in all four comparisons; tree 0 improves despite higher token counts, and tree 14 has small or mixed token changes. This is a useful targeting clue, not evidence that broad or chain shapes generally respond this way: these are individual questions and each arm’s critical path may select different nodes.

## Interpretation

The current records show that B2-tree materially changes routing, while total generated tokens are close in aggregate. This weakens a simple total-token-volume explanation, but neither token count nor the observed queue decomposition isolates a placement effect or yields a stable latency advantage across repeated runs and concurrency settings. Use the tree-1/tree-13 request paths to formulate a specific revised mechanism; do not launch a larger GPU matrix or infer causality from these outcome differences. If a new test follows, isolate one revised placement mechanism and preregister its effect threshold and stopping rule.

## Reproducibility

- Source arms and analysis: `/home/bumi/infra/cache/prismserve-direction1-replication-20261004/attempt2`
- Companion result report: `REPLICATION_ANALYSIS.md`
- Diagnostic script SHA-256: `de95ffeb68fa6bad6e0c4d1e2cb47d417f303714e6ef2e4180786ea369991fdc`

## Evidence existence check

The deterministic precheck found 64/64 cited diagnostic values in the analysis JSON. This verifies that the cited values are present; it does not validate their interpretation or establish causality. The precheck record remains local under `.aris/evidence_precheck_direction1_mechanism_diagnostics_20261004.json`.
