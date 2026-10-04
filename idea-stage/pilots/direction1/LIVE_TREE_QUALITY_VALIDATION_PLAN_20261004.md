# Live-tree strict-output quality validation

**Status:** completed; strict schema passed, answer-quality stop gate failed
**Purpose:** decide whether the online-tree output protocol is usable enough to justify preparing a scheduler-baseline screen. This is a protocol and answer-quality gate, not a scheduler performance experiment.
**Hard cap:** 100 GPU-min including server startup, both arms, and shutdown.

## Why this stage is needed

The B1 router changed assignments on all eight observed same-state question snapshots, but every tree reached the 9-node cap. Only 3/8 questions per arm had any terminal answer, with 1/8 exact match in each arm. The separate direct-answer probe reached 6/16 exact match under strict JSON Schema, but used a different cohort and therefore cannot explain the tree result. B2 would be premature without a same-question protocol check.

## Frozen cohort

- Dataset: GSM8K test, SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`, 1,319 rows.
- Seed: `20261006`; deterministic shuffle of dataset row indices, skipping every known prior-used/reserved index.
- Indices: `[844, 856, 376, 186, 709, 498, 103, 281, 838, 1242, 991, 652, 927, 713, 712, 661]`.
- Verified zero overlap with 44 historical used/reserved indices, B1's 12 indices, and the previous quality diagnostic's 16 indices. The exact exclusion lists and disjointness assertions are in the raw run's `COHORT.json`.
- This cohort is diagnostic only and is retired from confirmation, whether this stage passes or stops.

## Arms and fixed settings

1. **Direct-answer control:** one request per question, temperature 0, strict JSON Schema requiring `status="final"` and a numeric answer.
2. **Proposed live tree:** one online tree per question, temperature 0, strict schema for every node, `max_fanout=3`, `max_depth=3`, `max_nodes=9`, `max_tokens=256`, `max_inflight=4`, and the B1 unseen-work predictor/calibration. The API receives only the root question or currently released node history; it does not receive gold answers or future topology.

The search policy and budgets match B1. The structured-output contract is the only tree-generation change. The direct arm is a quality control, not a scheduler baseline.

## Execution order and interpretation

Run the direct block and then the tree block on the same two vLLM replicas at `127.0.0.1:8004` and `:8005`. Prefix cache will not be reset between these blocks. This order is recorded and latency is excluded from the decision. Request/cache counters are retained for provenance only. Do not report p50/p95 differences, SLO goodput, or a cache benefit from this stage.

## Frozen stop/continue gates

- Direct arm: all 16 requests succeed and produce schema-valid final answers.
- Tree arm: at least 15/16 trees have no malformed/schema-invalid node output.
- Answer coverage: at least 12/16 trees produce at least one final answer.
- Accuracy screen: tree exact-match count is no more than two below the direct arm on this paired cohort.
- Stop direction 1's scheduler-performance path if any of these gates fails or if trees again saturate the node cap without adequate final-answer coverage.

These are small-sample screening gates, not a formal quality non-inferiority test and not paper evidence. Regardless of outcome, any later confirmatory cohort must be new.

## Budget

The parent authorization is 400 GPU-min. Before this stage it records 230.282 used and 169.718 gross remaining; 20 GPU-min stay reserved for cleanup, leaving 149.718 usable. This stage's 100 GPU-min hard cap includes all four GPUs and both servers. On completion at the cap, at least 69.718 gross GPU-min remain, including the 20-minute reserve. Stop both servers immediately on completion or when the cap is reached, whichever comes first.

## B2 prerequisites if this gate passes

Before using any more GPUs for scheduler comparison:

- use a new cohort and identical search/quality settings in every arm;
- compare Preble-style queue plus prefix locality, TOPAS/FATE-style exposed-DAG remaining-path plus locality, Autellix-style attained-service, and the proposed unseen-descendant term;
- counterbalance execution order and reset or explicitly isolate prefix cache;
- preregister a tree-level SLO, answer-quality acceptance rule, and independent repetitions;
- record actual router decision CPU time, tree p50/p95, SLO goodput, correctness, failures, cache hit/recompute, queue/prefill/decode, and scheduler overhead;
- re-budget the full matrix. The remaining usable balance after this 100 GPU-min cap would be at most 49.718 GPU-min, which is insufficient for the planned four-arm screen.

If live-tree topology does not change actual scheduling decisions or quality gates fail, stop the unknown-descendant-work claim rather than spending the remaining allocation on another frozen-tree replay.

## Observed outcome

The run completed both arms and stopped both servers. Strict-schema direct answering returned 16/16 valid responses with 6/16 exact matches. The live tree completed 16/16 trees and all 136 completed nodes matched the response shape; it produced answers on only 3/16 questions and exact matches on 2/16. Thirteen trees reached the 9-node cap. The answer-coverage gate (12/16) and paired exact-match screen (no more than two below direct) both failed. The frozen decision is **stop before B2**; see `LIVE_TREE_QUALITY_VALIDATION_REPORT_20261004.md`.
