# Direction 1 Research Contract

**Idea:** Multi-GPU placement for branching reasoning/search trees.  
**Scope:** This contract is direction-specific and is stored with its pilot to avoid replacing the active direction-3 research contract/plan.

## Problem

Search workloads issue related inference requests whose dependency tree, remaining branch work, and reusable prompt prefixes may affect how work should be placed across model replicas. Existing direction-1 evidence did not show that the tested tree-aware heuristics outperform local-only or queue/KV-cost routing.

## Primary claim under test

With the same queue/KV service-cost model and sibling-ready assignment procedure, adding a calibrated estimate of a branch's remaining descendant work can reduce per-tree completion latency under imbalanced branches and replica queue pressure, while preserving answer quality.

## Mechanism

The scheduler estimates current-request service time from observed per-replica prefill/decode rates, current queue work, and locally reusable prefix tokens. The tree-aware variant additionally predicts descendant work from frozen descendant prompt lengths and an output-length profile calibrated on separate tree IDs. The control omits only this descendant-work term.

## Minimum evidence

1. Same prompts, model, hardware, cache reset, route capacity, service calibration, and sibling assignment logic in the tree-term-off/on pair.
2. At least two counterbalanced temporal blocks including local-only and least-loaded references.
3. Request-level generated work and client timing, server queue/prefill/decode aggregate metrics, tree completion latency, answer parsing/correctness, and exact placement choices.
4. Both blocks meet the preregistered 5% median improvement screen under at least 0.1 s mean vLLM server queue wait, with no majority-correct regression. This screen is exploratory and not a significance test.

## Evidence limitations

The pilot uses four evaluation trees and an output profile derived from earlier local-only pilot rows on the other four tree IDs from the same eight-question sample. It cannot establish a general latency claim, a reliable p95, online live-search benefit, SLO goodput, or cross-replica KV transfer.

## Decision rule

- If the mechanism screen is positive in both blocks, design a larger independent-trace confirmation with a sample size based on observed variance.
- If the server queue gate is not met, do not claim this tests queue pressure.
- If the mechanism screen is negative or inconsistent, stop extending this heuristic until a concrete explanation suggests a different test.
