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

1. Same prompts, model, hardware, cache reset, route capacity, service calibration, and sibling assignment logic in the tree-term-off/on pair. Both serving replicas receive the identical short/long calibration prompts.
2. Two counterbalanced temporal blocks for the paired flat/tree ablation. Run local-only and least-loaded once each as supplemental references only after all four primary runs complete and the compute cap allows.
3. Request-level generated work and client timing, server queue/prefill/decode aggregate metrics, tree completion latency, answer parsing/correctness, and exact placement choices.
4. Both blocks meet the preregistered 5% median improvement screen under at least 0.1 s request-count-weighted mean vLLM server queue wait for each primary condition run, with no majority-correct regression. This screen is exploratory and not a significance test.

## Evidence limitations

The pilot uses four evaluation trees, each representing a different GSM8K question from an eight-question sample, and an output profile derived from earlier local-only pilot rows on the other four tree IDs in that same sample. It cannot establish a general latency claim, a reliable p95, online live-search benefit, SLO goodput, or cross-replica KV transfer.

## Decision rule

- If the mechanism screen is positive in both blocks, design a larger independent-trace confirmation with a sample size based on observed variance.
- If the server queue gate is not met, do not claim this tests queue pressure.
- If the mechanism screen is negative or inconsistent, stop extending this heuristic until a concrete explanation suggests a different test.

## Execution log

The first server startup failed before calibration or inference because the process defaulted to `/usr/local/cuda`, which is absent, although a local CUDA 12.9 toolkit is available. A second startup with the correct toolkit path reached model warmup but FlashInfer 0.6.18.post1 `TopKMaskLogits` failed with `device kernel image is invalid` on the H20 GPUs; again, no calibration or inference ran. A final bounded startup attempt will set vLLM's supported `VLLM_USE_FLASHINFER_SAMPLER=0` on both replicas, fixing the native sampler for every condition. This preserves model, attention backend, hardware, traces, routing logic, and paired protocol, but changes the sampler kernel implementation, so results will be compared only within this experiment and not directly to older pilot latency. Both failed attempts' 14.148 GPU-min are charged to the unchanged 70 GPU-min cap; if this startup still fails, stop without changing the runtime/backend again.
