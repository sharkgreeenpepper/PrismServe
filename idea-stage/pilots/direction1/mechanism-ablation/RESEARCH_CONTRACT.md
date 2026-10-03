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

The first server startup failed before calibration or inference because the process defaulted to `/usr/local/cuda`, which is absent, although a local CUDA 12.9 toolkit is available. A second startup with the correct toolkit path reached model warmup but FlashInfer 0.6.18.post1 `TopKMaskLogits` failed with `device kernel image is invalid` on the H20 GPUs; again, no calibration or inference ran. A third startup with vLLM's supported `VLLM_USE_FLASHINFER_SAMPLER=0` on both replicas passed warmup and health/metrics/reset checks, but the calibration child lost the virtualenv because the controller resolved its `bin/python` symlink to `/usr/bin/python3.10`, which lacks `httpx`. The controller now preserves that interpreter symlink so Python can read the adjacent `pyvenv.cfg`. The fourth attempt keeps the native sampler fixed for every condition and leaves model, attention backend, hardware, traces, routing logic, and paired protocol unchanged. The three budget logs total 22.967 GPU-min; the next run conservatively charges 22.968 GPU-min against the same 70 GPU-min cap, leaving 630.48 seconds of work after the cleanup reserve. If the next run fails at any stage, stop without changing runtime/backend or restarting again.

## Final screening result (2026-10-01)

All four preregistered flat/tree runs completed, and the queue-pressure gate passed in every run. The tree-aware policy's median tree completion latency was 10.28% higher in B1 and 10.29% higher in B2, so it failed the required 5% improvement screen. Majority-correct counts were 4/4 for tree-aware and 3/4 for flat, but this four-tree sample does not establish a quality gain or non-inferiority. The optional local-only and least-loaded references were skipped because the remaining work budget was below their 360-second minimum. A same-family `gpt-6-astra/ultra` review returned `claim_supported: no`; the result remains provisional. The experiment-integrity audit returned WARN; see `EXPERIMENT_AUDIT.md`. See `RESULTS_ANALYSIS.md` and `findings.md` for the run-level evidence and caveats.
