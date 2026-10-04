# Live-tree B1 experiment audit

**Date:** 2026-10-04  
**Auditor:** fresh GPT-6-Astra reviewer, light reasoning, read-only  
**Review class:** same-family; acceptance is provisional  
**Run:** `/home/bumi/infra/cache/direction1-live-tree-b1-20261004_2029`

Machine-readable verdict: [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json). Review trace: `.aris/traces/experiment-audit/2026-10-04_run01/`.

## Overall verdict: WARN

B1 passes the implemented mechanism screen: on all eight tested questions, enabling the calibrated unseen-work term changed at least one same-snapshot routing batch, and both arms completed all trees with required request telemetry. This establishes that the signal can change decisions on observed live states. It does not establish useful natural tree completion, accurate prediction of future work, or a causal latency benefit.

## Checks

### A. Ground-truth provenance: PASS, evaluation caveat

The GSM8K reference answers are extracted from dataset rows after generation; only the question enters the root prompt. See `run_live_tree_smoke.py:64–67`, `:383–388`, and `:449`. The local dataset SHA-256 matches the cohort and run configurations. Calibration and test indices are disjoint, and the runner checks both dataset-hash equality and index overlap (`run_live_tree_smoke.py:280–292`).

This is a custom majority-of-terminal-JSON exact-match evaluator, not the official GSM8K evaluation pipeline (`run_live_tree_smoke.py:397–414`). The work predictor fits observed descendant counts and generated tokens, not correctness labels (`calibrate_live_tree_work.py:70–109`).

### B. Score normalization: PASS; calibration interpretation: WARN

Accuracy and latency are reported as raw counts and seconds. The scale fit in `calibrate_live_tree_work.py:35–42` is ordinary least-squares calibration, not score normalization. Both raw and fitted RMSE are saved (`:127–130`).

The calibration file reports fit error on 32 node observations from four trees that all reached the configured node cap. There is no held-out predictor-quality result; the saved RMSE is not evidence of generalization. The calibrated targets are also bounded by node/depth pruning and completion order.

### C. File/result/claim consistency: WARN

The comparison, tree summaries, events, configs, and logs exist. The reviewer found no discrepancies in released/completed counts, completion-token totals, terminal elapsed time, expected answers, or parent-completion-before-child-release order. Both test arms have 72 distinct request IDs and finite, nonnegative values for the required telemetry.

The meaning of “complete” is limited: the driver emits `tree_completed` when its work queues drain, including trees whose remaining branches were pruned (`live_tree_contract.py:392–401`). Malformed JSON stops a branch but does not fail the tree (`live_tree_contract.py:505–508`). The first calibration attempt was retained and identified as invalid in the run plan: all four root responses hit `finish_reason="length"`, stopped as malformed JSON, and produced no answers.

For the corrected run, all 20 calibration/test trees released exactly nine nodes, the configured maximum. Every test tree reached the budget cap; five of eight questions in each arm had no terminal answer. Exact match was 1/8 in both QKV and proposed. The implemented gate in `compare_live_tree_arms.py:156–185` checks routing change, tree execution completion, and telemetry presence. It does not check natural stopping, answer availability, answer quality, or predictor accuracy. Its `b1_screen_gate_pass=true` is arithmetically correct for a mechanism screen, but would be misleading if read as a successful useful-search or end-to-end result.

### D. Dead evaluation code: PASS for reported metrics; coverage: WARN

The answer parser, majority selector, latency aggregation, telemetry checks, and calibration RMSE are called and represented in outputs. The ordering/failure audits in `live_tree_contract.py:655–775` run through the fixture entry point (`:783–791`), not automatically in the live runner. The event records support parent-first ordering, but live runs do not inherit every fixture assertion automatically.

### E. Scope: WARN

The run used one model, one GSM8K test split, four calibration questions, eight paired test questions, and one run per test arm. Questions run sequentially; `max_inflight=4` is per tree, not a multi-tree concurrent serving workload (`run_live_tree_smoke.py:357`). The eight-question nearest-rank p95 is the maximum and is descriptive only (`compare_live_tree_arms.py:42–45`).

The two arms did not have cache-isolated or counterbalanced execution. Logged cached prompt tokens were 21,376 for QKV and 25,744 for proposed; for question 420's root they were 144 versus 224 despite identical output. This run order and retained prefix cache prevent attributing the small latency difference to route policy. Scheduler CPU overhead was not instrumented. No SLO threshold was fixed before the run, so no formal SLO-goodput value is claimed.

### F. Evaluation classification: PASS with separation

- Answer accuracy: `real_gt`, using the dataset-provided GSM8K answer in a custom offline evaluator.
- Live request, latency, and cache telemetry: measured vLLM execution. The adapter sends live HTTP requests (`live_tree_contract.py:568–580`) and reads usage/metrics (`:592–618`). Raw HTTP response bodies were not separately saved.
- Work calibration: `self_supervised_proxy`, from observed capped descendants and generated tokens.
- B0 fixtures: `simulation_only`; they validate code paths, not model performance.
- Same-snapshot route changes: counterfactual assignments computed on observed live states. They are not counterfactual executions or measured queue/latency outcomes.

The router's queue and locality values are estimates: client in-flight counts, historical service rates, prompt bytes divided by four, and a parent-locality fraction (`run_live_tree_smoke.py:130–151`). They must not be described as measured engine queue or KV state.

## Claim impact

- **Supported, narrowly:** the unseen-work term changed a same-state assignment on 8/8 observed questions.
- **Needs qualifier:** live model requests exercised parent-first expansion, budget/depth pruning, and occasional model-terminal branches under a 9-node cap.
- **Unsupported:** useful natural tree completion, accurate unseen-work prediction, stable tail-latency improvement, causal latency advantage, or broad quality preservation.

## Action items

1. Keep the B1 result labeled as a mechanism screen only; do not use `b1_screen_gate_pass` as an end-to-end success claim.
2. Before another GPU matrix, validate the tree protocol against a direct-answer quality baseline on the same cohort and add a separate quality gate.
3. Use strict structured output for tree responses and log natural terminal, depth-stop, and budget-prune rates separately.
4. Counterbalance arm order, reset prefix cache between arms, time router decision computation, and freeze an SLO target before collecting confirmatory results.
5. Report independent run blocks and hold out a new cohort after any prompt/protocol changes.
