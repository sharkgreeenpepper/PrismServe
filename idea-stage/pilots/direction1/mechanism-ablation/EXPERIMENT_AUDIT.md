# Experiment Audit Report

Date: 2026-10-03  
Auditor: GPT-6 Astra, light reasoning, fresh same-family reviewer  
Review class: same-family; provisional  
Project: PrismServe, direction 1 mechanism and independent-cohort experiments  
Reviewer trace: .aris/traces/experiment-audit/2026-10-03_run01/

## Overall verdict: WARN

The supplied result arithmetic is reproducible, and the reviewer found no evidence of fabrication. Provenance documentation and retained-output gaps prevent a full integrity pass. Checks describe the pre-remediation audit snapshot; this branch also records the corrections made after review. Keep the claims limited to these pilot screens.

## Checks

### A. Ground-truth and input provenance: WARN

Answer targets are dataset answers, not model predictions. Model-generated parent histories are frozen as inputs and disclosed; old and new dataset indices are disjoint. The exported dataset path is a placeholder, and the committed cohort trace digest differs from the runtime digest. Redacted path fields plausibly explain the byte difference, but the package has no reversible redaction manifest.

Evidence: idea-stage/pilots/direction1/run_vllm_placement_pilot.py:182; idea-stage/pilots/direction1/build_prompt_trace.py:88; idea-stage/pilots/direction1/mechanism-ablation/results/20261001_225325_independent_natural_generation/cohort_trace.json:3.

### B. Metrics and recomputation: WARN

The reviewer recomputed the supplied mechanism, fixed-output, and independent-cohort summary values from raw CSVs and counters; tree medians and weighted queue means matched. The structural CSV reproduced 2,700 rows, 540 pairs, a 0.02038% mean improvement, and 281 faster cases. No improper self-normalization was found.

An older pilot used nearest-rank p50, while later code uses the conventional median. The historical 18.0% comparison is valid for its p50 convention, but calling it a median increase is misleading. Under the conventional median, local-only is 140.674 s and least-loaded is 150.199 s, a 6.77% increase. The accompanying claim text now names the nearest-rank p50 convention.

Evidence: idea-stage/pilots/direction1/run_vllm_placement_pilot.py:1114; CLAIMS_FROM_RESULTS.md:28; CLAIMS_FROM_RESULTS.md:33.

### C. Claims, result files, and tracker: WARN

Main result files exist and the numerical screen conclusions reproduce. The runtime trace hash recorded in summaries is 51e7ea09d3d050c96718a1cef784f5c91500836de278e11b6fd7cdf003c5b483; the committed redacted file hashes to 3b4d99431e488e5b7a0328b5108e5c3238072221e7b09e893b085b51736b4dec. The difference is consistent with path redaction, but no transformation mapping is supplied. At the audit snapshot, findings.md:45 incorrectly described two completed B2-flat runs; there was one completion and two timed-out attempts. This branch corrects that count and labels the partial-run details as ledger-reported because their CSVs are absent.

Evidence: idea-stage/pilots/direction1/mechanism-ablation/results/20261001_225325_independent_natural_generation/INDEPENDENT_COHORT_RESULTS.md:9; idea-stage/pilots/direction1/mechanism-ablation/findings.md:45; idea-stage/pilots/direction1/mechanism-ablation/MECHANISM_ABLATION_TRACKER.md:14.

### D. Dead code and unused metrics: PASS

No material dead evaluation metric was found. Routing estimates feed placement decisions and request records; tree aggregation, phase metrics, and summaries run through the serving pipeline. Calibration rates are measured and loaded by the router. Incidental unused locals do not create phantom reported metrics.

Evidence: idea-stage/pilots/direction1/run_vllm_placement_pilot.py:561; idea-stage/pilots/direction1/run_vllm_placement_pilot.py:1042; idea-stage/pilots/direction1/mechanism-ablation/calibrate_vllm_service_rates.py:174.

### E. Scope, retries, and missing outputs: WARN

The small samples and lack of causal attribution are disclosed. Generated completion text was not retained, so correctness can be checked only as recorded, not independently reparsed. The timeout budget ledgers remain, but the partial CSVs are absent; the reported 30/32, 31/32, missing-node, and 13,002-token details cannot be independently recomputed from per-request records. Budget extensions and a restarted B2-flat session also weaken temporal pairing. The reviewer treats these as evidence limitations, not fabrication.

Evidence: idea-stage/pilots/direction1/mechanism-ablation/run_bounded_matrix.py:401; idea-stage/pilots/direction1/mechanism-ablation/results/20261001_225325_independent_natural_generation/INDEPENDENT_COHORT_RESULTS.md:63; idea-stage/pilots/direction1/mechanism-ablation/INDEPENDENT_COHORT_PLAN.md:42.

### F. Evaluation classification: PASS

- Structural pilot: structural synthetic CPU simulation with assumed GPU/KV costs.
- Original and mechanism pilots: real vLLM serving of frozen synthetic branching workloads with GSM8K answer targets.
- Fixed-output experiment: real-serving controlled-work diagnostic, not a quality evaluation.
- Independent cohort: real-serving frozen natural-history replay on new questions; its source pass alone uses current parent outputs.

The reports generally preserve these distinctions. The experiments do not establish live-search scheduling or KV-transfer performance.

Evidence: idea-stage/pilots/direction1/simulate_structural_pilot.py:4; idea-stage/pilots/direction1/run_vllm_placement_pilot.py:899; idea-stage/pilots/direction1/mechanism-ablation/FIXED_OUTPUT_DIAGNOSTIC_RESULTS.md:40.

## Claim impact

- Intended improvement of at least 5%: unsupported by the tested evidence.
- Original mechanism screen: no improvement demonstrated in this pilot; do not generalize beyond the tested workload and runs.
- Fixed-output and independent-cohort screens: inconclusive.
- Recorded correctness: descriptive only; full generated text was not retained for independent review.
- Structural pilot result: simulation-only evidence, not a real-serving performance claim.

## Action items

1. Add an explicit trace export manifest with both runtime and exported-file hashes and the path-redaction rule.
2. Recover the two excluded partial CSVs; until then, keep their detailed counts labeled as ledger-reported and not independently verified.
3. Retain generated outputs in future natural-generation runs.
4. Distinguish the historical nearest-rank p50 from the conventional median, and correct the B2-flat completed-run count; these text corrections are included in this branch.
5. Freeze restart and censoring rules before future independent cohorts; keep natural generation as the primary endpoint and fixed-output runs as diagnostics.

