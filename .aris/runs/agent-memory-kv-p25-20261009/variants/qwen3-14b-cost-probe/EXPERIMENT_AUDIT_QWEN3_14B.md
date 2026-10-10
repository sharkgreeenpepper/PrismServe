# Qwen3-14B Probe Experiment Integrity Audit

**Status: ERROR_PARTIAL_REVIEW; no final independent verdict was delivered.**

Auditor task: fresh GPT-6.1-sol (`/root/qwen14_integrity_audit`), read-only CPU review. The task hit the model usage limit before it could send the final report and final hashes. This record preserves the checks it reported; it does not convert them into a completed audit or PASS.

## Completed checks reported by the auditor

- 81,206 scalar, identity, event-map and CSV checks recomputed with zero mismatches.
- Reconstructed all 120 logical fixture rows from workload-generator seed 1709 and checked the target facts against the queried memory facts; replay RNG seed is 17. The labels are synthetic fixture ground truth, independent of model outputs.
- Recomputed Qwen3-14B accuracy, timing, bootstrap, host-stage values, cross-model ratios and the sampled GPU memory maximum. Reported values matched the saved artifacts.
- Re-generated the five 14B analysis outputs and the two cross-model report artifacts byte-for-byte in memory. Seven invalid-identity probes were rejected before output creation.
- Identified that the initial comparison CSV repeated a run-wide memory maximum on all 14B arm rows. The CSV was corrected to `run_sampled_gpu0_memory_max_mib`, populated once; the auditor reported that final2 CSV/report passed regeneration.
- Confirmed raw requests, maps, logs and summaries remained unchanged during the CSV correction.

## Unresolved audit limitation

The final manifest/hash sweep and formal A–F verdict were not delivered because the auditor hit its usage limit. Therefore this audit is **incomplete**, even though the completed numeric checks reported no discrepancies. The report does not claim a final independent PASS.

A separate provenance note records that the earlier Qwen3-8B audit hashed a pre-amendment manifest: [RUN_MANIFEST_AMENDMENT_20261010.md](../service-phase/RUN_MANIFEST_AMENDMENT_20261010.md). The old audit hash remains valid for the exact metadata version it inspected; the current service manifest adds metadata only.

## Scope

The experiments use 120 synthetic calibration requests per arm, one seed, fixed Full→Strict Prefix→Blend order, one physical GPU, and no production L1/L3 trace or holdout. Host phase spans include full generate calls and may be nested. GPU-ms and physical transfer bytes were not directly measured. The results cannot establish production quality or a production Go decision.
