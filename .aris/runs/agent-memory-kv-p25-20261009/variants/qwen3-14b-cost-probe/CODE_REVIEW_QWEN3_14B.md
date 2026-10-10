# Qwen3-14B Probe Code Review

Verdict: **PASS_WITH_WARNINGS**; no open blocking issues. Independent read-only reviewer: GPT-6.1-sol.

The first review found two blockers: the analyzer had copied a fixed Qwen3-8B gate conclusion, and it did not validate raw arm/model/session identity. Both were corrected before launch. The final review checked that gate text is computed from the observed quality upper bound and P95 improvement; the 10% engineering target is explicit. It verified rejection of mismatched model path, arm, session, query span, policy, and repair ratio; valid records pass. The shell harness requires an empty output directory and supplies seed 17 explicitly.

The reviewer also matched all 11 source hashes and the frozen input hash to the manifest and confirmed the model, GPU, cache reset, and invocation settings. Remaining warnings are scope limits: synthetic calibration, fixed arm order, one GPU, and inclusive host timings. No production-trace or formal holdout claim follows.
