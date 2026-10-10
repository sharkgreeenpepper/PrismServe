# 2026-10-10 Service-Phase Manifest Metadata Amendment

The read-only service-phase integrity audit recorded the SHA256 of `RUN_MANIFEST.json` as:

`2680acb8b9766f484a3bb9342d03abb96a7f687a16e1ed7c719579d4bc942d28`

After that audit, the manifest was amended to add final quality-latency figure/report hashes and the independent-audit metadata. Its current SHA256 is:

`e8417626c73432b925ffd590a3e1b753f9b0df309df32554cc1a4bbb33cf668a`

The amendment added metadata keys only; it did not change the previously audited raw request JSONL, phase maps, logs, summary, or metrics. The historical audit report and JSON remain unchanged and retain the hash they actually inspected. The current manifest is the metadata-complete version for subsequent runs and comparisons.

References:

- Historical audit: `EXPERIMENT_AUDIT_SERVICE_PHASE.md` and `.json`
- Current manifest: `RUN_MANIFEST.json`
- Raw/analysis artifacts: `paired-120-host-only/`
