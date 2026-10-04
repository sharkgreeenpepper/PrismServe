# Live-tree answer-quality diagnostic

**Date:** 2026-10-04  
**Purpose:** diagnose whether low terminal-answer coverage in B1 might come from the model or the output protocol. This is a development diagnostic, not a paired tree comparison or a quality claim.  
**Raw run:** `/home/bumi/infra/cache/direction1-live-tree-quality-probe-20261004`

## Cohort and method

The probe used 16 GSM8K test questions `[470, 1128, 1315, 527, 486, 817, 243, 1251, 458, 269, 934, 136, 705, 1241, 513, 425]`, selected after excluding the 44 prior-used/reserved indices and the 12 questions in B1. I reconstructed the historical set from the parent cohort manifest (16 selected + 12 reserved) and request-telemetry cohort (16 selected). The probe has zero overlap with all 44 and with the B1 12; the verified lists and checks are in the raw `COHORT.json`. The dataset hash matches B1. The model was DeepSeek-R1-Distill-Llama-70B, temperature 0, with a one-shot direct-answer prompt. No tree arm was run on these same questions, so this cannot quantify the tree-search quality gap.

## Results

| Request format | Successful requests | Exact matches | Interpretation |
|---|---:|---:|---|
| Unconstrained `json_object` | 16/16 | 4/16 | Invalid as a direct-answer quality estimate: 12/16 outputs used a function-shaped JSON object rather than the required `status`/`answer` form. The raw run plan's earlier 11/16 count was corrected from the retained records. |
| Strict `json_schema` requiring `status="final"` and numeric `answer` | 16/16 | 6/16 | Valid protocol response for this custom exact-match probe, but still a small one-run diagnostic. |

The corrected strict-schema run used no tree, performed no cache reset, and made no latency claim. Its request p50/p95 are recorded in the raw summary for reproducibility only. It does not establish the model's general GSM8K accuracy, nor can it be directly compared with B1's eight-question tree results because the cohorts differ.

## Implication for the next experiment

Before a baseline matrix, run direct-answer and tree arms on a new, shared cohort, with the tree arm using a strict schema for `final` and `expand` responses. Track answer availability and exact match separately from the mechanism gate. A new cohort is frozen for this protocol check; this diagnostic cohort and all prior probe indices are excluded from later confirmation.

The quality probe used 20.292 GPU-min against its 20 GPU-min stage cap, an overrun of 0.292 GPU-min. This is recorded in the child and parent ledgers. The parent authorization remained within 400 GPU-min and preserved its 20 GPU-min cleanup reserve.
