# Direction 1: Independent Natural-Generation Cohort Results

**Run ID:** `direction1-gsm8k-independent-20261003-seed20261003`  
**Decision under the frozen B1 screen:** **INCONCLUSIVE**  
**Claim supported:** No evidence supporting a ≥5% tree-aware latency benefit  
**Run date:** 2026-10-03; analysis and reviews completed 2026-10-04

## Setup and integrity

- Model: `deepseek-r1-distill-llama-70b`; vLLM native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`); two replicas on GPUs 4–7.
- Dataset: GSM8K test, SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`; seed `20261003`; 16 frozen indices: `[10, 37, 57, 104, 218, 293, 590, 610, 693, 793, 975, 1019, 1056, 1074, 1248, 1270]`, with four questions per shape. The executor sampled from the full candidate set with seeded rejection of the 12 historical indices; the plan's original shorthand “sample from the remaining indices” has been corrected.
- The source pass produced 128 frozen history nodes. Exported trace SHA-256: `67987c0056c8ed7c10458ea3718690181defe4060a9195b30c6099df7a6096bc`; manifest SHA-256: `ac4a953d0a0126216f41217584057d9122aec8dd55e36f8b8e689f774b681dd5`. Prompt-token mismatches: 0.
- All four policy arms completed with 128 request rows and 16 tree rows each, over identical question indices and trace hashes. Full completion text is present; there were no timeouts or partial request files at completion.
- The result schema has no explicit session-ID field. Arm order was reconstructed from arm labels and timestamps, so session effects cannot be isolated from the recorded data.

## Per-arm measurements

| Block / arm | Tree p50 (s) | Tree p95 (s) | Request p50 (s) | Request p95 (s) | Arm wall time (s) | Mean queue (s) | Majority exact match | Completion tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1-flat | 305.600464 | 373.986983 | 102.512920 | 137.630427 | 373.991424 | 11.080088 | 16/16 | 57,704 |
| B1-tree | 309.896202 | 382.495448 | 100.977975 | 142.129550 | 382.496438 | 10.673595 | 15/16 | 57,396 |
| B2-flat | 305.371022 | 373.737758 | 102.313612 | 137.785866 | 373.742344 | 11.073376 | 16/16 | 57,704 |
| B2-tree | 309.881953 | 382.424830 | 100.843701 | 142.178468 | 382.425812 | 10.672016 | 15/16 | 57,396 |

`tree p50` is the median of 16 tree completion times, while `request p50` is the median of 128 request latencies. Neither is the duration for the full arm to finish; that is reported separately as arm wall time. With only 16 trees, tree p95 is effectively the maximum observed tree latency and is exploratory.

## Frozen primary analysis

For each question, the flat and tree tree-completion medians were computed across the two order-reversed blocks. The per-question speedup is `1 - T_tree / T_flat`; the primary estimate is the median across 16 questions.

- Across-block median paired speedup: **−0.448935%**.
- Shape-stratified bootstrap 95% CI: **[−2.558685%, +1.253746%]**, 50,000 replicates, seed `20261003`.
- B1 median: **−0.417187%**; B2 median: **−0.480708%**.
- Positive per-question speedups: 6/16. Shape medians: balanced −0.918%, broad −2.818%, chain +2.297%, skewed −0.354%.
- Queue-pressure gate: **passed** in every replica and arm; replica mean queue time ranged from 10.448 to 11.087 seconds, above the frozen 0.1-second threshold.

The frozen positive screen requires a median speedup of at least 5%, a bootstrap lower bound above zero, positive medians in both blocks, and a passing queue gate. The point estimate and interval fail the positive rule. The reverse criterion also fails because the interval upper bound is above zero. Therefore this screen is **inconclusive**: it neither supports the ≥5% benefit claim nor establishes a slowdown. The thresholds were not changed after execution.

## Secondary observations and limits

- Tree arms used 57,396 completion tokens per block, 308 fewer (−0.53%) than flat arms. The tree arms had majority exact match 15/16 in both blocks versus 16/16 for flat; the same tree-arm error was GSM8K index 57. Leaf answer parse rate was 98.33% in each policy arm. These are descriptive metrics only; no correctness non-inferiority claim is made.
- One flat request per block and two tree requests per block ended at the configured length cap. The token-count and cap differences mean latency cannot be attributed solely to the subtree-work term. The optional fixed-output diagnostic was not run; the frozen screen produced no ≥5% signal and the aggregate token difference was 0.53%.
- Histories were frozen from a local source-generation pass. The study is a synthetic branching replay where the scheduler receives descendant prompt token sequences; it does not validate online search workloads with unknown future histories. There was no cross-replica KV transfer, and cached-prefix totals are estimates.
- One model and one 16-question GSM8K cohort do not establish a powered confirmation, causal effect independent of generated work, correctness non-inferiority, tail-SLO improvement, or generalization to other workloads.
- Optional reference arms were skipped under the frozen plan. All vLLM processes were stopped and GPUs were idle after cleanup.

## Budget and review

- GPU use: **142.655 / 400 GPU-min**; remaining: 257.345 GPU-min, including the retained 20 GPU-min cleanup reserve. All required stages returned code 0.
- Deterministic evidence precheck: 5/5 cited values found in the text; this verifies presence, not independent validity.
- Experiment-integrity audit: **WARN**, same-family GPT-6-Astra light review, provisional. It found no fabrication, self-normalization, missing primary output files, or mismatched summary metrics. Warnings cover study scope, custom-parser accuracy framing, generated-work differences, retrospective sample-selection wording, and missing explicit session IDs.
- Result-to-claim review: `claim_supported: no`; frozen screen `INCONCLUSIVE`; same-family GPT-6-Astra light review, provisional pending external review. This means the claim is unsupported by this screen, not that a slowdown is established.

## Key artifact hashes

All row-level data and logs are retained in the run artifact cache. SHA-256 hashes below identify the frozen inputs, analysis, ledger, and raw CSV/summary outputs.

| Artifact | SHA-256 |
|---|---|
| Dataset (`gsm8k:test`) | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| `cohort_trace.json` | `67987c0056c8ed7c10458ea3718690181defe4060a9195b30c6099df7a6096bc` |
| `cohort_trace.manifest.json` | `ac4a953d0a0126216f41217584057d9122aec8dd55e36f8b8e689f774b681dd5` |
| `service_rate_profile.json` | `6dd5e2d80c090bc9d474e1f5bf2bccf88a5b56e5558e6d1a5c7333600413224a` |
| `BUDGET_LOG.json` | `6c850c7474fe8ec5c110c45a87777381e4790ba6f958d63a333fadb8e3917d9d` |
| `RESULTS_ANALYSIS.json` | `51c401d765131d6456661ee9caa79193f657ece5b846bf326f37a23ba95beb3d` |
| `cohort-source-local` requests | `2fe37ca88383ae4006e6b6faa22388d50285f83d4ed534061d7d5de76f63a641` |
| `cohort-source-local` summary | `c7124cf2d2dfad8e9264e1d8536110d3930eb2a7b83078dd0a3005b1eaf2afb4` |
| `cohort-source-local` trees | `11178f74c5ac5b05b6a04bbf993d47679597aff7eb57780dc039b9592df65e73` |
| `B1-flat` requests | `f4dfbce063c5cf0879040612d4af6aa85d43b3cb8e68c230d0437d2e7487e9f9` |
| `B1-flat` summary | `940f12fa1a2180381010f42cabb1d14d35a625d4753d3e52b56d5e98a630fb41` |
| `B1-flat` trees | `ad50ef307b83cab15fbe91561e9dab75803940aab88f2edf88558b8062bb94b6` |
| `B1-tree` requests | `0d68c7a6325041405858b767b6cd0044a2f0171eef3ab242b235e877bb22a49e` |
| `B1-tree` summary | `bdfcb1d28e7415d552abdb2c94f564a5756b634e32514a80e97e7ef6f8755bdb` |
| `B1-tree` trees | `81f953f10578c5a18a8736f704bf1d89df0315deac413a47a8a96ee71f7dd862` |
| `B2-flat` requests | `0cd12347947cd2a398d46d9081b3d92d6a211c55119956eb11163d71715ccd5f` |
| `B2-flat` summary | `2b5d3e5143097eb78edb39bf7d5c5e645b996f00d736624942957acee2639069` |
| `B2-flat` trees | `70321cc2abb0086165f49779fdf22d29327b0a490396ceef41ceb907bdea2f15` |
| `B2-tree` requests | `aa7a2744a4d120d6ca38a149443a8ac1a6b2fdc676f6298894cbfa6b6490aac4` |
| `B2-tree` summary | `0ec0b56edf8cab1453020a865b450e6c0693bfa0fa4337a51f2b42dfda9206ca` |
| `B2-tree` trees | `9184ed7d5a56b1428662930c1dd78614d46fb2a89d84d1230b227862bf618a7d` |

The runner/artifact-contract fix was merged to `main` in commit `9b692374c26a26e9f7ac3b6401be2a5264348305` before the GPU run.
