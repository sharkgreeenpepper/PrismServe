# 方向一：在线展开推理树实验 Tracker

状态：B0/R001-R002 fixture 通过；B1 机制门通过但质量不足；B1.5 首轮严格 schema 质量门失败。B1.6 终答预算让新 cohort 覆盖达到 16/16，但 tree exact match 5/16 对 direct 8/16，超过预设允许差距 2。当前停止 B2/B3；不支持方向一当前调度器性能主张。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 冻结 online tree event/API contract | live_tree_contract.py Router/ExpansionPolicy boundary | fixture only | event order, future-info leakage, replayable decisions | DONE | PASS | 0 GPU; 现有 runner 预构树，不能用于 live-tree claim |
| R002 | M0 | 验证动态树展开/终止/剪枝/完成事件 | scripted fixture, not paper evidence | fixture only | event log, IDs, stop reason | DONE | PASS | 0 GPU; parent-first release；详见 LIVE_TREE_B0_CONTRACT.md |
| R003 | M1 | live-tree机制 smoke | 4 题校准 + 8 题 QKV/proposed | GSM8K test | topology, route changes, request/cache/tokens, exact match | MUST | MECHANISM_PASS_QUALITY_HOLD | 67.335 GPU-min；same-snapshot route changed 8/8；answers 3/8 and exact match 1/8 per arm；no latency claim |
| R003.1 | M1.5 | strict-schema paired quality gate | direct answer + proposed live tree | [844,856,376,186,709,498,103,281,838,1242,991,652,927,713,712,661] | schema validity, coverage/accuracy, terminal/cap rates | MUST | STOP_GATE_FAILED | 13.333 GPU-min; direct 6/16, tree 2/16, coverage 3/16; services stopped |
| R003.2 | M1.6 | terminal-budget fresh-cohort quality screen | strict-schema direct + live tree with reserved final nodes | [1036,350,1112,1102,142,741,791,85,755,678,1140,347,1208,434,488,1261] | answer coverage, paired exact match, finalization violations, node cap | MUST | STOP_ACCURACY_GATE_FAILED | 13.471 GPU-min; direct 8/16, tree coverage 16/16, tree exact match 5/16; allowed drop 2, observed 3; 23 forced finals; services stopped |
| R004 | M2 | 四臂主筛查 block A | Preble-style, exposed-DAG, attained-service, proposed | new held-out cohort | tree p95/p50, SLO goodput, quality, cache and scheduler cost | MUST | STOPPED | R003.2 failed exact-match gate; no GPU launch |
| R005 | M2 | 反序 block B | 同 R004，反转次序 | same cohort as R004 | paired p95, bootstrap, run-order/session | MUST | STOPPED | Block A did not run |
| R006 | M3 | 独立 powered confirmation + 第二工作流 | four arms | new cohort + structurally different workflow | powered p95/SLO + quality non-inferiority | MUST | NOT SCHEDULED | B2 did not pass; no power/budget plan |

## Latest result

Full report: idea-stage/pilots/direction1/LIVE_TREE_TERMINAL_BUDGET_VALIDATION_REPORT.md. Machine results and raw logs remain in /home/bumi/infra/cache/direction1-terminal-budget-quality-20261004.

The reserved terminal policy fixed answer coverage on this cohort but not exact-match quality. The earlier B1.5 result is a separate cohort; do not treat the between-cohort coverage difference as a paired improvement. One of 100 node outputs was malformed due to output-length truncation; the tree still completed with numeric answers. Tree latency percentiles are descriptive only. No latency, SLO, cache, or scheduler claim is supported.

The authorized parent ledger is 257.086/400 GPU-min used; 142.914 remains, including the preserved 20 GPU-min cleanup reserve (122.914 usable). Both model servers stopped, ports closed, all eight GPUs at 0 MiB. Do not spend the remainder on B2 without a materially different frozen policy that first passes a new quality gate.

A post-hoc selector sensitivity check on the already-used B1.6 cohort found 5/16 exact for both all-node and leaf-only plurality, 3/16 for branch-balanced and forced-final-only votes, and 5/16 after adding the paired direct answer as one equal vote (direct alone was 8/16). This does not identify a quality-preserving alternative; details are in `idea-stage/pilots/direction1/ANSWER_SELECTION_RETROSPECTIVE_DIAGNOSTIC_20261004.md`. Do not tune or confirm a selector on this cohort.
