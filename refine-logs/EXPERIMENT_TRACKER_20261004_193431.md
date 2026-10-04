# 方向一：在线展开推理树实验 Tracker

状态：计划已写；在线 tree driver 与预算合账尚未完成；GPU 运行未启动。B1/B2 必须分别通过门控。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 冻结 online tree event/API contract | parent-release driver + router visibility | fixture only | event order, future-info leakage, replayable decisions | MUST | TODO | 0 GPU; 查明现有 runner 是否有真正的 parent-completion→child-release |
| R002 | M0 | 验证动态树的展开/终止/剪枝/完成事件 | scripted fixture, not paper evidence | fixture only | complete event log, IDs, stop reason | MUST | TODO | 0 GPU; 禁止把 fixture/frozen trace当 live result |
| R003 | M1 | live-tree机制 smoke | QKV-only vs unseen-descendant+KV | 8 个新 GSM8K 问题 | topology releases, changed routes, latency, queue/KV/tokens, exact match | MUST | GATED | ≤80 GPU-min; 仅当 R001/R002 完成且总预算对账通过后 |
| R004 | M2 | 四臂主筛查 block A | QKV, exposed-DAG, attained-service, proposed | 16 个全新问题 | tree p95/p50, SLO goodput, quality, cost, prediction/route telemetry | MUST | GATED | B1 通过后；5% p95 gate, queue and quality rules frozen |
| R005 | M2 | 反序 block B | 同 R004，反转次序 | 与 R004 同一 held-out cohort | paired p95, bootstrap, run-order/session | MUST | GATED | 只有 block A 通过早停 gate 才启动 |
| R006 | M3 | 独立 powered confirmation + 第二工作流 | 四臂 | 新 cohort + structurally different workflow | powered p95/SLO + quality non-inferiority | MUST | NOT SCHEDULED | B2 正向后另做 power/budget plan |

## 既有证据边界

已完成的冻结树与 weighted-lookahead 实验只作为负面/不确定先导，不并入 live-tree 性能估计。Weighted screen 的三臂 p50 为 flat 333.744s、weight 1.0 327.680s、weight 0.5 330.590s；单个 block 未达到预设 5% 门槛，block B 按规则跳过。该结果不支持当前启发式的稳定延迟收益。
