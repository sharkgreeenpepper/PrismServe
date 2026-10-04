# 方向一：在线展开推理树实验 Tracker

状态：R001/R002 的零 GPU fixture 已通过；vLLM live-tree adapter 与预算合账尚未完成；GPU 运行未启动。B1/B2 必须分别通过门控。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 冻结 online tree event/API contract | `live_tree_contract.py` Router/ExpansionPolicy boundary | fixture only | event order, future-info leakage, replayable decisions | DONE | PASS | 0 GPU; 现有 `run_vllm_placement_pilot.py` 是预构树，不能用于 live-tree claim |
| R002 | M0 | 验证动态树的展开/终止/剪枝/完成事件 | scripted fixture, not paper evidence | fixture only | complete event log, IDs, stop reason | DONE | PASS | 0 GPU; 4 节点、变 fanout、终止、预算剪枝；详见 `LIVE_TREE_B0_CONTRACT.md` |
| R003 | M1 | live-tree机制 smoke | QKV-only vs unseen-descendant+KV | 8 个新 GSM8K 问题 | topology releases, changed routes, latency, queue/KV/tokens, exact match | MUST | GATED | ≤80 GPU-min; 仅当 R001/R002 完成且总预算对账通过后 |
| R004 | M2 | 四臂主筛查 block A | QKV, exposed-DAG, attained-service, proposed | 16 个全新问题 | tree p95/p50, SLO goodput, quality, cost, prediction/route telemetry | MUST | GATED | B1 通过后；5% p95 gate, queue and quality rules frozen |
| R005 | M2 | 反序 block B | 同 R004，反转次序 | 与 R004 同一 held-out cohort | paired p95, bootstrap, run-order/session | MUST | GATED | 只有 block A 通过早停 gate 才启动 |
| R006 | M3 | 独立 powered confirmation + 第二工作流 | 四臂 | 新 cohort + structurally different workflow | powered p95/SLO + quality non-inferiority | MUST | NOT SCHEDULED | B2 正向后另做 power/budget plan |

## 既有证据边界

已完成的冻结树与 weighted-lookahead 实验只作为负面/不确定先导，不并入 live-tree 性能估计。Weighted screen 的三臂 p50 为 flat 333.744s、weight 1.0 327.680s、weight 0.5 330.590s；单个 block 未达到预设 5% 门槛，block B 按规则跳过。该结果不支持当前启发式的稳定延迟收益。

## B0 fixture 结果与未完成项

- B0 event fixture：`B0_FIXTURE_PASS`；事件账本 `idea-stage/pilots/direction1/LIVE_TREE_B0_EVENTS_20261004_194422.jsonl`。
- Fixture 仅验证 API 可见性与事件先后，不是模型推理或性能证据。
- R003 仍 gated：先接入 vLLM `ModelAdapter`、冻结模型驱动的在线展开/停止协议，再合并核对 GPU 授权和剩余额度。不得用预生成拓扑替代。
