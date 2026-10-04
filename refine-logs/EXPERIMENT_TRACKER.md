# 方向一：在线展开推理树实验 Tracker

状态：R001/R002 fixture 通过。B1 的同快照路由变化机制门通过，但树均触及节点上限，质量证据不足；独立 GPT-6-Astra 轻度审计结论 WARN。B1.5 的严格结构输出有效，但只有 3/16 树题有终答，tree exact match 2/16（direct 6/16），冻结质量门失败。当前方向一调度器主张停止，B2 不启动。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 冻结 online tree event/API contract | `live_tree_contract.py` Router/ExpansionPolicy boundary | fixture only | event order, future-info leakage, replayable decisions | DONE | PASS | 0 GPU; 现有 `run_vllm_placement_pilot.py` 是预构树，不能用于 live-tree claim |
| R002 | M0 | 验证动态树的展开/终止/剪枝/完成事件 | scripted fixture, not paper evidence | fixture only | complete event log, IDs, stop reason | DONE | PASS | 0 GPU; 4 节点、变 fanout、终止、预算剪枝；详见 `LIVE_TREE_B0_CONTRACT.md` |
| R003 | M1 | live-tree机制 smoke | 4 题校准 + 8 题 QKV/proposed | 校准 `[18,56,300,340]`; test `[420,743,976,996,1049,1058,1166,1210]` | topology, same-snapshot route changes, request/cache/tokens, exact match | MUST | MECHANISM_PASS_QUALITY_HOLD | 67.335 GPU-min；same-snapshot route changed 8/8，test topology reached 9-node cap 8/8，terminal answers 3/8 and exact match 1/8 per arm；no latency claim |
| R003.1 | M1.5 | strict-schema paired quality gate | direct answer + one proposed live-tree arm | `[844,856,376,186,709,498,103,281,838,1242,991,652,927,713,712,661]` | schema validity, answer coverage/accuracy, natural stop and cap rates | MUST | STOP_GATE_FAILED | 13.333 GPU-min; direct 6/16, tree 2/16, answer coverage 3/16, cap hit 13/16; all services stopped |
| R004 | M2 | 四臂主筛查 block A | Preble-style, exposed-DAG, attained-service, proposed | new held-out cohort | tree p95/p50, SLO goodput, quality, cache and scheduler cost | MUST | STOPPED | B1.5 quality gate failed; no GPU launch |
| R005 | M2 | 反序 block B | 同 R004，反转次序 | 与 R004 同一 held-out cohort | paired p95, bootstrap, run-order/session | MUST | STOPPED | Block A did not run |
| R006 | M3 | 独立 powered confirmation + 第二工作流 | 四臂 | 新 cohort + structurally different workflow | powered p95/SLO + quality non-inferiority | MUST | NOT SCHEDULED | B2 正向后另做 power/budget plan |

## 既有证据边界

已完成的冻结树与 weighted-lookahead 实验只作为负面/不确定先导，不并入 live-tree 性能估计。Weighted screen 的三臂 p50 为 flat 333.744s、weight 1.0 327.680s、weight 0.5 330.590s；单个 block 未达到预设 5% 门槛，block B 按规则跳过。该结果不支持当前启发式的稳定延迟收益。

## B0/B1 证据与质量门

- B0 event fixture：`B0_FIXTURE_PASS`；事件账本 `idea-stage/pilots/direction1/LIVE_TREE_B0_EVENTS_20261004_194422.jsonl`。
- Fixture 仅验证 API 可见性与事件先后，不是模型推理或性能证据。
- B1 的同快照反事实显示 8/8 题至少一批派发改变，支持“该项会改变观测状态下的路由决策”；这不是替代策略的实际执行效果。
- 所有 16 个 test trees 都用了 9 节点上限；每臂只有 3/8 题产生终端答案，每臂 exact match 1/8。八题 p95 是最大观测值，且缓存与固定执行顺序混杂，不能支持延迟优势或正式 SLO goodput。
- 后续 direct-answer quality diagnostic 在一组独立 16 题上，strict schema exact match 为 6/16；首轮 JSON-object 尝试有 12/16 输出 function-shaped JSON。该小样本只定位协议问题，不是模型总体准确率估计。
- B1.5 使用新 16 题 `[844,856,376,186,709,498,103,281,838,1242,991,652,927,713,712,661]`，与 history44、B1 12题、旧质量诊断16题完全不重叠。严格 schema 结构通过，但答案覆盖与准确率门失败；详见 `idea-stage/pilots/direction1/LIVE_TREE_QUALITY_VALIDATION_REPORT_20261004.md`。
- 父账本批准上限 400 GPU-min；累计已用 243.615，剩余 156.385，保留清理 20 后可用 136.385。B1.5 实际使用 13.333 GPU-min；强基线矩阵未启动。
