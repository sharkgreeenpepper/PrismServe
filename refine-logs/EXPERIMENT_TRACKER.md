# 方向一：独立 cohort 扩展实验 Tracker

状态：主实验已完成并按冻结规则判为 INCONCLUSIVE。用户批准上限 400 GPU-min（含 20 GPU-min 清理预留）；实际使用 142.655 GPU-min。

| Run ID | 里程碑 | 目的 | 系统/变体 | 数据 | 指标 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 冻结 16 个新问题与索引 | GSM8K test sample manifest | seed 20261003；与已用 12 个索引互斥 | index set、dataset hash | MUST | FROZEN | `[10, 37, 57, 104, 218, 293, 590, 610, 693, 793, 975, 1019, 1056, 1074, 1248, 1270]`；数据集 SHA-256 见计划 |
| R002 | M0 | 检查 path-free trace/hash 与输出检查点 | trace exporter / runner | 16 个问题 | trace hash、tokenizer count、partial CSV 可恢复性 | MUST | COMPLETE | 既有 trace 32/32 节点、token mismatch=0；trace/manifest 两种哈希复算通过且无本机路径；CSV 进程退出后可恢复；0 GPU-min |
| R003 | M1 | 生成 natural-history traces | local-only source pass | 同一 16 题 | 完整 prompt/output、tokens、finish reason | MUST | COMPLETE | 128 节点；冻结 trace SHA-256 `67987c0056c8ed7c10458ea3718690181defe4060a9195b30c6099df7a6096bc`；prompt-token mismatch=0；完整输出留存 |
| R004 | M1 | 校准服务率与校验队列指标 | vLLM 双副本 | frozen trace | prefill/decode/queue histogram | MUST | COMPLETE | calibration 完成；四个主臂每副本 mean queue 均超过 0.1 秒门槛 |
| R005 | M2 | 主筛查 block 1 | flat → tree | 16 题、4 种 shape 各 4 题 | 配对 tree latency、queue gate、token/correctness | MUST | COMPLETE | 两臂各 128 request / 16 tree 行；配对 speedup 中位数 −0.417187%；queue gate 通过 |
| R006 | M2 | 主筛查 block 2 | tree → flat | 与 R005 完全相同的 frozen trace | 同上 | MUST | COMPLETE | 两臂各 128 request / 16 tree 行；配对 speedup 中位数 −0.480708%；queue gate 通过。结果 schema 未记录独立 session ID，顺序仅可按 arm 标签与启动时间回溯 |
| R007 | M3 | 审计原始产物并判断 claim | paired analysis | 两个完整 blocks | shape-stratified bootstrap CI、哈希、账本 | MUST | COMPLETE | 中位 speedup −0.448935%；95% CI [−2.558685%, +1.253746%]；冻结分类 INCONCLUSIVE；审计 WARN，result-to-claim 为 no（同家族、provisional） |
| R008 | M4 | 固定输出诊断 | flat vs tree, 256 tokens | R003 同一批 prompts | paired latency 与 token 工作量 | NICE | NOT TRIGGERED | 自然生成结果未达 ≥5% 信号，completion-token 差异 −0.53%；未运行，无额外 GPU-min |

**预算提示：** 上一轮 4 题使用 79.508 GPU-min；16 题粗略线性估算约 318 GPU-min。已批准总上限 400 GPU-min，其中 20 GPU-min 留给清理，最多 380 GPU-min 启动实验；各阶段共用一个执行器总账，不运行可选参考臂。
