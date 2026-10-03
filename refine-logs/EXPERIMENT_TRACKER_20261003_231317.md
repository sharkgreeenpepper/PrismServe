# 方向一：独立 cohort 扩展实验 Tracker

状态仅表示计划进度；用户已批准 400 GPU-min（含 20 GPU-min 清理预留），GPU 任务尚未启动。

| Run ID | 里程碑 | 目的 | 系统/变体 | 数据 | 指标 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 冻结 16 个新问题与索引 | GSM8K test sample manifest | seed 20261003；与已用 12 个索引互斥 | index set、dataset hash | MUST | FROZEN | `[10, 37, 57, 104, 218, 293, 590, 610, 693, 793, 975, 1019, 1056, 1074, 1248, 1270]`；数据集 SHA-256 见计划 |
| R002 | M0 | 检查 path-free trace/hash 与输出检查点 | trace exporter / runner | 16 个问题 | trace hash、tokenizer count、partial CSV 可恢复性 | MUST | COMPLETE | 既有 trace 32/32 节点、token mismatch=0；trace/manifest 两种哈希复算通过且无本机路径；CSV 进程退出后可恢复；0 GPU-min |
| R003 | M1 | 生成 natural-history traces | local-only source pass | 同一 16 题 | 完整 prompt/output、tokens、finish reason | MUST | READY | 仅在修正代码合并 main 后启动；source pass 后严格 token/hash gate；失败停止校准与主矩阵 |
| R004 | M1 | 校准服务率与校验队列指标 | vLLM 双副本 | frozen trace | prefill/decode/queue histogram | MUST | TODO | 仅 source trace 校验通过后运行 |
| R005 | M2 | 主筛查 block 1 | flat → tree | 16 题、4 种 shape 各 4 题 | 配对 tree latency、queue gate、token/correctness | MUST | TODO | 不重启单个失败 arm |
| R006 | M2 | 主筛查 block 2 | tree → flat | 与 R005 完全相同的 frozen trace | 同上 | MUST | TODO | 新 block/session ID |
| R007 | M3 | 审计原始产物并判断 claim | paired analysis | 两个完整 blocks | shape-stratified bootstrap CI、哈希、账本 | MUST | TODO | 只按冻结标准给结论 |
| R008 | M4 | 固定输出诊断 | flat vs tree, 256 tokens | R003 同一批 prompts | paired latency 与 token 工作量 | NICE | BLOCKED | 仅当 B1 有信号或输出工作量明显混淆时再评估预算 |

**预算提示：** 上一轮 4 题使用 79.508 GPU-min；16 题粗略线性估算约 318 GPU-min。已批准总上限 400 GPU-min，其中 20 GPU-min 留给清理，最多 380 GPU-min 启动实验；各阶段共用一个执行器总账，不运行可选参考臂。
