# 在线树终答预算质量复验计划

**状态：** 已执行；结构与覆盖门通过，exact-match 门失败；服务已停止。
**运行目录：** /home/bumi/infra/cache/direction1-terminal-budget-quality-20261004
**冻结时间：** 2026-10-04T15:04:30Z

## 目标与冻结条件

只检验为 live tree 加入显式终答预算能否修复答案覆盖并达到继续调度比较所需的质量筛查。此运行不是时延实验，也不是调度器比较。模型为 DeepSeek-R1-Distill-Llama-70B，vLLM 0.30.0+cu129，4 张 H20、两个 TP=2 副本。direct 严格 schema 与 online tree 使用同一批 16 道 GSM8K test 题，indices 为 [1036,350,1112,1102,142,741,791,85,755,678,1140,347,1208,434,488,1261]；数据集 SHA-256 为 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14。

树策略冻结为 unseen-work、temperature 0、最多 9 节点、深度 3、fanout 3、每节点最多 256 tokens、最多 4 个并发节点。达到最大深度或最后两个可分配节点时必须终答，严格 schema 禁止在终答模式返回孩子，强制终答节点的预测后代工作量为零。代码 SHA-256 记录在运行目录 RUN_PLAN.json。

阶段硬上限 40 GPU-min；父授权总额 400 GPU-min；启动前累计 243.615，清理预留 20。direct 后执行 live tree。冻结门：direct 有效答案至少 16/16；完整树至少 15/16；答案覆盖至少 12/16；tree exact-match 相对 direct 的落差不超过 2；强制终答 schema 违例为 0；深度剪枝为 0。延迟、缓存和 SLO 不作为门。

## 实际执行与结果

阶段消耗 13.471 GPU-min；父累计 257.086/400，剩余 142.914，扣除清理预留后可用 122.914。direct 为 16/16 有效、8/16 exact match。tree 为 16/16 完成、16/16 有数值终答、5/16 exact match。树准确题相对 direct 落后 3 题，超过允许差距 2；其余结构门通过。因此 overall gate failed，B2 未启动。100 节点里有 1 个 malformed JSON（finish reason: length），但对应树仍有数值终答。

两服务已关闭，端口 8004/8005 已关闭，8 张 GPU 显存使用均为 0 MiB。完整结果与核验数据见同目录 RESULTS_SUMMARY.json、RUN_PLAN.json、BUDGET_LOG.json、ARTIFACT_MANIFEST.json、stage_status.jsonl 和原始 direct/tree 输出。

## 裁决

覆盖问题得到修复，但准确率门失败；此小样本不是正式质量非劣检验。停止 B2/B3 和当前调度器性能主张。不得用该 cohort 调参后作为确认集。
