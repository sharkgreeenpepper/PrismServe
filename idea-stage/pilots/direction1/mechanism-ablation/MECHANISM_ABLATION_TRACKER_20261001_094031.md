# 方向一：机制消融执行记录

| 项目 | 状态 | 记录 |
|---|---|---|
| 逐节点旧结果检查 | done | 四种策略各 64 行，prompt token 完全配对；生成工作与单节点延迟存在差异；旧记录没有 queue/关键路径时间戳 |
| 实验契约与计划 | frozen | `RESEARCH_CONTRACT.md`、`MECHANISM_ABLATION_PLAN.md` |
| 校准数据与 held-out 输出长度 profile | pending | 用 tree IDs 0,2,4,6 校准，评估 IDs 1,3,5,7 |
| 路由器消融与逐请求 timing/metrics | pending | 新增 queue/KV grouped-flat 与 grouped-tree；审阅后再上 GPU |
| 本地 GPU sanity | pending | GPU 4–7；先单树短检查 |
| 两个反序 temporal blocks | pending | block1 与 block2 按冻结顺序各跑四个策略 |
| 汇总解释 | pending | 小样本筛查，不作 p95 总体推断 |
| PR → merge main | pending | 仅提交方向一实验代码和记录，保留用户方向三改动 |

## 运行预算

- 硬上限：70 GPU-min，四张 GPU × 被占用墙钟分钟；停止线保留 5 GPU-min 清理空间。
- 设备：GPU 4–7；GPU 0–3 留给方向三计划。
- 服务端：一次启动，两个 TP=2 vLLM 0.30.0 副本；每个运行前清 prefix cache。
- 当前未消耗预算；服务启动、校准、sanity 和完整策略矩阵均计入。
