# 方向一：机制消融执行记录

| 项目 | 状态 | 记录 |
|---|---|---|
| 逐节点旧结果检查 | done | 四种策略各 64 行，prompt token 完全配对；生成工作与单节点延迟存在差异；旧记录没有 queue/关键路径时间戳 |
| 实验契约与计划 | frozen | `RESEARCH_CONTRACT.md`、`MECHANISM_ABLATION_PLAN.md` |
| 输出长度 profile | done | 旧 local-only replay 上 tree IDs 0,2,4,6 校准；评估 IDs 1,3,5,7 |
| vLLM 启动兼容性 | second-failure | CUDA 路径修正生效；FlashInfer TopKMaskLogits 在 H20 上报 `device kernel image is invalid`，未运行校准、sanity 或推理；GPU 及端口已释放 |
| 采样器兼容性修正 | reviewed | vLLM 0.30.0 支持 `VLLM_USE_FLASHINFER_SAMPLER=0`；第三次尝试将所有副本固定到 native sampler；所有实验条件采用同一设置 |
| GPU 服务率校准 | pending | 待复核后重启；两个 endpoint 并行收到同一组 8 条短/长 prompt |
| 路由器消融与逐请求 timing/metrics | review-fixes-done | 两个副本使用相同 8 条短/长校准 prompt；增加每节点 CSV 检查点与 queue/prefill/decode histogram 计数完整性门；标准中位数用于 p50 和 screen |
| 更新后代码审查 | passed | 独立审阅确认 native sampler 环境变量对两副本生效、配对比较仍可解释，累计预算和清理逻辑一致；启动后须从日志确认 FlashInfer sampler 已关闭 |
| 本地 GPU sanity | pending | 即将使用 GPU 4–7；先单树短检查，计入 70 GPU-min 上限 |
| 两个反序 temporal blocks | pending | flat/tree 各跑两次并反转顺序；主矩阵后视剩余预算追加 local-only、least-loaded 各一次 |
| 汇总解释 | pending | 小样本筛查，不作 p95 总体推断 |
| PR → merge main | pending | 仅提交方向一实验代码和记录，保留用户方向三改动 |

## 运行预算

- 硬上限：70 GPU-min，四张 GPU × 被占用墙钟分钟；两次启动失败累计 14.148 GPU-min，剩余 55.852 GPU-min，停止线保留 5 GPU-min 清理空间。
- 第三次尝试工作时间：最多 762.78 秒，再按累计预算执行动态截止并预留清理时间；失败后停止重启。
- 设备：GPU 4–7；GPU 0–3 留给方向三计划。
- 服务端：一次启动，两个 TP=2 vLLM 0.30.0 副本；每个运行前清 prefix cache。
- 启动日志：`results/20261001_021124/BUDGET_LOG.json` 与 `results/20261001_021903/BUDGET_LOG.json`；服务日志分别在同名 `/home/bumi/infra/cache/prismserve-direction1-mechanism-*` 目录。服务启动、校准、sanity 和完整策略矩阵均计入同一累计上限。
