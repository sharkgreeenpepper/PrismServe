# 方向一：机制消融执行记录

| 项目 | 状态 | 记录 |
|---|---|---|
| 逐节点旧结果检查 | done | 四种策略各 64 行，prompt token 完全配对；生成工作与单节点延迟存在差异；旧记录没有 queue/关键路径时间戳 |
| 实验契约与计划 | frozen | `RESEARCH_CONTRACT.md`、`MECHANISM_ABLATION_PLAN.md` |
| 输出长度 profile | done | 旧 local-only replay 上 tree IDs 0,2,4,6 校准；评估 IDs 1,3,5,7 |
| vLLM 启动兼容性 | resolved | 第三次启动使用 CUDA Toolkit 路径及 native sampler 后，双副本预热、health/metrics/reset 均通过 |
| 校准启动器 Python 环境 | fixed-reviewed | 第三次校准进程因 `Path.resolve()` 把 venv `bin/python` 变成 `/usr/bin/python3.10` 而缺少 `httpx`；控制器现保留解释器 symlink，独立复核通过 |
| GPU 服务率校准 | done | 两个 endpoint 并行收到同一组 8 条短/长 prompt，输出 `results/20261001_023430/service_rate_profile.json` |
| 路由器消融与逐请求 timing/metrics | review-fixes-done | 两个副本使用相同 8 条短/长校准 prompt；增加每节点 CSV 检查点与 queue/prefill/decode histogram 计数完整性门；标准中位数用于 p50 和 screen |
| 更新后代码审查 | passed | 独立复核确认 native sampler、venv symlink 路径、保守累计预算 22.968 GPU-min 和进程清理均无阻塞；启动后须确认 venv 依赖可导入 |
| 本地 GPU sanity | done | `sanity-tree1-64tok` 完成；64-token 上限导致答案截断，仅验证服务与记录链路，不计入主矩阵 |
| 两个反序 temporal blocks | done | 四个 flat/tree 主运行均完整；local-only、least-loaded 因剩余工作预算不足 360 秒而跳过 |
| 汇总解释 | done | 两个 block 的 tree 中位完成时间均比 flat 慢约 10.3%；queue gate 通过；详见 `RESULTS_ANALYSIS.md` 与 `findings.md` |
| result-to-claim | no / provisional | gpt-6-astra/ultra 判定本轮不支持预设 latency claim；同族审阅，待外部复核；记录见 `.aris/traces/result-to-claim/2026-10-01_run02/` |
| PR → merge main | pending | 仅提交方向一实验代码和记录，保留用户方向三改动 |

## 运行预算

- 硬上限：70 GPU-min，四张 GPU × 被占用墙钟分钟；前三次账本累计 22.967 GPU-min，第四次按 22.968 GPU-min 保守扣账，剩余 47.032 GPU-min，停止线保留 5 GPU-min 清理空间。
- 第四次尝试工作时间：最多 630.48 秒，再按累计预算执行动态截止并预留清理时间；失败后停止重启。
- 设备：GPU 4–7；GPU 0–3 留给方向三计划。
- 服务端：第四次尝试启动两个 TP=2 vLLM 0.30.0 副本，固定 native sampler；每个策略运行前清 prefix cache。
- 启动日志：`results/20261001_021124/BUDGET_LOG.json`、`results/20261001_021903/BUDGET_LOG.json` 与 `results/20261001_022701/BUDGET_LOG.json`；服务日志分别在同名 `/home/bumi/infra/cache/prismserve-direction1-mechanism-*` 目录。服务启动、校准、sanity 和完整策略矩阵均计入同一累计上限。
- 最终账本：`results/20261001_023430/BUDGET_LOG.json` 显示四个主运行、校准及 sanity 均完成；累计 59.177/70 GPU-min，剩余 10.823 GPU-min，当前工作预算只剩 87.346 秒，低于参考策略每次要求的 360 秒，因此两个参考策略按计划跳过。服务进程已退出，GPU 4–7 空闲，未留下 partial CSV。
