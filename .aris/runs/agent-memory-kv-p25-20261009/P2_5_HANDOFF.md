# P2.5 阶段性交付（截至 2026-10-10）

本目录归档 P2.5 的实验计划、冻结环境、Qwen3-8B 与 Qwen3-14B 配对服务测试、逐请求输出、分析结果和审计记录。GPU 运行只使用物理 GPU0，BF16、TP=1；数据为固定种子生成的合成记忆工作负载，不是 `/home/bumi/git/Bumi_Mem` 的生产用户轨迹。

## 当前结论

- Qwen3-8B 服务阶段：Full、Strict Prefix、CacheBlend 各 120 条；Strict 与 Full 各正确 118/120，Blend 正确 117/120。Blend P95 TTFT（含上下文构建）为 Strict 的 2.497 倍；配对会话 bootstrap 的质量损失单侧 95% 上界为 2.5 个百分点。未过质量和时延联合门槛。
- Qwen3-14B：Full、Strict Prefix、CacheBlend 各 120 条，三组均正确 120/120。Blend P95 TTFT 为 Strict 的 2.386 倍；合成校准集上答案一致，但没有观察到非前缀复用的 P95 净收益。
- 两个模型的指标均是单次、低并发、合成负载结果，不支持规模规律或生产收益结论。完整 Prefill 与 Strict Prefix 的比较按每个模型自身基线归一化。
- 服务阶段的耗时钩子是 host 侧、嵌套/包含式时间，覆盖生成调用，不能加总为 TTFT，也不能解释为 GPU-ms。物理 H2D/RDMA/SSD 字节、精确 GPU-ms 和 HBM 峰值未直接测量；Qwen3-14B 仅保留每秒 `nvidia-smi` 采样值。
- 8B 审计结论为 `PASS_WITH_WARNINGS`。14B 审计完成大量独立重算，但复核代理在最终哈希核验前用量耗尽，故记录为 `ERROR_PARTIAL_REVIEW`，不能视为审计通过。
- Qwen3.5-9B 仅有 20 条 Full/Strict Prefix smoke；没有验证 Hybrid 状态卸载。P2.5 Stage D 及最终 Go/No-Go 尚未完成。

## 文件导航

- `EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`：计划和进度。
- `variants/service-phase/`：Qwen3-8B 服务阶段数据、审计和分析。
- `variants/qwen3-14b-cost-probe/`：Qwen3-14B 数据、分析和部分审计状态。
- `analysis-model-scale-final2/`：跨模型归一化报告、CSV 与图。
- `data/controlled-calibration-120-inputs/`：两模型配对的 120 请求固定工作负载。
- `configs/`：依赖冻结、环境验证及模型权重锁文件；不包含模型权重。
- `../..` 的仓库 `experiments/agent_memory_kv_service_phase/` 和 `experiments/nonprefix_kv/`：实验控制器、分析器、修复选择器和测试。

本次归档刻意排除模型权重、重复中间版本、完整 vLLM profiler trace 和下载缓存日志。运行时需要按环境锁重新准备模型与依赖。归档中的原始请求结果保留了审计所需的模型输出、prompt 标识和时间数据。
