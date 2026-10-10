# Host-only 配对服务回放报告

本报告覆盖固定校准轨迹，不是 holdout。每臂 120 请求、相同输入顺序、单并发、物理 GPU0、Qwen3-8B BF16 TP=1；每个会话切换时清空缓存。臂按 Full→Strict Prefix→Blend 固定顺序执行，未做顺序平衡。主机阶段插桩覆盖整次多 token generate，可能有测量开销。
阶段事件是 host elapsed，区间可能嵌套；不可相加为 TTFT，也不代表 GPU-ms。CUDA profiler、全词表 logprobs、搬运字节和峰值显存均未在这组计时中测量。

| Arm | 正确率 | TTFT P50/P95/P99 (s) | 含上下文 RPC 的 TTFT P95 (s) | 相对 Strict P95 | 与 Full 输出一致 | Repair tokens |
|---|---:|---:|---:|---:|---:|---:|
| full | 118/120 (98.33%) | 0.6802/2.6182/2.6391 | 2.6182 | -156.88% | 100.00% | 0 |
| strict-prefix | 118/120 (98.33%) | 0.2819/1.0192/2.6230 | 1.0192 | 0.00% | 100.00% | 0 |
| blend-window10 | 117/120 (97.50%) | 0.3814/2.5448/2.6796 | 2.5453 | -149.73% | 97.50% | 37458 |

按观测均值，Blend 相对 Full 的质量损失为 0.833 个百分点；按会话配对 bootstrap 的单侧 95% 上界为 2.500 个百分点。Blend 含上下文 RPC 的 P95 TTFT 相对 Strict Prefix 变化 -149.73%。
本组没有达到联合门槛：质量置信上界高于 1 个百分点，且 Blend 的 P95 TTFT 慢于 Strict Prefix。结果支持保留严格 Prefix；不支持将当前非前缀修复配置推进生产。
质量损失的按会话配对 bootstrap 95% 单侧上界、失败样本、逐请求指标与阶段事件分布见相邻 CSV/JSON/JSONL。全部结论是单 GPU、固定次序的校准轨迹描述统计；不得据此宣称正式 holdout 质量门通过或生产 Go。
