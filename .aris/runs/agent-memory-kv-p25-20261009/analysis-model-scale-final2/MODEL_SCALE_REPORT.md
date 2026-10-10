# Qwen3 模型规模敏感性：Host-only KV Cache 回放

数据来源是同一组 120 条独立合成事实校准轨迹（24 sessions × 5 requests，最长 prompt 18,702 tokens），不是生产 L1/L3 轨迹或 holdout。两个模型使用相同 tokenizer 文件、请求 token IDs、答案标签、生成设置和单 GPU0/TP=1/BF16。每个模型只与自身 Strict Prefix 基线比较，以下绝对时延不用于跨模型速度排名。

| 模型 | Arm | 正确率 | 含上下文 RPC 的 TTFT P50/P95/P99 (s) | P95 / 本模型 Strict Prefix | 与本模型 Full token 输出一致 | 跳过 prefill tokens | Repair tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | Full | 118/120 (98.33%) | 0.680/2.618/2.639 | 2.569× | 100.00% | 0 | 0 |
| Qwen3-8B | Strict Prefix | 118/120 (98.33%) | 0.282/1.019/2.623 | 1.000× | 100.00% | 473104 | 0 |
| Qwen3-8B | Blend window 10% | 117/120 (97.50%) | 0.382/2.545/2.680 | 2.497× | 97.50% | 506395 | 37458 |
| Qwen3-14B | Full | 120/120 (100.00%) | 1.238/4.541/4.576 | 2.469× | 100.00% | 0 | 0 |
| Qwen3-14B | Strict Prefix | 120/120 (100.00%) | 0.511/1.839/4.575 | 1.000× | 100.00% | 473104 | 0 |
| Qwen3-14B | Blend window 10% | 120/120 (100.00%) | 0.653/4.389/4.623 | 2.386× | 100.00% | 506395 | 37458 |

## 观察

- Qwen3-8B：Strict Prefix P95 为 1.019s；Blend P95 为 2.545s（2.497× Strict Prefix，变化 -149.73%）。Blend 质量损失均值 0.833pp，会话 bootstrap 单侧 95% 上界 2.500pp。
- Qwen3-14B：Strict Prefix P95 为 1.839s；Blend P95 为 4.389s（2.386× Strict Prefix，变化 -138.62%）。Blend 质量损失均值 0.000pp，会话 bootstrap 单侧 95% 上界 0.000pp。

### Blend 阶段 P95（host elapsed，描述性）

| 模型 | Lookup (ms) | Transfer (ms) | Store (ms) | Selective recompute (ms) |
|---|---:|---:|---:|---:|
| Qwen3-8B | 2.3 | 1004.8 | 1272.1 | 162.2 |
| Qwen3-14B | 2.6 | 1827.9 | 2290.7 | 245.2 |

14B 的 Blend 遥测显示 86 个修复请求，选择修复 37458 tokens；mandatory/reusable 分别 131471/374924，合计与引擎报告跳过的 506395 tokens 对齐。
14B Blend host-only 阶段观测 P50/P95：lookup 1.6/2.6ms，transfer 215.3/1827.9ms，store 0.5/2290.7ms，selective recompute 53.4/245.2ms。它们是可能嵌套的 host elapsed 区间，覆盖整个 generate 请求，不能相加成 TTFT 或 GPU-ms。
14B 回放期间 1 秒间隔 nvidia-smi 样本的 GPU0 显存最高为 60277 MiB；这是采样最大值，不是硬件精确峰值。GPU-ms 和物理 H2D/RDMA/SSD transfer bytes 未直接测量；报告不以 token 数或软件 payload 推算替代。
LMCache 日志（包括两个预热请求，不是纯计时子集）记录 35 个 Store 通知，连接器报告 payload size 合计 36.874 GB、cost 合计 54.371s。这是软件层日志值，不能解释为总线实测传输字节或独立 timed-request 开销。

## 结论范围

本次 14B calibration 中 Full、Strict Prefix、Blend 均为 120/120 正确，Blend 与 Full 输出 token 序列逐请求一致；这只描述这组固定合成样本。Blend 在两个模型上的 P95 都显著高于 Strict Prefix，未满足 P2.5 的 ≥10% P95 改善目标。当前证据支持优先保留严格 Prefix；不支持因为模型变大就认为非前缀修复获得净 TTFT 收益。由于缺少生产记忆长度/重复率分布、独立 holdout、随机化运行顺序和物理传输/GPU-ms测量，不能给出生产 Go/No-Go 或普适模型规模规律。

详细逐请求结果、阶段 CSV、质量错误文件和 JSON summary 位于两个 run 的 analysis-final 子目录；原始请求与映射 JSONL 保留在各自 run 根目录。
