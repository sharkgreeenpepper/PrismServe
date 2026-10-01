# 方向一：机制消融实验计划

**日期：** 2026-10-01  
**状态：** 第三次启动配置已独立复核；待执行  
**范围：** 小规模机制筛查；不用于支持普遍性或 SLO 主张

本版本因 GPU-min 预算在运行前收窄为 4 次必需的主消融运行（flat/tree 两种各两次），参考策略各最多追加一次，替代 `MECHANISM_ABLATION_PLAN_20261001_094031.md` 中的 8 次草案。校准 prompt 分配、中位数定义、阶段指标完整性、逐请求检查点和 GPU 截止控制已修正并通过独立复核。随后两次启动失败均发生在采集任何校准或推理数据之前；对第三次启动的 native sampler 配置和累计预算扣账也已通过独立复核。

## 研究问题与主张

**C1：** 在同一队列与 KV 前缀成本估计、同一 sibling-ready 分组分配器下，给分支分配加入“预测剩余子树工作量”是否能降低树完成延迟？

本轮只检验该调度项是否值得继续投入。正结果仍只是小样本机制信号；负结果也只否定当前实现与 workload 的组合，不构成所有树感知调度的反例。

## 逐节点记录审阅结果

- 主冻结回放的四种策略各有 64 条请求，四份 CSV 的 `prompt_tokens` 按 `(tree_id, node_id)` 完全一致。
- `local-only` 有 25/30 个叶回复解析出 `####` 最终答案，未见长度上限；`kv-cost` 为 24/30 且一条叶回复达到 1024 tokens；`sibling-lookahead` 为 25/30 且一条叶回复达到 1024 tokens。
- 三种非 local-only 路由相对 local-only 的节点 completion token 总量分别变化 -604、+1097、+1404。即使 prompt 冻结，生成工作量仍会因策略和运行而改变。
- 旧 CSV 只保存请求端到端延迟，不含调度时刻、客户端 semaphore 等待、服务端 queue wait 或每节点时间戳，无法复原实测关键路径或区分排队与推理时间。
- 因此本轮增加请求级单调时钟、客户端等待和调度预测记录，并从 vLLM 聚合指标读取实际 queue/prefill/decode 阶段时间。

## 实验设计

### 输入与校准

- 基础 trace：`../REAL_PLACEMENT_TRACE_20260930_074612.json`，仍使用原 8 题、相同 frozen node prompts。
- 校准树：tree IDs `0,2,4,6`（balanced/chain），仅从已有 local-only 记录得到 inner/leaf completion 长度中位数；这些树不进入本轮 latency 比较。
- 评估树：tree IDs `1,3,5,7`（两棵 broad、两棵 skewed），总计 38 个节点。每种形状内两棵树分别放在两个 worker，保留已有 local-only 的形状内均衡规则。
- vLLM 服务速率用 GPU 上低并发校准请求测量：prefill TPS = 新计算 KV tokens / prefill 阶段秒数；decode TPS = 返回生成 tokens / decode 阶段秒数。每个 endpoint 都收到完全相同、包含短长上下文的 8 条校准 prompt。校准及正式运行都要求 queue/prefill/decode histogram 的计数增量等于该 endpoint 完成请求数，否则该运行失败且不继续矩阵。保存每个 endpoint 的请求 queue、prefill、decode、TTFT 和 inter-token latency 指标。

### 对照策略

| 策略 | 含义 |
|---|---|
| `local-only` | 整棵树固定到一个副本，均衡参考 |
| `least-loaded` | 当前预测队列最短参考 |
| `kv-cost-group-flat` | 对当前 ready siblings 做联合放置，目标只用当前请求的队列、实测服务率和 prefix-cache 成本 |
| `kv-cost-group-tree` | 与 flat 完全相同，再加上从 held-out 校准树估出的后代工作量项 |

flat/tree 共享分组、候选分配、容量、KV 缓存状态和服务率。两者唯一的评分差异是是否把尚未执行的后代工作计入候选 worker 的 projected load。后代 prompt 长度来自冻结 trace，后代输出长度使用校准树按 inner/leaf 角色统计的中位数。该 profile 来自此前 pilot，不能视为独立训练集。

### 运行矩阵与顺序

每个 temporal block 在每个策略前清空两个 vLLM 副本的 prefix cache；同一 block 中间不重启模型。

| Block | 主消融顺序 |
|---|---|
| 1 | kv-cost-group-flat → kv-cost-group-tree |
| 2 | kv-cost-group-tree → kv-cost-group-flat |

flat/tree 两个主消融各运行两次并反转先后顺序，优先确保两组配对完整。主消融完成后，才运行 local-only 与 least-loaded 各一次，作为同一评估集上的描述性参考；每个参考 run 启动前至少需剩 360 秒工作预算。不为参考策略挤占主消融预算。

温度 0，inner/leaf 上限各 1024 tokens；两个 TP=2 副本使用 GPU 4–7，每副本 `max_num_seqs=2`。路由容量按 2 估计，HTTP 并发允许 4 个请求，以便服务端自身 queue 指标能够观测到排队。

## 指标与判断门槛

- **主要指标：** 每个 block 内四棵树的完成时间逐树记录；flat/tree 对比报告逐树差值及四棵树的标准中位数（中间两值取平均）。`p95` 仅作最大值描述，不作尾延迟证据。
- **排队压力门：** 每个 block 的 flat 和 tree 主消融 run，都必须达到按两个 endpoint 的请求数加权计算的平均 `request_queue_time_seconds` ≥ 0.1 秒；参考策略不参与该门槛。任一 run 未达标时，本实验不声称检验了服务端排队压力。
- **screen positive：** 两个 block 中 tree 方案的树完成时间中位数都比 flat 至少低 5%，且每个 block 的 majority-correct 树数不低于 flat。
- **其他结果：** 未达到上述门槛时记为“本轮未通过筛查”或“排队压力未建立”，不扩写为一般性无效结论。
- 同时报 prompt/completion tokens、finish reason、叶答案解析、逐节点调度决策、客户端等待、server queue/prefill/decode 聚合指标、校准参数和 cache 指标。

## 资源与停止规则

- 仅使用空闲 GPU 4–7；保留 GPU 0–3 给工作区里现有方向三计划。
- 本实验最多消耗 70 GPU-min（按四张被占用 GPU × 从首个 vLLM 进程启动到最后一个进程退出的墙钟分钟累计），其中保留 5 GPU-min 用于停止进程和落盘；每次启动的工作阶段截止时间都按累计剩余预算计算，并给每个运行设置 `min(360 秒, 截止时间剩余秒数)` timeout。
- 首次服务启动在 98.1 秒后因 `CUDA_HOME` 默认值 `/usr/local/cuda` 不存在而失败；没有进入校准或推理阶段，累计消耗 6.54 GPU-min。机器上已确认有完整 CUDA 12.9 Toolkit（`/home/bumi/infra/runtime/cuda-toolkit-12-9/usr/local/cuda-12.9`）。
- 第二次启动以显式 Toolkit 路径成功编译并加载模型，但在启动预热时 FlashInfer 0.6.18.post1 的 `TopKMaskLogits` 返回 `device kernel image is invalid`；同样没有进入校准或推理阶段。此次累计额外消耗 7.608 GPU-min，总计 14.148 GPU-min。
- 已定位的兼容性处理：本机 vLLM 0.30.0 支持 `VLLM_USE_FLASHINFER_SAMPLER=0`，会对两个服务副本一致地关闭 FlashInfer top-k/top-p sampler 并回退到 vLLM native sampler。下一次尝试固定使用该配置；模型、注意力后端、GPU、服务参数、路由矩阵和数据均不变。它会改变采样 kernel 实现，因此所有主消融、sanity 和参考策略都必须使用相同设置，且不与旧 pilot 的延迟直接比较。
- 第三次启动是最后一次兼容性诊断尝试；若仍在校准前失败，保存日志并停止，不再更换 runtime/backend 或继续重启。服务启动 OOM、模型/Tokenizer 不完整、metrics/reset endpoint 不可用也按此规则停止。
- 在累计 70 GPU-min 硬上限内，第三次尝试剩余 55.852 GPU-min，其中预留 5 GPU-min 清理，最多 762.78 秒工作时间。
- 单次策略运行若超过先验估计的两倍（360 秒），暂停后续矩阵，先分析日志与已落盘记录。
- 若四次 flat/tree 主消融未全部完成，状态记录为 `INCOMPLETE_BUDGET_EVIDENCE`，按证据不足处理，不得判作筛查阴性。中断策略 run 的请求级 partial CSV 保留已完成节点记录。

## 受限重启记录

- 2026-10-01 首次启动：vLLM 初始化模型权重后，FlashInfer sampler JIT 因默认 `/usr/local/cuda` 不存在而退出；未做服务率校准、sanity 或矩阵运行。原始日志见 `/home/bumi/infra/cache/prismserve-direction1-mechanism-20261001_021124/`，预算账本见 `results/20261001_021124/BUDGET_LOG.json`。
- 重启只显式设置本机 CUDA 12.9 Toolkit 的 `CUDA_HOME`、`CUDA_PATH`、`PATH` 和 `LD_LIBRARY_PATH`；账本以 `prior_gpu_min=6.54` 计入此前占用，硬上限仍为 70 GPU-min。运行前重新确认 GPU 4–7 空闲且端口 8004/8005 可用。
- 第二次启动证明 CUDA 路径已生效，但在首轮预热触发 FlashInfer `TopKMaskLogits` kernel 时因设备镜像无效而退出；日志见 `/home/bumi/infra/cache/prismserve-direction1-mechanism-20261001_021903/`，预算账本见 `results/20261001_021903/BUDGET_LOG.json`。第三次将显式设 `VLLM_USE_FLASHINFER_SAMPLER=0`，对所有 endpoint 固定使用 vLLM native sampler；`prior_gpu_min=14.148`，不超过累计总上限。

## 限制

- 评估集只有四棵树，来自原先 8 题；每棵树对应一个不同的 GSM8K 题目，包含两棵 broad 和两棵 skewed 树。结果是机制筛查，不估计总体 p95，也不证明答案质量非劣。
- 校准树和测试树来自同一小型 GSM8K 抽样；分开的是树 ID，不是独立数据集。
- KV 仍只有单副本 prefix caching；没有跨副本 KV 传输。
- vLLM 阶段时间是每副本聚合指标，不是逐请求 queue/prefill/decode 分解。
- 在线 frozen replay 固定 prompt，但各策略生成 token 数仍可能不同；需据实际 token/finish 记录解释延迟差异。
