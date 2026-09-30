# 大模型推理方向创新点报告

**研究方向：** 多 GPU 大模型推理系统：推理计算分配、KV/cache、prefill/decode、speculative decoding、跨卡通信与 SLO 之间的协同。  
**生成时间：** 2026-09-30（Asia/Shanghai）  
**评估数量：** 17 条原始候选 → 15 条去重后可行候选 → 1 项结构模拟、0 项真实推理 pilot → 3 条优先推荐。  
**证据截止：** 2026-09-30；包含 2026-09-28 的 Spexis 与 2026-09-27 的 SpecStream。

## 执行摘要

目前最值得验证的三个方向都处在“谨慎推进”而非“已确认新颖”的状态。排序考虑 PrismServe 的多 GPU 系统定位、可用实验规模和结果的信息价值：

1. **推理树分支的多 GPU 放置与 KV 选择**：最贴合跨卡资源协同，但 MemServe 和分布式 Tree-of-Thoughts 已分别覆盖了两侧，需要证明“搜索完成时间和动态分支结构”带来额外收益。
2. **根据实时服务状态选择推理策略**：上限最高；需要证明同一问题在不同排队、KV 和链路状态下会发生策略优劣反转，而不只是过载时统一少算。
3. **Speculative decoding 的 TP/互连/SLO 盈亏边界**：最容易做成短周期诊断；要超越已有动态 draft 长度控制与 Spexis 的互连对比，不能只交付更大的 benchmark 表。

三项的同族 GPT-6-Astra 新颖性评审都给出 **5/10，PROCEED WITH CAUTION**。这是早期、同一模型家族的暂定结论，不是独立新颖性确认。方向一已完成 CPU 结构模拟，但按剩余子树工作的 oracle 策略没有稳定胜过 KV 成本路由；真实 runtime 上的收益仍未验证。

## 领域版图

**推理时计算分配**已经从“多生成一些 token”扩展到按题目难度、置信度或阶段分配预算。近期工作包括自适应 test-time compute 分配、PASCAL 的阶段调度、Nitsum 的并行度适配，以及 UniScale 对 best-of-N、beam search、DVTS 等策略的在线选择。它们使“按题目选策略”本身很拥挤；更有空间的问题是，同一题的最佳算法是否会因为实时队列、KV 空间和链路状态改变。[Adaptive Test-Time Compute Allocation](https://arxiv.org/abs/2604.14853)、[PASCAL](https://arxiv.org/abs/2602.11530)、[Nitsum](https://arxiv.org/abs/2605.05467)、[UniScale](https://arxiv.org/html/2605.30898v1)

**推理服务系统**已经分别深入优化了请求调度、KV 复用/迁移、张量并行、prefill/decode 分离以及 SLO。比如 HW-Router 将队列、KV 使用率和 TTFT/TPOT 纳入模型/GPU 路由；MemServe 已用队列延迟、前缀重算成本比较 KV 传输与重算；MELL 也在 KV 迁移与 token 迁移/重算间选择。因此，把常见调度量再组合成一个通用路由器，难以单独构成贡献。[HW-Router](https://arxiv.org/html/2608.14575v1)、[MemServe](https://arxiv.org/html/2406.17565v3)、[MELL](https://arxiv.org/html/2501.06709v1)

**推理树与 KV 局部性**已出现树结构缓存保留、共享前缀调度、beam/Tree-of-Thoughts 的多 GPU 执行和跨副本缓存管理。Locality-aware Fair Scheduling 已指出把树集中在一张卡会排队并阻塞后续思考；MemServe 已有传输 KV 与重算前缀的成本比较。可能的窄缺口不是“在多卡上跑推理树”，而是显式用搜索剩余工作和整棵搜索的完成时间来决定分支复用、迁移或重算。[Locality-aware Fair Scheduling](https://arxiv.org/html/2501.14312v1)、[ArborKV](https://arxiv.org/abs/2605.22106)、[PEEK](https://arxiv.org/abs/2607.02525)、[PrefixPlace](https://arxiv.org/html/2608.01655v1)

**Speculative decoding**已经有硬件感知的 draft 长度/开关控制、接受率自适应、SLO-aware 服务，以及近期的跨互连对比。Cohere 的 DSD 已把测得的最优 draft 长度放进 runtime lookup；Spexis 已比较 PCIe 与 NVSwitch/NVLink 环境中张量并行和 speculation 的交叉结果；AdaServe 已针对异质 SLO 调整 speculative tree。因此“硬件和负载会影响 SD 收益”已不是新结论。值得检验的是标准 draft-target SD 的 collective 通信与 mixed-prefill 干扰能否形成可预测、能泛化的 SLO 盈亏边界。[Cohere DSD](https://cohere.com/blog/hardware-aware-dynamic-speculative-decoding)、[Spexis](https://arxiv.org/html/2609.34370v1)、[AdaServe](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf)、[SpecStream](https://arxiv.org/html/2609.33184v1)

## 推荐创新点（排序）

### 1. 推理树分支的多 GPU 放置与 KV 选择

- **方法（实际做什么）：** ① 固定模型、搜索算法和请求集，记录每个分支的父 GPU、共享前缀 KV、预计剩余生成工作；② 在每次扩展分支时读取 GPU 队列、KV 驻留和链路状态；③ 估算本地等待并复用 KV、迁移 KV 到远端、在远端重算前缀三种方案的预计完成时间，选择最短的一种；④ 与 local-only、least-loaded 以及 MemServe 风格的队列/前缀成本路由比较，同时保持搜索策略不变。
- **假设：** 当搜索分支不均衡且局部 GPU 排队时，以整棵搜索的剩余完成时间而非单次请求成本做放置，可以在不改变答案质量的前提下降低 solve latency 或提升 SLO goodput。
- **最小实验：** 2–4 GPU，在 GSM8K/MATH500 子集上固定搜索与验证器，控制互连带宽并制造不同程度的分支不均衡；比较三种基线和建议策略。主指标为 solve latency/p95、SLO goodput、重复 prefill 量和 KV 迁移字节。必须包含 MemServe 风格基线；只比 local-only 和 least-loaded 不够。
- **预期结果与判据：** 正结果需要在匹配准确率下稳定降低 p95 solve latency 或提高 SLO goodput，并且在未参与调参的到达率/带宽条件下保持收益。若策略只在合成极端负载下有效，或不优于通用队列+前缀成本路由，则搜索完成时间的差异化主张不成立，但可形成边界诊断。
- **新颖性：** **5/10，PROCEED WITH CAUTION**（早期同族评审，暂定）。MemServe 已有队列感知的缓存前缀路由及传输/重算成本比较；Locality-aware Fair Scheduling 已评测分布式 Tree-of-Thoughts 与局部性造成的阻塞。可验证的差异是按动态搜索的剩余分支工作优化整棵搜索完成时间，而不是一般请求完成时间。[MemServe §5.3](https://arxiv.org/html/2406.17565v3#S5.SS3)、[Locality-aware Fair Scheduling](https://arxiv.org/html/2501.14312v1)、[MELL](https://arxiv.org/html/2501.06709v1)、[PrefixPlace](https://arxiv.org/html/2608.01655v1)
- **可行性：** 约 4–12 GPU-hours 做小规模实验；完整多负载评估约 1–2 GPU-days，原型集成预计数周。H20 资源足够；数据集公开。
- **风险：** 中。分支生命周期和 KV 复用难预测，传输/重算估计可能无法战胜简单局部性策略。
- **贡献类型：** 系统方法 + 实证分析。
- **Pilot 结果：** CPU 结构模拟完成：20 个固定种子、27 个负载配置、540 组配对比较。oracle sibling-aware 策略相对 KV-cost 的平均 p95 改善约 **0.02%**（281/540 更快）；64 个并发搜索时整体差约 **0.23%**，mixed workload 差约 **0.81%**。local-only 到 KV-cost 有 5–14% 的模拟 p95 改善，但额外使用子树剩余工作没有稳定增益。完整结果见 [`STRUCTURAL_PILOT_REPORT.md`](pilots/direction1/STRUCTURAL_PILOT_REPORT.md) 和 CSV。此模拟使用人为设定的服务/链路成本，不是 GPU 实测；8 张 H20 上的真实 inference pilot 仍因缺 runtime、模型及项目推理代码未运行。
- **评审者最强反对意见：** 这可能只是把 MemServe/MELL 的队列与 KV 成本规则用于 Locality-aware Fair Scheduling 已研究过的分布式树工作负载。若不能测出“共享祖先、分支不均衡、未来后继工作或搜索完成时间”至少一个对普通请求路由有实质影响，差异会过薄。
- **为什么值得做：** 它正面对应 PrismServe 的多 GPU 资源协同，并能给出可证伪结论：搜索树结构是否值得进入服务调度器的成本模型。

### 2. 根据实时服务状态选择推理策略

- **方法（实际做什么）：** ① 对固定模型测量单链、并行采样、验证器/小树搜索在一组题目和负载下的质量、延迟与 KV 需求；② 对每个请求读取题目特征及队列长度、空闲 KV 块和链路压力；③ 训练一个小型策略选择器，在单链、并行采样和树搜索之间选择，并遵守每请求延迟/显存限制；④ 对比固定策略、只看题目的策略、只按资源压缩 token 预算的策略，以及“先按难度选策略、再按负载放置”的基线。
- **假设：** 对同一批题，实时服务压力会改变不同推理策略的质量—时延排名；同时观察题目和服务状态的策略选择器能在每请求延迟与显存限制下提高正确率或 goodput。
- **最小实验：** GSM8K/MATH500 约 200–500 题，1–2 GPU，至少三个到达率/队列压力档；为同一题在不同压力档运行候选策略。关键检查是是否出现可重复的策略排名反转，并且策略收益超过 state-aware budget-only 控制器和 AdaptiScale 式两阶段基线。
- **预期结果与判据：** 若能证明相同问题在不同实时状态下偏好的推理算法不同，且在线策略在独立负载下达到更高的准确率-SLO 曲线，贡献成立。若学到的规则退化为“忙时少生成 token”，则不支持策略选择的中心主张。
- **新颖性：** **5/10，PROCEED WITH CAUTION**（早期同族评审，暂定）。UniScale 已选择 BoN、beam 和 DVTS；AdaptiScale 已把难度驱动的策略选择与负载感知节点分配组合；HW-Router 已用队列/KV/TTFT/TPOT 做模型/GPU 路由。差异必须是固定模型下，实时服务状态让同一个请求切换推理算法，而非切换模型或仅改变并行度。[UniScale](https://arxiv.org/html/2605.30898v1)、[AdaptiScale 全文](https://www.researchgate.net/publication/403640207_Compute-Optimal_Resource_Allocation_for_Distributed_Large_Language_Model_Inference_in_Cloud-Scale_Intelligent_Systems)、[ISDFS 2026 官方摘要集](https://isdfs.org/wp-content/uploads/2026/03/ISDFS2026-Program_and_ABSTRACT_Book-2026.03.17.pdf)、[HW-Router](https://arxiv.org/html/2608.14575v1)、[Adaptive Test-Time Compute Allocation](https://arxiv.org/html/2604.14853v1)
- **可行性：** 初步约 4–10 GPU-hours；策略实现和误差分析约 1–3 周。公开数学推理数据可用。
- **风险：** 高。正确率、题目难度和实时资源状态难共同估计；服务压力可能只改变可用预算，并不改变最优算法。
- **贡献类型：** 方法 + 系统实证。
- **Pilot 结果：** **SKIPPED：需要先准备推理 runtime、模型和依赖**。
- **评审者最强反对意见：** 已有系统分别能按问题选 TTS 策略、按负载做分布式分配、按队列/KV 路由；把硬件特征加入策略网络本身不足以形成贡献。必须实测同一问题在不同 contention 下发生策略排名反转，并击败 state-aware budget-only 和 AdaptiScale 风格基线。
- **为什么值得做：** 该方向上限最高；无论正负结果都能回答一个实用问题：推理系统是否需要在“怎么推理”层面感知实时服务状态。

### 3. Speculative decoding 的 TP/互连/SLO 盈亏边界

- **方法（实际做什么）：** ① 固定一种标准 draft-target speculative decoding 实现和模型对；② 系统改变 target TP 宽度、链路条件、并发量、prefill/decode 混合比例及 draft 宽度；③ 分别记录草稿、验证、collective、排队和 KV 开销，并标注 TTFT/TPOT SLO；④ 从测量结果建立可预测的启用/关闭及 draft 宽度规则，在留出的拓扑/负载条件验证它，并与固定 SD、TurboSpec/DSD 风格控制器比较。
- **假设：** TP collective 开销与 mixed-prefill 干扰会共同移动标准 SD 的 SLO break-even 边界；用这些成本建立的预测器能在未见条件下做出比接受率或单点 profile 更好的选择。
- **最小实验：** 1–4 GPU、一个 target/draft 模型对；至少两档 TP、两档可控互连负载、并发/混合 prefill 请求和不同 draft width。先验证同一接受率附近是否出现配置收益反转，再测试留出条件下的边界预测。
- **预期结果与判据：** 正结果需要可复现的配置反转、对留出 TP/链路/prefill 条件的预测能力，以及比现有动态 draft 长度/开关规则更高的 SLO goodput。若只有更多测试点而没有可泛化预测关系，则只是 benchmark 扩充。
- **新颖性：** **5/10，PROCEED WITH CAUTION**（早期同族评审，暂定）。Cohere DSD 已在线使用 profile lookup 选择 draft 长度；TurboSpec 已使用反馈选择 speculation 深度和开关；Spexis 已比较 TP/互连下的 speculation 与并行策略；AdaServe 已对异质 SLO 控制 speculative tree。差异只能落在标准 draft-target SD 中 TP collective 与 mixed-prefill 对 SLO 盈亏点的联合影响及跨配置预测。[Cohere DSD](https://cohere.com/blog/hardware-aware-dynamic-speculative-decoding)、[TurboSpec](https://arxiv.org/html/2406.14066v3)、[Spexis](https://arxiv.org/html/2609.34370v1)、[AdaServe](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf)、[Performance or Illusion?](https://arxiv.org/html/2601.11580v1)
- **可行性：** pilot 约 2–6 GPU-hours；完整受控评估约 1–2 GPU-days。比 A/B 更容易在固定模型与实现下隔离变量。
- **风险：** 中。最大风险是最终结果退化成围绕既有 profile 控制器的大型 benchmark 矩阵。
- **贡献类型：** 诊断性实证 + 轻量运行时策略。
- **Pilot 结果：** **SKIPPED：需要先准备推理 runtime、模型和依赖**。
- **评审者最强反对意见：** Spexis 已展示 interconnect 会逆转 TP 与 speculation 的相对优势，Cohere 已有硬件感知动态 draft lookup；“硬件影响 SD”不够新。必须证明 mixed-prefill 下的 collective/verification 交互导致现有 controller 失误，并让边界规则泛化到未校准的配置。
- **为什么值得做：** 这是最便于控制变量和低成本否证的方向；即使没有通用规则，边界图也能指导 PrismServe 的多 GPU 配置和后续研究。

## 其他保留候选

以下 12 项均通过了客观预算门槛；表中 prior work 是检索到的近邻，不表示已确认重合或排除。

| 候选 | 主要近邻（差异仍需验证） | `so_what`：无论正负结果的意义 | 粗略算力 |
|---|---|---|---:|
| 负载感知地启用/停用 speculative decoding（与 prefill 干扰候选机械合并） | [TurboSpec](https://arxiv.org/html/2406.14066v3)、[Cohere DSD](https://cohere.com/blog/hardware-aware-dynamic-speculative-decoding)、[AdaServe](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf) 已覆盖动态 speculation/异质 SLO；差异需聚焦并发 prefill 对验证的外部干扰。 | 能说明 speculation 在什么负载边界开始损害其他请求 SLO。 | 8–16 GPU-hours |
| 联合回答质量与 GPU/KV 局部性的模型路由 | [HW-Router](https://arxiv.org/html/2608.14575v1)、[Preble](https://arxiv.org/abs/2407.00023) | 可量化为了回答质量多花的跨卡通信、排队与 cache 成本。 | 2–6 GPU-hours 校准，主要扫描可模拟 |
| 按专家路由重叠合批 MoE speculative verification（两条近重复候选合并） | [Semantic Parallelism](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html)、[EcoSpec](https://arxiv.org/abs/2607.12696)、[AcceptMoE](https://arxiv.org/abs/2608.02989) 已覆盖 MoE/speculation 或专家协同；跨请求验证 batch 仍需与它们对照。 | 可判断验证阶段的专家通信是否是 MoE 推测解码的真实瓶颈，避免无效合批。 | 8–20 GPU-hours |
| 按实时链路压力选择 context-parallel 宽度 | [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783)、[Medha](https://www.microsoft.com/en-us/research/publication/medha-efficient-llm-inference-on-multi-million-context-lengths-without-approximation/) | 可检验固定并行宽度是否在链路争用和短请求混合时引发可避免的长请求延迟。 | 8–20 GPU-hours |
| 根据 KV 传输争用调整 prefill/decode 放置 | [Nitsum](https://arxiv.org/abs/2605.05467)、[PLA-Serve](https://proceedings.mlsys.org/paper_files/paper/2026/hash/bbb7506579431a85861a05fff048d3e1-Abstract-Conference.html)、[Beyond the Buzz](https://proceedings.mlsys.org/paper_files/paper/2026/hash/d49cee5f3a79d97d719df255689d83d7-Abstract-Conference.html) | 能说明何时分离 prefill/decode 的收益会被 KV 转移成本抵消。 | 6–16 GPU-hours |
| 利用链路空闲窗口、按 SLO slack 安排 KV 迁移 | [MELL](https://arxiv.org/abs/2501.06709)、[SOLA](https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html)、[NetKV](https://arxiv.org/abs/2606.03910) | 能回答提前迁移带来的内存收益是否值得额外网络争用。 | 2–8 GPU-hours 校准，扩展评估可模拟 |
| 只压缩经过慢链路的 KV shard | [LMCache](https://arxiv.org/abs/2510.09665)、[SmartGen](https://arxiv.org/abs/2607.28150)、[KV cache quantization work](https://proceedings.mlr.press/v267/tiwari25b.html) | 能判断压缩开销和精度损失是否只在慢路径上值得付出。 | 1–3 GPU-hours pilot，1–2 GPU-days 扩展 |
| 测量 KV 量化与 SD 在不同 TP 拓扑下是否互补 | [QuantSpec](https://arxiv.org/abs/2502.10424)、[QSpec](https://aclanthology.org/2025.emnlp-main.240/) 已把量化与 speculation 组合；新增问题限于多卡 TP 及通信成本交互。 | 可避免把两种单项加速的收益错误相加，给出是否联合部署的边界。 | 1–3 GPU-hours pilot，约 1 GPU-day 扩展 |
| 用 GPU 秒/KV GiB-time/跨卡流量的影子价格调度继续、分叉或验证 | [PASCAL](https://arxiv.org/abs/2602.11530)、[Plan and Budget](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae8d4084f418bb51575c2ca6c658a05b-Abstract-Conference.html)、[Adaptive Test-Time Compute Allocation](https://arxiv.org/abs/2604.14853) | 能比较 token 预算与真实服务资源预算，回答瓶颈从计算转向显存/互连时策略如何改变。 | 8–16 GPU-hours |
| 面向 SLO 的推理树前沿批处理：结合共享前缀与 decode 截止时间 | [FastTree](https://proceedings.mlsys.org/paper_files/paper/2025/hash/96894468eb44631a32d7ebd56f9892c7-Abstract-Conference.html)、[Locality-Aware Beam Scheduling](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c74b624843218d9b6713fcf299d6d5e4-Abstract-Conference.html) | 能测试共享前缀批处理是否会因短请求截止时间而伤害交互式服务。 | 6–16 GPU-hours |
| 将 speculative decoding 节省的时间再投入未解决分支 | 邻近 test-time budget 工作如 [PASCAL](https://arxiv.org/abs/2602.11530) 已做阶段预算；新增点是把实测 decode 节省转为额外搜索，而非只减少延迟。 | 能判定提速应兑现为更快回答，还是可换成更高解题率。 | 8–16 GPU-hours |
| 比较不同推理策略在等真实资源预算下的效果，而非等 token 数 | [Adaptive Test-Time Compute Allocation](https://arxiv.org/abs/2604.14853)、[Towards Thinking-Optimal Scaling](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e22bea3b170f4c2aebb9c48d98ae64d-Abstract-Conference.html)；差异是同时报告 GPU 时间、KV 驻留和通信。 | 可能改写方法排序，也可能证明 token 预算足够准确；两种结果都能形成评测基线。 | 2–6 GPU-hours |

## 合并与淘汰记录

| 处理 | 条目 | 原因 |
|---|---|---|
| 合并 | “繁忙多 GPU 服务中何时启用 SD” + “prefill 干扰时按 SLO 启用 SD” | 都研究服务争用条件下 SD 开关/准入；保留为一项更一般的负载感知候选。 |
| 合并 | “MoE speculative verification 专家流量共调度” + “按专家重叠组成验证 batch” | 都按专家激活重叠安排跨请求验证；保留一项候选。 |
| 淘汰 | 无 | 没有候选因预判新颖性或影响不足而被淘汰；其余全部低于一周 GPU 预算且依赖公开任务。 |

## Pilot 状态

本轮为方向一执行了 CPU 离散事件结构模拟，没有执行真实推理 GPU pilot。只读检查发现 8 张空闲 NVIDIA H20（每张约 97.8 GiB 显存），但项目只有 README/配置类文件，当前 Python 环境无法导入 `torch`、`transformers`、`vllm`、`sglang`、`datasets`；也没有可直接运行的 inference runtime 或模型。模拟显示多卡分支路由优于整棵树固定单卡，但 oracle 的剩余工作量策略没有稳定优于 KV 成本基线。所有方向仍需真实 runtime pilot 验证。

| 优先方向 | Pilot GPU/时长 | 最关键的首轮观察 | 当前状态 |
|---|---|---|---|
| 推理树分支放置 | 2–4 H20；单卡 1–2 小时先跑小轨迹 | 搜索准确率匹配时，p95 solve latency 是否优于 MemServe 风格成本路由 | 结构模拟无稳定增益；真实 GPU pilot pending（缺 runtime/模型） |
| 服务状态选策略 | 1–2 H20；单卡 1–2 小时 | 同一题是否随队列压力发生策略排名反转 | SKIPPED：缺 runtime/模型 |
| SD 盈亏边界 | 1–2 H20；单卡 1–2 小时 | 匹配接受率时是否出现 TP/链路/prefill 引起的收益反转 | SKIPPED：缺 runtime/模型 |

## 建议推进顺序

1. **方向一先做真实 runtime 小 pilot，再决定是否集成调度器**：结构模拟没有显示子树工作量规则比 KV 成本路由有稳定增益；真实轨迹、KV 容量和链路争用可能改变结论。
2. **若准备 runtime 的成本过高，转做 SD 盈亏边界的受控测量**：变量更容易隔离；若没有跨配置预测能力，应收敛为边界图或停止方法主张。
3. **服务状态选策略保留为高上限方向**：先离线检查同题在不同服务状态下是否发生策略排名反转，再投资在线策略实现。

## 下一步

- [ ] 为方向一准备隔离的多 GPU inference runtime、固定模型和实际搜索树轨迹，再测 p95 solve latency、SLO goodput、重算 token 与 KV 传输字节。
- [ ] 真实 pilot 必须对比 local-only、least-loaded、MemServe 风格 KV 成本路由及搜索树感知策略，并匹配答案质量。
- [ ] 只有真实 pilot 显示稳定差异后，才对该主张运行完整 `/novelty-check` 与外部批判审查；本报告里的同族评审只是前置筛查。

## 评审追踪

本轮候选由三个只生成、不排序的分析 shard 汇总；后续 3 个 GPT-6-Astra 同族审稿 shard 分别核验 B、E、A，并由同族 reviewer 对 top ideas 排序。审稿 verdict 为 provisional。17 条去重前候选及合并映射见 `.aris/traces/idea-creator/2026-09-30_run01/candidate-pool.md`；请求/响应 trace 同目录保存。A 的 novelty prompt 与 shortlist follow-up prompt 按评审任务重建并在 request 文件中标注；B/E 保留评审返回的原始完整请求文本。

