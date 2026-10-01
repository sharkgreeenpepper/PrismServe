# 方向一：独立样本自然生成复验计划

- **冻结时间：** 2026-10-01 22:59（Asia/Shanghai）
- **状态：** 已完成；原冻结设计与判断门槛保持不变，结果见 `results/20261001_225325_independent_natural_generation/INDEPENDENT_COHORT_RESULTS.md`。
- **目的：** 在全新 GSM8K 样本上复验 flat 与 tree 放置策略的自然生成延迟差异。

## 研究问题

固定输出诊断中，四树总体中位数差异不足 1%，但树 3 与树 5 的变化方向相反。下一步检验这种树级异质性是否仍会出现在未参与先前实验的推理问题上。此轮仍是小规模 pilot，不用于支持总体显著性或普遍收益主张。

## 冻结样本与流程

- **数据集：** GSM8K test，共 1,319 道题。
- **随机种子：** `20261002`；按 `load_trees` 的确定性采样规则抽取并排序，得到 dataset indices `[289, 329, 758, 1108]`。此前方向一所有请求 CSV 与 trace 共使用 `{455, 630, 830, 947, 1027, 1291, 1310, 1318}`，本轮四题与之无重叠。
- **树形：** tree IDs `0–3` 分别为 balanced、broad、chain、skewed，共 32 个节点请求。
- **自然生成：** temperature=0；inner/leaf 的 max tokens 均为 1,024；保留模型正常 EOS 停止，不启用固定输出长度。答案正确率和终端节点解析率作为次级结果报告。
- **上下文准备：** 先以 `local-only` 对新样本生成一次自然树输出，用其父节点回复构造冻结 prompt trace。此 source pass 仅准备固定上下文，不计入主比较。要求本地 tokenizer 重建的每个 prompt token 数与 source 服务端记录完全一致；任一不一致则停止主矩阵。
- **路由先验：** 两臂共用此前自然生成运行得到的 inner=529、leaf=532.5 token 输出长度先验；先验在本轮前冻结，不根据新样本调参。服务速率在冻结的新 trace 上重新校准。
- **主比较：** `kv-cost-group-flat` 对比 `kv-cost-group-tree`，其余运行参数相同。B1 次序 flat→tree，B2 次序 tree→flat。每个主运行前清空两个 vLLM 副本的 prefix cache。

## 指标与判断门槛

- **主要指标：** 每轮四棵树各自的 completion latency 中位数；同时报告整轮 wall time。该 median 是四棵树的中位数，不是整轮端到端 wall time，也不称为 p50 请求延迟。
- **配对有效性：** 每个 arm 必须包含全部四个 tree IDs；同一 tree/node 的 trace prompt token 数必须一致；服务端 queue/prefill/decode 观测必须完整。
- **队列门槛：** 每个主运行的按副本汇总加权 mean queue wait ≥0.1 秒，才将结果描述为队列压力下的诊断。
- **正向筛查：** tree 在 B1、B2 中的四树中位数都至少快 5%，才记为“独立样本出现初步机制信号”；仍需更大样本确认。
- **反向筛查：** tree 在两个 block 中都至少慢 5%，记为当前启发式在此 cohort 下的反向证据。
- **其他结果：** 记为 inconclusive；不在这四道题上追加重复，不合并两轮小样本生成显著性或 p95 结论。

## 预算与停止条件

- 总预算上限 **55 GPU-min**，使用空闲 GPU 4–7，保留 5 GPU-min 用于退出和落盘，最多 50 GPU-min 用于启动、source pass、校准和四个主运行。
- 不运行可选 local-only / least-loaded 对照；预算不足时不启动新的主运行。任何阶段超时、tokenizer prompt 校验失败、队列观测不全或服务未能清理时停止并记录状态。

## 执行修正记录（2026-10-01）

首次启动已消耗约 **6.016 GPU-min**（仅服务启动、模型加载与 CUDA graph 初始化；source requests 未发出）。运行器发现 `kv-cost-group-flat` 在没有 frozen trace 时会拒绝运行，因此它不能负责生成自身所需 trace。为避免循环依赖，source pass 改用可基于实时父节点回复生成后续 prompt 的 `local-only`；它仍使用相同的新 cohort 和自然 EOS 输出，且仍只用于构造 trace、不纳入结果比较。两个主比较策略、运行顺序、样本、指标与判定门槛均保持冻结。后续重试预算计入已消耗的 6.016 GPU-min，保留原总上限 55 GPU-min 与 5 GPU-min 清理预留。

## 执行预算修订（2026-10-01）

在累计消耗 **52.134 GPU-min** 时，B1-flat、B1-tree、B2-tree 三个主条件已完整结束；B2-flat 在 78.5 秒运行窗口到期时有 30/32 个请求完成，被标记为超时，不纳入结果。原预算不足以完整执行四个条件。为完成已冻结的四条件矩阵，将累计 GPU-min 上限修订为 **70**，继续保留 5 GPU-min 清理预留。续跑复用同一 cohort trace 和已校准的 service-rate profile，仅重跑完整的 B2-flat 条件；不复用超时 partial CSV，也不改变样本、输出模式、缓存重置或分析规则。

## 续跑预算再修订（2026-10-01）

在累计 **67.134 GPU-min** 时，B2-flat 第二次重跑仍在 117.0 秒工作窗口到期时达到 31/32 个请求；该 partial 仍不纳入结果。此次超时主要消耗在一个尚未完成的请求上。为避免再次重复启动开销，将累计上限调整为 **87 GPU-min**，仍保留 5 GPU-min 清理预留。最后一次 continuation 仅运行 B2-flat，复用原 trace 与校准文件，并跳过已由 trace/tokenizer 严格校验和前三个主条件覆盖的 sanity replay；完整运行仍须满足原始请求数、队列观测与输出要求。

## 执行完成记录（2026-10-01）

四个主条件均取得完整 32/32 请求和 queue/prefill/decode 观测。累计使用 79.508 GPU-min，低于修订后的 87 GPU-min 上限，且保留了 5 GPU-min 清理预算。两次未完成的 B2-flat attempt 已排除。B2-flat 在两次超时重启后于独立服务会话完成，因此虽然保留了 tree→flat 的顺序，仍不能视为连续的时间反转 crossover。冻结的正向与反向筛查均未通过，结果分类为 inconclusive；详细数字与限制见结果报告。
- 沿用已验证的本地 vLLM 0.30.0+cu129、PyTorch 2.13.0+cu129、CUDA 12.9 与 DeepSeek-R1-Distill-Llama-70B 环境，不重建依赖或模型。

## 结论边界

本轮与上一轮分别报告，不把固定输出诊断与自然生成 cohort 混为一个统计样本。四棵树只用于决定是否值得扩大独立数据验证；若观察到树形异质性，后续还需逐请求与服务端分阶段数据才能判断其来源。
