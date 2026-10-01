# 方向一：独立样本自然生成复验计划

- **冻结时间：** 2026-10-01 22:59（Asia/Shanghai）
- **状态：** 执行前冻结
- **目的：** 在全新 GSM8K 样本上复验 flat 与 tree 放置策略的自然生成延迟差异。

## 研究问题

固定输出诊断中，四树总体中位数差异不足 1%，但树 3 与树 5 的变化方向相反。下一步检验这种树级异质性是否仍会出现在未参与先前实验的推理问题上。此轮仍是小规模 pilot，不用于支持总体显著性或普遍收益主张。

## 冻结样本与流程

- **数据集：** GSM8K test，共 1,319 道题。
- **随机种子：** `20261002`；按 `load_trees` 的确定性采样规则抽取并排序，得到 dataset indices `[289, 329, 758, 1108]`。此前方向一所有请求 CSV 与 trace 共使用 `{455, 630, 830, 947, 1027, 1291, 1310, 1318}`，本轮四题与之无重叠。
- **树形：** tree IDs `0–3` 分别为 balanced、broad、chain、skewed，共 32 个节点请求。
- **自然生成：** temperature=0；inner/leaf 的 max tokens 均为 1,024；保留模型正常 EOS 停止，不启用固定输出长度。答案正确率和终端节点解析率作为次级结果报告。
- **上下文准备：** 先以 `kv-cost-group-flat` 对新样本生成一次自然树输出，用其父节点回复构造冻结 prompt trace。此 source pass 仅准备固定上下文，不计入主比较。要求本地 tokenizer 重建的每个 prompt token 数与 source 服务端记录完全一致；任一不一致则停止主矩阵。
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
- 沿用已验证的本地 vLLM 0.30.0+cu129、PyTorch 2.13.0+cu129、CUDA 12.9 与 DeepSeek-R1-Distill-Llama-70B 环境，不重建依赖或模型。

## 结论边界

本轮与上一轮分别报告，不把固定输出诊断与自然生成 cohort 混为一个统计样本。四棵树只用于决定是否值得扩大独立数据验证；若观察到树形异质性，后续还需逐请求与服务端分阶段数据才能判断其来源。
