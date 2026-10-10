# P2.5 实现约定（运行前）

- L3 扫描 token 档位暂解释为 Top-5 合计，不是每条记忆；保存每条长度及合计，真实分布另列。
- 数据入口为用户提供的只读脱敏 JSONL 导出，保留 L1 原消息字段；不写 SQLite，不调用 Mem0 的增删改 API。
- 当前 Query 及 assistant generation suffix 在非前缀路径不可作为复用候选，候选加载前排除；Strict Prefix 仍按完全相同因果前缀判定。
- 四个 arm 使用完全相同的 token IDs、模板和生成配置。同模型内比较，不跨模型比较绝对 TTFT。
- 关闭 Thinking 通过 apply_chat_template(enable_thinking=False) 完成，不只加 /no_think 文本。
- Stage A 的确定性 greedy 只用于机制/一致性诊断；正式任务生成配置在运行前固定，不把官方模型建议参数与诊断条件混同。
- 测量 service 与 profiling 分开。First-token logits/sampling 属于 TTFT；首个 decode forward 在首 token 后，单列且不重复计入 TTFT。
- CPU 阶段、GPU event/kernel 时间、Memcpy 计数有不同口径。允许重叠，不将各阶段时长之和当 TTFT。不可观测字段为 null。
- 质量必须有独立答案/任务结果；缺标签的真实 trace 只能做成本/一致性测量，不能宣称质量达标。
- 用户已授权GPU使用无小时限制，仍仅GPU0–3；框架仓库已提供，真实生产轨迹尚未准备。先完成模型/接口正确性，再有限校准与正式评估。
- Hybrid 只测 native Full/Strict Prefix 和缓存状态；禁止复用标准 Attention 的 CacheBlend 状态修复路径。
