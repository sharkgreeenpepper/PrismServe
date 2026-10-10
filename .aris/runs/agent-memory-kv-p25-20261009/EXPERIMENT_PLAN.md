# P2.5 动态记忆 KV 推理实验

## 目标与资源

用户授权GPU使用无小时限额，仍仅GPU0–3；GPU4–7不使用。10工作日是阶段路线，不将旧24h/96GPUh预算继承或伪装为本轮限额。资源实际消耗逐进程记录。

保留Bumi_Mem源码及现有数据，只读访问已有导出或数据库缓存；新框架功能验证的数据/SQLite在本实验隔离目录，明确标synthetic。当前仓库未附生产数据库/响应轨迹，正式真实分布评估仍缺数据；不把样例/LoCoMo适配标成生产真实轨迹。

## 阶段A

统一token输入四arm：Full、Strict Prefix、Blend、Blend+COW。同一模型同一请求完全相同tokens；Thinking关闭。非前缀路径在lookup/retrieve/store之前排除Query重叠chunk，Query及生成前缀走当前上下文计算。Bumi 0.6当前Query位于short_term末尾，只拆一次；保留所有返回的long_term，不将public context误认为原始Top5；owner/time语义保留。Mem0 ID/version未公开，内部内容身份明确标来源。

新模型层数/KV bytes从config读取；Qwen3的QK norm使用现有LMCQwen3Model。100%修复与Native Full输出/logit必须先对齐，才允许5/10/20%独立校准。检查各来源生产者/上下文/位置/COW代数、非前缀命中和清空。已有1440回退：保守区间到末端需覆盖所有下游reusable，预算不足1368、下游不足72；不是写回失效，不是原生Full bypass。

## 阶段B

Service与profiling隔离。queue、lookup、H2D/D2H、选择性GPU工作、residual prefill、first-token sampling建立独立时间戳；允许重叠，不做简单时长相加。first decode forward在TTFT后单列。GPU-ms、CUPTI memcpy bytes、HBM要实际采集，否则null；软件payload单列。记录COW存储事件和成本，保持并发及主机负载匹配，交错/轮换物理GPU比较。

## 阶段C

主线Qwen3-8B；14B仅规模敏感性；每个模型分别对自己的Full/Prefix。L1扫描1K/4K/8K/16K，L3 Top5合计128/512/2K/4K，重复率0/20/60/100。不是自动展开全笛卡尔海量请求；校准后冻结有限主配置。真实数据分布与controlled scan单列。生成参数机制诊断greedy，真实质量参数运行前另固定。独立GT才计算正确率；Full输出只用于一致性/KL。

## 阶段D

HBM/DRAM两级、异步加载/计算重叠收益；短L3是否值得存储由实际成本判断。不移植SGLang源码。Qwen3.5-9B只做Native Full/Strict Prefix/state cache、观察Full KV与GatedDeltaNet状态布局/生命周期；禁止标准Blend套到Hybrid。27B等主线成本与兼容明确后再决定。

## 阶段E

按会话配对bootstrap，质量损失95%上界≤1pp且P95≤各自Prefix的90%，GPU-ms不明显恶化、数据移动不抵消Prefill收益、更新/冲突/跨会话稳定性符合要求，才判Go。缺独立GT或真实分布/观测不足为uncertain。未过联合门槛转严格Prefix+有收益的分层缓存。KV-aware Routing只设计接口/成本输入，暂无生产开发。
