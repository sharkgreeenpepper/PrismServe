# P2.5 阶段 A 初步机制验证

ModelScope 下载并校验 Qwen3-8B、Qwen3-14B、Qwen3.5-9B 完成。master 为可变分支，复现以下载文件 SHA256 锁定清单为准。初始机制验证采用Qwen3-8B；后续Qwen3-14B和Qwen3.5-9B已完成下面列明的短文本Native检查，不能据此声称Blend或状态offload兼容。

隔离运行时使用 BF16、TP=1、Qwen3-8B 36 层，关闭 Thinking。四组各20个受控请求：Full、Strict Prefix、Blend100、Blend+COW100，合计80个请求均正确，输出 token 全部一致。Blend与COW各18个暖请求实际逐层修复，记录36层；加载块均未覆盖当前 Query。原始缓存来源代数0，COW最高来源代数8；两会话在清空后的首请求都没有修复调用。四组同时占用GPU0–3，因此不用于公平性能比较。

另有60个单 token 全词表诊断请求：Full、Blend100、COW100各20个。完整词表 token IDs 对齐，首 token 全部一致；两个修复组原始KL负浮点值截断后平均1.08e-8，原始最大KL9.02e-8；原始值和截断值均保存，不能把截断均值写成未处理原始均值。数值接近但不声明浮点逐位一致。诊断结果排除任务正确率和时延统计。

Packed NHD/HND两种布局×两种token顺序的GPU往返与未触及槽位检查通过。任意选择位置的原生FA3与dense最大绝对差0.001953125，未来token不可见检查通过。

输入来自 Bumi renderer 的纯函数与受控事实，origin=synthetic/split=profile，未调用生产SQLite或Mem0。public LTM不是纯Mem0 Top-5；其所有条目及owner/time均保留，内容哈希身份不伪装为Mem0原生版本。当前缺真实脱敏生产轨迹。

历史1440次编辑回退的原因是原策略将修改点至prompt末端视为必须修复的依赖区间：1368次超出预算、72次无法填满等预算。它是保守算法回退，不是已证实的引擎兼容错误，也不能证明短窗口性能收益。新实验需要显式区分强制新增计算、候选修复和Full回退。

当前仅通过有限机制检查，未完成正式质量门槛、P95净收益、GPU成本归因、规模对照或Hybrid状态兼容。独立环境 follows-doc 复核和分离profiling继续执行。原实验文件、Bumi_Mem和旧运行环境均保留。

追加独立文档见证：9条命令exit0，5组各20请求正确；5%真正减少高层计算，COW来源1–8、Query隔离与清空均通过。此环境仅机制ready。

规模/Hybrid短请求兼容性：Qwen3-14B Native Full20/20正确；Qwen3.5-9B Native Full/Prefix各20/20正确，输出token一致，其中8个Prefix请求记录528缓存token。Hybrid32层中8层Full Attention，尚未测量DeltaNet状态大小或验证状态offload。

原生CUPTI链路捕获16份有效CUDA trace（四组各4请求）。采样包含全词表logprobs导出，其D2H不全部归为KV；阶段归因尚不完整，结果不进入P95或任务质量统计。暖Blend中36次10,485,760byte H2D与RoPE表逐层复制路径一致；可选表驻留优化等待复核及GPU对照。

审计后补充：results/logit-kl-normalized-diagnostic.jsonl以CPU float64 logsumexp对每个完整分布归一化，单独记录raw/normalized KL；不改原始NPZ或先前comparison。版本快照说明在variants/grouped/LOCKS.md。
