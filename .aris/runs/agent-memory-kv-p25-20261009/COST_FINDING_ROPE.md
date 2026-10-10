# RoPE 表重复搬运：有限样本成本归因

Qwen3-8B 的patched LMCache legacy Blend路径，每层调用FusedRope.fused_encode时执行self.cos_sin_cache.to(k.device)，但原属性仍引用CPU表。有效CUPTI trace显示每个暖请求36次10,485,760-byte H2D。模型40960位置、128维、BF16的cos/sin表对应10MiB；独立GPU见证直接读取tensor storage确认10485760bytes。

可选优化PRISMSERVE_CACHE_ROPE_TABLE=1把.to(k.device)结果保留在FusedRope实例，再调用原函数。默认关闭。原Kernel、position、dtype不变；驻留表随实例生命周期保留，KV清空不主动释放，worker结束释放。该10MiB是表storage，不是总峰值HBM。

|暖请求|原H2D bytes|驻留H2D bytes|减少bytes|
|---|---:|---:|---:|
|1|391652634|14165274|377487360|
|2|391655058|14167698|377487360|
|3|391661690|14174330|377487360|

每次准确减少360MiB，H2D总量减少约96.4%。原36个10MiB复制事件变为0。其余复制不能仅按尺寸认定为KV；采样包含全词表输出的诊断开销。

Full-vocabulary logprobs：原/驻留Blend100与COW100各20请求，共40对，输入hash、完整token IDs一致，logprobs逐位一致。独立fresh follows-doc复跑60请求全部正确，Blend100输出token与Native Full一致，36层repair、部分计算缩减、Query隔离、COW来源和会话清空均通过。

局限：受控synthetic/profile输入，四短请求中三暖请求；两次profiling时间/物理GPU/主机共享资源状态不同，因此不据此比较TTFT。未完成普通服务采样的分阶段时间、真实生产分布、质量置信区间或P95收益门槛。修复确认了可消除的数据搬运问题，但不证明整个非前缀复用架构优于Strict Prefix。

证据：profiles/blend与profiles/blend-rope-resident逐请求trace-map及原始CUPTI；results/rope-resident-logit-comparison.jsonl及80个原始NPZ；logs/rope-environment-witness.md/json；figures/rope-h2d-comparison.png/pdf/svg。下一步冻结该可选配置，运行独立校准集后再选择正式评估参数。
