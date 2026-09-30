# 方向一：真实 vLLM 多 GPU 放置 Pilot

**方向：** 推理树分支的多 GPU 放置与 KV 选择  
**模型：** DeepSeek-R1-Distill-Llama-70B（bf16）  
**运行日期：** 2026-09-30  
**结果类型：** 真实 vLLM 推理；冻结 prompt trace；探索性小样本比较

## 结论

本轮在 4 张 GPU 上对 4 种放置策略各运行 8 棵推理树、64 个节点请求，使用同一份冻结 prompt trace。四种策略的 majority exact-match 都是 7/8。least-loaded 的树级 p95 比 local-only 低 2.3%，但 p50 高 18.0%；KV-cost 和 sibling-lookahead 的 p95 分别比 local-only 高 5.6% 和 4.1%。因此这组数据没有证明树感知调度稳定优于常规路由，也不支持将该策略集成进 PrismServe。

这只是一个小型 pilot：每种策略单次运行，只有 8 棵树；按 nearest-rank 计算时 p95 等于观测最大值。least-loaded 的 p95 小幅改善同时伴随 p50 变差，不能据此声称稳定收益。当前结论是**未验证出收益，方向仍可由更真实的动态树轨迹和重复实验继续检验**。

## 实验设置

- 模型：DeepSeek-R1-Distill-Llama-70B，bf16；vLLM 0.30.0，PyTorch 2.13.0，固定 temperature=0。
- 硬件：GPU 0–3；两个 vLLM 副本各使用 TP=2。每个副本 `max_num_seqs=2`，启用 vLLM prefix caching。每种策略运行前重启服务以清空缓存。
- 数据与树：GSM8K 测试集固定抽取 8 道题，每种合成树形（balanced、broad、chain、skewed）各出现两次；每棵树 6–10 个节点，总计 64 个请求。每个节点最多生成 1024 tokens。
- 冻结输入：[`REAL_PLACEMENT_TRACE_20260930_074612.json`](REAL_PLACEMENT_TRACE_20260930_074612.json) 保存每个节点完整 chat messages。四种策略使用相同 64 份 prompt；本地 tokenizer 与服务端报告的 prompt token 数全部匹配（0 个不匹配）。冻结 trace 从 local-only 结果构建，目的是隔离放置策略对同一组请求的影响。
- 指标：每棵树完成时间的 p50/p95、答案 exact match、叶节点答案解析率，以及 vLLM prefix-cache token 计数器增量。缓存命中率按 `hit_delta / query_delta` 对两个副本的 token 计数加权。
- 策略：local-only、least-loaded、kv-cost、sibling-lookahead。后两者是本 pilot 中实现的启发式；kv-cost 使用预设的 25k prefill tokens/s 与 60 decode tokens/s 服务速率假设，sibling-lookahead 是拓扑感知启发式，不是 oracle。

## 冻结输入下的结果

| 策略 | 树完成 p50 (s) | 树完成 p95 (s) | p95 相对 local-only | majority exact match | 叶答案解析 | Prefix-cache 命中 token | 达到长度上限 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local-only | 124.02 | 196.26 | 基线 | 7/8 | 25/30（83.3%） | 25,664 / 34,164（75.1%） | 0 |
| least-loaded | 146.34 | 191.76 | -2.3% | 7/8 | 26/30（86.7%） | 22,032 / 34,164（64.5%） | 0 |
| kv-cost | 145.34 | 207.27 | +5.6% | 7/8 | 24/30（80.0%） | 21,280 / 34,164（62.3%） | 1 |
| sibling-lookahead | 147.35 | 204.39 | +4.1% | 7/8 | 25/30（83.3%） | 22,160 / 34,164（64.9%） | 1 |

四种策略均为 7/8 majority exact-match。答案解析率是能否从叶节点回复中解析出规定格式的数值答案，不等同于正确率。kv-cost 与 sibling-lookahead 各有一个叶请求达到 1024-token 上限；kv-cost 的该响应没有闭合 reasoning 标记。

least-loaded 的 p95 只比 local-only 快约 4.50 秒，但它的 p50 慢约 22.31 秒；小样本的尾部值受单棵树影响很大。local-only 的 prefix-cache token 命中率最高，不过这些计数不能直接换算成端到端延迟收益。

## 解释与限制

1. **当前没有树感知策略优势的证据。** kv-cost 和 sibling-lookahead 均未降低 p95；least-loaded 的小幅 p95 改善没有反映在 p50。答案准确率也没有因策略改变而提升。
2. **样本量和重复不足。** 每个策略只有 8 棵树且只运行一次；这里的 p95 是最大观测值，不是稳定的尾延迟估计，也没有置信区间。
3. **这是冻结请求回放，不是完整动态搜索。** 子节点只有在父请求完成后才释放，但子节点 prompt 来自 local-only trace，并不采用当前策略新生成的父节点输出。这样能固定比较输入，却不能测量真实搜索过程中的分支变化。
4. **不测量跨副本 KV 传输。** DeepSeek chat template 可能移除或改写带 `think` 标记的 assistant 历史；trace 里的后续消息不代表 parent-output reasoning KV 可复用。远端前缀会重新计算，未实现 KV 迁移。
5. **启发式和缓存统计有限。** kv-cost 的 prefill/decode 速率是固定假设；估算不覆盖运行队列 API、链路争用、KV 驱逐、活跃兄弟节点共享或精确生成 token IDs。vLLM 的 prefix-cache 指标是 token 计数，不是缓存块计数。
6. **树与语料是合成的小规模配置。** 8 个 GSM8K 样本和四类固定树形不能代表真实搜索算法的动态分支分布。

## 产物

- Runner：[`run_vllm_placement_pilot.py`](run_vllm_placement_pilot.py)
- 冻结 trace 构建器：[`build_prompt_trace.py`](build_prompt_trace.py)
- 冻结输入：[`REAL_PLACEMENT_TRACE_20260930_074612.json`](REAL_PLACEMENT_TRACE_20260930_074612.json)
- 每种策略的原始请求、树级 CSV 和汇总 JSON：四组 `REAL_PLACEMENT_*` 冻结运行文件（时间戳分别为 075200、075650、080157、080712，按策略对应）。
- 更早的 256/512-token 预算调试运行及非冻结动态历史也保留为迭代记录；它们不属于上表配对主结果。动态历史在不同策略下生成了不同的对话内容，不能用于因果比较。
