# P2.5 Tracker

|工作|状态|证据/限制|
|---|---|---|
|Bumi仓库只读核对|完成|0.6.0/commit5628ba1；Query已在L1尾，public LTM含core/history|
|真实生产轨迹|缺数据|用户仅提供仓库；无本地DB/响应导出|
|Prompt/data/cache候选契约|CPU完成|9测试；实际ModelScope tokenizer与LMCache切块一致|
|Query不复用修复|GPU已验证|80条中加载chunk均不重叠当前Query|
|Qwen3-8B/14B下载|完成|ModelScope下载及SHA256锁定|
|Qwen3.5-9B下载|下载完成|Native Full/Prefix短检查通过；state offload未验证|
|Qwen3.8-27B|未启动|优先级P2|
|新GPU环境|有限机制ready|独立9命令全部通过；不代表生产质量|
|四arm正确性|机制通过|初始80+独立100正确；另60 logits诊断|
|成本profiling|有限阶段证据完成|原CUPTI 16 trace + instrumented 8 trace；lookup/transfer/store/recompute 已观测，见 `variants/instrumented/PHASE_SMOKE_REPORT.md`；全词表一 token 诊断，不是服务P95/GPU-ms|
|质量/净收益边界|未运行|不以CPU测试或模型导入代替真实结果|

Qwen3-14B Full 20/20正确；Qwen3.5-9B Full/Prefix各20/20正确且token一致，8个请求观测528个严格Prefix缓存token。仅短文本兼容性，不证明状态offload或生产质量。

RoPE驻留可选配置通过独立fresh文档复核：60/60受控请求正确，40对logprobs逐位一致，3warm trace各36×10MiB复制消失，实际表storage10MiB。默认关闭，生产/P95门槛仍未通过。20配置×120请求校准全部完成；20/20 completed、2400条、无runtime错误；正式holdout尚未启动。

2026-10-10：calibration/analysis-final和CALIBRATION_REPORT已完成。分组元数据隔离副本通过12-case CUDA primitive、20 repair100 logits、40 short COW off/on gate及80条独立 long follows-doc。GPU0同卡120×4配对结果显示group-on P95为2.546s，低于group-off的4.089s，但仍是Strict Prefix 1.021s的2.494倍；未通过性能联合门槛。

2026-10-10：物理GPU3完成 instrumented CUDA trace smoke，Full/Blend各4条synthetic短输入，8份trace与请求一一映射；Blend的三条暖请求确实命中缓存并修复36层。按CUDA External ID/correlation关联到最内层启动阶段后，观察到每暖请求约14.16MB H2D拷贝落在kv_transfer范围。全词表采样本身产生额外D2H/D2D；kernel duration为启动范围归类、主机区间存在嵌套，不能相加为GPU-ms或服务TTFT。代码与产物复核7 PASS/7 WARN/0 ERROR；[阶段报告](variants/instrumented/PHASE_SMOKE_REPORT.md)。GPU0–3已释放。

实验诚信审计此前未覆盖新的long gate、480条paired replay及本次instrumented trace；补充独立审计待完成。真实脱敏生产轨迹、正式holdout、8K/16K成本对照和质量/P95联合门槛仍未完成，当前没有生产Go结论。

2026-10-10：Qwen3-8B host-only service phase 完成并独立审计（120×Full/Strict Prefix/Blend，共360请求，GPU0，单并发）。Full/Strict 各118/120，Blend 117/120；Blend相对Full质量损失0.833pp，会话bootstrap单侧95%上界2.5pp；含上下文RPC的P95为2.545s，Strict Prefix为1.019s（Blend慢149.728%）。联合门槛未达。审计6386/6386确定性检查通过，PASS_WITH_WARNINGS；结论仅适用于合成校准轨迹。历史同轨迹未插桩各臂输出120/120一致，观测P95差<0.14%，但Full/Strict grouped flag不一致且运行未随机化，不能因果估计插桩开销。主机阶段覆盖整次多token generate且可嵌套；GPU-ms、搬运bytes、HBM峰值仍缺失。见 `variants/service-phase/paired-120-host-only/analysis-final/HOST_PHASE_REPORT.md`、`variants/service-phase/EXPERIMENT_AUDIT_SERVICE_PHASE.md` 和更新后的 `variants/service-phase/RUN_MANIFEST.json`。

2026-10-10：Qwen3-14B阶段C成本探测完成。4条host-only冒烟4/4正确、事件映射齐全；正式Full/Strict Prefix/Blend-window10各120条，三臂均120/120正确且Blend与Full输出逐请求完全一致。Strict Prefix含RPC P95=1.839s；Blend=4.389s，是2.386× Strict，质量均值损失/会话bootstrap上界均0pp（仅固定校准集）；联合门槛未达。Blend记录86修复请求、37,458 repair tokens、374,924 reusable、131,471 mandatory；host-only阶段transfer/store P95为1.828/2.291s，但区间可能嵌套，非物理传输/GPU-ms。GPU0一秒采样显存最高60,277MiB，非精确峰值；GPU0–3运行后释放。相对8B，Blend/各自Strict P95仍为2.386× vs2.497×，未观察到非前缀收益交叉点；单模型/单轨迹不足以支持规模规律。独立代码审查PASS_WITH_WARNINGS；实验诚信审计只完成部分检查（审计员报告81,206项零差异、fixture seed1709复建），但最终A–F verdict/最终哈希因GPT-6.1-sol使用额度耗尽未交付，审计状态ERROR_PARTIAL_REVIEW，不视为独立PASS。报告：`analysis-model-scale-final2/MODEL_SCALE_REPORT.md` 与 `analysis-model-scale-final2/figures/model-scale-quality-latency.png`。14B细项见 `variants/qwen3-14b-cost-probe/paired-120-host-only/analysis-final/HOST_PHASE_REPORT.md` 和 `RUN_MANIFEST.json`。生产L1/L3轨迹与holdout仍缺。
