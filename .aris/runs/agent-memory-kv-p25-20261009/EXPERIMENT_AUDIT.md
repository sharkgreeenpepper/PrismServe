# Experiment Audit Report

日期：2026-10-10（Asia/Shanghai）。Auditor：fresh GPT-6-Astra / ultra，独立只读审计；`review_independence: same-family`，`acceptance_status: provisional`。审计没有启动GPU、没有修改实验源码/输入/原始结果，也没有向外部服务发送消息。只有本审计和forensic材料由审计员写入。

## Overall Verdict: WARN

**Integrity Status: warn。** 未发现伪造任务GT、按自身最优预测重标质量、删除失败请求或不存在的完成结果。原指定48个replay文件共2880条（含2400条校准）、140份NPZ、20份实际CUDA trace完成程序核对；数值核对零残留错误。WARN来自尚未执行的服务测量、单seed/有限合成范围、不均衡硬件对照及少量记录说明。这个结论不等于生产Go，也不是跨模型家族独立认证。先前额度中断及失败子审计标为ERROR，不计PASS；本结论来自本次重新读文件和CPU复算。

审计根目录：`/home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009`。下面相对实验文件均以此目录为根；`experiments/...`以项目根`/home/bumi/git/PrismServe`为根。未另列路径的源码短文件名均指`/home/bumi/git/PrismServe/experiments/agent_memory_kv/`；grouped源码另标。可复运行核对脚本及逐项数字在`.aris/traces/experiment-audit/20261009_p25/`。SHA清单共1254项；报告与tracker的本次审阅文本已另存`claims-snapshot/`。后续`witness-long-*`/long gate正在运行，明确不在本次完成范围。

## A. Ground Truth Provenance: PASS

`make_workload.py:30–71`的随机状态先生成项目金额，再同时写入当前L3事实和target；脚本历史也来自该独立状态，未调用模型。120条calibration可从seed1709、sessions=1、steps=5精确再生成。140条smoke/calibration可用冻结tokenizer重新组装，token IDs、SHA、Query边界和重复比例一致。每条target均可从输入权威事实独立读出；`contracts.py:69–74`要求独立来源标记，`run_replay.py:11–15,112`按strip后精确字符串比较。

`data/controlled-renderer-smoke.jsonl:1`明确synthetic/profile，并把public-content identity与Mem0原生ID区分。`bumi_adapter.py:8–38`只转换已有response，保留owner/time；只读SQLite入口在`:41–61`，本轮不构成真实生产轨迹。没有正式公开benchmark可要求其official evaluator。

Full只用于fidelity：`analyze_calibration.py:54`比较输出tokens，`run_replay.py:112–114`将logit诊断正确率设null并排除质量/时延统计。Full/Prefix均有2个真实任务错误，位于各自raw的第39、68行（输出9577/5193，GT8475/2316）；这些错误没有变为“标准答案”。因此100% Full token一致不代表100%任务正确。

## B. Score Normalization: PASS

正确率=正确条数/120；质量损失=逐请求Full.correct−trial.correct；会话bootstrap为24个会话均值的10000次配对重采样；P95改善=(1−trial P95/Strict Prefix P95)×100。`analyze_calibration.py:15–22,51–57`中的分母来自样本量或明确基线，不是某方法自身最优预测。repair ratio分母是reusable token数（`engine_bridge.py:83–87`），不代表总计算比例。

从全部NPZ读到完整151936个token IDs及未改动raw logprobs。初始40对raw KL均值为1.795702241335e−9；逐条max(0,raw KL)后均值为1.076766707355e−8，raw最大9.016150236340e−8。最初文字把截断均值简称“平均KL”；已在本审计期间改为`STAGE_A_REPORT.md:7`的明确说明，并保留`results/logit-kl-comparison.jsonl`两种值，重新核对通过。

新增`results/logit-kl-normalized-diagnostic.jsonl:1–40`的CPU float64 logsumexp是对各完整概率分布纠正有限精度归一性，不是把任务分数除以自身最大值。原NPZ/raw KL保持不变；独立重算normalized KL均值4.387522001116e−9、最大1.887776699773e−8，与新增文件逐项一致。它仍只证明固定位置fidelity。远小于1的KL不能替代任务指标，也不保证尾部每个logprob逐位相同。

## C. Result Existence / Completion / Provenance: WARN（数值核对PASS）

- **校准完整且数值一致。** 20个配置各120个唯一有序request ID，全部status=ok、一次attempt、exit0、持久化completed；manifest SHA与queue_state一致。20份stdout各120个request/correct与raw一致，queue-manager恰有20次started/finished。全部summary.csv列及报告20行的四舍五入数值复算一致；69条任务错误与`analysis-final/errors.jsonl`完全相同，没有删除或重复请求。
- **输入与真实调用一致。** 所有raw的input_hash、target、session、origin/split、Query span与冻结输入匹配。各修复请求都记录层0–35且一次repair invocation，预算、选择位置和mandatory计数一致；加载chunk与选择修复位置均不覆盖Query。所有缓存来源为同会话的早前request且context hash匹配。各新会话首请求无load/repair，original-only来源代数0，COW smoke代数最高8。校准每个Blend配置86个实际修复请求，每个COW配置92个，其高层QKV行均减少；不是声称全部120条都修复。
- **初始/模型数值有raw。** 初始4×20任务均正确且tokens一致；独立环境5×20及RoPE环境3×20均正确。14B Full20条、Hybrid Full/Prefix各20条正确，Hybrid八条记录528个prefix cached tokens，32层/8 full-attention层符合config；state-offload字段为false/null。`results/stage-a-initial-summary.json:1`和`results/model-smoke-validation.json:1`与这些raw吻合。
- **CUPTI是真实事件文件。** 20份trace均存在、可解压且有CUDA kernel事件，trace-map与对应raw IDs/hash匹配；原16份与追加RoPE4份分开。三个warm pair直接重算H2D为391652634→14165274、391655058→14167698、391661690→14174330；每对差377487360 byte=360 MiB，36个10485760-byte事件变为0（`profiles/blend/trace-map.jsonl:2–4`和resident对应行；`COST_FINDING_ROPE.md:9–13`）。没有把所有D2H认成KV，也没有把10 MiB表大小认成总HBM。
- **grouped有限gate有效。** 初始primitive日志8 case，扩展日志12 case，最大绝对差0.001953125；future_invisible=true。20条repair100 NPZ与Full比较数值吻合、argmax一致；`grouped/logit-comparison.jsonl:1`的隐含参考确实是Full，建议明写reference路径/hash。20对short COW off/on输入、repair位置/预算、load provenance、输出tokens均一致且40条任务正确；on的18暖请求有实际region telemetry，off没有。没有把此短检查扩展成long质量或性能认证。
- **锁与原件。** Stage-A源锁808项：现源码变化的2文件可在stage-a-initial快照匹配，其他一致；RoPE锁15项、calibration锁17项一致。grouped旧锁19项中witness扩展差异已补充`LOCKS.md:1`、旧SHA精确匹配的重建文件和当前19项新锁；其余实现未变。重建文件不证明同时代保存，说明已诚实保留此限制。3模型26个已锁文件（含17个权重分片）实际流式SHA全部一致，索引引用分片完整；运行时233个固定package版本一致。
- **遗留数字另取原raw核对。** 历史edit-window的10份/2000条原结果中恰1440次fallback，其中1368次预算不足、72次下游不足；示例见旧run的`extra/edit-window/results/guard-edit-16384-continuous-17.jsonl:2,92`。这只是fallback原因，不是非前缀速度收益或引擎失败证明。

剩余记录WARN：tracker总述已更新为20/20校准完成，long gate未计完成；但`EXPERIMENT_TRACKER.md:10`表格仍写“Hybrid推理尚未验证”，与`:17`的native短检查并存。应显式标初始快照或统一为“native短检查通过，状态offload未验证”。`environment-validation.json:6–7`的false属于初始8B环境范围，新增additional_native_model_checks链接指向后续native结果；读取者需要保留这个层次。这是状态表达问题，没有将未完成实验冒称完成。

## D. Dead Code / Missing Measurements: WARN

已报告的score、bootstrap、CUPTI汇总函数均有实际调用/输出。修复控制从runtime `vllm/v1/worker/gpu_worker.py:735,761–767`安装，LMCache `cache_engine.py:674–678,1047–1074,1215–1219`在lookup/retrieve/store前排除Query；实际层遥测与trace支持有限调用证据。

但`timing.py:8–19`的HostSpans未接入run_replay；`:36–54`的summarize_host只见`test_contracts.py:47`测试调用。现`run_replay.py:115–120`只有引擎TTFT+controller RPC及e2e，服务GPU-ms、H2D/D2H、峰值HBM保持null；`analyze_calibration.py:62`也没有生成这些值。first-decode和phase列表只有设计/工具支架，不能声称完成分阶段服务成本归因。`analyze_profiles.py:24–27`的kernel duration sum仅为诊断trace事件求和，不是无重叠服务GPU-ms。报告目前明确承认缺项，因此是未完成WARN而非phantom-result FAIL。

## E. Scope / Fairness / Confidence: WARN

实际校准为Qwen3-8B、24会话×5步、8anchor×3scenario、20配置各一次；数据seed1709，推理seed17。总2400条含同输入跨方法重复，不能当2400个独立会话。增长L1是脚本GT历史，非闭环Agent rollout（`make_workload.py:68–71`）。smoke仅2会话/20请求；14B和Hybrid只做native短输入，不能证明规模收益或Hybrid缓存状态生命周期。

全部18非前缀配置P95比Prefix的1.022090852528秒更高；Full为2.618646230910秒，最小非前缀为3.323120920465秒。可以报告当前校准描述，不可以据此做严格因果速度量化：manifest每配置固定一张物理GPU、没有均衡重复；其他GPU的微检查可能共享主机资源。RoPE历史trace也非同时间/同物理GPU/同负载，所以只支持搬运字节变化。

bootstrap确为会话配对经验95%单侧上界（`analyze_calibration.py:15–22`），不是独立holdout或分布无关保证；两个topk配置零观测correct差得到上界0，不能证明总体质量等价，也不能在18配置中选优后仍把同一CI当正式验收。`CALIBRATION_REPORT.md:30–38`及`scope.json:4–8`均正确限定这一点。

真实生产分布、独立GT holdout、更新/冲突更复杂任务、公平硬件重复、GPU-ms/普通服务移动/HBM仍缺。当前数据不支持Research Contract B的质量约束净P95改善≥10%，也不支持生产Go。

## F. Evaluation Type: PASS（类型明确）

|评估|分类|准确含义与claim上限|
|---|---|---|
|受控金额任务正确率|`simulation_only`|合成记忆/脚本会话+独立生成状态GT；实际GPU模型推理，非模拟执行时间。仅对此合成任务计正确率。|
|与Native Full的token/logprob比较|`synthetic_proxy`|模型参考fidelity，不是任务GT或真实生产质量。|
|随机tensor attention/packed primitive|`simulation_only`|数值/因果/布局机制检查；不等于模型质量或系统收益。|
|CUPTI拷贝/kernel诊断|真实硬件诊断（无任务GT）|可报告实际事件与字节，不据诊断采样报告服务P95。|

没有生产`real_gt`数据、human_eval或正式closed-loop Agent成功率。本表使用skill的标签体系；simulation_only指任务/张量由人工控制，绝不意指模型或CUPTI为假执行。

## Claim Impact

- **A：needs qualifier。** Query隔离、来源控制、短输入repair100 fidelity和部分修复调用有支持；正式质量保持、真实Agent多轮泛化未证明。
- **B：unsupported。** 当前18非前缀配置均无P95正收益，且成本测量/公平对照未齐；不能声称净改善≥10%。
- **C：needs qualifier。** 支持14B/Hybrid native短检查；不支持Hybrid状态offload或非前缀state repair。
- **RoPE：supported with scope。** 三个warm诊断请求每次消除360 MiB搬运、40对完整词表logprobs一致有直接证据；不蕴含端到端速度改善。
- **Grouped：supported with scope。** 限12-case primitive、20条repair100和20对short partial COW；后续long/fair-performance gate不在此结论。

## Action Items

1. 对正式质量/生产claim：冻结候选后做独立seed/会话holdout，补真实脱敏轨迹；保持独立GT与Full fidelity分离。
2. 对性能/成本claim：完成实际服务phase/GPU-ms/普通采样传输/HBM/首decode测量，做均衡GPU、负载、顺序及重复次数控制。
3. 同步tracker的历史表格与追加状态，报告中区分数据seed1709与推理seed17；给grouped comparison写明参考路径/hash及方向。
4. 保留旧结果、旧锁和raw KL；对后重建的primitive快照保留“重建而非同时代保存”说明，新运行使用新源码锁。

审计边界：文件/代码/日志一致性检查不能提供硬件执行的加密证明；模型家族仍相同，因此语义审计为provisional。原始指标可确定复算，但正式质量、性能或生产验收仍未完成。
