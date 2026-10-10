# P2.5 Experiment Audit 独立补充报告（2026-10-10）

**Overall verdict: WARN；8 PASS / 7 WARN / 0 ERROR。** 新增文件的确定性复算没有残留数值不一致。支持受限的同GPU校准、long follows-doc执行、host钩子与CUDA启动归因；**不支持生产Go、正式质量保持、正式服务P95、KV-only成本或服务GPU-ms结论**。

审计员：fresh `gpt-6.1-sol`子代理；`review_independence=same-family`，`acceptance_status=provisional`。独立从文件读取证据，不采用执行者的解释作为判断依据。只运行CPU读取/解压/数值运算；未启动GPU、未修改源码、raw JSONL、trace、NPZ或原审计。只新增本补充报告/JSON与forensic复算材料。

本报告内未注明项目根的相对路径以 `/home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009` 为根；`experiments/`以 `/home/bumi/git/PrismServe` 为根。

## 方法与样本边界

- 对480条paired、80条long witness、8条instrumented逐行验证唯一ID、顺序、input hash、Query边界、来源和GT。120个calibration target从源数据L3当前项目事实独立读出，逐行复算exact-label score；Full仅作fidelity参考，不充当GT。
- P50/P95采用NumPy默认linear quantile，对raw `ttft_with_context_s`（引擎first-token latency加context RPC）计算。复算paired-summary全部数值及96个cell。mandatory由`[0,loaded_span_tokens)`减去loaded ranges独立重建；L1均值取其中System+L1 segment交集，分母是有repair telemetry的暖请求。
- 读取8份gzip trace的实际kernel/memcpy和byte字段；用独立实现复算External ID→CUDA API→同pid/tid最窄包围`prism:*`标记。External ID无关联API时用correlation，冲突则保留未归因。逐trace输出和总summary逐字段比对；摘要重建另调用作者request_summary后，以独立直接事件计数/bytes交叉核验。
- 加载8份NPZ，验证全151,936词表ID、有限raw logprobs、argmax与1-token输出。Full额外读取4份既有NPZ做逐位比较；Blend只比较matching window10的首token，未冒充全词表对照。
- paired有120 unique inputs/24会话；long有20 unique inputs/4会话，完全是原calibration子集；instrumented只有4 unique inputs/1会话，跨2 arm得到8条记录。不能把跨arm重复计作独立样本。
- 文件SHA-256、可复算脚本、全部细节存于 `.aris/traces/experiment-audit/20261010_p25_addendum/{recompute.py,recomputed-evidence.json,recompute.stdout.log}`。原审计与读取文件在审计前后hash保持一致。

## 同GPU paired校准复算

|配置|raw正确数|与Full token序列一致|P95 `ttft_with_context_s`秒|
|---|---:|---:|---:|
|Full|118/120|120/120|2.6188794970628804|
|Strict Prefix|118/120|120/120|1.020544907229486|
|Window10 group on|117/120|117/120|2.5457398061989807|
|Window10 group off|117/120|117/120|4.089310134202241|

`1 − P95(on)/P95(off)` = **37.746472567417186%**，与报告37.746%一致。`P95(on)/P95(Prefix)` = **2.4944907256555733**，与2.494倍一致。Full与group on正确数差1/120，即**0.8333333333333334pp**。这里是这一次样本的描述性P95，不是正式服务验收结论。

120/120 off/on输出token序列相同。两arm各86条repair，全部选择位置/预算/36层计数/load provenance相同；96个cell逐项一致。Full/Prefix错误保留在`paired-{full,prefix}.jsonl:39,68`，group on/off错误在`paired-window10-{on,off}.jsonl:39,68,69`。第39行GT8475，Full输出9577，group输出9760；第68行GT2316，Full5193、group6235；第69行group6235仍错误。不能以off/on一致率代替任务质量。

## Long follows-doc见证复算

4 arm各20条，0 runtime errors，独立GT正确80/80；Full/Prefix、Full/off、off/on分别20/20 token序列一致。五个保存的exitcode均0，primitive原stdout有12 cases、future_invisible=true、max delta=0.001953125。

两COW arm各16条repair；各有51个`namespace=repaired`加载chunk（不是字段值`cow`），source generation最高3；on每个repair有35个region条目，region数1–6，最大region token数518–17340；off无region条目。各repair实际36层、loaded/mandatory/repair budget/Query排除独立核对通过。prompt长度为5234及18550–18702 tokens。

该long gate使用原calibration的20条完全相同input/target，并非独立seed/holdout。80/80只说明这个子集的四次执行都正确，未弥补原120条population中的partial-repair错误。GPU3前后0MiB是已有环境见证JSON保存的记录，本审计未重新查询GPU；worker module __file__仍无直接记录。

## Instrumented机制与CUDA归因复算

Full/Blend各4条成功，长度223、324、425、526；全部1-token全词表诊断、`correct=null`、两项eligibility=false。每条唯一trace及NPZ存在可解析，全部151936个token ID完整、raw logprobs有限、argmax匹配输出。Full4份与旧Full逐位相同；Blend首token与window10 off/on相同，完整logprob匹配对照仍缺。

Blend首请求无repair，随后三请求各加载4个旧chunk、实际36层repair；reusable=96、repair budget=repair count=9，mandatory=165/266/367，loaded span=261/362/463。loaded range为同会话早前请求的相同token chunk，不含Query；on有35个region条目。层0/1 QKV均是整个loaded span，层2–35 QKV为mandatory+9，不能称36层都仅计算9 tokens。

|暖请求|trace总H2D bytes|归到`kv_transfer`的H2D bytes|host transfer/recompute spans|
|---|---:|---:|---:|
|controlled-0:1|14,166,574|14,155,848|39 / 37|
|controlled-0:2|14,169,278|14,155,848|39 / 37|
|controlled-0:3|14,175,630|14,155,848|39 / 37|

`kv_transfer` H2D合计**42,467,544 bytes**；每条**14.155848 MB**（十进制，约14.16 MB）。它是启动范围内直接CUPTI流量，**不是经识别的纯KV payload**。直接读取memcpy bytes，无缺失byte count；不能拿raw token数乘公式替代这些观察。

|arm|kernel总数/归因数/未归因数|memcpy总数/未归因数|kernel correlation回退|kernel / memcpy嵌套scope解析数|
|---|---:|---:|---:|---:|
|Full|2188 / 2140 / 48|68 / 40|4|0 / 0|
|Blend|6026 / 5978 / 48|2062 / 46|112|295 / 540|

所有未归因事件都因API落在`prism`范围外；没有missing API/等窄冲突。各trace External ID或correlation首先关联CUDA runtime/driver API，再按API所在CPU pid/tid归因，未将GPU执行时间戳与host monotonic timestamp直接相减。

Blend `kv_transfer` 1206 memcpy/1110 kernels、kernel duration sum **4.279981ms**；`selective_recompute` 2289 kernels、**100.643927ms**、DtoD **2,224,128bytes**；`native_model_forward` 2136 kernels、**55.599985ms**。`kv_store` 540 memcpy，DtoH **22,118,472bytes**、DtoD **22,118,400bytes**。两arm各first-token diagnostic DtoH **4,862,032bytes**、DtoD **9,723,904bytes**。均与PHASE_SMOKE_REPORT表值四舍五入一致。

冷Blend host区间总和**122.894543ms**、并集**79.845864ms**，与报告122.895/79.846ms一致；nested span不可相加当TTFT。末暖请求`controlled-0:3`另有38个host kv_store span，未产生追加归为store的CUDA event。每请求lookup有1个host span，但CUDA trace没有lookup user annotation；这是host reachability证据，lookup CUDA成本仍未知。37/39/38个wrapper span均不应当作模型层数。

## PASS / WARN / ERROR逐项记录

### P1 PASS — 独立GT与请求完整性

480条paired和80条witness的请求ID、顺序、hash、target、Query span与冻结input一致；120个GT均能由源L3当前项目事实独立读出，correct按strip后的exact label复算，错误未删除。

证据：`data/controlled-calibration-120.jsonl:1–120; variants/grouped/paired-*.jsonl:1–120; variants/grouped/witness-long-*.jsonl:1–20`。

### P2 PASS — paired汇总与96个cell

正确数、Full token一致数、线性P50/P95与paired-summary逐项一致；96行paired-cells的5请求计数/正确数/P95、暖repair请求mandatory∩System+L1均值一致。

证据：`variants/grouped/paired-summary.json; variants/grouped/paired-cells.csv:2–97; variants/grouped/PAIRED_REPORT.md:7–16`。

### P3 PASS — group开关机制与输出

120/120 off/on token序列一致；两侧各86个repair请求，选择位置、预算、加载来源、36层QKV计数均一致，独立loaded-range补集与mandatory计数吻合；来源均为同会话早前request且chunk token identity一致。

证据：`variants/grouped/paired-window10-{on,off}.jsonl:1–120`。

### P4 PASS — long follows-doc见证

4×20条均成功且独立label正确；五个exitcode为0；12-case primitive原stdout与见证一致。两COW arm各16条repair、51个namespace=repaired加载chunk，generation最高3；on每repair 35个region条目，范围[1,6]、max-region tokens[518,17340]。

证据：`variants/grouped/witness-long-*.jsonl:1–20; variants/grouped/ENVIRONMENT_WITNESS.md:3–24; variants/grouped/witness-primitive.stdout.log`。

### P5 PASS — 8请求/trace/NPZ映射及fidelity

8条raw各成功、1 token、correct=null、timing/task_accuracy=false；各有唯一可解析gzip CUDA trace和151936维有限raw logprobs NPZ，argmax等于输出。Full4份NPZ与既有Full逐位一致；Blend4个首token与window10 off/on一致。

证据：`variants/instrumented/{full,blend}.jsonl:1–4; variants/instrumented/{full,blend}-traces/trace-map.jsonl:1–4; raw logit_artifact路径; FIDELITY_CHECK.json`。

### P6 PASS — phase-smoke summary/CSV与命中修复

8个逐请求JSONL/CSV、旧trace-summary与直接CUDA事件计数/byte count/host span重建一致；summary arms计数与流量一致。Blend3暖请求每条4旧chunk、96 reusable、9 repair budget、36层；mandatory为165/266/367。

证据：`variants/instrumented/phase-smoke-requests.{jsonl,csv}; phase-smoke-summary.json; trace-summary.jsonl; blend.jsonl:2–4`。

### P7 PASS — CUDA独立启动归因复算

独立parser按External id→API，缺API时用correlation→同pid/tid最窄包围prism范围，逐trace kernel/memcpy全部字段与归因JSONL一致，汇总一致；未发现等长scope冲突或missing API，未归因事件保留。

证据：`variants/instrumented/cuda-phase-attribution.jsonl:1–8; cuda-phase-attribution-summary.json; 各trace-map.files[0]; experiments/agent_memory_kv_instrumented/attribute_cuda_phases.py:30–101`。

### P8 PASS — 诊断口径与原审计保留

报告明确kernel duration sum可重叠，memcpy不是KV-only，未将阶段host加总当TTFT；服务GPU-ms/HBM字段保持null。原EXPERIMENT_AUDIT.md/json及本次读取的68项文件SHA在审计前后相同。

证据：`variants/instrumented/PHASE_SMOKE_REPORT.md:5,11,28–37; EXPERIMENT_AUDIT.md:43–53; recomputed-evidence.json.audited_input_hashes`。

### W1 WARN — 单次校准性能边界

同GPU0复测消除了这4组之间固定GPU差异，但顺序Full→Prefix→on→off、单数据seed1709/推理seed17、无均衡顺序/多重复；37.746%只是这次样本P95差，不能作为因果收益或正式服务P95验收。

证据：`variants/grouped/run-paired-120.sh; PAIRED_REPORT.md:3,12`。

### W2 WARN — long见证不是holdout

80条输出是20个unique input跨4 arm重复；4会话全部来自原controlled-calibration-120-inputs的完全相同子集。独立见证执行及独立GT不等于独立统计样本/holdout；原partial-repair任务错误仍保留。

证据：`variants/grouped/long-gate-inputs.jsonl:1–20; data/controlled-calibration-120-inputs/paired-inputs.jsonl; paired-window10-on.jsonl:39,68–69`。

### W3 WARN — 服务成本与比较缺项

instrumented仅1会话4个短prompt(223/324/425/526)，全词表1-token诊断；缺Strict Prefix诊断arm、8K/16K同轨迹、普通服务采样、TPOT/并发、服务GPU-ms、峰值HBM、KV-only流量。不能将8条diagnostic计入任务质量或服务P95。

证据：`variants/instrumented/{full,blend}.jsonl:1–4; PHASE_SMOKE_REPORT.md:10–11,37`。

### W4 WARN — lookup及store span解释边界

lookup只有每Blend请求1个host span，CUDA user_annotation不存在，故lookup获host reachability见证而非CUDA成本归因。controlled-0:3另有38个host kv_store span但无新增store CUDA event；不能把host span数视为层数/实际copy数。

证据：`variants/instrumented/blend-traces/trace-map.jsonl:1–4; PHASE_SMOKE_REPORT.md:17,21–22; experiments/agent_memory_kv_instrumented/phase_hooks.py:32–54`。

### W5 WARN — Blend数值fidelity范围

Blend只核对4个首token，未获得window10一致配置的未插桩全词表NPZ；repair100与COW运行不能扩展成任意input、任意logit逐位等价。

证据：`PHASE_SMOKE_REPORT.md:34–35; FIDELITY_CHECK.json; variants/grouped/cow-window10-{on,off}.jsonl:1–4`。

### W6 WARN — provenance与环境见证边界

manifest为raw/log重建，fingerprint为post-run当前源码；grouped worker绝对__file__未直接发出。GPU3前后0MiB是保存的见证记录而非本审计新采样。文件一致性不提供硬件执行加密证明；同家族语义审计为provisional。

证据：`variants/instrumented/RUN_MANIFEST.json.note; variants/grouped/ENVIRONMENT_WITNESS.json.worker_import_evidence; ENVIRONMENT_WITNESS.md:20,26; EXPERIMENT_AUDIT.md:81`。

### W7 WARN — 质量与Strict Prefix优势仍未建立

两group arm均117/120，Full/Prefix118/120，质量点估计损失0.833333pp；on P95为Prefix的2.494491倍。无holdout/正式置信质量门槛、真实Agent轨迹，Research Contract B及生产Go继续unsupported。

证据：`variants/grouped/PAIRED_REPORT.md:7–12,18; EXPERIMENT_AUDIT.md:53,68–72`。

## 对原审计缺项的影响

|原审计缺项/边界|本次新增证据|本次可关闭的有限缺项|继续未完成|
|---|---|---|---|
|原审计明确long gate未纳入，grouped仅short/primitive|20个calibration input ×4 arm的long follows-doc见证|这个long子集的执行、Query排除、COW来源、36层和grouped region|独立holdout/真实GT/复杂更新冲突/生产质量|
|原配置固定不同GPU，无公平性能复测|GPU0上4 arm顺序跑120请求|这4组间固定GPU身份差异|顺序平衡、重复/seed/负载控制、正式质量约束服务P95|
|原phase只有支架，未接入live path|8条真实路径host钩子、8 trace与CUDA launch scope归因|有限profiling reachability及事件/字节关联|lookup CUDA覆盖、普通服务计时、Strict Prefix同轨迹、GPU-ms/KV-only/HBM/first-decode|

原`EXPERIMENT_AUDIT.md:43–53,68–81`的WARN和A/B/C边界继续有效。新证据不会将原整个结果集、或历史运行的缺项倒推为已完成。原B“质量约束净P95改善≥10%”继续unsupported；本次同GPUon仍比Prefix慢2.49449倍。原文继续保留，增量范围由本报告与JSON单独记录。

## 修正建议与未发现的问题

未发现需要改raw或改表格数字的不一致；ERROR=0。建议仅补足三处容易误读的口径：

1. `variants/grouped/ENVIRONMENT_WITNESS.md:24`可明写20个input来自原calibration子集，80是跨arm记录数，避免将独立执行见证误读为holdout。
2. `variants/grouped/PAIRED_REPORT.md:16`可明写mandatory-L1均值只对有repair telemetry的暖请求取平均；CSV中requests=5并不等于该均值分母。
3. `variants/instrumented/PHASE_SMOKE_REPORT.md:40`可将lookup注明仅host证据；`:22`可补末暖请求也有38 host store span，虽然无追加store CUDA copy。

这些是透明度建议，不构成数值错误。本审计没有修改上述文件。报告目前已明确不能把诊断流量称KV-only或把kernel duration sum称服务GPU-ms，也已明确不能作正式P95/质量或生产Go。

## 原审计保留证明

- `EXPERIMENT_AUDIT.md`：`sha256:9c626d348ea8ddf6ebc1891be8ebd807473e8c20c8a3eee3bd9e52f3699a562b`，审计前后相同。
- `EXPERIMENT_AUDIT.json`：`sha256:986a969b1dd2b040151d69fd60b1e69d320793b9608dd50542da5d551d103f6f`，审计前后相同。

机器可读详项见同级 `EXPERIMENT_AUDIT_ADDENDUM_20261010.json`；68项读取文件SHA及逐请求复算详项保留在forensic目录。文件一致性不能提供硬件执行加密证明；语义判断仍为同家族provisional。
