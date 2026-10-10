# 独立服务回放实验完整性审计

结论：**PASS_WITH_WARNINGS**。确定性检查 6386/6386 通过；语义审计 same-family/provisional。

审计员：独立 gpt-6.1-sol 代理；仅 CPU 读取、逐行核算和 SHA 核验。未导入或启动 GPU/推理后端，未改写原始 JSONL、日志、manifest、统计或报告。

A 目标来源 PASS：固定合成数据的独立事实标签，与查询指向的原始记忆事实逐行相符；不是模型自参照代理。B 归一化 PASS：正确数/120 与整条 token 序列一致请求数/120。C 文件/数字按全部确定性检查结果判定。D 指标调用路径 PASS（静态检查）：run_replay.py:113 调用 score；analyze_service_phase.py 在 summary 构造中调用 quantiles 和 paired_bootstrap_loss，未重跑导出器。E 范围 WARN：单 GPU/seed、固定次序校准。F 类型为 real_gt（独立合成事实标签），实际 GPU 回放由保存日志支持，审计仅 CPU。

## 独立复算

| Arm | 正确率 | TTFT P50 / P95 / P99 秒 | 含 RPC P95 秒 | 与 Full token 序列一致 | Repair requests / tokens |
|---|---:|---:|---:|---:|---:|
| full | 118/120 (98.333333%) | 0.680173159 / 2.618166256 / 2.639060607 | 2.618167589 | 100.000% | 0 / 0 |
| strict-prefix | 118/120 (98.333333%) | 0.281945586 / 1.019210529 / 2.622990782 | 1.019211445 | 100.000% | 0 / 0 |
| blend-window10 | 117/120 (97.500000%) | 0.381408811 / 2.544759643 / 2.679618649 | 2.545259961 | 97.500% | 86 / 37458 |

Blend 相对 Full 质量损失 0.833333333 pp；24 个等长 session 的 10,000 次配对重采样单侧 95% 上界 2.500000000 pp（独立 stdlib RNG 上界 2.500000000 pp）。
Blend 含 RPC P95 相对 Strict 的 improvement = (1 − Blend/Strict)×100 = -149.728353606%；即慢 149.728353606%。质量上界 >1pp 且 P95 变慢，联合门槛不满足。

三臂各 120 原始行/120 phase map，逐行核验输入顺序、request_id/input_hash/session/target/query_span、map 索引、事件身份/时钟/资格位；独立从文本精确匹配目标复算 correct。输入 hash 从 token IDs 重算，目标还与原始逻辑轨迹查询所指向的四位金额事实交叉核验，未从 Full 输出推导。

Full/Strict 输出 token 序列逐请求完全一致；Blend 与 Full 117/120 一致。该值是“整条输出 token 序列相等的请求比例”，不是逐 token micro accuracy。两份 analysis/analysis-final 的所有 summary 标量、分位数、stage 汇总、360 行逐请求 CSV 所有单元格、18 行 stage CSV、最终 7 条质量错误 JSONL 均核验。

Blend mandatory=131471、reusable=374924、engine skip=506395；86 个修复请求的 36 层索引、repair position 数量/预算、已加载 span 与 reusable/mandatory 划分、Query 排除、pristine generation=0 和会话内先前 producer 身份均交叉核验。Repair tokens 为请求级选择位置之和，不能当作跨层 GPU 工作量。

## 历史未插桩校准回放对照

| Arm | 历史 / 当前正确数 | 跨回放 token 序列一致 | 历史含 RPC P95 秒 | 当前含 RPC P95 秒 | 当前观测 P95 增幅 |
|---|---:|---:|---:|---:|---:|
| full | 118 / 118 | 100.000% | 2.618879497 | 2.618167589 | -0.027% |
| strict-prefix | 118 / 118 | 100.000% | 1.020544907 | 1.019211445 | -0.131% |
| blend-window10 | 117 / 117 | 100.000% | 2.545739806 | 2.545259961 | -0.019% |

上述配对已确认同一 120 输入、相同模型路径/GPU0/seed17/输出上限32/rope cache/policy/ratio。历史 Full/Strict 的 group_contiguous_queries 字段为 False，当前为 True；Blend 两次均 True。原生臂未走 Blend selective-recompute 路径，但不能声称所有配置元数据完全相同。没有明显 P95 增幅（均略微下降，绝对变化 <0.14%）。差异是两次历史运行的观测时延，不能因果归于插桩：运行时间、引擎进程和环境可能不同，且没有随机化或 A/B 去插桩重复。两次都仍是 calibration；相对 Strict 的本次内部时延比较也只描述已插桩条件。

## SHA、复现与证据边界

RUN_MANIFEST 的输入、源码、配置、raw/map/log 和分析脚本 SHA；代码 review 的 22 个源码 SHA；权重 lock 和其中实际模型文件 SHA 均核验（若命令使用 --skip-weight-files，则仅核验 lock 并明确警告）。审核前后所有已审计文件 SHA 保持一致。详见 JSON 的 audited_input_hashes 与逐项 checks。

CODE_REVIEW_HOST_ONLY 的 PASS_WITH_WARNINGS/0 必修项与 manifest 一致；W1–W4 保留。真实回放文件和日志补充 W4 中“尚未运行实机”的历史证据，但本审计没有重新验证库版本或 GPU 兼容性。GPU_RELEASE_AFTER_RUN 的 GPU0–3 零显存/零利用率文本与 manifest 一致；这是保存的快照，缺少采集时间/命令及进程清单，不证明目前设备状态。Blend teardown 的强杀警告如实保留。

报告明确说明整次 generate、多 token/decode、嵌套 host elapsed、无 GPU-ms/TTFT 分解，未越界宣称 holdout/生产 Go。bootstrap 上界对固定校准 session 的重采样有意义，不能升级为正式验收。

## 局限和警告

- Older analysis/HOST_PHASE_REPORT.md omits the fixed arm-order and numeric quality-upper-bound narrative; analysis-final is the manifest-designated final report and includes both. Old summary/CSV numbers match final.
- Historical full raw has group_contiguous_queries=False; current=True. Native arms do not take the Blend selective-recompute branch, but strict all-settings equivalence is not established.
- Historical strict-prefix raw has group_contiguous_queries=False; current=True. Native arms do not take the Blend selective-recompute branch, but strict all-settings equivalence is not established.
- Synthetic fact-retrieval calibration (24 sessions × 5 requests), not real workload, holdout, or autonomous agent rollout.
- Single seed, fixed Full→Strict Prefix→Blend sequence, one run/arm, concurrency=1; no order balancing or independent replication.
- Host spans include decode and are inclusive/nested; no GPU-ms, transfer bytes, HBM peak or complete TTFT decomposition.
- Same-family fresh-agent semantic review is provisional; deterministic numeric checks do not certify evidence provenance against tampering.
- Release snapshot text matches manifest but contains no acquisition timestamp/command/PID listing; no GPU query was performed during audit.
- Blend log force-kills remaining EngineCore after processing/teardown. Complete 120-row evidence and saved empty-GPU snapshot exist; snapshot cannot retrospectively certify graceful teardown.
- Paired session percentile bootstrap is descriptive for this calibration; upper bound is not a validated production noninferiority guarantee.
- Previous grouped versus current host-instrumented run is a nonrandomized historical comparison; observed timing differences cannot isolate instrumentation overhead from runtime/environment drift.
- Model lock resolved_revision=master is mutable; actual local locked file hashes verify bytes, not an immutable remote revision.
- Reproduction shell/script and review are statically inspected; no replay, GPU action or original analysis exporter was rerun.

## CPU 复现

```bash
PYTHONDONTWRITEBYTECODE=1 /home/bumi/infra/runtime/agent-memory-kv-p25-20261009/bin/python /home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009/variants/service-phase/audit_service_phase_cpu.py
```

仅覆盖本审计脚本与上述两份审计输出；复现会更新本审计输出，不执行真实服务回放。
