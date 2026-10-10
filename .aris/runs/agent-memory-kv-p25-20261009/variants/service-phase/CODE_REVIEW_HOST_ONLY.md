# Host-only 副本独立代码复核

**结论：PASS_WITH_WARNINGS。无未关闭必修项。** 修复后的副本满足本次 GPU 前代码门：host-only 参数正确剥离，禁用 Torch/vLLM profiler 和全词表 logprobs，完整 120 条请求的事件映射、warmup 索引与异常资格标记正确；原有 CUDA 分支和 grouped/instrumented 源目录隔离得到纯 CPU 证据支持。

本评审只读取实现文件；只在本报告目录写评审工具和结果。没有导入真实 Torch/vLLM/LMCache，没有执行 GPU 操作、查询 GPU、启动真实推理或写入原有正式回放结果。CPU 替身事件数不能作为真实性能或 GPU 行为证据。

## 证据与版本

- 实现：`experiments/agent_memory_kv_service_phase/profile_replay.py`、`phase_hooks.py`、`engine_bridge.py`、`run_replay.py`，及与 grouped 对照的全部 Python 源文件。
- 真实输入清单：`.aris/runs/agent-memory-kv-p25-20261009/data/controlled-calibration-120-inputs/paired-inputs.jsonl`，120 行；CPU 替身消费原始 prompt token IDs、request ID、input hash。
- 最终 profile_replay.py SHA-256：`e8d103d83cc5e5ceb5c16e3659c642d569da4356c75b97fa14ce18943f9940e5`。
- 最终 phase_hooks.py SHA-256：`f25d7e09cc3f805dde12b3d7cbe7db03e3cb7bbbb8cd5fd889dfe762543f6839`。
- 可复现工具：[CPU_REVIEW_HOST_ONLY.py](/home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009/variants/service-phase/CPU_REVIEW_HOST_ONLY.py)；详细结果：[CPU_REVIEW_HOST_ONLY_EVIDENCE.json](/home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009/variants/service-phase/CPU_REVIEW_HOST_ONLY_EVIDENCE.json)。该工具让真实 wrapper、真实 run_replay 和真实 phase_hooks 执行，只替换推理后端；host-only 中真实/伪 Torch import 都会立刻失败。CUDA 回归使用假 profiler/record_function，不运行真实 CUDA。

## PASS 检查

| 检查 | 结果与证据 |
|---|---|
| wrapper 参数剥离 | `profile_replay.py:15-23,137-141` 消费 `--mode`、`--event-dir`、`--trace-dir` 和 capture 标志，底层 real run_replay parser 正常接受余下参数；四个 arm 全部完成。CUDA 替身另断言以上三个 wrapper 参数未进入底层 argv，capture 标志以规范拼写回传。 |
| host-only 不启动 profiler | `profile_replay.py:61-65,92-94,107-112` 均以 cuda 模式门控。full、strict_prefix、blend、blend_cow 每 arm 120 条回放的 profiler_config 不存在，start/stop 各为 0，Torch import 为 0。 |
| 不启用全词表 logprobs | `profile_replay.py:22,35-36` 在 engine 启动前拒绝 `--capture-logprobs`、`--capture-logprob`、`--capture-log`、`--capture`、`--cap`；五条测试均 ValueError 且 LLM 构造次数为 0。四 arm 普通回放 SamplingParams.logprobs=None、无 max_logprobs=-1、无 logit_artifact。 |
| 完整 120 条映射 | 四 arm 各 120 条 request records 和 120 条 phase-map，均 status=ok；request_index 恰为 0..119，request_id/input_hash 与输入逐行一致，所有条目非空事件且事件身份匹配本请求。每 arm 的 1200 个事件来自可控 CPU 假后端。 |
| warmup 对齐 | `profile_replay.py:48-53,74-82` 先应用与底层一致的 limit，再以 min(2,len(rows)) 推导 warmup。120 条输入各 122 次 generate、120 次 phase_begin/end。limit=1 与 limit=2 各生成 1/2 条映射，索引从 0 起，无丢失首条。 |
| host-only annotation 关闭 | `phase_hooks.py:23-28,78-80` 在 host_only 时只记录 perf_counter_ns 阶段事件，无 record_function；phase RPC 在 `engine_bridge.py:132-134` 提前分派，未落入非 phase 的 Torch/cache-engine lookup 分支。 |
| host event 语义 | 四 arm 事件均 clock=host_monotonic、measurement=host_elapsed、end_ns>=start_ns；包含 kv_lookup、kv_transfer、kv_store、selective_recompute、native_model_forward、first_token_sampling。worker phase_end 正常时 ACTIVE=None。 |
| 失败映射与原异常 | phase_begin、generate、phase_end、generate+phase_end 四种注入均写 1 条 error mapping 和 1 条底层 error record，再 fail-fast。error mapping 的 timing/task_accuracy_eligible 均 false。begin/generate 失败仍尝试 phase_end；双故障保留原 generate 异常。 |
| CUDA 分支回归 | 成功、begin/generate/end/双故障、start/stop 故障共 7 条 CPU 回归。仍传 profiler_config，成功 start 后各调用 stop 一次，正常 CUDA 开启 prism:* annotation；所有 CUDA mapping 的 timing/task_accuracy_eligible 为 false。start 本身报错时不调用 stop，与 profile_started 状态一致。 |
| 副本隔离 | service_phase 的 run_replay.py、causal_flash.py、packed_kv.py 等共享实现与 grouped 保持字节一致；只有 instrumentation 扩展文件/分支不同。基础、grouped、instrumented、service_phase 四目录的全 Python 文件测试前后哈希完全一致。grouped 历史 source-current.lock 全部匹配；instrumented smoke-source-fingerprints 的 10 项也全部匹配。 |
| 既有合同 | `python3 -m unittest discover -s experiments/agent_memory_kv_service_phase -p test_contracts.py -v`，9/9 PASS；22 个 Python 文件 AST 解析通过。 |

## 已关闭必修项

1. **CLOSED：参数缩写绕过 host-only logprobs 禁令。** 初始实现仅检查精确字符串；中间版只拒绝 `--capture-log*` 仍被 CPU 注入见证 `--capture` 到达 max_logprobs=-1 构造路径。最终通过 wrapper argparse 注册 capture 标志统一识别缩写，五个拼写均提前拒绝。
2. **CLOSED：phase_end 报错仍把结果标为可用。** 最终 `profile_replay.py:122-123` 要求 request_error 与 phase_error 都为空；phase_end 单故障与双故障均得到 status=error、两个资格位 false。
3. **CLOSED：单请求输入固定两次 warmup 导致测量缺失。** 最终从实际 limit 后输入长度推导 warmup；limit=1 与 limit=2 的完整真实底层回放由 CPU 替身验证。

## 非阻塞警告与证据边界

- **W1：RPC 不可达时无法保证 worker ACTIVE 清零。** phase_end 故障在调用 worker control 前注入，ACTIVE 仍为当前 ID；wrapper 记录失败并终止，未继续复用此 engine。下一次正式尝试需新进程/新 engine，不能把此 run 当完成，也不能宣称 crash-safe 清理。该行为不影响正常 120 条序列。
- **W2：多 token 阶段事件覆盖整次 generate。** ACTIVE 在整个 generate 期间保持设置，native_model_forward 和名为 first_token_sampling 的 hook 会覆盖后续 decode 调用。现阶段可报告请求级 host 阶段观测；不能把这些事件总和直接称为 TTFT 或 GPU compute，不含排队、完整调度/IPC/postprocess 的完整分解，worker 与 driver 时钟也未做跨进程校准。
- **W3：插桩本身有开销且说明字符串仍偏向旧诊断用途。** `phase_hooks.py:1,72` 仍称 one-token/profiling-only，host-only 实际允许普通多 token 回放。资格位表示该模式没有 diagnostic logprobs，不证明与未插桩正式时延等价。正式比较需要明确插桩条件。
- **W4：真实 GPU 和安装环境未在本评审验证。** CUDA profiler 真正启停、LMCache/vLLM worker 类型、实机事件覆盖仍须实际执行证据；本报告仅给出代码和 CPU 故障路径的 PASS。

复现命令（在项目根目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 .aris/runs/agent-memory-kv-p25-20261009/variants/service-phase/CPU_REVIEW_HOST_ONLY.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/agent_memory_kv_service_phase -p test_contracts.py -v
```

最终统计：22 个带断言 CPU 场景通过，9 个合同测试通过，未关闭必修项 0。此结论只适用于上述 SHA-256 的最终源码。
