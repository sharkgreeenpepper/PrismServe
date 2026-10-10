# 本地非前缀 KV 实验

本工具执行真实 vLLM 单卡推理。P0目录为 `.aris/runs/nonprefix-kv-20261008/`；继续执行的P1/P2目录为 `.aris/runs/nonprefix-kv-p1p2-20261008-2023/`。所有 GPU 子进程只能绑定 0–3。结果使用引擎 TTFT 和同步调用 E2E，不冒充在线 HTTP 负载数据。

## 当前能力

- P0：Full Prefill / 严格 Prefix，8K/16K，冷缓存、连续会话、三个随机历史顺序，物理 GPU 轮换。
- 原始 JSONL、输入哈希、逐请求缓存命中、错误记录、配对会话 bootstrap、历史答案变动率、静态图。
- 24h 截止时间与 96 GPUh 上限；预算耗尽停止本控制器拥有的进程组。
- 原P0环境的P1/P2因现代4-D fused KV布局阻塞，历史失败证据保留；`run_engine.py`仍仅负责P0。
- 新隔离环境的`run_repairs.py`接通公共逐层CacheBlend。`packed_kv.py`处理请求范围I/O，`causal_flash.py`调用既有FA3 paged API维持原始因果位置，不修改Attention Kernel。
- `engine_bridge.py`接入相同layer1偏差评分和同实际预算Top-K/window/document选择；记录28层实际QKV/attention行数、强制缺失token、来源、COW代数。不能只凭引擎skip-prefill span认定缓存命中。
- `run_repair_diagnostics.py`另行采集prompt末端的全词表KL，以及保守长依赖编辑窗口的全重算回退；诊断不混入性能或任务正确率。

## 环境与运行

```bash
cd /home/bumi/git/PrismServe
source .aris/runs/nonprefix-kv-20261008/activate.sh
# 仅用于恢复本次运行，原始截止时间不会延长。
python experiments/nonprefix_kv/control.py --resume --spread-pending --nonprefix-status blocked
python experiments/nonprefix_kv/analyze.py .aris/runs/nonprefix-kv-20261008
PYTHONPATH=experiments/nonprefix_kv python -m unittest discover -s experiments/nonprefix_kv -p 'test_*.py'
```

不要同时运行两个控制器；运行前读取 status.json 中的 controller_pid。`--resume` 仅跳过 1000 条唯一请求、配置匹配且无错误的完整配置，不继承其他策略的缓存。

P1/P2启动命令如下，适用于本次尚未写入结果时的启动；控制器不覆盖已有正式JSONL。原全局截止时间不延长。完整配置在`configs/eval-gpu*.json`，修复比例在评估前的`frozen-selection.json`中固定。

```bash
cd /home/bumi/git/PrismServe
source .aris/runs/nonprefix-kv-p1p2-20261008-2023/activate.sh
python experiments/nonprefix_kv/control_repairs.py .aris/runs/nonprefix-kv-p1p2-20261008-2023
```

只做CPU结果复核（矩阵、诊断及自有GPU进程清理后）：

```bash
source .aris/runs/nonprefix-kv-p1p2-20261008-2023/activate.sh
python experiments/nonprefix_kv/finalize_repairs.py .aris/runs/nonprefix-kv-p1p2-20261008-2023
PYTHONPATH=experiments/nonprefix_kv python -m unittest discover -s experiments/nonprefix_kv -p 'test_*.py'
```

冻结环境的独立GPU复核命令见P1/P2目录的`ENVIRONMENT.md`，只在GPU已空闲且结果文件名未使用时执行。新复现轮次需要新结果目录及用户授权的预算；不要删除原始JSONL来绕过覆盖保护。

P2编辑窗口补充单独存于`extra/edit-window/`，启动入口为`control_edit_windows.py`，使用GPU0/2和同一个原截止时间。`run_edit_windows.py`仅增加首次token变更与保守依赖区间元数据，实际调用原模型和已冻结adapter。10配置×200请求；不要将全跨度fallback称为实际5%修复。其计时排除依赖区间的CPU构造。

```bash
source .aris/runs/nonprefix-kv-p1p2-20261008-2023/activate.sh
python experiments/nonprefix_kv/control_edit_windows.py .aris/runs/nonprefix-kv-p1p2-20261008-2023/extra/edit-window
python experiments/nonprefix_kv/analyze_edit_windows.py .aris/runs/nonprefix-kv-p1p2-20261008-2023/extra/edit-window
```

上面的启动同样拒绝覆盖已完成JSONL；分析命令仅重新计算CPU统计。补充结果及审计不混入主60,000请求门槛。

## 数据与限制

结构化轨迹从独立记录状态生成答案。`audit_data.py` 从实际模型输入再次提取答案，验证长度/hash/document span。校准与评估使用不同种子和命名空间；独立校准完成后冻结统一5%正式预算，禁止用正式评估调参。五类轨迹是机制级工具/RAG测试，非端到端 Agent。历史顺序测试是在每个会话10请求内部打乱，并非无限跨会话常驻缓存。

LongBench-v2 补充只选择完整多文档上下文能放入16K的样本，最多20条；不截断证据补齐数量。选择存在长度偏差，报告单独列出。

logical_exact_reuse_bytes / effective_reused_kv_bytes 是根据真实命中或选择位置计算的逻辑字节；kv_host_load_bytes由实际复制范围乘格式统计，都不是PCIe硬件计数。GPU-ms、硬件传输量、单请求 HBM 未采集时写 null。正式计时中的Logit KL为null，单独诊断位于`diagnostics/`。低输出不代表长输出 Agent 的 TPOT。

原始正式JSONL不修改；`finalize_repairs.py`在`results/normalized/`中附加模型和配置身份，并写完整配对输入、预算、来源、计数及进程释放验收。数据结果CSV、质量—时延图及中文报告都来自真实模型输出，不生成模拟性能。
