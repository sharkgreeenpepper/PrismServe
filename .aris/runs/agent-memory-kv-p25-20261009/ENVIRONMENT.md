# P2.5 环境复核

范围：独立运行时、Qwen3-8B BF16 TP=1、GPU0–3。以下是机制检查，不是正式质量或时延验收。不要改动 Bumi_Mem、既有运行时或 GPU4–7。模型来自 ModelScope，实际文件 SHA256 在 configs/*weights.lock.json。

从项目根目录执行；每条 GPU 命令必须明确 CUDA_VISIBLE_DEVICES。复核文件使用 witness-* 名称，拒绝覆盖已有结果。supervise.py 只清理其子进程组。

```bash
cd /home/bumi/git/PrismServe
source .aris/runs/agent-memory-kv-p25-20261009/activate.sh
python -m pip check
python -m unittest discover -s experiments/agent_memory_kv -p test_contracts.py
CUDA_VISIBLE_DEVICES=0 python experiments/nonprefix_kv/witness_packed.py
CUDA_VISIBLE_DEVICES=0 python experiments/nonprefix_kv/witness_causal.py
```

复核只用 GPU0，在显存空闲后顺序运行，不能把这些耗时与其他阶段作性能比较。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/agent_memory_kv/supervise.py python experiments/agent_memory_kv/run_replay.py --model /home/bumi/infra/models/agent-memory-kv-p25-20261009/Qwen3-8B --inputs .aris/runs/agent-memory-kv-p25-20261009/data/controlled-renderer-smoke-inputs/paired-inputs.jsonl --output .aris/runs/agent-memory-kv-p25-20261009/results/witness-full.jsonl --gpu 0 --arm full --max-model-len 8192
CUDA_VISIBLE_DEVICES=0 python experiments/agent_memory_kv/supervise.py python experiments/agent_memory_kv/run_replay.py --model /home/bumi/infra/models/agent-memory-kv-p25-20261009/Qwen3-8B --inputs .aris/runs/agent-memory-kv-p25-20261009/data/controlled-renderer-smoke-inputs/paired-inputs.jsonl --output .aris/runs/agent-memory-kv-p25-20261009/results/witness-prefix.jsonl --gpu 0 --arm strict_prefix --max-model-len 8192
CUDA_VISIBLE_DEVICES=0 python experiments/agent_memory_kv/supervise.py python experiments/agent_memory_kv/run_replay.py --model /home/bumi/infra/models/agent-memory-kv-p25-20261009/Qwen3-8B --inputs .aris/runs/agent-memory-kv-p25-20261009/data/controlled-renderer-smoke-inputs/paired-inputs.jsonl --output .aris/runs/agent-memory-kv-p25-20261009/results/witness-blend100.jsonl --gpu 0 --arm blend --ratio 1 --lmcache-config .aris/runs/agent-memory-kv-p25-20261009/configs/blend.yaml --max-model-len 8192
CUDA_VISIBLE_DEVICES=0 python experiments/agent_memory_kv/supervise.py python experiments/agent_memory_kv/run_replay.py --model /home/bumi/infra/models/agent-memory-kv-p25-20261009/Qwen3-8B --inputs .aris/runs/agent-memory-kv-p25-20261009/data/controlled-renderer-smoke-inputs/paired-inputs.jsonl --output .aris/runs/agent-memory-kv-p25-20261009/results/witness-blend05.jsonl --gpu 0 --arm blend --ratio .05 --lmcache-config .aris/runs/agent-memory-kv-p25-20261009/configs/blend.yaml --max-model-len 8192
CUDA_VISIBLE_DEVICES=0 python experiments/agent_memory_kv/supervise.py python experiments/agent_memory_kv/run_replay.py --model /home/bumi/infra/models/agent-memory-kv-p25-20261009/Qwen3-8B --inputs .aris/runs/agent-memory-kv-p25-20261009/data/controlled-renderer-smoke-inputs/paired-inputs.jsonl --output .aris/runs/agent-memory-kv-p25-20261009/results/witness-cow05.jsonl --gpu 0 --arm blend_cow --ratio .05 --lmcache-config .aris/runs/agent-memory-kv-p25-20261009/configs/blend.yaml --max-model-len 8192
```

检查结果：每组20条，无错误；100% repair 输出 token 与 Native Full 相同；部分重算有36层记录且高层 QKV 行数小于完整 span；loaded_chunks 不与当前 query_span 重叠；原始模式只加载 generation0；COW 模式暖请求出现 generation>0；两个会话首请求在清空后不修复。正确率逐组如实记录，即使低比例错误也不能删样本。清理后 nvidia-smi 验证 GPU0 释放。独立代理把命令、退出码、逐项检查和限制写入 logs/environment-witness.md。

本数据由真实 Bumi renderer 的纯函数和受控事实生成，origin=synthetic/split=profile，不含真实生产 Mem0 检索。因此通过这里也不证明生产负载质量、10% P95 收益或 Hybrid 兼容性。GPU-ms/传输bytes尚未采集，维持 null。
