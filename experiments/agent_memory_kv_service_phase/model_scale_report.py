"""Build model-normalized CSV/Markdown for the paired Qwen3-8B/14B probes."""
import argparse
import csv
import json
import re
from pathlib import Path


ARMS = ('full', 'strict-prefix', 'blend-window10')
ROOT_DEFAULT = Path('/home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009')


def load_json(path):
    return json.loads(path.read_text())


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = {
        'Qwen3-8B': args.root / 'variants/service-phase/paired-120-host-only',
        'Qwen3-14B': args.root / 'variants/qwen3-14b-cost-probe/paired-120-host-only',
    }
    summaries = {model: load_json(path / 'analysis-final/summary.json') for model, path in runs.items()}
    raw = {model: {arm: load_jsonl(path / f'{arm}.jsonl') for arm in ARMS}
           for model, path in runs.items()}
    reference_model = 'Qwen3-8B'
    reference_ids = [r['request_id'] for r in raw[reference_model]['full']]
    reference_hashes = [r['input_hash'] for r in raw[reference_model]['full']]
    if len(reference_ids) != 120 or len(set(reference_ids)) != 120:
        raise ValueError('Expected 120 unique paired requests')
    for model in runs:
        for arm in ARMS:
            rows = raw[model][arm]
            if len(rows) != 120 or [r['request_id'] for r in rows] != reference_ids:
                raise ValueError(f'{model}/{arm}: request IDs/order differ')
            if [r['input_hash'] for r in rows] != reference_hashes:
                raise ValueError(f'{model}/{arm}: input hashes differ from paired reference')
            if any(r.get('status') != 'ok' for r in rows):
                raise ValueError(f'{model}/{arm}: request errors present')
    prefix_p95 = {m: summaries[m]['strict-prefix']['ttft_with_context_s']['p95'] for m in runs}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out_csv = args.output_dir / 'model-scale-comparison.csv'
    fields = ['model', 'arm', 'requests', 'correct', 'accuracy_percent',
              'output_token_agreement_full_percent', 'quality_loss_vs_full_pp',
              'quality_loss_upper95_pp', 'ttft_p50_s', 'ttft_p95_s', 'ttft_p99_s',
              'ttft_with_context_p95_s', 'p95_ratio_vs_own_strict_prefix',
              'e2e_p95_s', 'engine_skipped_tokens', 'repair_requests', 'repair_tokens',
              'mandatory_prefix_tokens', 'reusable_tokens', 'gpu_ms_measured',
              'physical_transfer_bytes_measured', 'run_sampled_gpu0_memory_max_mib',
              'host_lookup_p95_ms', 'host_transfer_p95_ms', 'host_store_p95_ms',
              'host_selective_recompute_p95_ms']
    rows_out = []
    peak14 = None
    samples = runs['Qwen3-14B'] / 'gpu0-samples.csv'
    if samples.exists():
        values = []
        for line in samples.read_text().splitlines():
            columns = [c.strip() for c in line.split(',')]
            try:
                values.append(int(columns[2].split()[0]))
            except (IndexError, ValueError):
                continue
        peak14 = max(values) if values else None
    for model in runs:
        summary = summaries[model]
        for arm in ARMS:
            s = summary[arm]
            rows_out.append({
                'model': model, 'arm': arm, 'requests': s['requests'], 'correct': s['correct'],
                'accuracy_percent': s['accuracy_percent'],
                'output_token_agreement_full_percent': s['output_token_agreement_full_percent'],
                'quality_loss_vs_full_pp': s['quality_loss_vs_full_pp'],
                'quality_loss_upper95_pp': s['paired_session_bootstrap']['one_sided_upper95_pp'],
                'ttft_p50_s': s['ttft_s']['p50'], 'ttft_p95_s': s['ttft_s']['p95'],
                'ttft_p99_s': s['ttft_s']['p99'],
                'ttft_with_context_p95_s': s['ttft_with_context_s']['p95'],
                'p95_ratio_vs_own_strict_prefix': s['ttft_with_context_s']['p95'] / prefix_p95[model],
                'e2e_p95_s': s['e2e_s']['p95'],
                'engine_skipped_tokens': s['tokens_skipped_by_engine_prefix'],
                'repair_requests': s['blend_repair_requests'], 'repair_tokens': s['repair_tokens'],
                'mandatory_prefix_tokens': s['mandatory_tokens'], 'reusable_tokens': s['reusable_tokens'],
                'gpu_ms_measured': '', 'physical_transfer_bytes_measured': '',
                'run_sampled_gpu0_memory_max_mib': peak14 if model == 'Qwen3-14B' and arm == 'full' else '',
                'host_lookup_p95_ms': (s['host_stages']['kv_lookup']['inclusive_event_sum_ms_per_request'] or {}).get('p95'),
                'host_transfer_p95_ms': (s['host_stages']['kv_transfer']['inclusive_event_sum_ms_per_request'] or {}).get('p95'),
                'host_store_p95_ms': (s['host_stages']['kv_store']['inclusive_event_sum_ms_per_request'] or {}).get('p95'),
                'host_selective_recompute_p95_ms': (s['host_stages']['selective_recompute']['inclusive_event_sum_ms_per_request'] or {}).get('p95'),
            })
    with out_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows_out)

    lines = [
        '# Qwen3 模型规模敏感性：Host-only KV Cache 回放', '',
        '数据来源是同一组 120 条独立合成事实校准轨迹（24 sessions × 5 requests，最长 prompt 18,702 tokens），不是生产 L1/L3 轨迹或 holdout。两个模型使用相同 tokenizer 文件、请求 token IDs、答案标签、生成设置和单 GPU0/TP=1/BF16。每个模型只与自身 Strict Prefix 基线比较，以下绝对时延不用于跨模型速度排名。', '',
        '| 模型 | Arm | 正确率 | 含上下文 RPC 的 TTFT P50/P95/P99 (s) | P95 / 本模型 Strict Prefix | 与本模型 Full token 输出一致 | 跳过 prefill tokens | Repair tokens |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    label = {'full': 'Full', 'strict-prefix': 'Strict Prefix', 'blend-window10': 'Blend window 10%'}
    for model in runs:
        for arm in ARMS:
            s = summaries[model][arm]
            t = s['ttft_with_context_s']
            lines.append(f"| {model} | {label[arm]} | {s['correct']}/{s['requests']} ({s['accuracy_percent']:.2f}%) | {t['p50']:.3f}/{t['p95']:.3f}/{t['p99']:.3f} | {t['p95']/prefix_p95[model]:.3f}× | {s['output_token_agreement_full_percent']:.2f}% | {s['tokens_skipped_by_engine_prefix']} | {s['repair_tokens']} |")
    lines.extend(['', '## 观察', ''])
    for model in runs:
        s = summaries[model]
        strict = s['strict-prefix']['ttft_with_context_s']['p95']
        blend = s['blend-window10']
        ratio = blend['ttft_with_context_s']['p95'] / strict
        lines.append(f"- {model}：Strict Prefix P95 为 {strict:.3f}s；Blend P95 为 {blend['ttft_with_context_s']['p95']:.3f}s（{ratio:.3f}× Strict Prefix，变化 {blend['p95_ttft_with_context_improvement_vs_strict_prefix_percent']:.2f}%）。Blend 质量损失均值 {blend['quality_loss_vs_full_pp']:.3f}pp，会话 bootstrap 单侧 95% 上界 {blend['paired_session_bootstrap']['one_sided_upper95_pp']:.3f}pp。")
    blend14 = summaries['Qwen3-14B']['blend-window10']
    stages = blend14['host_stages']
    lines.extend([
        '',
        '### Blend 阶段 P95（host elapsed，描述性）',
        '',
        '| 模型 | Lookup (ms) | Transfer (ms) | Store (ms) | Selective recompute (ms) |',
        '|---|---:|---:|---:|---:|',
    ])
    for model in runs:
        stage = summaries[model]['blend-window10']['host_stages']
        vals = [stage[k]['inclusive_event_sum_ms_per_request']['p95']
                for k in ('kv_lookup', 'kv_transfer', 'kv_store', 'selective_recompute')]
        lines.append(f'| {model} | ' + ' | '.join(f'{x:.1f}' for x in vals) + ' |')
    log_path = runs['Qwen3-14B'] / 'blend-window10.log'
    log_text = log_path.read_text(errors='replace')
    store_events = re.findall(
        r'Stored (\d+) out of total (\d+) tokens\. size: ([\d.]+) GB, cost ([\d.]+) ms, throughput: ([\d.]+) GB/s',
        log_text)
    logged_store_gb = sum(float(row[2]) for row in store_events)
    logged_store_ms = sum(float(row[3]) for row in store_events)
    lines.extend([
        '',
        f"14B 的 Blend 遥测显示 {blend14['blend_repair_requests']} 个修复请求，选择修复 {blend14['repair_tokens']} tokens；mandatory/reusable 分别 {blend14['mandatory_tokens']}/{blend14['reusable_tokens']}，合计与引擎报告跳过的 {blend14['tokens_skipped_by_engine_prefix']} tokens 对齐。",
        f"14B Blend host-only 阶段观测 P50/P95：lookup {stages['kv_lookup']['inclusive_event_sum_ms_per_request']['p50']:.1f}/{stages['kv_lookup']['inclusive_event_sum_ms_per_request']['p95']:.1f}ms，transfer {stages['kv_transfer']['inclusive_event_sum_ms_per_request']['p50']:.1f}/{stages['kv_transfer']['inclusive_event_sum_ms_per_request']['p95']:.1f}ms，store {stages['kv_store']['inclusive_event_sum_ms_per_request']['p50']:.1f}/{stages['kv_store']['inclusive_event_sum_ms_per_request']['p95']:.1f}ms，selective recompute {stages['selective_recompute']['inclusive_event_sum_ms_per_request']['p50']:.1f}/{stages['selective_recompute']['inclusive_event_sum_ms_per_request']['p95']:.1f}ms。它们是可能嵌套的 host elapsed 区间，覆盖整个 generate 请求，不能相加成 TTFT 或 GPU-ms。",
        f"14B 回放期间 1 秒间隔 nvidia-smi 样本的 GPU0 显存最高为 {peak14} MiB；这是采样最大值，不是硬件精确峰值。GPU-ms 和物理 H2D/RDMA/SSD transfer bytes 未直接测量；报告不以 token 数或软件 payload 推算替代。",
        f"LMCache 日志（包括两个预热请求，不是纯计时子集）记录 {len(store_events)} 个 Store 通知，连接器报告 payload size 合计 {logged_store_gb:.3f} GB、cost 合计 {logged_store_ms/1000:.3f}s。这是软件层日志值，不能解释为总线实测传输字节或独立 timed-request 开销。",
        '',
        '## 结论范围',
        '',
        '本次 14B calibration 中 Full、Strict Prefix、Blend 均为 120/120 正确，Blend 与 Full 输出 token 序列逐请求一致；这只描述这组固定合成样本。Blend 在两个模型上的 P95 都显著高于 Strict Prefix，未满足 P2.5 的 ≥10% P95 改善目标。当前证据支持优先保留严格 Prefix；不支持因为模型变大就认为非前缀修复获得净 TTFT 收益。由于缺少生产记忆长度/重复率分布、独立 holdout、随机化运行顺序和物理传输/GPU-ms测量，不能给出生产 Go/No-Go 或普适模型规模规律。',
        '',
        '详细逐请求结果、阶段 CSV、质量错误文件和 JSON summary 位于两个 run 的 analysis-final 子目录；原始请求与映射 JSONL 保留在各自 run 根目录。',
    ])
    (args.output_dir / 'MODEL_SCALE_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'csv': str(out_csv.resolve()), 'report': str((args.output_dir / 'MODEL_SCALE_REPORT.md').resolve()),
                      'models': list(runs), 'rows': len(rows_out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
