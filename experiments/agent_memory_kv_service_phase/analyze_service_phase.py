"""Validate and summarize paired host-only service runs and event mappings."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARMS = ('full', 'strict-prefix', 'blend-window10')
STAGE_NAMES = ('kv_lookup', 'kv_transfer', 'kv_store', 'selective_recompute',
               'native_model_forward', 'first_token_sampling')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {'p50': float(np.quantile(values, .50)),
            'p95': float(np.quantile(values, .95)),
            'p99': float(np.quantile(values, .99))}


def paired_bootstrap_loss(full, trial):
    sessions = sorted({r['session_id'] for r in full.values()})
    per_session = np.asarray([
        np.mean([int(full[k]['correct']) - int(trial[k]['correct'])
                 for k in full if full[k]['session_id'] == session])
        for session in sessions], dtype=np.float64)
    rng = np.random.default_rng(517)
    indices = rng.integers(0, len(sessions), size=(10000, len(sessions)))
    distribution = per_session[indices].mean(axis=1) * 100
    return {'session_count': len(sessions), 'loss_pp': float(per_session.mean() * 100),
            'one_sided_upper95_pp': float(np.quantile(distribution, .95)),
            'bootstrap': '10,000 paired session resamples; descriptive calibration only'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs_path = Path('/home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009/data/controlled-calibration-120-inputs/paired-inputs.jsonl')
    source = read_jsonl(inputs_path)
    if len(source) != 120 or len({r['request_id'] for r in source}) != 120:
        raise ValueError('Expected the frozen 120-request paired input set')
    expected = {r['request_id']: r for r in source}

    rows_by_arm = {}
    maps_by_arm = {}
    for arm in ARMS:
        raw = read_jsonl(args.root / f'{arm}.jsonl')
        phase_map = read_jsonl(args.root / f'{arm}-events/phase-map.jsonl')
        if len(raw) != 120 or len(phase_map) != 120:
            raise ValueError(f'{arm}: incomplete records or event map')
        if any(r.get('status') != 'ok' for r in raw):
            raise ValueError(f'{arm}: request errors present')
        if any(m.get('status') != 'ok' or m.get('mode') != 'host-only' for m in phase_map):
            raise ValueError(f'{arm}: event map errors or wrong instrumentation mode')
        raw_ids = [r['request_id'] for r in raw]
        map_ids = [m['request_id'] for m in phase_map]
        source_ids = [r['request_id'] for r in source]
        if raw_ids != source_ids or map_ids != source_ids:
            raise ValueError(f'{arm}: request order or identity mismatch')
        for idx, (record, mapped, original) in enumerate(zip(raw, phase_map, source)):
            if record['input_hash'] != original['input_hash'] or mapped['input_hash'] != original['input_hash']:
                raise ValueError(f'{arm}: input hash mismatch at {idx}')
            if record['target'] != original['target'] or mapped['request_index'] != idx:
                raise ValueError(f'{arm}: target or request index mismatch at {idx}')
            if not mapped['phase_host_events']:
                raise ValueError(f'{arm}: no host phase events for {record["request_id"]}')
            if not record.get('timing_eligible') or not record.get('task_accuracy_eligible'):
                raise ValueError(f'{arm}: raw request marked ineligible')
        rows_by_arm[arm] = {r['request_id']: r for r in raw}
        maps_by_arm[arm] = {m['request_id']: m for m in phase_map}

    full = rows_by_arm['full']
    summary = {}
    for arm in ARMS:
        rows = rows_by_arm[arm]
        correct = [bool(rows[k]['correct']) for k in expected]
        stage_values = {name: [] for name in STAGE_NAMES}
        stage_event_counts = {name: 0 for name in STAGE_NAMES}
        stage_request_counts = {name: 0 for name in STAGE_NAMES}
        for key in expected:
            events = maps_by_arm[arm][key]['phase_host_events']
            by_name = {}
            for event in events:
                by_name.setdefault(event['name'], []).append((event['end_ns'] - event['start_ns']) / 1e6)
            for name in STAGE_NAMES:
                values = by_name.get(name, [])
                if values:
                    stage_values[name].append(sum(values))
                    stage_event_counts[name] += len(values)
                    stage_request_counts[name] += 1
        summary[arm] = {
            'requests': len(rows), 'accuracy_percent': float(np.mean(correct) * 100),
            'correct': int(sum(correct)), 'incorrect': int(len(correct) - sum(correct)),
            'output_token_agreement_full_percent': float(np.mean([
                rows[k]['output_token_ids'] == full[k]['output_token_ids'] for k in expected]) * 100),
            'ttft_s': quantiles([rows[k]['ttft_s'] for k in expected]),
            'ttft_with_context_s': quantiles([rows[k]['ttft_with_context_s'] for k in expected]),
            'e2e_s': quantiles([rows[k]['e2e_s'] for k in expected]),
            'tokens_skipped_by_engine_prefix': int(sum(rows[k].get('engine_skip_prefill_span_tokens') or 0 for k in expected)),
            'blend_repair_requests': int(sum(bool((rows[k].get('telemetry') or {}).get('repair_invocations')) for k in expected)),
            'repair_tokens': int(sum((rows[k].get('telemetry') or {}).get('repair_tokens', 0) for k in expected)),
            'mandatory_tokens': int(sum((rows[k].get('telemetry') or {}).get('mandatory_prefix_tokens', 0) for k in expected)),
            'reusable_tokens': int(sum((rows[k].get('telemetry') or {}).get('reusable_tokens', 0) for k in expected)),
            'host_stages': {name: {
                'requests_covered': stage_request_counts[name], 'event_count': stage_event_counts[name],
                'inclusive_event_sum_ms_per_request': quantiles(stage_values[name]) if stage_values[name] else None}
                for name in STAGE_NAMES},
            'GPU_ms_measured': None, 'transfer_bytes_measured': None, 'peak_HBM_measured': None}

    baseline_p95 = summary['strict-prefix']['ttft_with_context_s']['p95']
    for arm in ARMS:
        summary[arm]['p95_ttft_with_context_improvement_vs_strict_prefix_percent'] = (
            (1 - summary[arm]['ttft_with_context_s']['p95'] / baseline_p95) * 100)
        summary[arm]['quality_loss_vs_full_pp'] = (
            float(np.mean([int(full[k]['correct']) - int(rows_by_arm[arm][k]['correct']) for k in expected]) * 100))
        summary[arm]['paired_session_bootstrap'] = paired_bootstrap_loss(full, rows_by_arm[arm])

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'quality-errors.jsonl').open('w') as handle:
        for arm in ARMS:
            for key in expected:
                row = rows_by_arm[arm][key]
                if not row['correct']:
                    handle.write(json.dumps({'arm': arm, 'request_id': key,
                        'session_id': row['session_id'], 'input_hash': row['input_hash'],
                        'full_correct': full[key]['correct'], 'trial_correct': row['correct'],
                        'output_agrees_with_full': row['output_token_ids'] == full[key]['output_token_ids'],
                        'target': row['target'], 'output_text': row['text'],
                        'output_token_ids': row['output_token_ids']}, ensure_ascii=False) + '\n')
    per_request_path = args.output / 'paired-request-metrics.csv'
    with per_request_path.open('w', newline='') as handle:
        fields = ['request_id', 'session_id', 'input_hash', 'arm', 'correct', 'output_agrees_with_full',
                  'ttft_s', 'context_rpc_s', 'ttft_with_context_s', 'e2e_s',
                  'engine_skip_prefill_span_tokens', 'repair_invocations', 'repair_tokens',
                  'mandatory_prefix_tokens', 'reusable_tokens']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            for key in expected:
                row = rows_by_arm[arm][key]
                telemetry = row.get('telemetry') or {}
                writer.writerow({'request_id': key, 'session_id': row['session_id'], 'input_hash': row['input_hash'],
                    'arm': arm, 'correct': row['correct'],
                    'output_agrees_with_full': row['output_token_ids'] == full[key]['output_token_ids'],
                    'ttft_s': row['ttft_s'], 'context_rpc_s': row['context_rpc_s'],
                    'ttft_with_context_s': row['ttft_with_context_s'], 'e2e_s': row['e2e_s'],
                    'engine_skip_prefill_span_tokens': row.get('engine_skip_prefill_span_tokens'),
                    'repair_invocations': telemetry.get('repair_invocations', 0),
                    'repair_tokens': telemetry.get('repair_tokens', 0),
                    'mandatory_prefix_tokens': telemetry.get('mandatory_prefix_tokens', 0),
                    'reusable_tokens': telemetry.get('reusable_tokens', 0)})

    event_path = args.output / 'host-phase-summary.csv'
    with event_path.open('w', newline='') as handle:
        fields = ['arm', 'stage', 'requests_covered', 'event_count', 'p50_inclusive_sum_ms',
                  'p95_inclusive_sum_ms', 'p99_inclusive_sum_ms', 'semantics']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            for stage, metrics in summary[arm]['host_stages'].items():
                q = metrics['inclusive_event_sum_ms_per_request'] or {}
                writer.writerow({'arm': arm, 'stage': stage, 'requests_covered': metrics['requests_covered'],
                    'event_count': metrics['event_count'], 'p50_inclusive_sum_ms': q.get('p50'),
                    'p95_inclusive_sum_ms': q.get('p95'), 'p99_inclusive_sum_ms': q.get('p99'),
                    'semantics': 'host elapsed spans; inclusive/possibly nested; not GPU-ms or TTFT decomposition'})

    lines = ['# Host-only 配对服务回放报告', '',
        '本报告覆盖固定校准轨迹，不是 holdout。每臂 120 请求、相同输入顺序、单并发、物理 GPU0、Qwen3-8B BF16 TP=1；每个会话切换时清空缓存。臂按 Full→Strict Prefix→Blend 固定顺序执行，未做顺序平衡。主机阶段插桩覆盖整次多 token generate，可能有测量开销。',
        '阶段事件是 host elapsed，区间可能嵌套；不可相加为 TTFT，也不代表 GPU-ms。CUDA profiler、全词表 logprobs、搬运字节和峰值显存均未在这组计时中测量。', '',
        '| Arm | 正确率 | TTFT P50/P95/P99 (s) | 含上下文 RPC 的 TTFT P95 (s) | 相对 Strict P95 | 与 Full 输出一致 | Repair tokens |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s = summary[arm]
        t = s['ttft_s']; c = s['ttft_with_context_s']
        lines.append(f"| {arm} | {s['correct']}/120 ({s['accuracy_percent']:.2f}%) | {t['p50']:.4f}/{t['p95']:.4f}/{t['p99']:.4f} | {c['p95']:.4f} | {s['p95_ttft_with_context_improvement_vs_strict_prefix_percent']:.2f}% | {s['output_token_agreement_full_percent']:.2f}% | {s['repair_tokens']} |")
    blend = summary['blend-window10']
    lines.extend(['',
        f"按观测均值，Blend 相对 Full 的质量损失为 {blend['quality_loss_vs_full_pp']:.3f} 个百分点；按会话配对 bootstrap 的单侧 95% 上界为 {blend['paired_session_bootstrap']['one_sided_upper95_pp']:.3f} 个百分点。Blend 含上下文 RPC 的 P95 TTFT 相对 Strict Prefix 变化 {blend['p95_ttft_with_context_improvement_vs_strict_prefix_percent']:.2f}%。",
        '本组没有达到联合门槛：质量置信上界高于 1 个百分点，且 Blend 的 P95 TTFT 慢于 Strict Prefix。结果支持保留严格 Prefix；不支持将当前非前缀修复配置推进生产。',
        '质量损失的按会话配对 bootstrap 95% 单侧上界、失败样本、逐请求指标与阶段事件分布见相邻 CSV/JSON/JSONL。全部结论是单 GPU、固定次序的校准轨迹描述统计；不得据此宣称正式 holdout 质量门通过或生产 Go。'])
    (args.output / 'HOST_PHASE_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'output': str(args.output.resolve()), 'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
