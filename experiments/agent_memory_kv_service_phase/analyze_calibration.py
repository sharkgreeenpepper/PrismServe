"""Descriptive paired calibration statistics; never a held-out Go verdict."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(r['status'] != 'ok' for r in rows):raise ValueError('Incomplete/error input')
    return {r['request_id']: r for r in rows}


def bootstrap_loss(full, trial):
    sessions = sorted({r['session_id'] for r in full.values()})
    differences = np.array([np.mean([int(full[k]['correct']) - int(trial[k]['correct'])
                                    for k in full if full[k]['session_id'] == session])
                            for session in sessions])
    rng = np.random.default_rng(517)
    indices = rng.integers(0, len(sessions), size=(10000, len(sessions)))
    return float(np.quantile(differences[indices].mean(axis=1), .95) * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.root / 'queue_state.json').read_text())
    manifest = json.loads((args.root / 'manifest.json').read_text())
    full = load(args.root / 'results/full-topk-05.jsonl')
    prefix = load(args.root / 'results/strict_prefix-topk-05.jsonl')
    source = {r['request_id']: r for r in map(json.loads, Path(manifest['inputs']).read_text().splitlines())}
    assert full.keys() == prefix.keys() == source.keys()
    baseline_p95 = np.quantile([r['ttft_with_context_s'] for r in prefix.values()], .95)
    summary = [];errors = []
    for job in manifest['jobs']:
        if state['jobs'][job['id']]['status'] != 'completed':continue
        rows = load(Path(job['output']))
        if rows.keys() != full.keys():raise ValueError('Request set mismatch')
        for key, row in rows.items():
            if row['input_hash'] != full[key]['input_hash'] or row['target'] != full[key]['target']:
                raise ValueError('Paired input/target mismatch')
        p95 = float(np.quantile([r['ttft_with_context_s'] for r in rows.values()], .95))
        telemetry = [r.get('telemetry') or {} for r in rows.values()]
        sample = next(iter(rows.values()))
        summary.append({'configuration': job['id'], 'arm': sample['arm'], 'policy': sample['policy'],
            'nominal_ratio': sample['ratio'], 'gpu': job['gpu'], 'requests': len(rows),
            'sessions': len({r['session_id'] for r in rows.values()}),
            'accuracy_percent': np.mean([r['correct'] for r in rows.values()]) * 100,
            'quality_loss_pp': np.mean([int(full[k]['correct']) - int(rows[k]['correct']) for k in full]) * 100,
            'paired_bootstrap_loss_upper95_pp': bootstrap_loss(full, rows),
            'token_agreement_full_percent': np.mean([rows[k]['output_token_ids'] == full[k]['output_token_ids'] for k in full]) * 100,
            'ttft_p50_s': float(np.quantile([r['ttft_with_context_s'] for r in rows.values()], .5)),
            'ttft_p95_s': p95, 'ttft_p99_s': float(np.quantile([r['ttft_with_context_s'] for r in rows.values()], .99)),
            'p95_improvement_prefix_percent': (1 - p95 / baseline_p95) * 100,
            'actual_repair_requests': sum(bool(t.get('repair_invocations')) for t in telemetry),
            'selected_repair_tokens': sum(t.get('repair_tokens', 0) for t in telemetry),
            'mandatory_blend_tokens': sum(t.get('mandatory_prefix_tokens', 0) for t in telemetry),
            'reusable_blend_tokens': sum(t.get('reusable_tokens', 0) for t in telemetry),
            'GPU_ms_measured': None, 'transfer_bytes_measured': None, 'peak_HBM_measured': None,
            'scope': 'synthetic calibration only; not a held-out gate'})
        for key, r in rows.items():
            if not r['correct']:
                errors.append({'configuration': job['id'], 'request_id': key,
                               'full_correct': full[key]['correct'], 'text': r['text'],
                               'expected': r['target']['value'], 'finish_reason': r['finish_reason']})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'summary.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (args.output / 'errors.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in errors))
    (args.output / 'scope.json').write_text(json.dumps({'completed_configs':len(summary), 'total_configs':len(manifest['jobs']),
        'calibration_only': True, 'held_out_quality_gate': 'not evaluated', 'production_Go': False,
        'bootstrap': 'empirical paired session resampling; zero observed differences do not establish population equivalence',
        'performance': 'descriptive sequential single-GPU calibration; formal balanced controls still required'},indent=2))
    print(json.dumps({'completed_configurations': len(summary), 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
