"""Read a redacted export and freeze common token inputs for all four arms."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from contracts import Geometry, assemble, check_split_sessions, exact_repeat_fraction, validate_trace


def prepare(data, tokenizer, config, output):
    rows = [validate_trace(json.loads(line)) for line in data.open() if line.strip()]
    check_split_sessions(rows)
    if len({(r['session_id'], r['request_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate session/request identity')
    geometry = Geometry.from_config(config)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'paired-inputs.jsonl'
    if path.exists():
        raise FileExistsError('Preserve prior manifests; choose a new output directory')
    previous = {};stats = []
    with path.open('w') as f:
        for row in rows:
            prompt = assemble(row, tokenizer)
            session = row['session_id']
            repeat = exact_repeat_fraction(previous.get(session, []), row['l3'], tokenizer)
            previous[session] = row['l3']
            identity = {k: row[k] for k in ('request_id', 'session_id', 'origin', 'split')}
            record = {**identity, **asdict(prompt), 'target': row.get('target'),
                      'has_independent_quality_target': row.get('target') is not None,
                      'l3_exact_token_repeat_fraction': repeat,
                      'arms': ['full', 'strict_prefix', 'blend', 'blend_cow'] if not geometry.hybrid else ['full', 'strict_prefix']}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            stats.append({**identity, 'prompt_tokens': len(prompt.prompt_token_ids),
                          'l1_and_system_context_tokens': prompt.segments[0]['range'][1],
                          'l3_content_tokens': prompt.l3_content_tokens,
                          'l3_context_tokens': sum(b - a for s in prompt.segments if s['kind'] == 'l3' for a, b in [s['range']]),
                          'query_tokens': prompt.query_span[1] - prompt.query_span[0],
                          'l3_count': len(row['l3']), 'repeat_fraction': repeat})
    manifest = {'requests': len(rows), 'origin_counts': dict(Counter(r['origin'] for r in rows)),
                'split_counts': dict(Counter(r['split'] for r in rows)),
                'model_geometry': asdict(geometry), 'full_attention_kv_bytes_per_token_bf16': geometry.full_kv_bytes_per_token(),
                'hybrid_recurrent_state_bytes': None, 'enable_thinking': False,
                'query_cache_candidate_policy': 'reject overlapping chunk before any approximate KV transfer',
                'real_distribution_measured': any(r['origin'] == 'real_redacted' for r in rows),
                'GPU_inference_run': False, 'quality_result': None,
                'source': str(data.resolve())}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / 'workload-statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(prepare(args.data, tokenizer, json.loads(args.config.read_text()), args.output), ensure_ascii=False))
