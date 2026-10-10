"""Summarize directly observed Kineto events; never infer absent byte counts."""
import argparse
import gzip
import json
from pathlib import Path


def event_bytes(event):
    args = event.get('args', {})
    for key in ('bytes', 'Bytes', 'size (bytes)'):
        value = args.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def summarize(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt') as f:
        events = json.load(f).get('traceEvents', [])
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
    copies = [e for e in events if e.get('cat') == 'gpu_memcpy' and 'dur' in e]
    result = {'trace': str(path.resolve()), 'kernel_events': len(kernels),
              'kernel_duration_sum_ms': sum(e['dur'] for e in kernels) / 1000 if kernels else None,
              'gpu_memcpy_events': len(copies), 'includes_full_vocabulary_diagnostic': True,
              'KV_transfer_bytes': None, 'peak_hbm_bytes': None,
              'stage_attribution_complete': False, 'timing_eligible': False}
    for direction in ('HtoD', 'DtoH', 'DtoD'):
        subset = [e for e in copies if direction in e.get('name', '')]
        observed = [event_bytes(e) for e in subset]
        result[direction] = {'events': len(subset),
                             'duration_sum_ms': sum(e['dur'] for e in subset) / 1000 if subset else None,
                             'events_with_byte_count': sum(b is not None for b in observed),
                             'bytes': sum(observed) if subset and all(b is not None for b in observed) else None}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows = []
    for mapping in sorted(args.profiles.glob('*/trace-map.jsonl')):
        for line in mapping.read_text().splitlines():
            row = json.loads(line)
            for filename in row['files']:
                rows.append({**row, 'arm': mapping.parent.name, **summarize(Path(filename))})
    if not rows:raise ValueError('No actual traces available')
    args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps({'trace_requests': len(rows), 'requests_with_CUDA_kernels': sum(r['kernel_events'] > 0 for r in rows)}))
