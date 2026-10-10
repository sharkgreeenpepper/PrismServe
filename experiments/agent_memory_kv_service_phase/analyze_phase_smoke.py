"""Summarize the isolated phase-hook smoke without claiming service timings.

All CUDA observations come from separate one-token, full-vocabulary diagnostic
traces. Host spans can overlap; kernel-duration sums and transfer events are
trace observations, not additive TTFT components or KV-only traffic.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def byte_count(event):
    args = event.get('args', {})
    for key in ('bytes', 'Bytes', 'size (bytes)'):
        value = args.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def trace_summary(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt') as f:
        events = json.load(f).get('traceEvents', [])
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
    copies = [e for e in events if e.get('cat') == 'gpu_memcpy' and 'dur' in e]
    transfers = {}
    for direction in ('HtoD', 'DtoH', 'DtoD'):
        subset = [e for e in copies if direction in e.get('name', '')]
        sizes = [byte_count(e) for e in subset]
        transfers[direction] = {
            'events': len(subset),
            'duration_sum_ms': sum(e['dur'] for e in subset) / 1000 if subset else None,
            'events_with_byte_count': sum(x is not None for x in sizes),
            'bytes': sum(sizes) if subset and all(x is not None for x in sizes) else None,
        }
    annotations = Counter(e.get('name') for e in events
                          if e.get('cat') == 'user_annotation' and
                          str(e.get('name', '')).startswith('prism:'))
    return {
        'trace_path': str(path.resolve()),
        'trace_events': len(events),
        'kernel_events': len(kernels),
        'kernel_duration_sum_ms': sum(e['dur'] for e in kernels) / 1000 if kernels else None,
        'gpu_memcpy_events': len(copies),
        'copies': transfers,
        'prism_user_annotation_events': dict(annotations),
        'peak_hbm_bytes': None,
        'kv_only_transfer_bytes': None,
    }


def request_summary(arm, raw, mapping, input_row):
    spans = defaultdict(list)
    for event in mapping.get('phase_host_events', []):
        duration_ms = (event['end_ns'] - event['start_ns']) / 1e6
        spans[event['name']].append(duration_ms)
    host = {
        name: {
            'clock': 'host_monotonic',
            'measurement': 'host_elapsed_interval_ms',
            'count': len(values),
            'sum_of_observed_intervals_ms': sum(values),
            'max_interval_ms': max(values),
            'may_overlap_other_spans': True,
        }
        for name, values in sorted(spans.items())
    }
    telemetry = raw.get('telemetry') or {}
    loaded = telemetry.get('loaded_chunks', [])
    return {
        'arm': arm,
        'request_id': raw['request_id'],
        'input_hash': raw['input_hash'],
        'input_origin': raw.get('origin'),
        'input_tokens': len(input_row['prompt_token_ids']),
        'status': raw.get('status'),
        'output_token_ids': raw.get('output_token_ids'),
        'timing_eligible': raw.get('timing_eligible'),
        'task_accuracy_eligible': raw.get('task_accuracy_eligible'),
        'host_spans': host,
        'repair': {
            'repair_invocations': telemetry.get('repair_invocations'),
            'loaded_chunks': len(loaded),
            'loaded_span_tokens': telemetry.get('loaded_span_tokens'),
            'mandatory_prefix_tokens': telemetry.get('mandatory_prefix_tokens'),
            'reusable_tokens': telemetry.get('reusable_tokens'),
            'repair_budget': telemetry.get('repair_budget'),
            'repair_tokens': telemetry.get('repair_tokens'),
            'excluded_query_chunks': telemetry.get('excluded_query_chunks'),
            'fallback_full': telemetry.get('fallback_full'),
        },
        'cuda_trace': [trace_summary(Path(filename)) for filename in mapping['files']],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_json, args.output_jsonl, args.output_csv):
        if path.exists():
            raise FileExistsError(path)

    input_rows = {x['input_hash']: x for x in read_jsonl(args.inputs)}
    requests = []
    for arm in ('full', 'blend'):
        raw_rows = read_jsonl(args.variant_dir / f'{arm}.jsonl')
        maps = read_jsonl(args.variant_dir / f'{arm}-traces' / 'trace-map.jsonl')
        raw_by_id = {x['request_id']: x for x in raw_rows}
        map_by_id = {x['request_id']: x for x in maps}
        if len(raw_rows) != 4 or len(raw_by_id) != len(raw_rows):
            raise ValueError(f'{arm}: expected four unique successful requests')
        if set(raw_by_id) != set(map_by_id):
            raise ValueError(f'{arm}: request IDs differ between JSONL and trace map')
        if any(x.get('status') != 'ok' for x in raw_rows):
            raise ValueError(f'{arm}: non-success request present')
        if any(x.get('timing_eligible') or x.get('task_accuracy_eligible') for x in raw_rows):
            raise ValueError(f'{arm}: diagnostic rows unexpectedly marked eligible')
        for request_id, raw in raw_by_id.items():
            input_row = input_rows[raw['input_hash']]
            if input_row['request_id'] != request_id:
                raise ValueError(f'{arm}: input request identity mismatch')
            mapping = map_by_id[request_id]
            if len(mapping.get('files', [])) != 1:
                raise ValueError(f'{arm}/{request_id}: expected one unique trace')
            requests.append(request_summary(arm, raw, mapping, input_row))

    requests.sort(key=lambda x: (x['arm'], x['request_id']))
    summary = {
        'scope': 'finite CUDA trace smoke; synthetic inputs; one generated token with full-vocabulary logprob capture',
        'requests': len(requests),
        'arms': {},
        'interpretation_limits': [
            'All request rows are timing-ineligible and task-accuracy-ineligible.',
            'Host elapsed intervals may overlap and are not additive TTFT stages.',
            'Kernel duration sums may overlap across CUDA streams and include diagnostic overhead.',
            'Memcpy observations include all runtime and diagnostic transfers; KV-only bytes are unknown.',
            'Peak HBM and production service GPU-ms are not measured by this smoke.',
        ],
    }
    for arm in ('full', 'blend'):
        rows = [x for x in requests if x['arm'] == arm]
        summary['arms'][arm] = {
            'requests': len(rows),
            'input_token_lengths': [x['input_tokens'] for x in rows],
            'kernel_events': sum(t['kernel_events'] for x in rows for t in x['cuda_trace']),
            'kernel_duration_sum_ms_diagnostic_only': sum(
                t['kernel_duration_sum_ms'] or 0 for x in rows for t in x['cuda_trace']),
            'gpu_memcpy_events': sum(t['gpu_memcpy_events'] for x in rows for t in x['cuda_trace']),
            'copy_bytes_observed': {
                direction: (sum(t['copies'][direction]['bytes'] or 0
                                for x in rows for t in x['cuda_trace'])
                            if not any(t['copies'][direction]['events'] !=
                                       t['copies'][direction]['events_with_byte_count']
                                       for x in rows for t in x['cuda_trace']) else None)
                for direction in ('HtoD', 'DtoH', 'DtoD')
            },
            'copy_events_missing_byte_counts': {
                direction: sum(t['copies'][direction]['events'] -
                               t['copies'][direction]['events_with_byte_count']
                               for x in rows for t in x['cuda_trace'])
                for direction in ('HtoD', 'DtoH', 'DtoD')
            },
        }
    args.output_json.write_text(json.dumps(summary, indent=2) + '\n')
    args.output_jsonl.write_text(''.join(json.dumps(x) + '\n' for x in requests))
    with args.output_csv.open('w', newline='') as f:
        fields = ['arm', 'request_id', 'input_hash', 'input_tokens', 'status',
                  'repair_invocations', 'loaded_chunks', 'loaded_span_tokens',
                  'mandatory_prefix_tokens', 'reusable_tokens', 'repair_budget',
                  'repair_tokens', 'excluded_query_chunks', 'kernel_events',
                  'kernel_duration_sum_ms_diagnostic_only', 'gpu_memcpy_events',
                  'HtoD_bytes_observed', 'DtoH_bytes_observed', 'DtoD_bytes_observed',
                  'timing_eligible', 'task_accuracy_eligible']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in requests:
            trace = row['cuda_trace'][0]
            writer.writerow({
                **{k: row.get(k) for k in ('arm', 'request_id', 'input_hash', 'input_tokens',
                   'status', 'timing_eligible', 'task_accuracy_eligible')},
                **{k: row['repair'].get(k) for k in (
                    'repair_invocations', 'loaded_chunks', 'loaded_span_tokens',
                    'mandatory_prefix_tokens', 'reusable_tokens', 'repair_budget',
                    'repair_tokens', 'excluded_query_chunks')},
                'kernel_events': trace['kernel_events'],
                'kernel_duration_sum_ms_diagnostic_only': trace['kernel_duration_sum_ms'],
                'gpu_memcpy_events': trace['gpu_memcpy_events'],
                **{f'{direction}_bytes_observed': trace['copies'][direction]['bytes']
                   for direction in ('HtoD', 'DtoH', 'DtoD')},
            })


if __name__ == '__main__':
    main()
