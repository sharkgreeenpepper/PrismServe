"""Attribute CUDA events to the narrowest matching launch-time host scope.

The method matches a CUDA event's External id (falling back to correlation id)
to its CUDA runtime/driver API, then finds the narrowest ``prism:*``
user_annotation on the same CPU thread that encloses the API call. This is
launch-origin attribution. Kernel-duration sums can overlap across streams;
memcpy bytes include all memory traffic in the named scope and are not
automatically KV-only.
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_trace(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f).get('traceEvents', [])


def byte_count(event):
    args = event.get('args', {})
    for key in ('bytes', 'Bytes', 'size (bytes)'):
        value = args.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def api_phase(api, scopes):
    candidates = [scope for scope in scopes
                  if api.get('pid') == scope.get('pid') and
                  api.get('tid') == scope.get('tid') and
                  scope['ts'] <= api['ts'] and
                  api['ts'] + api.get('dur', 0) <= scope['ts'] + scope.get('dur', 0)]
    if not candidates:
        return None, set(), None
    # Nested scopes are expected: e.g. kv_store inside native_model_forward.
    # Resolve by choosing the smallest enclosing annotation; do not sum scopes.
    candidates.sort(key=lambda x: (x.get('dur', float('inf')), x.get('name', '')))
    labels = {x['name'].removeprefix('prism:') for x in candidates}
    return (candidates[0]['name'].removeprefix('prism:'), labels,
            candidates[0].get('dur', float('inf')))


def classify_events(events, target_category, scopes, apis_by_external_id, apis_by_correlation):
    targets = [e for e in events if e.get('cat') == target_category and 'dur' in e]
    by_phase = defaultdict(lambda: {'events': 0, 'duration_sum_ms': 0.0,
                                    'bytes': 0, 'events_with_bytes': 0,
                                    'events_missing_bytes': 0})
    counts = Counter()
    missing_api = 0
    no_scoped_api = 0
    nested_resolved = 0
    unresolved = 0
    correlation_fallback = 0
    for event in targets:
        external_id = event.get('args', {}).get('External id')
        apis = apis_by_external_id.get(external_id, []) if external_id is not None else []
        if not apis:
            correlation = event.get('args', {}).get('correlation')
            apis = apis_by_correlation.get(correlation, []) if correlation is not None else []
            if apis:
                correlation_fallback += 1
        if not apis:
            missing_api += 1
            continue
        possible = []
        possible_labels = set()
        for api in apis:
            label, labels, scope_duration = api_phase(api, scopes)
            if label is not None:
                possible.append((scope_duration, label))
                possible_labels.update(labels)
        if not possible:
            no_scoped_api += 1
            continue
        if len(possible_labels) > 1:
            nested_resolved += 1
        # Prefer the API invocation itself with the most-specific phase scope.
        # If equally specific calls disagree, leave the target unattributed.
        min_scope_duration = min(x[0] for x in possible)
        best_scopes = [x[1] for x in possible if x[0] == min_scope_duration]
        if len(set(best_scopes)) != 1:
            unresolved += 1
            continue
        phase = best_scopes[0]
        item = by_phase[phase]
        item['events'] += 1
        item['duration_sum_ms'] += event['dur'] / 1000
        counts[phase] += 1
        if target_category == 'gpu_memcpy':
            size = byte_count(event)
            if size is None:
                item['events_missing_bytes'] += 1
            else:
                item['events_with_bytes'] += 1
                item['bytes'] += size
            name = event.get('name', '')
            direction = next((d for d in ('HtoD', 'DtoH', 'DtoD') if d in name), 'other')
            item.setdefault('directions', defaultdict(lambda: {
                'events': 0, 'duration_sum_ms': 0.0, 'bytes': 0,
                'events_with_bytes': 0, 'events_missing_bytes': 0}))
            direction_item = item['directions'][direction]
            direction_item['events'] += 1
            direction_item['duration_sum_ms'] += event['dur'] / 1000
            if size is None:
                direction_item['events_missing_bytes'] += 1
            else:
                direction_item['events_with_bytes'] += 1
                direction_item['bytes'] += size
    # Convert nested defaultdicts to plain JSON values.
    clean = {}
    for phase, item in by_phase.items():
        item = dict(item)
        if 'directions' in item:
            item['directions'] = {k: dict(v) for k, v in item['directions'].items()}
        clean[phase] = item
    return {
        'target_category': target_category,
        'events_total': len(targets),
        'by_launch_phase': clean,
        'unattributed': {
            'missing_external_id_or_api': missing_api,
            'api_outside_prism_host_scope': no_scoped_api,
            'equally_specific_phase_conflict': unresolved,
        },
        'nested_scope_resolution_count': nested_resolved,
        'correlation_fallback_count': correlation_fallback,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_jsonl.exists() or args.output_json.exists():
        raise FileExistsError('Refusing to overwrite attribution outputs')
    rows = []
    for mapping_path in sorted(args.trace_dir.glob('*/trace-map.jsonl')):
        for mapping in (json.loads(s) for s in mapping_path.read_text().splitlines() if s.strip()):
            if len(mapping.get('files', [])) != 1:
                raise ValueError(f"{mapping['request_id']}: expected one trace")
            path = Path(mapping['files'][0])
            events = read_trace(path)
            scopes = [e for e in events if e.get('cat') == 'user_annotation' and
                      e.get('ph') == 'X' and str(e.get('name', '')).startswith('prism:')]
            apis_by_external_id = defaultdict(list)
            apis_by_correlation = defaultdict(list)
            for event in events:
                if event.get('cat') in ('cuda_runtime', 'cuda_driver'):
                    external_id = event.get('args', {}).get('External id')
                    if external_id is not None:
                        apis_by_external_id[external_id].append(event)
                    correlation = event.get('args', {}).get('correlation')
                    if correlation is not None:
                        apis_by_correlation[correlation].append(event)
            row = {
                'arm': mapping_path.parent.name.removesuffix('-traces'),
                'request_id': mapping['request_id'],
                'input_hash': mapping['input_hash'],
                'trace': str(path.resolve()),
                'kernel_launch_attribution': classify_events(
                    events, 'kernel', scopes, apis_by_external_id, apis_by_correlation),
                'memcpy_launch_attribution': classify_events(
                    events, 'gpu_memcpy', scopes, apis_by_external_id, apis_by_correlation),
                'timing_eligible': False,
                'task_accuracy_eligible': False,
            }
            rows.append(row)

    if len(rows) != 8:
        raise ValueError(f'Expected 8 traces, found {len(rows)}')
    rows.sort(key=lambda x: (x['arm'], x['request_id']))
    summary = {
        'method': 'CUDA event External id (correlation fallback) -> cuda_runtime/cuda_driver API ID -> narrowest same-thread prism user_annotation enclosing API call',
        'scope': 'launch-origin attribution for the finite one-token full-vocabulary profiling smoke',
        'traces': len(rows),
        'arms': {},
        'limits': [
            'An assigned phase identifies the host scope that launched the CUDA event; it is not exclusive GPU execution time.',
            'Kernel duration sums may overlap across GPU streams and are not additive phase latency.',
            'Memcpy bytes are directly observed CUPTI bytes inside the assigned launch scope; they are not automatically KV-only.',
            'Unassigned events remain visible and are not redistributed.',
            'Host monotonic timestamps are not subtracted from trace timestamps.',
        ],
    }
    for arm in ('full', 'blend'):
        group = [x for x in rows if x['arm'] == arm]
        by_phase = defaultdict(lambda: {'kernel_events': 0, 'kernel_duration_sum_ms': 0.0,
                                         'memcpy_events': 0, 'memcpy_duration_sum_ms': 0.0,
                                         'memcpy_bytes': defaultdict(int),
                                         'memcpy_events_missing_bytes': defaultdict(int)})
        totals = {'kernel_events': 0, 'kernel_unattributed': 0,
                  'memcpy_events': 0, 'memcpy_unattributed': 0}
        for row in group:
            ka = row['kernel_launch_attribution']
            totals['kernel_events'] += ka['events_total']
            totals['kernel_unattributed'] += sum(ka['unattributed'].values())
            for phase, metrics in ka['by_launch_phase'].items():
                by_phase[phase]['kernel_events'] += metrics['events']
                by_phase[phase]['kernel_duration_sum_ms'] += metrics['duration_sum_ms']
            ma = row['memcpy_launch_attribution']
            totals['memcpy_events'] += ma['events_total']
            totals['memcpy_unattributed'] += sum(ma['unattributed'].values())
            for phase, metrics in ma['by_launch_phase'].items():
                by_phase[phase]['memcpy_events'] += metrics['events']
                by_phase[phase]['memcpy_duration_sum_ms'] += metrics['duration_sum_ms']
                for direction, dmetrics in metrics.get('directions', {}).items():
                    by_phase[phase]['memcpy_bytes'][direction] += dmetrics['bytes']
                    by_phase[phase]['memcpy_events_missing_bytes'][direction] += dmetrics['events_missing_bytes']
        summary['arms'][arm] = {
            **totals,
            'by_launch_phase': {
                phase: {
                    'kernel_events': values['kernel_events'],
                    'kernel_duration_sum_ms': values['kernel_duration_sum_ms'],
                    'memcpy_events': values['memcpy_events'],
                    'memcpy_duration_sum_ms': values['memcpy_duration_sum_ms'],
                    'memcpy_bytes_observed': dict(values['memcpy_bytes']),
                    'memcpy_events_missing_byte_count': dict(values['memcpy_events_missing_bytes']),
                }
                for phase, values in by_phase.items()
            },
        }
    args.output_jsonl.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    args.output_json.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
