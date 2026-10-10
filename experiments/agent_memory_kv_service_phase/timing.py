"""Host timing and trace attribution; missing CUDA measurements stay missing."""
from contextlib import contextmanager
import time

TTFT_PHASES = ('kv_lookup', 'kv_transfer', 'selective_recompute', 'residual_prefill', 'first_token_sampling')


class HostSpans:
    def __init__(self):
        self.events = []

    @contextmanager
    def measure(self, name):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.events.append({'name': name, 'start_ns': start, 'end_ns': time.perf_counter_ns(),
                                'clock': 'host_monotonic', 'measurement': 'host_elapsed'})


def interval_union_ns(events):
    spans = sorted((e['start_ns'], e['end_ns']) for e in events)
    total = 0;end = None
    for start, stop in spans:
        if stop < start:
            raise ValueError('Negative interval')
        if end is None or start > end:
            total += stop - start
        else:
            total += max(0, stop - end)
        end = max(end if end is not None else stop, stop)
    return total


def summarize_host(events, request_start_ns, first_token_ns, queue_ns=None):
    if first_token_ns < request_start_ns:
        raise ValueError('Invalid first-token timestamp')
    relevant = []
    phases = {}
    for name in TTFT_PHASES:
        subset = [e for e in events if e['name'] == name]
        if any(e['clock'] != 'host_monotonic' for e in subset):
            raise ValueError('Do not add CUDA and host clocks')
        clipped = [{**e, 'start_ns': max(request_start_ns, e['start_ns']),
                    'end_ns': min(first_token_ns, e['end_ns'])} for e in subset
                   if e['start_ns'] < first_token_ns and e['end_ns'] > request_start_ns]
        phases[name + '_host_ms'] = interval_union_ns(clipped) / 1e6 if clipped else None
        relevant.extend(clipped)
    ttft = (first_token_ns - request_start_ns) / 1e6
    return {'ttft_ms': ttft, 'queue_ms': queue_ns / 1e6 if queue_ns is not None else None,
            **phases, 'instrumented_union_host_ms': interval_union_ns(relevant) / 1e6,
            'first_decode_forward_in_ttft': False, 'gpu_compute_ms': None,
            'h2d_bytes': None, 'd2h_bytes': None, 'peak_hbm_bytes': None}
