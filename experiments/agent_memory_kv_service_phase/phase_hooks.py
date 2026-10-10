"""Profiling-only CPU scopes, never injected into service benchmark runs."""
from contextlib import contextmanager
from functools import wraps
import types
import time
import threading

ACTIVE = None
EVENTS = []
LOCK = threading.Lock()
INSTALLED = False
ANNOTATE_WITH_TORCH_PROFILER = True


@contextmanager
def span(name):
    request = ACTIVE
    if request is None:
        yield
        return
    start = time.perf_counter_ns()
    try:
        if ANNOTATE_WITH_TORCH_PROFILER:
            import torch
            with torch.profiler.record_function('prism:' + name):
                yield
        else:
            yield
    finally:
        with LOCK:
            EVENTS.append({'name': name, 'start_ns': start, 'end_ns': time.perf_counter_ns(),
                           'clock': 'host_monotonic', 'measurement': 'host_elapsed',
                           'request_id': request})


class MeasuredIterator:
    def __init__(self, generator, name):self.generator = generator;self.name = name
    def __iter__(self):return self
    def __next__(self):
        with span(self.name):return next(self.generator)
    def send(self, value):
        with span(self.name):return self.generator.send(value)
    def throw(self, *args):
        with span(self.name):return self.generator.throw(*args)
    def close(self):
        with span(self.name):return self.generator.close()


def wrap(function, name):
    @wraps(function)
    def measured(*args, **kwargs):
        with span(name):result = function(*args, **kwargs)
        # LMCache's decorators can hide the original generator function type.
        if isinstance(result, types.GeneratorType):return MeasuredIterator(result, name)
        return result
    return measured


def install(worker):
    global INSTALLED
    if INSTALLED:return {'already_installed': True}
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.compute.models.base import LMCBaseModel
    LMCacheEngine.lookup = wrap(LMCacheEngine.lookup, 'kv_lookup')
    LMCacheEngine.retrieve_layer = wrap(LMCacheEngine.retrieve_layer, 'kv_transfer')
    LMCacheEngine.store_layer = wrap(LMCacheEngine.store_layer, 'kv_store')
    LMCBaseModel.compute_layer = wrap(LMCBaseModel.compute_layer, 'selective_recompute')
    model = worker.model_runner.get_model()
    model.forward = wrap(model.forward, 'native_model_forward')
    worker.model_runner.sample_tokens = wrap(worker.model_runner.sample_tokens, 'first_token_sampling')
    INSTALLED = True
    return {'installed': True, 'scope': 'one-token profiling only; no service timing claims'}


def control(worker, payload):
    global ACTIVE, EVENTS, ANNOTATE_WITH_TORCH_PROFILER
    action = payload['action']
    if action == 'phase_install':
        ANNOTATE_WITH_TORCH_PROFILER = not payload.get('host_only', False)
        return install(worker)
    if action == 'phase_begin':
        ACTIVE = payload['request_id'];EVENTS = []
        return {'request_id': ACTIVE}
    if action == 'phase_end':
        ACTIVE = None
        with LOCK:return {'events': list(EVENTS), 'clock': 'host_monotonic'}
    raise ValueError('Unknown phase profiling action')
