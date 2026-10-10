"""Replay requests with either CUDA traces or lightweight host phase events.

CUDA mode is a one-token diagnostic and is excluded from timing/quality claims.
Host-only mode avoids the Torch profiler and full-vocabulary logprobs so ordinary
service measurements can be collected alongside host-side phase spans.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=('cuda', 'host-only'), default='cuda')
    parser.add_argument('--trace-dir', type=Path)
    parser.add_argument('--event-dir', type=Path)
    # Register this option here so argparse's accepted abbreviations are
    # consumed too (notably --capture and --cap), then forward the canonical
    # spelling only in the CUDA diagnostic mode.
    parser.add_argument('--capture-logprobs', action='store_true')
    args, remainder = parser.parse_known_args()
    if args.mode == 'cuda':
        if args.trace_dir is None:
            raise ValueError('CUDA mode requires --trace-dir')
        if not args.capture_logprobs:
            raise ValueError('CUDA trace mode requires the separate one-token diagnostic mode')
        if args.event_dir is not None:
            raise ValueError('--event-dir is only for host-only mode')
        output_dir = args.trace_dir
    else:
        if args.event_dir is None:
            raise ValueError('Host-only mode requires --event-dir')
        if args.capture_logprobs:
            raise ValueError('Host-only service mode must not capture full-vocabulary logprobs')
        if args.trace_dir is not None:
            raise ValueError('--trace-dir is only for CUDA mode')
        output_dir = args.event_dir
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    # GPU selection precedes vLLM/Torch import, even for this wrapper.
    gpu_parser = argparse.ArgumentParser(add_help=False)
    gpu_parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    gpu_parser.add_argument('--inputs', type=Path, required=True)
    gpu_parser.add_argument('--limit', type=int, default=0)
    gpu_args, _ = gpu_parser.parse_known_args(remainder)
    rows = [json.loads(line) for line in gpu_args.inputs.read_text().splitlines() if line.strip()]
    if gpu_args.limit:
        rows = rows[:gpu_args.limit]
    warmup_calls = min(2, len(rows))
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_args.gpu)
    import vllm
    native_llm = vllm.LLM

    class ProfiledLLM:
        def __init__(self, **kwargs):
            if args.mode == 'cuda':
                kwargs['profiler_config'] = {
                    'profiler': 'torch', 'torch_profiler_dir': str(args.trace_dir.resolve()),
                    'torch_profiler_with_stack': False, 'torch_profiler_with_memory': True,
                    'torch_profiler_record_shapes': True, 'torch_profiler_use_gzip': True}
            self.engine = native_llm(**kwargs)
            self.engine.collective_rpc('prismserve_control', args=({
                'action': 'phase_install', 'host_only': args.mode == 'host-only'},))
            self.calls = 0

        def __getattr__(self, name):
            return getattr(self.engine, name)

        def generate(self, *positional, **kwargs):
            self.calls += 1
            if self.calls <= warmup_calls:
                return self.engine.generate(*positional, **kwargs)
            request_index = self.calls - warmup_calls - 1
            if request_index >= len(rows):
                raise IndexError(f'LLM.generate call {self.calls} exceeds {len(rows)} replay rows')
            row = rows[request_index]
            assert positional[0]['prompt_token_ids'] == row['prompt_token_ids']
            before = ({p.resolve() for p in args.trace_dir.rglob('*') if p.is_file()}
                      if args.mode == 'cuda' else set())
            profile_started = False
            phase = None
            phase_error = None
            result = None
            request_error = None
            request_traceback = None
            try:
                if args.mode == 'cuda':
                    self.engine.start_profile(profile_prefix=f'request-{request_index:04d}')
                    profile_started = True
                self.engine.collective_rpc('prismserve_control', args=({'action': 'phase_begin',
                                                   'request_id': row['request_id']},))
                result = self.engine.generate(*positional, **kwargs)
            except BaseException as exc:
                request_error = exc
                request_traceback = exc.__traceback__
            finally:
                try:
                    phase = self.engine.collective_rpc('prismserve_control',
                                                       args=({'action': 'phase_end'},))[0]
                except BaseException as exc:
                    phase_error = exc
                if profile_started:
                    try:
                        self.engine.stop_profile()
                    except BaseException as exc:
                        if phase_error is None:
                            phase_error = exc
                after = ({p.resolve() for p in args.trace_dir.rglob('*') if p.is_file()}
                         if args.mode == 'cuda' else set())
                trace_files = sorted(str(p) for p in after - before
                                     if p.name.endswith(('.pt.trace.json', '.pt.trace.json.gz')))
                mapping = {'request_id': row['request_id'], 'input_hash': row['input_hash'],
                           'request_index': request_index, 'mode': args.mode,
                           'files': trace_files, 'phase_host_events': phase['events'] if phase else [],
                           'status': 'ok' if request_error is None and phase_error is None else 'error',
                           'error': repr(request_error or phase_error) if request_error or phase_error else None,
                           'timing_eligible': args.mode == 'host-only' and request_error is None and phase_error is None,
                           'task_accuracy_eligible': args.mode == 'host-only' and request_error is None and phase_error is None,
                           'phase_event_clock': phase.get('clock') if phase else None,
                           'phase_events_are_host_elapsed': True}
                map_name = 'trace-map.jsonl' if args.mode == 'cuda' else 'phase-map.jsonl'
                with (output_dir / map_name).open('a') as out:
                    out.write(json.dumps(mapping) + '\n')
            if request_error is not None:
                raise request_error.with_traceback(request_traceback)
            if phase_error is not None:
                raise phase_error
            return result

    vllm.LLM = ProfiledLLM
    import runpy
    replay_args = [*remainder]
    if args.capture_logprobs:
        replay_args.append('--capture-logprobs')
    sys.argv = ['run_replay.py', *replay_args]
    runpy.run_path(str(Path(__file__).with_name('run_replay.py')), run_name='__main__')


if __name__ == '__main__':
    main()
