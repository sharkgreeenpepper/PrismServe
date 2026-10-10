"""Local sequential adaptation of experiment-queue with immutable attempts.

One active GPU job keeps calibration contention controlled. Physical GPU
assignment is explicit in the manifest. No other user's process is touched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def save(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def free_mib(gpu):
    value = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    return int(value)


def complete_output(path, expected_ids):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return ([r['request_id'] for r in rows] == expected_ids
            and all(r.get('status') == 'ok' for r in rows))


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    expected_ids = [json.loads(line)['request_id'] for line in Path(manifest['inputs']).read_text().splitlines()]
    gate = json.loads(Path(manifest['environment_gate']).read_text())
    if gate.get('state') != 'ready_for_limited_calibration':raise ValueError('Environment gate not ready')
    for filename, expected in manifest['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed after freeze: ' + filename)
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state['manifest_sha256'] != digest:raise ValueError('Manifest changed on resume')
    else:
        state = {'manifest_sha256': digest, 'started_epoch': time.time(),
                 'jobs': {j['id']: {'status': 'pending', 'attempts': []} for j in manifest['jobs']}}
    save(args.state, state)
    for job in manifest['jobs']:
        entry = state['jobs'][job['id']]
        if entry['status'] == 'completed':continue
        if entry['status'] != 'pending':raise RuntimeError('Inspect non-pending attempt before resuming: ' + job['id'])
        gpu = job['gpu']
        if gpu not in (0,1,2,3):raise ValueError('GPU outside authorization')
        while free_mib(gpu) >= 500:
            print(json.dumps({'waiting_for_gpu': gpu, 'job': job['id']}), flush=True)
            time.sleep(15)
        output = Path(job['output']);log = Path(job['log'])
        if output.exists() or log.exists():raise FileExistsError('Preserve existing attempt artifacts')
        env = {**os.environ, **manifest['env'], 'CUDA_VISIBLE_DEVICES': str(gpu)}
        command = [manifest['python'], manifest['runner'], '--inputs', manifest['inputs'],
                   '--model', manifest['model'], '--output', str(output), '--gpu', str(gpu), *job['args']]
        output.parent.mkdir(parents=True, exist_ok=True);log.parent.mkdir(parents=True, exist_ok=True)
        attempt = {'started_epoch': time.time(), 'gpu': gpu, 'output': str(output), 'log': str(log), 'command': command}
        entry.update(status='running');entry['attempts'].append(attempt);save(args.state, state)
        print(json.dumps({'started': job['id'], 'gpu': gpu}), flush=True)
        with log.open('x') as stream:
            process = None
            try:
                process = subprocess.Popen(command, cwd=manifest['cwd'], env=env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                attempt['pid'] = process.pid;save(args.state, state)
                code = process.wait()
            except BaseException as exc:
                attempt.update(exit_code=process.poll() if process is not None else None,
                               finished_epoch=time.time(), error=repr(exc), interrupted=True)
                entry['status'] = 'stuck';save(args.state, state)
                raise
            finally:
                if process is not None:
                    try:os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:pass
        attempt.update(exit_code=code, finished_epoch=time.time())
        save(args.state, state)
        try:
            passed = code == 0 and output.exists() and complete_output(output, expected_ids)
        except Exception as exc:
            attempt['validation_error'] = repr(exc);passed = False
        entry['status'] = 'completed' if passed else 'stuck'
        save(args.state, state)
        print(json.dumps({'finished': job['id'], 'status': entry['status']}), flush=True)
        if not passed:raise RuntimeError('Failed attempt preserved; inspect before retry')
        # HBM release gates the next job; it cannot delay failure persistence.
        while free_mib(gpu) >= 500:time.sleep(5)
    state['finished_epoch'] = time.time();save(args.state, state)


if __name__ == '__main__':
    main()
