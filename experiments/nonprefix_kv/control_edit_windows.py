"""Bounded two-GPU supervisor for the predeclared P2 edit supplement."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('root', type=Path);a = p.parse_args()
    root = a.root.resolve();code = Path(__file__).parent;repo = code.parents[1]
    lock = (root / 'controller.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget = json.loads((root / 'budget.json').read_text());jobs = {};logs = {}
    state = {'state': 'running', 'start_epoch': time.time(), 'gpu_ids': [0, 2], 'jobs': {}}
    try:
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
        mem = {int(x.split(',')[0]): int(x.split(',')[1]) for x in usage.splitlines()}
        assert all(mem[g] < 500 for g in (0, 2)), mem
        for gpu in (0, 2):
            env = os.environ.copy();env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            logs[gpu] = (root / f'logs/gpu{gpu}.log').open('w')
            cmd = [sys.executable, str(code / 'run_edit_windows.py'), '--run-root', str(root),
                   '--data', str(Path(budget['original_results']) / 'data/eval.jsonl'),
                   '--cases', str(root / f'configs/gpu{gpu}.json')]
            jobs[gpu] = subprocess.Popen(cmd, env=env, cwd=repo, stdout=logs[gpu],
                                         stderr=subprocess.STDOUT, start_new_session=True)
            state['jobs'][str(gpu)] = {'pid': jobs[gpu].pid, 'command': cmd}
        while True:
            counts = {p.stem: sum(1 for _ in p.open()) for p in (root / 'results').glob('guard-*.jsonl')}
            state.update(update_epoch=time.time(), case_counts=counts, requests=sum(counts.values()),
                         complete_cases=sum(n == 200 for n in counts.values()))
            for gpu, proc in jobs.items():state['jobs'][str(gpu)]['exit_code'] = proc.poll()
            temp = root / 'status.tmp.json';temp.write_text(json.dumps(state, indent=2));temp.replace(root / 'status.json')
            if all(p.poll() is not None for p in jobs.values()):
                state['state'] = 'complete' if all(p.returncode == 0 for p in jobs.values()) else 'partial_failure';break
            if time.time() >= budget['deadline_epoch'] - 180:
                state['state'] = 'budget_exhausted';break
            time.sleep(20)
    except BaseException:
        state['state'] = 'failed';raise
    finally:
        for gpu, proc in jobs.items():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid, signal.SIGKILL);proc.wait(timeout=15)
            try:os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:pass
            logs[gpu].close();state['jobs'][str(gpu)]['exit_code'] = proc.returncode
        state['finish_epoch'] = time.time();(root / 'status.json').write_text(json.dumps(state, indent=2))
        print('FINISHED', state['state'], flush=True)
