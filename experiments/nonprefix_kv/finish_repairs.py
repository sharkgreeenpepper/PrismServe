"""Wait for the owned matrix, run untimed diagnostics, and finalize evidence."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    code = Path(__file__).parent
    budget = json.loads((root / 'budget.json').read_text())
    last_progress = 0
    while True:
        status = json.loads((root / 'status.json').read_text())
        if status['state'] != 'running':
            assert status['state'] == 'complete', status['state']
            break
        assert time.time() < budget['deadline_epoch'] - 180
        if time.time() - last_progress > 300:
            print('MATRIX_PROGRESS', status['requests'], status['complete_cases'], flush=True)
            last_progress = time.time()
        time.sleep(20)
    # The matrix controller writes its final state after owned group cleanup.
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    mem = {int(s.split(',')[0]): int(s.split(',')[1]) for s in usage.splitlines()}
    # GPU0 may be reserved concurrently for the independent follows-doc pass.
    assert mem[2] < 500, mem
    log = (root / 'logs/diagnostics.log').open('w')
    env = os.environ.copy();env['CUDA_VISIBLE_DEVICES'] = '2'
    job = subprocess.Popen([sys.executable, str(code / 'run_repair_diagnostics.py'),
                            '--run-root', str(root), '--data',
                            str(Path(budget['original_results']) / 'data/eval.jsonl')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = job.wait(timeout=min(1800, budget['deadline_epoch'] - time.time() - 180))
    finally:
        if job.poll() is None:
            os.killpg(job.pid, signal.SIGTERM)
            try:job.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(job.pid, signal.SIGKILL);job.wait(timeout=15)
        try:os.killpg(job.pid, signal.SIGKILL)
        except ProcessLookupError:pass
        log.close()
    (root / 'diagnostic-status.json').write_text(json.dumps({'exit_code': rc, 'gpu': 2,
          'finish_epoch': time.time(), 'timing_eligible': False}, indent=2))
    print('DIAGNOSTIC_EXIT', rc, flush=True)
    # Give the separate short environment witness time to finish on GPU0.
    wait_end = min(time.time() + 300, budget['deadline_epoch'] - 180)
    while True:
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        mem = {int(s.split(',')[0]): int(s.split(',')[1]) for s in usage.splitlines()}
        if all(mem[g] < 500 for g in range(4)):break
        assert time.time() < wait_end, mem
        time.sleep(10)
    with (root / 'logs/final-validation.log').open('w') as log:
        result = subprocess.run([sys.executable, str(code / 'finalize_repairs.py'), str(root)],
                                stdout=log, stderr=subprocess.STDOUT)
    print('FINALIZATION_EXIT', result.returncode, flush=True)
    assert result.returncode == 0


if __name__ == '__main__':main()
