"""Drain the reserved GPU's current job, then resume remaining work on idle GPUs."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

repo=Path(__file__).resolve().parents[2]
root=repo/'.aris/runs/nonprefix-kv-20261008'
while True:
    s=json.loads((root/'status.json').read_text())
    if time.time()>=s['deadline_epoch']-180:
        raise TimeoutError('Budget exhausted while draining')
    if not s['running'] and len(s['completed'])>=16:
        (root/'status-before-resume.json').write_text(json.dumps(s,indent=2))
        os.kill(s['controller_pid'],signal.SIGTERM)
        for _ in range(30):
            try:os.kill(s['controller_pid'],0)
            except ProcessLookupError:break
            time.sleep(1)
        else:raise RuntimeError('Previous controller failed to stop; do not double launch')
        (root/'reserved-gpus.json').write_text('[]')
        with (root/'logs/controller-resume.log').open('w') as log:
            result=subprocess.run([sys.executable,str(repo/'experiments/nonprefix_kv/control.py'),
                                   '--resume','--spread-pending','--nonprefix-status','blocked'],
                                  cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        raise SystemExit(result.returncode)
    if s['state']!='running':raise RuntimeError(f'Unexpected controller state {s["state"]}')
    time.sleep(10)
