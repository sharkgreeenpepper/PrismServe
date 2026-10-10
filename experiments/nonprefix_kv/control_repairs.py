"""Four-GPU bounded supervisor for the frozen P1/P2 matrix."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root.resolve();repo=Path(__file__).resolve().parents[2]
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=json.loads((root/'budget.json').read_text());jobs={};logs={}
    status={'state':'running','controller_pid':os.getpid(),'global_deadline':budget['deadline_epoch'],
            'frozen_selection':str(root/'frozen-selection.json'),'gpu_ids':[0,1,2,3],'jobs':{}}
    def interrupted(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={int(s.split(',')[0]):int(s.split(',')[1]) for s in usage.splitlines()}
        assert all(used[i]<500 for i in range(4)),used
        for gpu in range(4):
            log=(root/'logs'/f'eval-gpu{gpu}.log').open('w');logs[gpu]=log
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            cmd=[sys.executable,str(repo/'experiments/nonprefix_kv/run_repairs.py'),
                 '--run-root',str(root),'--data',str(repo/'.aris/runs/nonprefix-kv-20261008/data/eval.jsonl'),
                 '--cases',str(root/f'configs/eval-gpu{gpu}.json')]
            jobs[gpu]=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['jobs'][str(gpu)]={'pid':jobs[gpu].pid,'command':cmd,'start_epoch':time.time()}
        last_analysis=0
        while True:
            now=time.time();counts={}
            for path in (root/'results').glob('eval-*.jsonl'):
                counts[path.stem]=sum(1 for _ in path.open())
            status.update(update_epoch=now,requests=sum(counts.values()),complete_cases=sum(n==1000 for n in counts.values()),
                          case_counts=counts,conservative_global_gpu_hours=4*(now-budget['global_start_epoch'])/3600)
            for gpu,proc in jobs.items():status['jobs'][str(gpu)]['exit_code']=proc.poll()
            temp=root/'status.tmp.json';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'status.json')
            if now-last_analysis>120 and counts:
                result=subprocess.run([sys.executable,str(repo/'experiments/nonprefix_kv/analyze_repairs.py'),str(root)],
                                      cwd=repo,stdout=subprocess.DEVNULL)
                status['analysis_exit_code']=result.returncode
                last_analysis=now
            if all(proc.poll() is not None for proc in jobs.values()):
                status['state']='complete' if all(proc.returncode==0 for proc in jobs.values()) else 'partial_failure';break
            if now>=budget['deadline_epoch']-180:
                status['state']='budget_exhausted';break
            time.sleep(20)
    except KeyboardInterrupt:
        status['state']='interrupted';raise
    except Exception:
        status['state']='failed';raise
    finally:
        for gpu,proc in jobs.items():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)
            # The frontend can exit before its LMCache background threads stop.
            # Terminate residual members of this owned, isolated process group.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            logs[gpu].close();status['jobs'][str(gpu)]['exit_code']=proc.returncode
        status['finish_epoch']=time.time()
        (root/'status.json').write_text(json.dumps(status,indent=2))
        subprocess.run([sys.executable,str(repo/'experiments/nonprefix_kv/analyze_repairs.py'),str(root)],cwd=repo,check=True)
        print('FINISHED',status['state'],flush=True)
