"""Bounded local job controller with durable status and owned-process cleanup."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'.aris/runs/nonprefix-kv-20261008'


def save_status(status):
    temporary=ROOT/'status.tmp.json'
    temporary.write_text(json.dumps(status,indent=2))
    temporary.replace(ROOT/'status.json')


def stop(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nonprefix-status',default='blocked')
    p.add_argument('--resume',action='store_true',help='Skip verified complete output files')
    p.add_argument('--spread-pending',action='store_true',help='Assign remaining jobs to idle authorized GPUs')
    a=p.parse_args()
    lock=(ROOT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another controller owns this run')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    budget=json.loads((ROOT/'budget.json').read_text())
    deadline=budget['deadline_epoch']
    data=ROOT/'data/eval.jsonl'
    rows=[json.loads(s) for s in data.read_text().splitlines()]
    for length in (8192,16384):
        assert len([r for r in rows if r['length']==length])==1000
    assert len({r['request_id'] for r in rows})==2000
    expected_hashes={r['request_id']:r['input_hash'] for r in rows}
    tasks=[]
    cases=[('cold',17),('continuous',17),('history',17),('history',29),('history',43)]
    for round_id,(mode,seed) in enumerate(cases):
        for length,initial_full,initial_prefix in [(8192,0,1),(16384,2,3)]:
            for arm,initial in [('full',initial_full),('prefix',initial_prefix)]:
                # Rotate each arm's physical GPU; one process per GPU at a time.
                gpu=(initial+round_id)%4
                name=f'p0-{arm}-{length}-{mode}-{seed}'
                tasks.append(dict(name=name,arm=arm,gpu=gpu,length=length,mode=mode,seed=seed))
    status=dict(state='running',nonprefix_status=a.nonprefix_status,
                start_epoch=budget['start_epoch'],deadline_epoch=deadline,
                wall_budget_seconds=86400,gpu_budget_hours=96,
                controller_pid=os.getpid(),completed=[],failed=[],running=[],pending=tasks.copy())
    if a.resume:
        pending=[]
        for task in tasks:
            output=ROOT/'results'/f"{task['name']}.jsonl"
            if output.exists():
                records=[json.loads(s) for s in output.read_text().splitlines()]
                if (len(records)==1000 and len({r['request_id'] for r in records})==1000
                    and all(r['arm']==task['arm'] and r['mode']==task['mode']
                            and r['length']==task['length'] and r['seed']==task['seed']
                            and r['status']=='ok'
                            and expected_hashes.get(r['request_id'])==r['input_hash'] for r in records)):
                    task.update(gpu=records[0]['gpu'],rows=1000,complete=True,exit_code=0,recovered=True)
                    status['completed'].append(task);continue
            pending.append(task)
        tasks=pending
    active={}
    try:
        while tasks or active:
            now=time.time()
            if now>=deadline-180:
                status['state']='budget_exhausted';break
            for gpu in range(4):
                if gpu in active:continue
                reserved_path=ROOT/'reserved-gpus.json'
                reserved=json.loads(reserved_path.read_text()) if reserved_path.exists() else []
                if gpu in reserved:continue
                idx=next((i for i,t in enumerate(tasks) if t['gpu']==gpu),None)
                if idx is None and a.spread_pending and tasks:idx=0
                if idx is None:continue
                memory=subprocess.check_output(
                    ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                usage={int(line.split(',')[0]):int(line.split(',')[1]) for line in memory.splitlines()}
                if usage[gpu]>=500:continue
                task=tasks.pop(idx)
                task['gpu']=gpu
                log=(ROOT/'logs'/f"{task['name']}.log").open('w')
                command=[sys.executable,str(REPO/'experiments/nonprefix_kv/run_engine.py'),
                         '--model','/home/bumi/infra/models/Qwen/Qwen2.5-7B-Instruct',
                         '--data',str(data),'--output',str(ROOT/'results'/f"{task['name']}.jsonl"),
                         '--arm',task['arm'],'--mode',task['mode'],'--seed',str(task['seed']),
                         '--length',str(task['length']),'--deadline',str(deadline)]
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO,
                                      env=env,start_new_session=True)
                task.update(pid=proc.pid,launch_epoch=now,command=command)
                active[gpu]=(proc,task,log)
                print('LAUNCHED',task['name'],gpu,proc.pid,flush=True)
            for gpu,(proc,task,log) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                log.close();task.update(exit_code=rc,finish_epoch=time.time())
                output=ROOT/'results'/f"{task['name']}.jsonl"
                task['rows']=sum(1 for _ in output.open()) if output.exists() else 0
                task['complete']=rc==0 and task['rows']==1000
                status['completed' if task['complete'] else 'failed'].append(task)
                del active[gpu]
                print('FINISHED',task['name'],rc,task['rows'],flush=True)
                subprocess.run([sys.executable,str(REPO/'experiments/nonprefix_kv/analyze.py'),str(ROOT)],
                               stdout=subprocess.DEVNULL,check=True,cwd=REPO)
            status.update(running=[t for _,t,_ in active.values()],pending=tasks.copy(),
                          update_epoch=time.time(),
                          conservative_gpu_hours=4*(time.time()-budget['start_epoch'])/3600)
            compatibility_path=ROOT/'compatibility.json'
            if compatibility_path.exists():
                compatibility=json.loads(compatibility_path.read_text())
                status['nonprefix_status']=compatibility['status']
            save_status(status)
            if tasks or active:time.sleep(10)
        if not tasks and not active:
            status['state']='complete' if not status['failed'] else 'partial_failure'
    except KeyboardInterrupt:
        status['state']='interrupted'
        raise
    finally:
        for proc,task,log in active.values():
            stop(proc);log.close()
            task.update(exit_code=proc.returncode,finish_epoch=time.time(),complete=False)
            status['failed'].append(task)
        status.update(running=[],pending=tasks,finish_epoch=time.time(),
                      conservative_gpu_hours=4*(time.time()-budget['start_epoch'])/3600)
        save_status(status)
        subprocess.run([sys.executable,str(REPO/'experiments/nonprefix_kv/analyze.py'),str(ROOT)],check=True,cwd=REPO)
        print('CONTROLLER_FINISHED',status['state'],flush=True)


if __name__=='__main__':main()
