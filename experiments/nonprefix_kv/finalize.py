"""Final evidence checks after jobs exit; preserve captured records before metadata fixes."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

root=Path('.aris/runs/nonprefix-kv-20261008')
status=json.loads((root/'status.json').read_text())
assert status['state']=='complete' and not status['running'] and not status['pending'] and not status['failed']
assert len(status['completed'])==20
expected={r['request_id']:r for r in map(json.loads,(root/'data/eval.jsonl').read_text().splitlines())}
backup=root/'results/raw-captured'
backup.mkdir(exist_ok=True)
corrections=0;total=0;cases=0
for path in sorted((root/'results').glob('*.jsonl')):
    if not (path.name.startswith('p0-') or path.name.startswith('supplement-')):continue
    records=[json.loads(s) for s in path.read_text().splitlines()]
    if path.name.startswith('p0-'):
        assert len(records)==1000 and len({r['request_id'] for r in records})==1000
        cases+=1;total+=len(records)
        for r in records:
            assert r['input_hash']==expected[r['request_id']]['input_hash']
            assert r['answer']==expected[r['request_id']]['answer']
            assert r['status']=='ok' and r['gpu'] in (0,1,2,3)
            assert r['cached_tokens'] is not None and 0<=r['cached_tokens']<=r['length']
            if r['arm']=='full' or r['mode']=='cold':assert r['cached_tokens']==0
    if not (backup/path.name).exists():shutil.copy2(path,backup/path.name)
    for r in records:
        r['cache_policy']='none' if r['arm']=='full' else 'exact_prefix'
        actual='unknown' if r.get('cached_tokens') is None else 'exact_prefix' if r['cached_tokens']>0 else 'none'
        corrections+=int(r['cache_source']!=actual)
        r['cache_source']=actual
    temp=path.with_suffix('.final.tmp')
    temp.write_text(''.join(json.dumps(r)+'\n' for r in records));temp.replace(path)
assert cases==20 and total==20000
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
gpu_usage={int(s.split(',')[0]):int(s.split(',')[1]) for s in memory.splitlines()}
assert all(gpu_usage[i]<500 for i in range(4)),gpu_usage
report={'main_cases':cases,'main_requests':total,'input_hash_and_label_checks':'passed',
        'full_and_cold_have_zero_cache_hits':True,'source_metadata_corrections':corrections,
        'correction':'cache_policy retains configured backend; cache_source denotes actual measured hit source',
        'original_records_preserved':str(backup),'metrics_and_predictions_unchanged':True,
        'gpu_memory_mib_at_cleanup':{i:gpu_usage[i] for i in range(4)},
        'wall_hours':(status['finish_epoch']-status['start_epoch'])/3600,
        'conservative_gpu_hours':status['conservative_gpu_hours'],
        'nonprefix_status':status['nonprefix_status']}
assert report['wall_hours']<=24 and report['conservative_gpu_hours']<=96
(root/'results/final-validation.json').write_text(json.dumps(report,indent=2))
source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('experiments/nonprefix_kv').glob('*.py')}
(root/'configs/experiment-source.lock.json').write_text(json.dumps(source,indent=2))
print(report)
