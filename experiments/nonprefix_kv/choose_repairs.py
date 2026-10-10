"""Freeze repair ratios using calibration results only, before reading evaluation."""
import argparse
import json
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    summaries=[];choices=[]
    for wb in ('original_only','persistent_repair'):
        for policy in ('topk','window','document'):
            options=[]
            for ratio in (.05,.10,.20):
                name=f'cal-{policy}-{int(ratio*100):02d}-{wb}'
                path=a.root/'results'/f'{name}.jsonl'
                rows=[json.loads(s) for s in path.read_text().splitlines()]
                assert len(rows)==100 and all(r['status']=='ok' for r in rows),name
                correct=sum(r['correct'] for r in rows)
                repairs=[r for r in rows if r['telemetry'].get('repair_invocations')]
                assert repairs,name
                item={'name':name,'policy':policy,'writeback':wb,'ratio':ratio,'n':len(rows),
                      'correct':correct,'accuracy':correct/len(rows),
                      'repair_requests':len(repairs),'max_generation':max(r['telemetry'].get('max_source_generation',0) for r in rows)}
                summaries.append(item);options.append(item)
            passed=[o for o in options if o['correct']==o['n']]
            selected=min(passed,key=lambda o:o['ratio']) if passed else max(options,key=lambda o:(o['accuracy'],o['ratio']))
            choices.append({k:selected[k] for k in ('policy','writeback','ratio','accuracy')})
            choices[-1]['calibration_eligible']=bool(passed)
    common=[s['ratio'] for s in summaries if s['policy']=='document' and s['writeback']=='original_only' and s['correct']==s['n']]
    common_ratio=min(common) if common else .20
    output={'rule':'primary experiment uses one shared ratio: smallest zero-error document/original-only calibration ratio, else diagnostic 20%; individual frontier choices are reported separately',
            'calibration_only':True,'primary_equal_budget_ratio':common_ratio,'summaries':summaries,'selected':choices}
    target=a.root/'frozen-selection.json'
    if target.exists():raise RuntimeError('Selection already frozen; do not tune evaluation')
    target.write_text(json.dumps(output,indent=2))
    cases=[]
    for c in choices:
        for length in (8192,16384):
            for mode,seed in [('cold',17),('continuous',17),('history',17),('history',29),('history',43)]:
                cases.append(dict(name=f'eval-{c["policy"]}-{int(common_ratio*100):02d}-{c["writeback"]}-{length}-{mode}-{seed}',
                                  policy=c['policy'],ratio=common_ratio,writeback=c['writeback'],length=length,mode=mode,seed=seed,
                                  calibration_eligible=any(s['correct']==s['n'] and s['policy']==c['policy'] and s['writeback']==c['writeback'] and s['ratio']==common_ratio for s in summaries)))
    for gpu in range(4):
        (a.root/f'configs/eval-gpu{gpu}.json').write_text(json.dumps(cases[gpu::4],indent=2))
    print(json.dumps(output,indent=2))
