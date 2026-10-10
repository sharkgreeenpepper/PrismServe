"""Paired fidelity and history stability for actual GPU repair evidence."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze import loss_upper,quantile


def analyze(root):
    parent=Path(json.loads((root/'budget.json').read_text())['original_results'])
    reference={}
    for path in (parent/'results').glob('p0-full-*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line);reference[(r['length'],r['mode'],r['seed'],r['request_id'])]=r
    prefix={}
    for path in (parent/'results').glob('p0-prefix-*.jsonl'):
        rr=[json.loads(s) for s in path.read_text().splitlines()]
        if rr:prefix[(rr[0]['length'],rr[0]['mode'],rr[0]['seed'])]=quantile(rr,'ttft_s',.95)
    cases=[];all_rows=[]
    for path in sorted((root/'results').glob('eval-*.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if not rows:continue
        r0=rows[0];pairs=[]
        for r in rows:
            key=(r['length'],r['mode'],r['seed'],r['request_id'])
            ref=reference[key];assert ref['input_hash']==r['input_hash']
            pairs.append((ref,r))
        n=len(rows);errors=sum(r['status']!='ok' for r in rows)
        upper=loss_upper(pairs)
        p95=quantile(rows,'ttft_with_control_s',.95)
        prefix95=prefix[(r0['length'],r0['mode'],r0['seed'])]
        quality=upper is not None and upper<=.01 and n==1000 and errors==0
        repairs=[r for r in rows if r.get('telemetry',{}).get('repair_invocations')]
        info={'case':path.stem,'policy':r0['policy'],'ratio':r0['ratio'],'writeback':r0['writeback'],
              'length':r0['length'],'mode':r0['mode'],'seed':r0['seed'],'n':n,'errors':errors,
              'accuracy':sum(r.get('correct',False) for r in rows)/n,
              'fidelity_to_full':sum(ref['output_token_ids']==r.get('output_token_ids') for ref,r in pairs)/n,
              'loss_upper95':upper,'ttft_p50':quantile(rows,'ttft_s',.5),'ttft_p95':quantile(rows,'ttft_s',.95),
              'ttft_with_control_p95':p95,'prefix_p95':prefix95,
              'e2e_p95':quantile(rows,'e2e_s',.95),'repair_requests':len(repairs),
              'effective_reused_kv_bytes':sum(r.get('effective_reused_kv_bytes',0) for r in rows),
              'kv_host_load_bytes':sum(r.get('kv_host_load_bytes',0) for r in rows),
              'max_source_generation':max(r.get('telemetry',{}).get('max_source_generation',0) for r in rows),
              'quality_gate':'pass' if quality else 'fail' if n==1000 and upper is not None else 'uncertain',
              'latency_gate':'pass' if p95 is not None and p95<prefix95 and n==1000 else 'fail' if n==1000 else 'uncertain',
              'overall_gate':'pass' if quality and p95 is not None and p95<prefix95 else 'fail' if n==1000 else 'uncertain'}
        cases.append(info);all_rows.extend(rows)
    targets=defaultdict(list)
    for r in all_rows:
        if r['mode']=='history':targets[(r['policy'],r['ratio'],r['writeback'],r['length'],r['request_id'])].append(r)
    histories=defaultdict(lambda:{'targets':0,'varied':0})
    for key,rows in targets.items():
        if {r['seed'] for r in rows}!={17,29,43}:continue
        assert len({r['input_hash'] for r in rows})==1
        name='|'.join(map(str,key[:-1]));h=histories[name]
        h['targets']+=1;h['varied']+=int(len({r.get('prediction') for r in rows})>1)
    for h in histories.values():h['variation_rate']=h['varied']/h['targets']
    (root/'summary.json').write_text(json.dumps({'cases':cases,'history':dict(histories)},indent=2))
    if cases:
        with (root/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    report=['# P1/P2 真实 GPU 复用与修复报告','',
            '所有性能记录来自实际模型执行；高层 QKV 行数、实际修复预算和来源代数逐请求记录。',
            'pristine 是 Full-produced 的片段来源，跨上下文使用仍为近似；repaired 版本采用独立 key 和 COW。',
            '','|配置|n|正确率|Full token 一致率|质量损失95%上界|TTFT P95含控制RPC|Prefix P95|门槛|',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    for c in cases:
        upper='uncertain' if c['loss_upper95'] is None else f'{c["loss_upper95"]:.4f}'
        ttft='missing' if c['ttft_with_control_p95'] is None else f'{c["ttft_with_control_p95"]:.4f}'
        report.append(f'|{c["case"]}|{c["n"]}|{c["accuracy"]:.2%}|{c["fidelity_to_full"]:.2%}|{upper}|{ttft}|{c["prefix_p95"]:.4f}|{c["overall_gate"]}|')
    report+=['','## 历史稳定性','','```json',json.dumps(dict(histories),ensure_ascii=False,indent=2),'```',
             '', '## 解释边界','',
             '基线为此前同模型/输入/FA3、greedy、TP1 的 P0；先验选择只使用 calibration，正式评估后不调修复比例。',
             '控制RPC纳入额外 TTFT 口径，collector telemetry RPC 不计为模型推理。',
             'kv_host_load_bytes 由实际加载范围、层数和 dtype 统计复制负载；不是 PCIe 硬件计数器。',
             'Logit KL 和 GPU-ms 若缺失保持 null；本工作负载是结构化工具/RAG，不是端到端 Agent。',
             '小样本 sanity 与正式质量门槛分别报告。不能仅因缓存命中或一条 canary 正确而认定优化达标。']
    (root/'EXPERIMENT_REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'cases':len(cases),'complete':sum(c['n']==1000 for c in cases),'requests':len(all_rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root)
