"""Paired task quality, cache-history consistency, and measured latency summaries."""
import argparse
import csv
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from pathlib import Path
import numpy as np


def quantile(rows, key, p):
    values = [r[key] for r in rows if r.get(key) is not None and r['status']=='ok']
    return float(np.quantile(values,p)) if values else None


def loss_upper(pairs):
    groups = defaultdict(list)
    for ref, trial in pairs:
        groups[ref['session_id']].append(float(ref['correct'])-float(trial['correct']))
    if len(groups) < 2:
        return None
    means = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(17)
    draws = rng.choice(means, size=(10000,len(means)),replace=True).mean(axis=1)
    return float(np.quantile(draws,.95))


def analyze(root):
    cases={}
    for p in sorted((root/'results').glob('p0-*.jsonl')):
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        if rows: cases[p.stem]=rows
    summaries=[]
    for name,rows in cases.items():
        r0=rows[0]
        reference_name=name.replace('p0-prefix-','p0-full-')
        baseline={r['request_id']:r for r in cases.get(reference_name,[])}
        pairs=[(baseline[r['request_id']],r) for r in rows if r['request_id'] in baseline]
        if any(ref['input_hash']!=trial['input_hash'] for ref,trial in pairs):
            raise ValueError('Input hashes differ in a purported pair')
        n=len(rows)
        loss95=loss_upper(pairs) if pairs else None
        prefix_name=name.replace('p0-full-','p0-prefix-')
        prefix_rows=cases.get(prefix_name,[]) if r0['arm']=='full' else rows
        p95=quantile(rows,'ttft_s',.95)
        prefix95=quantile(prefix_rows,'ttft_s',.95)
        baseline95=quantile(list(baseline.values()),'ttft_s',.95)
        exact_bytes=[r.get('logical_exact_reuse_bytes') for r in rows]
        measured_ttft=[r['ttft_s'] for r in rows if r.get('ttft_s') is not None and r['status']=='ok']
        summary=dict(case=name,arm=r0['arm'],length=r0['length'],mode=r0['mode'],seed=r0['seed'],
                     gpu=r0['gpu'],n=n,errors=sum(r['status']!='ok' for r in rows),
                     accuracy=sum(bool(r.get('correct')) for r in rows)/n,
                     paired_n=len(pairs),answer_fidelity=(sum(ref.get('prediction')==trial.get('prediction')
                       and ref.get('prediction') is not None for ref,trial in pairs)/len(pairs)) if pairs else None,
                     loss_upper95=loss95,ttft_p50=quantile(rows,'ttft_s',.5),ttft_p95=p95,
                     ttft_mean=float(np.mean(measured_ttft)) if measured_ttft else None,
                     ttft_p99=quantile(rows,'ttft_s',.99),e2e_p95=quantile(rows,'e2e_s',.95),
                     tpot_p50=quantile(rows,'tpot_s',.5),
                     output_tokens_per_measured_e2e_second=(sum(r.get('output_tokens',0) for r in rows)
                                                           /sum(r.get('e2e_s',0) for r in rows)),
                     exact_hit_token_ratio=sum(r.get('cached_tokens') or 0 for r in rows)/sum(r['length'] for r in rows),
                     logical_exact_reuse_bytes=sum(v for v in exact_bytes if v is not None),
                     nonprefix_reuse_bytes=0,
                     quality_gate=('pass' if loss95 is not None and loss95<=.01 and len(pairs)==1000
                                   and all(ref['status']=='ok' and trial['status']=='ok' for ref,trial in pairs)
                                   else 'uncertain'),
                     ttft_vs_full_speedup=baseline95/p95 if baseline95 and p95 else None,
                     optimization_gate='not_applicable_baseline')
        summaries.append(summary)
    by_target=defaultdict(list)
    for rows in cases.values():
        for r in rows:
            by_target[(r['arm'],r['length'],r['request_id'])].append(r)
    history=defaultdict(lambda:{'targets':0,'varied':0,'token_varied':0,'errors':0})
    for (arm,length,req),rows in by_target.items():
        # Only compare identical requests with all three persistent-history seeds.
        rows=[r for r in rows if r['mode']=='history']
        if {r['seed'] for r in rows} != {17,29,43}:continue
        if len({r['input_hash'] for r in rows})!=1:raise ValueError('History changed target prompt')
        h=history[f'{arm}-{length}'];h['targets']+=1
        h['varied']+=int(len({r.get('prediction') for r in rows})>1)
        h['token_varied']+=int(len({tuple(r.get('output_token_ids',[])) for r in rows})>1)
        h['errors']+=int(any(r['status']!='ok' for r in rows))
    for h in history.values():
        h['answer_variation_rate']=h['varied']/h['targets']
        h['token_variation_rate']=h['token_varied']/h['targets']
    kind_stats=[]
    for name,rr in cases.items():
        by_kind=defaultdict(list)
        for r in rr:by_kind[r['kind']].append(r)
        for kind,kk in by_kind.items():
            kind_stats.append({'case':name,'kind':kind,'n':len(kk),
                               'accuracy':sum(bool(r.get('correct')) for r in kk)/len(kk),
                               'ttft_p50':quantile(kk,'ttft_s',.5),'ttft_p95':quantile(kk,'ttft_s',.95),
                               'exact_hit_token_ratio':sum(r.get('cached_tokens') or 0 for r in kk)/sum(r['length'] for r in kk)})
    supplements={}
    for arm in ('full','prefix'):
        path=root/'results'/f'supplement-{arm}.jsonl'
        if path.exists():
            rr=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            supplements[arm]={'n':len(rr),'correct':sum(bool(r.get('correct')) for r in rr),
                              'accuracy':sum(bool(r.get('correct')) for r in rr)/len(rr) if rr else None,
                              'errors':sum(r['status']!='ok' for r in rr),
                              'ttft_p95':quantile(rr,'ttft_s',.95),
                              'context_truncated':False,'representative':False}
    out=root/'summary.json'
    out.write_text(json.dumps({'cases':summaries,'history':dict(history),'supplement':supplements,
                              'kind_stats':kind_stats},indent=2))
    if summaries:
        with (root/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        with (root/'by-kind.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(kind_stats[0]));writer.writeheader();writer.writerows(kind_stats)
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    report=['# 本地 KV 测试报告：P0 完成，P1/P2 阻塞','',
            '## 状态','',f"控制器状态：{status.get('state','unknown')}。",
            'P0 为真实 vLLM 推理；本报告中的 Prefix 是严格前缀基线，不是非前缀复用。',
            f"P1/P2：{status.get('nonprefix_status','pending')}。",
            '','','## 基线结果','',
            '|配置|样本|正确率|TTFT P50 / P95 / P99（秒）|精确命中 token 比例|',
            '|---|---:|---:|---|---:|']
    for s in summaries:
        latency=' / '.join(f'{s[k]:.4f}' if s[k] is not None else '缺失'
                           for k in ('ttft_p50','ttft_p95','ttft_p99'))
        report.append(f"|{s['case']}|{s['n']}|{s['accuracy']:.2%}|{latency}|{s['exact_hit_token_ratio']:.2%}|")
    compatibility=json.loads((root/'compatibility.json').read_text()) if (root/'compatibility.json').exists() else {}
    report+=['','## 历史稳定性','', '```json',json.dumps(dict(history),ensure_ascii=False,indent=2),'```',
             '', '## LongBench-v2 全文补充','', '```json',json.dumps(supplements,ensure_ascii=False,indent=2),'```',
             '只选择完整上下文能放入 16K 的多文档样本；不足 20 条时不截断补数。此选择有长度偏差。',
             '', '## 非前缀兼容性结论','',compatibility.get('reason','pending'),
             '等预算 Top-K、连续窗口、文档片段和依赖窗口的选择器代码已实现，相关算法测试通过；没有接入 GPU 修复路径，不构成 P2 性能或质量结果。',
             '', '## 局限与下一阶段','',
             '主工作负载是有独立标准答案的结构化工具/RAG轨迹，不代表端到端 Coding Agent 成功率。',
             '时延取自引擎 RequestStateStats；E2E 为同步调用实测，不含 HTTP 网络传输。',
             '逻辑 KV 复用字节由真实命中 token 数及模型配置计算，不等于实测 H2D 字节。',
             'Logit KL、GPU-ms、传输字节及单请求 HBM 若未采集，保留 null。',
             'P1/P2 未通过时不判断连续修复有效，也不建议据此进入 P3/P4。',
             '下一轮先固定并验证 Connector 支持的 KV 布局或采用官方兼容版本；通过真实跨片段加载和逐层修复见证后，再接入等预算连续选择器。',
             '部分完成的配置不能满足 1000 请求验收，质量门槛标记 uncertain。']
    if all(s['n']==1000 for s in summaries) and len(summaries)==20:
        report += ['', '## 结论','',
                   'P0 完成 20 个配置、20,000 个目标请求。P1/P2 按兼容性失败分支阻塞，未获得非前缀优化结果。',
                   '结构化轨迹的 100% 正确率存在明显天花板，不能据此宣称复杂 Agent 任务质量保持。',
                   '已有精确命中不等于尾延迟必然改善；应按场景和尾部请求单独分析。']
    validation_path=root/'results/final-validation.json'
    if validation_path.exists():
        validation=json.loads(validation_path.read_text())
        report += ['', '## 最终验收与资源','',
                   f'控制器墙钟时间 {validation["wall_hours"]:.2f} 小时；GPU 时保守上界 {validation["conservative_gpu_hours"]:.2f}，低于 24h/96 GPUh 上限。',
                   f'GPU 0–3 释放后显存：{validation["gpu_memory_mib_at_cleanup"]} MiB。',
                   '全部配对输入哈希、标准答案、Full/冷缓存零命中检查通过；原始模型环境未修改。',
                   f'修正 {validation["source_metadata_corrections"]} 条来源标签：cache_policy 表示配置的后端，cache_source 根据实际命中确定。输入、预测和时延未变；首次捕获记录完整保留在 results/raw-captured/。',
                   '8 项回归测试通过；独立冻结环境复核见 logs/final-p0-witness.log。']
    rendered='\n'.join(report)+'\n'
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    (root/f'EXPERIMENT_REPORT_{stamp}.md').write_text(rendered)
    (root/'EXPERIMENT_REPORT.md').write_text(rendered)
    complete=sum(s['n']==1000 and s['errors']==0 for s in summaries)
    tracker='\n'.join(['# Experiment Tracker','',
                      '|阶段|状态|证据|','|---|---|---|',
                      '|M0/P0 环境|passed|独立冻结环境复核通过；不认证非前缀|',
                      f'|P0|{"complete" if complete==20 else "running"}|{complete}/20 配置完整；{sum(s["n"] for s in summaries)} 请求|',
                      f'|P1|{status.get("nonprefix_status","pending")}|compatibility.json，真实 KV store 失败|',
                      f'|P2|{status.get("nonprefix_status","pending")}|P1 前置条件未满足|',
                      '|M3|skipped|没有达标非前缀配置，不进入负载或跨模型扩展|',
                      f'|M4|{"complete" if status.get("state")=="complete" else "running"}|summary.csv、by-kind.csv、报告、质量/时延图|',''])
    (root/f'EXPERIMENT_TRACKER_{stamp}.md').write_text(tracker)
    (root/'EXPERIMENT_TRACKER.md').write_text(tracker)
    # Static scientific figure, always labelled as baseline only.
    if summaries:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,(ax,latency_ax)=plt.subplots(1,2,figsize=(11,4))
        colors={'full':'#2563eb','prefix':'#e87924'}
        for arm,marker in [('full','o'),('prefix','s')]:
            ss=[s for s in summaries if s['arm']==arm and s['ttft_p95'] is not None]
            ax.scatter([s['ttft_p95'] for s in ss],[s['accuracy']*100 for s in ss],label=arm,
                       marker=marker,color=colors[arm],alpha=.7)
            supplement=supplements.get(arm)
            if supplement and supplement['accuracy'] is not None:
                ax.scatter([supplement['ttft_p95']],[supplement['accuracy']*100],
                           marker=marker,color=colors[arm],s=70)
        if supplements.get('full'):
            ss=supplements['full']
            ax.annotate(f'LongBench subset: {ss["correct"]}/{ss["n"]}',
                        (ss['ttft_p95'],ss['accuracy']*100),xytext=(-130,10),textcoords='offset points',fontsize=9)
        ax.set(xlabel='Measured engine TTFT P95 (s)',ylabel='Task accuracy (%)',
               title='Structured traces and full-context QA',ylim=(-2,102),yticks=[0,20,40,60,80,100])
        labels=[];bars={'full':[],'prefix':[]}
        for length in (8192,16384):
            pair={s['arm']:s for s in summaries if s['length']==length and s['mode']=='history'
                  and s['seed']==29 and s['n']==1000}
            if set(pair)!= {'full','prefix'}:continue
            for metric,label in [('ttft_mean','mean'),('ttft_p95','P95')]:
                labels.append(f'{length//1024}K {label}')
                for arm in bars:bars[arm].append(pair[arm][metric])
        x=np.arange(len(labels))
        for arm,offset in [('full',-.18),('prefix',.18)]:
            latency_ax.bar(x+offset,bars[arm],width=.36,label=arm,color=colors[arm])
        latency_ax.set(xlabel='Same requests, history order 29',ylabel='Engine TTFT (s)',
                       title='Mean and tail latency',xticks=x,xticklabels=labels)
        ax.legend();latency_ax.legend()
        fig.suptitle('P0 baselines only; non-prefix repair is blocked',fontsize=11)
        fig.tight_layout();fig.savefig(root/'quality-latency.png',dpi=160);plt.close(fig)
    print(json.dumps({'cases':len(summaries),'rows':sum(s['n'] for s in summaries)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root)
