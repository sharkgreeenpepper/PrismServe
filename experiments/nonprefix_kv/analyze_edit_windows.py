"""Paired real-GPU edit fallback results, separate from the frozen main matrix."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from analyze import loss_upper, quantile
from run_engine import parse_answer


def analyze(root):
    b = json.loads((root / 'budget.json').read_text());parent = Path(b['original_results'])
    reference = {};prefix = {}
    for arm, dest in (('full', reference), ('prefix', prefix)):
        for path in (parent / 'results').glob(f'p0-{arm}-*.jsonl'):
            for line in path.open():
                r = json.loads(line)
                if r['kind'] == 'local_edit':dest[(r['length'], r['mode'], r['seed'], r['request_id'])] = r
    summaries = [];total = 0
    for path in sorted((root / 'results').glob('guard-*.jsonl')):
        rr = [json.loads(line) for line in path.open()]
        if not rr:continue
        cfg = json.loads(path.with_suffix('.config.json').read_text())
        for r in rr:
            assert all(r[k] == cfg[k] for k in ('length', 'mode', 'seed', 'policy', 'ratio', 'writeback'))
        complete = len(rr) == 200 and len({r['request_id'] for r in rr}) == 200
        if complete:
            sessions = Counter(r['session_id'] for r in rr)
            assert len(sessions) == 20 and set(sessions.values()) == {10}
        healthy = all(r['status'] == 'ok' for r in rr)
        pairs = [];pref = [];reasons = Counter();repairs = []
        for r in rr:
            key = (r['length'], r['mode'], r['seed'], r['request_id']);ref = reference[key]
            assert r['input_hash'] == ref['input_hash'] and r['answer'] == ref['answer']
            assert r['prediction'] == parse_answer(r['text']) and r['correct'] == (r['prediction'] == r['answer'])
            pairs.append((ref, r));pref.append(prefix[key]);t = r['telemetry']
            if t.get('repair_invocations'):
                assert t['repair_invocations'] == 1 and len(t['layers']) == 28
                assert [x['layer'] for x in t['layers']] == list(range(28))
                assert t['attention_backend'].startswith('FA3')
                if t['fallback_full']:
                    assert t['repair_tokens'] == t['reusable_tokens']
                    assert t['loaded_span_tokens'] == r['length']
                    assert t['reusable_tokens'] + t['mandatory_prefix_tokens'] == r['length']
                    assert all(l['qkv_rows'] == l['attention_rows'] == l['kv_rows'] == r['length'] for l in t['layers'])
                    assert len(set(t['repair_positions'])) == t['reusable_tokens']
                    assert r['effective_reused_tokens'] == r['effective_reused_kv_bytes'] == 0
                    reasons[t['fallback_reason']] += 1
                else:
                    assert t['repair_tokens'] == t['repair_budget']
                repairs.append(r)
        n = len(rr);total += n;upper = loss_upper(pairs);r0 = rr[0]
        info = {'case': path.stem, 'length': r0['length'], 'mode': r0['mode'], 'seed': r0['seed'],
                'n': n, 'errors': sum(r['status'] != 'ok' for r in rr),
                'accuracy': sum(r['correct'] for r in rr) / n,
                'answer_fidelity_to_full': sum(ref['prediction'] == r['prediction'] for ref, r in pairs) / n,
                'token_fidelity_to_full': sum(ref['output_token_ids'] == r['output_token_ids'] for ref, r in pairs) / n,
                'loss_upper95': upper, 'fallback_full_requests': sum(reasons.values()),
                'actual_repair_requests': len(repairs), 'ttft_p50': quantile(rr, 'ttft_with_control_s', .5),
                'ttft_p95': quantile(rr, 'ttft_with_control_s', .95),
                'full_p95': quantile([a for a, _ in pairs], 'ttft_s', .95),
                'prefix_p95': quantile(pref, 'ttft_s', .95),
                'fallback_reasons': dict(reasons),
                'quality_gate': ('pass' if upper <= .01 else 'fail') if complete and healthy and upper is not None else 'uncertain',
                'prefill_gpu_ms': None, 'hardware_transfer_bytes': None, 'peak_hbm_bytes': None}
        summaries.append(info)
    (root / 'summary.json').write_text(json.dumps({'cases': summaries, 'requests': total}, indent=2))
    if summaries:
        with (root / 'summary.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    text = ['# P2 保守编辑窗口补充', '',
            '主60,000请求矩阵不变；本补充只覆盖local_edit类的20会话/长度、每会话10请求。',
            '首次变化位置至prompt末端查询作为保守依赖范围，超预算或下游可复用位置不足时走既有全重算路径。',
            '该回退仍加载缓存并使用FA3 paged路径，不代表最快原生Full Prefill。质量与时延单独报告。', '',
            'TTFT口径为engine+context RPC；首次变更token/依赖区间的CPU构造在计时前，不含telemetry RPC。不能把该口径描述为包含所有控制器成本。', '',
            '|配置|n|正确率|Full token一致率|质量损失95%上界|全重算回退|P95 / Full / Prefix（秒）|',
            '|---|---:|---:|---:|---:|---:|---|']
    for c in summaries:
        upper = '缺失' if c['loss_upper95'] is None else f'{c["loss_upper95"]:.4f}'
        text.append(f'|{c["case"]}|{c["n"]}|{c["accuracy"]:.1%}|{c["token_fidelity_to_full"]:.1%}|{upper}|{c["fallback_full_requests"]}|{c["ttft_p95"]:.3f} / {c["full_p95"]:.3f} / {c["prefix_p95"]:.3f}|')
    text += ['', '此任务没有独立标注的较短隐状态因果传播路径，不能据保守全重算证明短窗口选择性修复有效。',
             '三个主策略在同一local_edit子集的正确率均可从父目录by-kind.csv查看；不与另外四类任务混合。',
             '原始JSONL、回退原因及28层计数保留。未观测GPU-ms/硬件bytes/HBM为缺失，不从时延推算。']
    text += ['名义5%只规定候选修复预算；Full fallback重算全部reusable，不称为实际5%或等预算性能。',
             '编辑区间由相邻请求差异构造，未与各缓存producer的独立依赖逐项对齐。本固定长度、单目标记录变化及全跨度回退不能外推为混合来源或变长上下文的一般编辑保护器。',
             '零损失的20会话经验bootstrap会退化为0，不代表总体质量风险已保证≤1个百分点。']
    (root / 'EXPERIMENT_REPORT.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'cases': len(summaries), 'requests': total, 'complete': sum(c['n'] == 200 for c in summaries)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('root', type=Path);a = p.parse_args();analyze(a.root)
