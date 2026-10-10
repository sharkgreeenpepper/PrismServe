"""Validate immutable GPU evidence, enrich identity, and render final artifacts.

This is CPU postprocessing. It never generates model predictions or timings.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from analyze import quantile
from analyze_repairs import analyze
from run_engine import parse_answer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if rows:
        with path.open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def finalize(root):
    status = json.loads((root / 'status.json').read_text())
    assert status['state'] == 'complete', status['state']
    assert all(j['exit_code'] == 0 for j in status['jobs'].values())
    budget = json.loads((root / 'budget.json').read_text())
    parent = Path(budget['original_results'])
    spec = json.loads((root / 'configs/env-spec.json').read_text())
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for name, expected in spec['adapter_modules'].items():
        assert digest(Path(__file__).parent / name) == expected, f'Frozen adapter changed: {name}'
    analyze(root)
    summary = json.loads((root / 'summary.json').read_text())
    cases = {c['case']: c for c in summary['cases']}
    expected_cases = {c['name']: c for gpu in range(4)
                      for c in json.loads((root / f'configs/eval-gpu{gpu}.json').read_text())}
    assert set(cases) == set(expected_cases) and len(cases) == 60
    reference = {}
    for path in (parent / 'results').glob('p0-full-*.jsonl'):
        for line in path.open():
            r = json.loads(line)
            reference[(r['length'], r['mode'], r['seed'], r['request_id'])] = r
    identity = {'model_path': spec['model'], 'dtype': spec['dtype'],
                'tensor_parallel_size': spec['tensor_parallel_size'],
                'environment_spec_sha256': spec_hash,
                'model_files_lock_sha256': digest(parent / 'configs/model-files.lock.json'),
                'requirements_sha256': digest(root / 'configs/requirements.freeze.txt')}
    (root / 'results/model-identity.json').write_text(json.dumps(identity, indent=2))
    normalized = root / 'results/normalized'
    normalized.mkdir(exist_ok=True)
    failures = []
    by_kind = []
    budgets = defaultdict(dict)
    checks = defaultdict(int)
    source_cross_context = 0
    source_moved = 0
    generations = defaultdict(int)
    for name, c in cases.items():
        path = root / 'results' / f'{name}.jsonl'
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == len({r['request_id'] for r in rows}) == 1000, name
        cfg_hash = digest(path.with_suffix('.config.json'))
        groups = defaultdict(list)
        with (normalized / path.name).open('w') as out:
            for r in rows:
                assert r['status'] == 'ok' and r['gpu'] in (0, 1, 2, 3)
                assert all(r[k] == expected_cases[name][k] for k in ('policy', 'ratio', 'writeback', 'mode', 'seed', 'length'))
                ref = reference[(r['length'], r['mode'], r['seed'], r['request_id'])]
                assert r['input_hash'] == ref['input_hash'] and r['answer'] == ref['answer']
                assert r['prediction'] == parse_answer(r['text'], r.get('answer_type', 'amount'))
                assert r['correct'] == (r['prediction'] == r['answer'])
                t = r['telemetry']
                repaired = bool(t.get('repair_invocations'))
                if repaired:
                    assert t['repair_invocations'] == 1
                    assert len(t['layers']) == 28 and [x['layer'] for x in t['layers']] == list(range(28))
                    assert t['attention_backend'].startswith('FA3')
                    n, k, m = t['reusable_tokens'], t['repair_tokens'], t['mandatory_prefix_tokens']
                    assert not t['fallback_full']
                    assert k == t['repair_budget'] == max(1, int(.05 * n))
                    checks['minimum_one_token_budget_requests'] += int(int(.05 * n) == 0)
                    assert len(set(t['repair_positions'])) == k
                    available = set()
                    for chunk in t['loaded_chunks']:
                        start, end = chunk['range']
                        chunk_positions = set(range(start, end))
                        assert not (available & chunk_positions)
                        available.update(chunk_positions)
                        assert chunk['producer_request_id'] and chunk['context_fingerprint']
                        assert chunk['namespace'] in ('pristine', 'repaired')
                        assert (chunk['namespace'] == 'pristine') == (chunk['repair_generation'] == 0)
                        source_cross_context += int(chunk['context_fingerprint'] != r['input_hash'])
                        source_moved += int(chunk['original_position'] != chunk['range'])
                    assert len(available) == n and set(t['repair_positions']) <= available
                    assert n + m == t['loaded_span_tokens']
                    assert all(x['qkv_rows'] == k + m for x in t['layers'][2:])
                    assert all(x['attention_rows'] == k + m for x in t['layers'][1:])
                    assert t['layers'][-1]['qkv_rows'] < t['layers'][-1]['kv_rows']
                    assert r['effective_reused_tokens'] == n - k
                    assert r['effective_reused_kv_bytes'] == (n - k) * 57344
                    assert r['kv_host_load_bytes'] == n * 57344
                    if r['writeback'] == 'original_only':
                        assert t['max_source_generation'] == 0
                    generations[r['writeback']] = max(generations[r['writeback']], t['max_source_generation'])
                    checks['actual_selective_repair_requests'] += 1
                else:
                    n = k = m = 0
                    assert r['effective_reused_tokens'] == 0
                    checks['full_requests'] += 1
                if r['mode'] == 'cold':
                    assert not repaired and r['engine_skip_prefill_span_tokens'] == 0
                    checks['cold_zero_hit_requests'] += 1
                key = (r['writeback'], r['length'], r['mode'], r['seed'], r['request_id'])
                budgets[key][r['policy']] = (n, k, m)
                groups[r['kind']].append(r)
                out.write(json.dumps({**r, 'model_identity': identity, 'config_sha256': cfg_hash,
                                      'raw_source_file': str(path.resolve()),
                                      'peak_hbm_bytes': None, 'h2d_bytes': None, 'd2h_bytes': None,
                                      'hardware_transfer_bytes': None,
                                      'retained_nonprefix_kv_logical_bytes': (n - k) * 2048 * 27 if repaired else 0,
                                      'cache_source': 'nonprefix_repaired' if repaired and t['max_source_generation']
                                      else 'nonprefix_pristine' if repaired else 'full_compute'}) + '\n')
                if not r['correct']:
                    failures.append({k: r[k] for k in ('case', 'request_id', 'session_id', 'kind',
                                     'length', 'mode', 'seed', 'policy', 'writeback', 'answer', 'prediction', 'text')})
            checks['validated_requests'] += len(rows)
        for kind, rr in groups.items():
            by_kind.append({'case': name, 'kind': kind, 'n': len(rr),
                            'accuracy': sum(r['correct'] for r in rr) / len(rr),
                            'ttft_p95': quantile(rr, 'ttft_s', .95)})
        c.update(ttft_p99=quantile(rows, 'ttft_s', .99),
                 ttft_mean=float(np.mean([r['ttft_s'] for r in rows])),
                 ttft_with_control_p50=quantile(rows, 'ttft_with_control_s', .5),
                 ttft_with_control_p99=quantile(rows, 'ttft_with_control_s', .99),
                 mean_repair_fraction_of_reusable=float(np.mean([r['telemetry']['ratio_realized'] for r in rows
                                                       if r['telemetry'].get('repair_invocations')] )) if c['repair_requests'] else None,
                 mean_high_layer_fraction_of_prompt=float(np.mean([r['telemetry']['layers'][-1]['qkv_rows'] / r['length']
                                                       for r in rows if r['telemetry'].get('repair_invocations')])) if c['repair_requests'] else None,
                 tpot_p50=float(np.median([(r['metrics']['last_token_ts'] - r['metrics']['first_token_ts']) /
                                           (r['output_tokens'] - 1) for r in rows if r['output_tokens'] > 1])),
                 output_tokens_per_measured_e2e_second=sum(r['output_tokens'] for r in rows) / sum(r['e2e_s'] for r in rows),
                 ttft_p95_ratio_to_prefix=c['ttft_with_control_p95'] / c['prefix_p95'],
                 prefill_gpu_ms=None, logit_kl=None, hardware_transfer_bytes=None, peak_hbm_bytes=None)
        c['answer_fidelity_to_full'] = sum(reference[(r['length'], r['mode'], r['seed'], r['request_id'])]['prediction']
                                            == r['prediction'] for r in rows) / len(rows)
        elapsed = rows[-1]['metrics']['arrival_time'] - rows[0]['metrics']['arrival_time'] + rows[-1]['e2e_s']
        assert elapsed > 0
        c['closed_loop_observed_requests_per_second'] = len(rows) / elapsed
        c['closed_loop_observed_output_tokens_per_second'] = sum(r['output_tokens'] for r in rows) / elapsed
        repaired_rows = [r for r in rows if r['telemetry'].get('repair_invocations')]
        c.update(repaired_only_ttft_with_control_p50=quantile(repaired_rows, 'ttft_with_control_s', .5),
                 repaired_only_ttft_with_control_p95=quantile(repaired_rows, 'ttft_with_control_s', .95),
                 full_request_ttft_with_control_p95=quantile([r for r in rows if not r['telemetry'].get('repair_invocations')],
                                                           'ttft_with_control_s', .95),
                 retained_nonprefix_kv_logical_bytes=sum(r['effective_reused_tokens'] * 2048 * 27 for r in repaired_rows))
    for key, methods in budgets.items():
        assert set(methods) == {'topk', 'window', 'document'}, key
        assert len(set(methods.values())) == 1, (key, methods)
        checks['equal_actual_budget_triplets'] += 1
    assert checks['validated_requests'] == 60000
    diagnose_spans(root, parent)
    write_csv(root / 'summary.csv', list(cases.values()))
    write_csv(root / 'by-kind.csv', by_kind)
    (root / 'results/failures.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in failures))
    summary['cases'] = list(cases.values())
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    gpu_mem = {int(s.split(',')[0]): int(s.split(',')[1]) for s in usage.splitlines() if int(s.split(',')[0]) < 4}
    wall_hours = (time.time() - budget['global_start_epoch']) / 3600
    validation = {'checks': dict(checks), 'max_source_generation': dict(generations),
                  'cross_context_loaded_chunks': source_cross_context, 'moved_loaded_chunks': source_moved,
                  'wall_hours_including_p0_and_retries': wall_hours,
                  'conservative_gpu_hours_upper_bound': 4 * wall_hours,
                  'gpu_memory_mib_at_finalization': gpu_mem,
                  'all_owned_workers_exit_zero': True,
                  'partial_attempts_preserved': ['results/monitor-attempt1', 'results/sanity-attempt1'],
                  'quality_pass_cases': sum(c['quality_gate'] == 'pass' for c in cases.values()),
                  'overall_pass_cases': sum(c['overall_gate'] == 'pass' for c in cases.values()),
                  'failure_predictions': len(failures),
                  'missing_metrics': ['fixed_position_logit_kl', 'GPU_ms', 'hardware_H2D_D2H_bytes',
                                      'per_request_peak_HBM'],
                 'bootstrap': '10000 paired session draws; fixed seed17; one-sided95 percentile',
                  'budget_rounding': 'max(1,floor(0.05 * reusable_tokens)); nominal5%; actual fraction reported',
                  'bootstrap_limit': 'Zero observed errors produce a degenerate empirical bootstrap; no proof of zero population risk.'}
    diagnostic_path = root / 'diagnostics/logit-kl.jsonl'
    if diagnostic_path.exists():
        diagnostic_rows = [json.loads(line) for line in diagnostic_path.open()]
        grouped = defaultdict(list)
        for r in diagnostic_rows:
            assert r['timing_eligible'] is False and r['task_accuracy_eligible'] is False
            grouped[r['case']].append(r)
        diagnostic_summary = {}
        for name, rr in grouped.items():
            diagnostic_summary[name] = {'n': len(rr),
                'kl_mean': float(np.mean([r['logit_kl_full_to_trial'] for r in rr])),
                'kl_max': max(r['logit_kl_full_to_trial'] for r in rr),
                'full_fallback_requests': sum(bool(r['telemetry'].get('fallback_full')) for r in rr)}
        validation['separate_logit_diagnostics'] = diagnostic_summary
        if len(diagnostic_rows) == 220 and len(grouped) == 6 and all(len(rr) == (20 if name == 'edit_guard05' else 40)
                                                                 for name, rr in grouped.items()):
            validation['missing_metrics'].remove('fixed_position_logit_kl')
        (root / 'diagnostics/summary.json').write_text(json.dumps(diagnostic_summary, indent=2))
    assert wall_hours <= 24 and all(v < 500 for v in gpu_mem.values()), validation
    (root / 'results/final-validation.json').write_text(json.dumps(validation, indent=2))
    plot(root, summary)
    report(root, summary, validation)
    print(json.dumps(validation, indent=2))


def diagnose_spans(root, parent):
    """Post-hoc descriptive overlap; oracle identities never enter selection."""
    data = {}
    for line in (parent / 'data/eval.jsonl').open():
        r = json.loads(line)
        if r['length'] == 8192 and r['kind'] == 'document_reorder':
            data[r['request_id']] = r
    summaries, details = [], []
    for policy in ('topk', 'window', 'document'):
        totals = defaultdict(int)
        path = root / f'results/eval-{policy}-05-original_only-8192-history-17.jsonl'
        for line in path.open():
            r = json.loads(line)
            if r['kind'] != 'document_reorder' or not r['telemetry'].get('repair_invocations'):continue
            d = data[r['request_id']]
            spans = d['document_spans']
            target = spans[1 + d['document_order'].index(d['target_entity'])]
            selected = r['telemetry']['repair_positions']
            target_count = sum(target[0] <= i < target[1] for i in selected)
            query_count = sum(spans[-1][0] <= i < spans[-1][1] for i in selected)
            totals['n'] += 1;totals['correct'] += int(r['correct'])
            totals['target_repair_tokens'] += target_count
            totals['query_suffix_repair_tokens'] += query_count
            totals['query_suffix_repair_requests'] += int(query_count > 0)
            details.append({'policy': policy, 'request_id': r['request_id'],
                            'query_repair_tokens': query_count, 'target_repair_tokens': target_count})
        summaries.append({'policy': policy, **totals})
    (root / 'results/target-span-diagnostic.json').write_text(json.dumps({
        'scope': '8K/document_reorder/history17/original-only; descriptive after evaluation',
        'fed_to_selector': False, 'rows': summaries, 'requests': details}, indent=2))


def plot(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'topk': '#dd8844', 'window': '#647cad', 'document': '#23866c'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, length in zip(axes, (8192, 16384)):
        for policy, color in colors.items():
            for wb, marker in (('original_only', 'o'), ('persistent_repair', '^')):
                cc = [c for c in summary['cases'] if c['length'] == length and c['policy'] == policy
                      and c['writeback'] == wb and c['mode'] != 'cold']
                ax.scatter([c['ttft_p95_ratio_to_prefix'] for c in cc],
                           [100 * c['loss_upper95'] for c in cc], marker=marker, color=color,
                           label=f'{policy}, {"original" if wb == "original_only" else "COW"}', s=42)
        ax.axhline(1, color='#777777', linestyle='--', linewidth=1)
        ax.axvline(1, color='#777777', linestyle='--', linewidth=1)
        ax.set(title=f'{length // 1024}K: non-cold, concurrency 1',
               xlabel='P95 TTFT / strict prefix (lower is better)',
               ylabel='Accuracy loss: one-sided 95% upper bound (pp)')
        ax.set_xlim(left=.8);ax.set_ylim(bottom=-.4)
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=8)
    fig.savefig(root / 'quality-latency.png', dpi=180)
    fig.savefig(root / 'quality-latency.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for ax, length in zip(axes, (8192, 16384)):
        labels, values, cs = [], [], []
        for policy, color in colors.items():
            for wb in ('original_only', 'persistent_repair'):
                h = summary['history'][f'{policy}|0.05|{wb}|{length}']
                labels.append(f'{policy}\n{"original" if wb == "original_only" else "COW"}')
                values.append(100 * h['variation_rate']);cs.append(color)
        ax.bar(labels, values, color=cs)
        ax.set(title=f'{length // 1024}K: 3 history seeds, 1000 identical targets', ylabel='Answer variation (%)')
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.15)
    fig.savefig(root / 'history-stability.png', dpi=180)
    fig.savefig(root / 'history-stability.svg')
    plt.close(fig)


def report(root, summary, validation):
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    rows = summary['cases']
    text = ['# P1/P2 本地非前缀 KV 实验报告', '',
            f'完成时间：{stamp}（Asia/Shanghai）。60 个正式配置、60,000 次真实 Qwen2.5-7B 请求；18 个校准配置另有1,800次请求。', '',
            f'**机制验收通过，生产优化门槛通过 {validation["overall_pass_cases"]}/60。** 质量门槛通过 {validation["quality_pass_cases"]}/60；验收要求同时满足质量损失单侧95%上界≤1个百分点及P95 TTFT优于严格Prefix。', '',
            '## 实现与范围', '',
            'P1 接通 vLLM 0.30.0 / CUDA12.9 / LMCache0.5.5 的公共逐层 CacheBlend 路径。现代4-D fused KV通过请求范围 gather/scatter 适配；不修改 Attention Kernel。',
            '这是私有实验环境中的适配实现，不是未修改LMCache的开箱性能。三策略统一使用FP32的layer1 K平方偏差；Top-K保留离散排序机制，不声称与上游原dtype评分逐位相同。',
            'P2 复用相同layer1 K偏差评分，比较离散Top-K、最大偏差连续窗口、按平均偏差排序的文档片段。既有FA3 paged API保持每个选中query的原始因果位置，排除未来token。',
            '因果见证仅证明query不直接读取其当前位置之后的KV；被移动的旧KV隐状态仍可能包含旧上下文信息，不能因此宣称非前缀缓存已严格因果等价。',
            'document分段实际覆盖chat前缀、16个文档和查询后缀，以覆盖全部可复用位置；这不是仅检索文档边界的严格论文复现。选择器不接收标准答案或target_entity。',
            'Full-produced pristine来源和COW repaired来源分离；后者只有全部28层保存成功才发布别名。跨上下文使用pristine片段仍是近似，不是严格Prefix。',
            '5%/10%/20%仅在独立100请求校准集扫描。主实验共同名义5%预算在评估前冻结；未使用正式评估调参。执行规则为max(1,floor(0.05×reusable))，三策略实际reusable/repair/mandatory预算逐目标完全一致。',
            f'其中{validation["checks"].get("minimum_one_token_budget_requests",0)}条只命中18个reusable token，修复1个，实际比例5.56%；不将名义比例描述为所有请求严格5%。',
            'GPU0–3，TP1、BF16、greedy、FA3、并发1；物理GPU按配置轮换，缓存每会话清空、会话内驻留10请求。历史条件指三个会话内请求顺序，未验证跨数百请求的无界常驻历史。', '',
            '## 正式结果', '',
            '|策略|写回|长度|连续正确率|历史正确率范围|三顺序答案变动率|连续P95 TTFT / Prefix|',
            '|---|---|---:|---:|---:|---:|---:|']
    for length in (8192, 16384):
        for policy in ('topk', 'window', 'document'):
            for wb in ('original_only', 'persistent_repair'):
                cc = [c for c in rows if c['length'] == length and c['policy'] == policy and c['writeback'] == wb]
                cont = next(c for c in cc if c['mode'] == 'continuous')
                hh = [c['accuracy'] for c in cc if c['mode'] == 'history']
                hist = summary['history'][f'{policy}|0.05|{wb}|{length}']
                text.append(f'|{policy}|{wb}|{length}|{cont["accuracy"]:.1%}|{min(hh):.1%}–{max(hh):.1%}|{hist["variation_rate"]:.1%}|{cont["ttft_with_control_p95"]:.3f}s / {cont["prefix_p95"]:.3f}s|')
    text += ['', '逐配置原始数字、bootstrap上界、P50/P95/P99、TPOT及吞吐见summary.csv；五种变更分别见by-kind.csv。', '',
             '![质量与尾延迟](quality-latency.png)', '', '![历史答案变化](history-stability.png)', '',
             '## 解释与限制', '',
             '文档与窗口结果必须分别判断；连续性本身不足以保证保留当前问题需要的事实。当前排序及模板使文档策略更有利，但该机制实验不能确证跨真实文本或模型的收益。',
             '事后位置核对（8K文档换序、history17、original-only）：document对180/180个修复请求选择查询后缀，共2880个后缀token；Top-K/window均为0/180。优势可能部分来自查询片段被保护，不能归因为文档边界或连续性本身。核对保存在results/target-span-diagnostic.json，未反馈选择器或改动正式配置。',
            'P95包含每会话首个冷请求以及缓存保存开销。CPU分层复制和Python控制的实际成本保留在测量中；没有删除慢请求来制造加速。GPU核时间和PCIe计数缺失，不能将尾部回退因果归给某一种硬件瓶颈。',
             'summary.csv另列repaired-only的P50/P95与Full请求P95，作为冷热分解诊断；条件统计不替代包含全部目标的验收门槛。每个worker使用20GB CPU缓存池，未启用SSD/RDMA或P4联合调度。',
             '控制RPC计入ttft_with_control；telemetry collector RPC位于计时外。没有HTTP服务、网络或并发4/8负载结果。基线来自此前同模型/输入/参数/FA3的P0；两阶段时间不同，无同时对照的漂移控制。',
             'original-only与COW配置按固定顺序执行，后期活跃worker数下降，共享CPU/PCIe负载不恒定。COW时延较低的观测不能单独归因于写回；下一轮需交错执行并固定主机负载。',
             'closed_loop_observed吞吐由该配置首末请求的实测arrival_time和最后E2E计算，包含请求间控制/收集/缓存清空间隔；output_tokens_per_measured_e2e_second只统计模型调用区间，两者分开列出。',
             'effective_reused_kv_bytes是实际未选择修复的位置乘模型KV格式的逻辑字节，不代表所有层省掉同等计算。layer0和layer1探测投影仍处理完整加载范围；高层真实QKV行数另列。',
             'layer0的KV全部被当前结果覆盖，因此另列retained_nonprefix_kv_logical_bytes=(reusable−repair)×2048×27，只计layer1–27真正保留的旧K/V位置；该逻辑计数也不等价于硬件带宽或算力节省。',
             'kv_host_load_bytes由实际加载范围及dtype统计复制负载，是主机加载的数据体积，不是PCIe/RDMA硬件计数。引擎skip-prefill span包含无缓存孔洞，不能直接当成命中token数。',
             '原始正式JSONL的logit_kl与prefill_gpu_ms为null，未采集的峰值HBM和硬件传输字段缺省；normalized补充记录中peak_hbm_bytes、h2d_bytes、d2h_bytes、hardware_transfer_bytes显式为null。另行KL诊断不混入性能统计。LongBench-v2只在P0测试13个全文适配样本，未获得P1/P2真实文本外推证据。',
             'edit_window尚无独立大规模GPU质量/性能矩阵；下游覆盖回退的短诊断只用于验证安全分支。',
             '独立标准答案来自结构化权威记录，不依赖模型输出。Full基线100%正确带来明显天花板；零损失配置的经验bootstrap退化为0，不能证明总体零风险。',
             '跨模型、并发扩展按先验条件仅对达标配置执行；本轮未得到同时达标配置，因此取消。', '',
             '## 失败记录与完整性', '',
             '所有错误答案保留在原始JSONL及results/failures.jsonl。第一次正式监督器因小样本CI为None的报告格式异常中断；相关部分结果保存在results/monitor-attempt1，正式矩阵重新完整执行，不混入统计。',
             '独立审计在最终验收执行前发现floor预算检查遗漏正预算最少1 token规则，已修正CPU验收；没有改动45条低命中请求或其计时/预测。',
             '环境兼容性和冻结生成器修复的失败尝试完整保留；失败耗时计入全局预算。最终独立环境复核见logs/final-environment-report.md：5条原样命令通过、40真实请求正确，只认证小样本机制。原P0环境及源运行环境保留。',
             'GPU0完成正式任务后运行了40请求独立复核，与最后GPU1/3任务短暂并行；共享主机资源可能影响这段时延，未作为严格隔离的硬件因果比较。', '',
             '## 资源与下一阶段', '',
             f'全局墙钟（含P0、环境准备和失败重试）{validation["wall_hours_including_p0_and_retries"]:.2f}h；保守GPU时上界{validation["conservative_gpu_hours_upper_bound"]:.2f}，低于24h/96GPUh。GPU0–3收尾显存：{validation["gpu_memory_mib_at_finalization"]} MiB。',
             '当前建议：保留严格Prefix作为生产默认。下一轮首先加入查询后缀始终重算的统一对照，隔离查询保护与文档连续性的作用；再优化适配器的缓存保存、搬运成本，并用真实文本和长会话复核。现有结果不足以启动P3/P4生产改造。']
    kind_errors = defaultdict(lambda: [0, 0])
    for row in csv.DictReader((root / 'by-kind.csv').open()):
        n = int(row['n'])
        kind_errors[row['kind']][0] += round((1 - float(row['accuracy'])) * n)
        kind_errors[row['kind']][1] += n
    text += ['', '## 错误按变更类型汇总', '', '|变更|错误数|请求数|', '|---|---:|---:|']
    for kind, (errors, n) in kind_errors.items():
        text.append(f'|{kind}|{errors}|{n}|')
    text += ['', '局部改值通常使对应chunk失效并被强制重算；这四类较简单任务的高正确率不单独证明复杂下游依赖已被近似修复。文档换序保持chunk内容，暴露位置与前置上下文依赖，是本矩阵的重要压力条件。']
    if validation.get('separate_logit_diagnostics'):
        text += ['', '## 独立Logit KL与编辑回退诊断', '',
                 '在8K local_edit和document_reorder各两个会话、共40个固定目标prompt末端位置，单独读取全词表raw logprobs，计算KL(Full||Trial)；edit_guard只对20个local_edit目标执行。只生成一个token；不计入正确率及性能。此事后诊断不用于参数选择或达标判断。原始分布NPZ和逐请求JSONL在diagnostics/。',
                 'edit_guard05以首次变化位置到末端查询作为保守依赖范围；超出预算时重算全部可复用位置。该分支仍经过缓存加载，不能当作最快的原生Full回退。', '',
                 '|诊断|n|KL均值|KL最大值|全重算回退次数|', '|---|---:|---:|---:|---:|']
        for name, d in validation['separate_logit_diagnostics'].items():
            text.append(f'|{name}|{d["n"]}|{d["kl_mean"]:.6g}|{d["kl_max"]:.6g}|{d["full_fallback_requests"]}|')
    rendered = '\n'.join(text) + '\n'
    (root / 'EXPERIMENT_REPORT.md').write_text(rendered)
    (root / f'EXPERIMENT_REPORT_{stamp}.md').write_text(rendered)
    (root / 'EXPERIMENT_TRACKER.md').write_text('\n'.join([
        '# P1/P2 Tracker', '', '|阶段|状态|证据|', '|---|---|---|',
        '|P1 跨位置/前置上下文真实复用|通过|canary + packed/causal独立见证|',
        '|P2 Top-K/window/document真实选择性计算|通过|60,000请求；43,200次修复逐28层计数，20,000组等预算配对|',
        '|18校准配置|完成|frozen-selection.json，1,800请求|',
        '|60正式配置|完成|summary.csv，60,000请求|',
        f'|联合质量与P95验收|{validation["overall_pass_cases"]}/60通过|逐配置quality_gate/latency_gate|',
        '|edit_window GPU独立矩阵|未完成|仅短诊断中的覆盖回退分支|',
        '|Logit KL诊断|' + ('完成' if validation.get('separate_logit_diagnostics') else '未采集') + '|diagnostics/，与计时分离|',
        '|并发4/8与跨模型|取消|未得到联合达标配置|',
        '|P3/P4|不建议启动|需要先解决时延和外部有效性|',
        '|进程清理|完成|final-validation.json，仅清理自有进程组|', '']) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    finalize(args.root.resolve())
