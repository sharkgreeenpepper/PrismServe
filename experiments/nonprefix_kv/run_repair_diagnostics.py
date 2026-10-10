"""Separate untimed full-vocabulary KL and conservative edit fallback witness.

One next-token distribution is captured at the fixed end-of-prompt position.
These short diagnostics do not count as task-accuracy or performance evidence.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from run_engine import rows_for, plan_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1', '2', '3')
    status = json.loads((root / 'status.json').read_text())
    assert status['state'] == 'complete', 'Diagnostics must wait for the formal matrix'
    budget = json.loads((root / 'budget.json').read_text())
    assert time.time() < budget['deadline_epoch'] - 600
    outdir = root / 'diagnostics'
    outdir.mkdir(exist_ok=True)
    assert not (outdir / 'logit-kl.jsonl').exists(), 'Preserve existing diagnostics'
    os.environ.update(LMCACHE_ENABLE_BLENDING='True', LMCACHE_USE_LAYERWISE='True',
                      LMCACHE_CONFIG_FILE=str(root / 'configs/probe-blend.yaml'),
                      LMCACHE_EXTRA_CONFIG='{"enable_sparse":false}')
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    llm = LLM(model='/home/bumi/infra/models/Qwen/Qwen2.5-7B-Instruct', dtype='bfloat16',
              tensor_parallel_size=1, max_model_len=17408, max_num_batched_tokens=17408,
              max_num_seqs=8, gpu_memory_utilization=.50, enforce_eager=True,
              enable_prefix_caching=False, disable_log_stats=False, seed=17,
              max_logprobs=-1, logprobs_mode='raw_logprobs',
              attention_config={'backend': 'FLASH_ATTN'},
              kv_transfer_config=KVTransferConfig(kv_connector='LMCacheConnectorV1', kv_role='kv_both'))
    params = SamplingParams(temperature=0, max_tokens=1, seed=17, logprobs=-1)
    def rpc(payload):
        return llm.collective_rpc('prismserve_control', args=(payload,))[0]
    all_rows = [r for r in rows_for(args.data, 8192, 0)
                if r['kind'] in ('local_edit', 'document_reorder') and r['session_id'].endswith(('-00', '-01'))]
    assert len(all_rows) == 40
    full = {}
    policies = [('full', 'topk', 1., True), ('repair100', 'topk', 1., False),
                ('topk05', 'topk', .05, False), ('window05', 'window', .05, False),
                ('document05', 'document', .05, False), ('edit_guard05', 'edit_window', .05, False)]
    with (outdir / 'logit-kl.jsonl').open('w', buffering=1) as out:
        for name, policy, ratio, force_full in policies:
            diagnostic_rows = [r for r in all_rows if policy != 'edit_window' or r['kind'] == 'local_edit']
            for _, group in plan_order(diagnostic_rows, 'continuous', 17):
                rpc({'action': 'clear'});assert llm.reset_prefix_cache()
                previous = None
                for row in group:
                    assert time.time() < budget['deadline_epoch'] - 180
                    ctx = {k: row[k] for k in ('request_id', 'input_hash', 'document_spans')}
                    ctx.update(policy=policy, ratio=ratio, writeback='original_only')
                    if policy == 'edit_window' and previous is not None:
                        changed = [i for i, (a, b) in enumerate(zip(previous, row['prompt_token_ids'])) if a != b]
                        # The query at prompt end consumes the edited fact; use
                        # a conservative causal span rather than attention as proof.
                        assert changed
                        ctx.update(edit_start=min(changed), dependency_end=row['length'])
                    rpc({'action': 'context', 'request': ctx, 'force_full': force_full, 'no_store': force_full})
                    result = llm.generate({'prompt_token_ids': row['prompt_token_ids']}, params, use_tqdm=False)[0]
                    telemetry = rpc({'action': 'telemetry'})
                    probs = result.outputs[0].logprobs[0]
                    ids = np.array(sorted(probs), dtype=np.int64)
                    logp = np.array([probs[int(i)].logprob for i in ids], dtype=np.float64)
                    assert len(ids) > 150000 and np.isfinite(logp).all()
                    assert abs(np.exp(logp).sum() - 1) < 1e-4
                    np.savez_compressed(outdir / f'{name}-{row["request_id"]}.npz', token_ids=ids, raw_logprobs=logp)
                    if force_full:
                        full[row['request_id']] = (ids, logp)
                    ref_ids, ref_lp = full[row['request_id']]
                    assert np.array_equal(ids, ref_ids)
                    kl = float(np.sum(np.exp(ref_lp) * (ref_lp - logp)))
                    assert kl >= -1e-6
                    if policy == 'edit_window' and telemetry.get('repair_invocations'):
                        assert telemetry['fallback_full']
                        assert telemetry['repair_tokens'] == telemetry['reusable_tokens']
                        assert all(l['qkv_rows'] == l['kv_rows'] for l in telemetry['layers'])
                    out.write(json.dumps({'case': name, 'request_id': row['request_id'],
                                          'kind': row['kind'],
                                          'input_hash': row['input_hash'], 'fixed_target_position': row['length'],
                                          'vocabulary_tokens': len(ids), 'logit_kl_full_to_trial': max(0., kl),
                                          'argmax_token': int(ids[np.argmax(logp)]), 'telemetry': telemetry,
                                          'timing_eligible': False, 'task_accuracy_eligible': False}) + '\n')
                    previous = row['prompt_token_ids']
            print('DIAGNOSTIC_COMPLETE', name, flush=True)
    rpc({'action': 'clear'})


if __name__ == '__main__':
    main()
