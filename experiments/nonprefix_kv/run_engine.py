"""Real vLLM single-engine replay. No surrogate caches or timing estimates."""
import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path


def rows_for(path, length, limit):
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row['length'] == length:
                rows.append(row)
    return rows[:limit] if limit else rows


def plan_order(rows, mode, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row['session_id'], []).append(row)
    rng = random.Random(seed)
    for session, group in groups.items():
        group.sort(key=lambda x: x['step'])
        if mode == 'history':
            rng.shuffle(group)
        yield session, group


def parse_answer(text, answer_type='amount'):
    # Reject multiple answers or explanations; whitespace/JSON scalar wrappers allowed.
    pattern = r'\s*"?([ABCD])"?\s*' if answer_type == 'choice' else r'\s*"?(\d{4})"?\s*'
    m = re.fullmatch(pattern, text)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['full', 'prefix', 'topk', 'window', 'document', 'edit_window'], required=True)
    p.add_argument('--mode', choices=['cold', 'continuous', 'history'], default='continuous')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--length', type=int, default=8192)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--ratio', type=float, default=.05)
    p.add_argument('--writeback', choices=['original_only','persistent_repair'], default='original_only')
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    ids = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(ids) != 1 or ids[0] not in ('0','1','2','3'):
        raise RuntimeError('Exactly one authorized GPU 0–3 must be explicitly bound')
    rows = rows_for(a.data, a.length, a.limit)
    if not rows:
        raise ValueError('Empty dataset')
    if a.arm not in ('full','prefix'):
        raise RuntimeError('P1/P2 blocked: see compatibility.json; no validated repair runtime exists')
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    model_cfg=AutoConfig.from_pretrained(a.model,local_files_only=True)
    kv_bytes_per_token=(2 * model_cfg.num_hidden_layers * model_cfg.num_key_value_heads
                        * (model_cfg.hidden_size//model_cfg.num_attention_heads) * 2)
    kw = dict(model=a.model, tokenizer=a.model, dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=17408, max_num_batched_tokens=17408, max_num_seqs=8,
              gpu_memory_utilization=.50, enforce_eager=True,
              enable_prefix_caching=a.arm == 'prefix', seed=17,
              disable_log_stats=False, attention_config={'backend':'FLASH_ATTN'})
    llm = LLM(**kw)
    sampling = SamplingParams(temperature=0, max_tokens=32, seed=17)
    llm.generate({'prompt_token_ids':rows[0]['prompt_token_ids']}, sampling, use_tqdm=False)
    if not llm.reset_prefix_cache():
        raise RuntimeError('Prefix cache reset failed after warmup')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(gpu=int(ids[0]), timing_source='vLLM RequestStateStats + monotonic host E2E',
                  generation={'temperature':0,'max_tokens':32,'seed':17}, engine=kw)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True,default=str).encode()).hexdigest()
    a.output.with_suffix('.config.json').write_text(json.dumps(config,indent=2,default=str))
    completed = 0
    with a.output.open('w', buffering=1) as out:
        for session, group in plan_order(rows, a.mode, a.seed):
            if not llm.reset_prefix_cache():
                raise RuntimeError('Session cache reset failed')
            history_hashes = []
            for row in group:
                if time.time() >= a.deadline - 120:
                    raise TimeoutError('Global budget reached; reserve cleanup time')
                if a.mode == 'cold' and not llm.reset_prefix_cache():
                    raise RuntimeError('Cold cache reset failed')
                record = {k:v for k,v in row.items() if k not in ('prompt_token_ids','document_spans')}
                record.update(arm=a.arm, mode=a.mode, seed=a.seed, gpu=int(ids[0]),
                              config_hash=config_hash, status='ok', error=None,
                              prior_input_hashes=history_hashes.copy(), cache_source='none' if a.arm=='full' else 'exact_prefix',
                              repair_generation=0, repair_tokens=0, nonprefix_reuse_bytes=0,
                              h2d_bytes=None,d2h_bytes=None,prefill_gpu_ms=None,peak_hbm_bytes=None,
                              logit_kl=None)
                record['cache_policy']='none' if a.arm=='full' else 'exact_prefix'
                start = time.perf_counter()
                try:
                    generated = llm.generate({'prompt_token_ids':row['prompt_token_ids']}, sampling, use_tqdm=False)[0]
                    record['e2e_s'] = time.perf_counter() - start
                    output = generated.outputs[0]
                    record.update(text=output.text, prediction=parse_answer(output.text,row.get('answer_type','amount')),
                                  output_tokens=len(output.token_ids), output_token_ids=list(output.token_ids),
                                  finish_reason=output.finish_reason, cached_tokens=generated.num_cached_tokens)
                    record['correct'] = record['prediction'] == row['answer']
                    record['cache_source']=('unknown' if generated.num_cached_tokens is None else
                                            'exact_prefix' if generated.num_cached_tokens>0 else 'none')
                    record['metrics'] = dataclasses.asdict(generated.metrics) if generated.metrics else None
                    record['ttft_s'] = generated.metrics.first_token_latency if generated.metrics else None
                    m = generated.metrics
                    record['tpot_s'] = ((m.last_token_ts-m.first_token_ts)/(len(output.token_ids)-1)
                                        if m and len(output.token_ids)>1 else None)
                    record['actual_prefill_tokens'] = (a.length-generated.num_cached_tokens
                                                       if generated.num_cached_tokens is not None else None)
                    # Logical KV payload computed from model config; not measured transfer bytes.
                    record['logical_exact_reuse_bytes'] = (generated.num_cached_tokens * kv_bytes_per_token
                                                           if generated.num_cached_tokens is not None else None)
                    if a.arm=='full' and (generated.num_cached_tokens or 0) != 0:
                        raise RuntimeError('Full baseline unexpectedly reused cache')
                except Exception as exc:
                    record.update(status='error',error=repr(exc),correct=False,e2e_s=time.perf_counter()-start)
                    out.write(json.dumps(record)+'\n')
                    raise
                out.write(json.dumps(record)+'\n')
                history_hashes.append(row['input_hash'])
                completed += 1
                if completed % 100 == 0 or a.smoke:
                    print(json.dumps({'completed':completed,'request':row['request_id'],
                                      'ttft':record['ttft_s'],'cached':record['cached_tokens'],
                                      'correct':record['correct']}),flush=True)
    print(f'COMPLETE {completed}', flush=True)


if __name__ == '__main__':
    main()
