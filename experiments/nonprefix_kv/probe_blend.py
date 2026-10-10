"""Compatibility witness for the public in-process CacheBlend path."""
import json
import os
import time
import argparse
from pathlib import Path


def clear_worker(worker):
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is None:
        raise RuntimeError('LMCache worker engine missing')
    return engine.clear()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,default=Path('.aris/runs/nonprefix-kv-20261008'))
    p.add_argument('--data',type=Path,default=Path('.aris/runs/nonprefix-kv-20261008/data/eval.jsonl'))
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--canary',action='store_true')
    p.add_argument('--policy',choices=['topk','window','document'],default='topk')
    p.add_argument('--ratio',type=float,default=.05)
    p.add_argument('--writeback',choices=['original_only','persistent_repair'],default='original_only')
    p.add_argument('--probe-name',default='blend-probe')
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2','3')
    budget=json.loads((a.run_root/'budget.json').read_text())
    env_deadline=budget.get('environment_deadline_epoch',budget.get('start_epoch',0)+10800)
    if time.time()>env_deadline:
        raise TimeoutError('The approved 3-hour environment window has expired; preserve blocked status')
    os.environ.update(LMCACHE_CHUNK_SIZE='256', LMCACHE_ENABLE_BLENDING='True',
                      LMCACHE_BLEND_SPECIAL_STR=' # # ', LMCACHE_USE_LAYERWISE='True',
                      LMCACHE_BLEND_CHECK_LAYERS='1', LMCACHE_BLEND_RECOMPUTE_RATIOS='0.05',
                      LMCACHE_LOCAL_CPU='True', LMCACHE_MAX_LOCAL_CPU_SIZE='20',
                      LMCACHE_EXTRA_CONFIG='{"enable_sparse":false}')
    os.environ.pop('LMCACHE_BLEND_SPECIAL_STR')
    os.environ['LMCACHE_CONFIG_FILE']=str((a.run_root/'configs/probe-blend.yaml').resolve())
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from run_engine import rows_for
    root = a.run_root
    rows = rows_for(a.data,8192,a.limit)
    if a.canary:
        from transformers import AutoTokenizer
        import hashlib
        t=AutoTokenizer.from_pretrained('/home/bumi/infra/models/Qwen/Qwen2.5-7B-Instruct')
        enc=lambda s:t.encode(s,add_special_tokens=False)
        prefix=enc('<|im_start|>system\nRead records and return only the requested four digit amount.<|im_end|>\n<|im_start|>user\n')
        suffix=enc('<|im_end|>\n<|im_start|>assistant\n')
        sep=enc(' # #')
        docs=[enc(f'\nRecord ITEM-{i:02d} has current amount {3456+i:04d}.\n'+
                  (f'Background for item {i}: packaging and delivery records are archived. '*40)) for i in range(4)]
        rows=[]
        for j,order in enumerate(([0,1,2,3],[2,0,3,1],[3,2,1,0])):
            tokens=prefix+sep
            for i in order:tokens+=docs[i]+sep
            tokens+=enc(f'\nQuery number {j}: what is the amount for ITEM-00?\n')+suffix
            rows.append({'prompt_token_ids':tokens,'input_hash':hashlib.sha256(json.dumps(tokens).encode()).hexdigest()})
    llm = LLM(model='/home/bumi/infra/models/Qwen/Qwen2.5-7B-Instruct',
              dtype='bfloat16', tensor_parallel_size=1, max_model_len=17408,
              max_num_batched_tokens=17408, max_num_seqs=8, gpu_memory_utilization=.5,
              enforce_eager=True, enable_prefix_caching=False, disable_log_stats=False,
              seed=17, attention_config={'backend':'FLASH_ATTN'},
              kv_transfer_config=KVTransferConfig(kv_connector='LMCacheConnectorV1',kv_role='kv_both'))
    sampling = SamplingParams(temperature=0,max_tokens=32,seed=17)
    with (root/'results'/f'{a.probe_name}.jsonl').open('w',buffering=1) as out:
        for j,row in enumerate(rows):
            context={'request_id':row.get('request_id',f'canary-{j}'),'input_hash':row['input_hash'],
                     'policy':a.policy,'ratio':a.ratio,'writeback':a.writeback,
                     'document_spans':row.get('document_spans',[(0,len(row['prompt_token_ids']))])}
            llm.collective_rpc('prismserve_control',args=({'action':'context','request':context},))
            begin=time.perf_counter()
            result=llm.generate({'prompt_token_ids':row['prompt_token_ids']},sampling,use_tqdm=False)[0]
            elapsed=time.perf_counter()-begin
            telemetry=llm.collective_rpc('prismserve_control',args=({'action':'telemetry'},))[0]
            out.write(json.dumps({'input_hash':row['input_hash'],'text':result.outputs[0].text,
                                  'cached_tokens':result.num_cached_tokens,'e2e_s':elapsed,
                                  'telemetry':telemetry})+'\n')
    print('CLEAR',llm.collective_rpc('prismserve_control',args=({'action':'clear'},)),flush=True)
    print('BLEND_PROBE_COMPLETE',flush=True)
