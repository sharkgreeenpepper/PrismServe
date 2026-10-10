"""Real layerwise KV replay with per-request controls, source provenance, and counts."""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from run_engine import rows_for,plan_order,parse_answer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert gpu in ('0','1','2','3')
    root=a.run_root
    budget=json.loads((root/'budget.json').read_text())
    os.environ.update(LMCACHE_ENABLE_BLENDING='True',LMCACHE_USE_LAYERWISE='True',
                      LMCACHE_CONFIG_FILE=str((root/'configs/probe-blend.yaml').resolve()),
                      LMCACHE_EXTRA_CONFIG='{"enable_sparse":false}')
    from vllm import LLM,SamplingParams
    from vllm.config import KVTransferConfig
    llm=LLM(model='/home/bumi/infra/models/Qwen/Qwen2.5-7B-Instruct',dtype='bfloat16',
            tensor_parallel_size=1,max_model_len=17408,max_num_batched_tokens=17408,
            max_num_seqs=8,gpu_memory_utilization=.50,enforce_eager=True,
            enable_prefix_caching=False,disable_log_stats=False,seed=17,
            attention_config={'backend':'FLASH_ATTN'},
            kv_transfer_config=KVTransferConfig(kv_connector='LMCacheConnectorV1',kv_role='kv_both'))
    params=SamplingParams(temperature=0,max_tokens=32,seed=17)
    cases=json.loads(a.cases.read_text())
    def rpc(payload):return llm.collective_rpc('prismserve_control',args=(payload,))[0]
    # Warm actual repair shapes, then reset both sources and native prefix caches.
    warm=rows_for(a.data,cases[0]['length'],3)
    for row in warm:
        ctx={k:row[k] for k in ('request_id','input_hash','document_spans')}
        ctx.update(policy='topk',ratio=.05,writeback='original_only')
        rpc({'action':'context','request':ctx})
        llm.generate({'prompt_token_ids':row['prompt_token_ids']},params,use_tqdm=False)
    rpc({'action':'clear'});assert llm.reset_prefix_cache()
    for case in cases:
        if time.time()>=budget['deadline_epoch']-180:raise TimeoutError('Global 24h budget reached')
        rows=rows_for(a.data,case['length'],a.limit)
        if 'kinds' in case:rows=[r for r in rows if r['kind'] in case['kinds']]
        path=root/'results'/f'{case["name"]}.jsonl'
        if path.exists():raise RuntimeError(f'Refuse to overwrite {path}')
        path.with_suffix('.config.json').write_text(json.dumps({**case,'gpu':int(gpu),'data':str(a.data)},indent=2))
        count=0
        with path.open('w',buffering=1) as out:
            for session,group in plan_order(rows,case['mode'],case['seed']):
                rpc({'action':'clear'});assert llm.reset_prefix_cache()
                for row in group:
                    if time.time()>=budget['deadline_epoch']-120:raise TimeoutError('Budget reached')
                    if case['mode']=='cold':rpc({'action':'clear'})
                    ctx={k:row[k] for k in ('request_id','input_hash','document_spans')}
                    ctx.update(policy=case['policy'],ratio=case['ratio'],writeback=case['writeback'])
                    begin=time.perf_counter()
                    rpc({'action':'context','request':ctx,'force_full':case.get('force_full',False),
                         'no_store':case.get('force_full',False)})
                    control_time=time.perf_counter()-begin
                    record={k:v for k,v in row.items() if k not in ('prompt_token_ids','document_spans')}
                    record.update(case=case['name'],policy=case['policy'],ratio=case['ratio'],
                                  writeback=case['writeback'],mode=case['mode'],seed=case['seed'],gpu=int(gpu),
                                  status='ok',error=None,logit_kl=None,prefill_gpu_ms=None)
                    generation_start=time.perf_counter()
                    try:
                        result=llm.generate({'prompt_token_ids':row['prompt_token_ids']},params,use_tqdm=False)[0]
                        e2e=time.perf_counter()-generation_start
                        telemetry=rpc({'action':'telemetry'})
                        text=result.outputs[0].text
                        record.update(text=text,prediction=parse_answer(text,row.get('answer_type','amount')),
                                      output_tokens=len(result.outputs[0].token_ids),
                                      output_token_ids=list(result.outputs[0].token_ids),
                                      correct=parse_answer(text,row.get('answer_type','amount'))==row['answer'],
                                      e2e_s=e2e,control_rpc_s=control_time,
                                      end_to_end_controller_s=control_time+e2e,
                                      ttft_s=result.metrics.first_token_latency,
                                      ttft_with_control_s=control_time+result.metrics.first_token_latency,
                                      metrics=dataclasses.asdict(result.metrics),
                                      engine_skip_prefill_span_tokens=result.num_cached_tokens,
                                      telemetry=telemetry)
                        if telemetry.get('repair_invocations'):
                            assert len(telemetry['layers'])==28
                            assert telemetry['attention_backend'].startswith('FA3')
                            n=telemetry['reusable_tokens'];k=telemetry['repair_tokens']
                            assert k==telemetry['repair_budget'] or telemetry['fallback_full']
                            record['effective_reused_tokens']=n-k
                            record['effective_reused_kv_bytes']=(n-k)*57344
                            record['kv_host_load_bytes']=n*57344
                        else:
                            record.update(effective_reused_tokens=0,effective_reused_kv_bytes=0,kv_host_load_bytes=0)
                    except Exception as exc:
                        record.update(status='error',error=repr(exc),correct=False)
                        out.write(json.dumps(record)+'\n');raise
                    out.write(json.dumps(record)+'\n');count+=1
                    if count%100==0:print(json.dumps({'case':case['name'],'completed':count,'correct':record['correct']}),flush=True)
        print('CASE_COMPLETE',case['name'],count,flush=True)
    rpc({'action':'clear'})


if __name__=='__main__':main()
