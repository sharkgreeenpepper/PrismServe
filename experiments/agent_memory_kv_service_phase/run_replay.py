"""Native/Blend paired real inference; no SQLite or Mem0 writes."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import time
from contracts import Geometry


def score(text, target):
    if target is None or target['type'] == 'task_success':
        return None
    answer = text.strip()
    return answer == str(target['value']).strip()


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    config = json.loads((args.model / 'config.json').read_text());geometry = Geometry.from_config(config)
    blend = args.arm in ('blend', 'blend_cow')
    if blend:
        geometry.require_blend()
        if args.lmcache_config is None:raise ValueError('Blend config required')
        os.environ.update(LMCACHE_ENABLE_BLENDING='True', LMCACHE_USE_LAYERWISE='True',
                          LMCACHE_EXTRA_CONFIG='{"enable_sparse":false}', LMCACHE_CONFIG_FILE=str(args.lmcache_config.resolve()))
    else:
        os.environ['LMCACHE_ENABLE_BLENDING'] = 'False'
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    kw = dict(model=str(args.model.resolve()), dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=args.max_model_len, max_num_batched_tokens=args.max_model_len,
              max_num_seqs=8, gpu_memory_utilization=.50, enforce_eager=True,
              enable_prefix_caching=args.arm == 'strict_prefix', disable_log_stats=False, seed=args.seed)
    if not geometry.hybrid:kw['attention_config'] = {'backend': 'FLASH_ATTN'}
    if blend:kw['kv_transfer_config'] = KVTransferConfig(kv_connector='LMCacheConnectorV1', kv_role='kv_both')
    if args.capture_logprobs:
        if args.max_output_tokens != 1:
            raise ValueError('Logit diagnostics require one output token')
        kw.update(max_logprobs=-1, logprobs_mode='raw_logprobs')
    try:
        llm = LLM(**kw)
    except Exception as exc:
        args.output.with_suffix('.startup-error.json').write_text(json.dumps({
            'status': 'startup_error', 'error': repr(exc), 'model': str(args.model.resolve()),
            'arm': args.arm, 'gpu': args.gpu, 'epoch': time.time()}, indent=2))
        raise
    params = SamplingParams(temperature=0, max_tokens=args.max_output_tokens, seed=args.seed,
                            logprobs=-1 if args.capture_logprobs else None)
    rows = [json.loads(line) for line in args.inputs.open() if line.strip()]
    if args.limit:rows = rows[:args.limit]
    if any(len(r['prompt_token_ids']) + args.max_output_tokens > args.max_model_len for r in rows):
        raise ValueError('Input would exceed context limit; do not truncate memory')
    def rpc(payload):return llm.collective_rpc('prismserve_control', args=(payload,))[0]
    def clear():
        if blend:rpc({'action': 'clear'})
        assert llm.reset_prefix_cache()
    def context(row):
        if not blend:return
        segments = [s['range'] for s in row['segments']]
        rpc({'action': 'context', 'request': {'request_id': row['request_id'], 'input_hash': row['input_hash'],
             'document_spans': segments, 'query_span': row['query_span'], 'policy': args.policy,
             'ratio': args.ratio, 'writeback': 'persistent_repair' if args.arm == 'blend_cow' else 'original_only'}})
    # Warm the actual model separately, then discard all resulting caches.
    for row in rows[:2]:
        context(row);llm.generate({'prompt_token_ids': row['prompt_token_ids']}, params, use_tqdm=False)
    clear()
    session = None
    with args.output.open('w', buffering=1) as out:
        for row in rows:
            try:
                if row['session_id'] != session or args.cold:
                    clear();session = row['session_id']
                start_control = time.perf_counter();context(row)
                control_s = time.perf_counter() - start_control
                started = time.perf_counter()
                result = llm.generate({'prompt_token_ids': row['prompt_token_ids']}, params, use_tqdm=False)[0]
                e2e = time.perf_counter() - started
                telemetry = rpc({'action': 'telemetry'}) if blend else None
                if telemetry and telemetry.get('repair_invocations'):
                    assert len(telemetry['layers']) == geometry.layers
                    assert all(c['range'][1] <= row['query_span'][0] for c in telemetry['loaded_chunks'])
            except Exception as exc:
                out.write(json.dumps({'request_id': row['request_id'], 'session_id': row['session_id'],
                                      'input_hash': row['input_hash'], 'arm': args.arm, 'gpu': args.gpu,
                                      'model': str(args.model.resolve()), 'status': 'error',
                                      'error': repr(exc), 'correct': None, 'epoch': time.time()}) + '\n')
                raise
            try:
                logit_artifact = None
                if args.capture_logprobs:
                    import numpy as np
                    probabilities = result.outputs[0].logprobs[0]
                    token_ids = np.array(sorted(probabilities), dtype=np.int64)
                    logp = np.array([probabilities[int(i)].logprob for i in token_ids], dtype=np.float64)
                    assert len(token_ids) >= 150000 and np.isfinite(logp).all()
                    assert abs(np.exp(logp).sum() - 1) < 1e-4
                    artifact_dir = args.output.parent / (args.output.stem + '-logits')
                    artifact_dir.mkdir(exist_ok=True)
                    artifact = artifact_dir / (row['input_hash'] + '.npz')
                    if artifact.exists():raise FileExistsError(artifact)
                    np.savez_compressed(artifact, token_ids=token_ids, raw_logprobs=logp)
                    logit_artifact = str(artifact.resolve())
                record = {'request_id': row['request_id'], 'session_id': row['session_id'], 'input_hash': row['input_hash'],
                          'origin': row['origin'], 'split': row['split'], 'model': str(args.model.resolve()),
                          'model_geometry': dataclasses.asdict(geometry), 'arm': args.arm, 'policy': args.policy,
                          'ratio': args.ratio, 'gpu': args.gpu, 'seed': args.seed, 'status': 'ok',
                          'rope_table_device_cache': os.environ.get('PRISMSERVE_CACHE_ROPE_TABLE') == '1',
                          'group_contiguous_queries': os.environ.get('PRISMSERVE_GROUP_CONTIGUOUS_Q') == '1',
                          'text': result.outputs[0].text, 'output_token_ids': list(result.outputs[0].token_ids),
                          'correct': None if args.capture_logprobs else score(result.outputs[0].text, row.get('target')), 'target': row.get('target'),
                          'logit_artifact': logit_artifact, 'fixed_target_position': len(row['prompt_token_ids']),
                          'timing_eligible': not args.capture_logprobs, 'task_accuracy_eligible': not args.capture_logprobs,
                          'ttft_s': result.metrics.first_token_latency, 'context_rpc_s': control_s,
                          'ttft_with_context_s': result.metrics.first_token_latency + control_s,
                          'e2e_s': e2e, 'metrics': dataclasses.asdict(result.metrics),
                          'engine_skip_prefill_span_tokens': result.num_cached_tokens, 'telemetry': telemetry,
                          'query_span': row['query_span'], 'GPU_ms': None, 'h2d_bytes': None,
                          'd2h_bytes': None, 'peak_hbm_bytes': None, 'enable_thinking': False, 'finish_reason': result.outputs[0].finish_reason,
                          'stop_reason': result.outputs[0].stop_reason, 'max_output_tokens': args.max_output_tokens}
                serialized = json.dumps(record, ensure_ascii=False)
            except Exception as exc:
                out.write(json.dumps({"request_id": row["request_id"], "session_id": row["session_id"], "input_hash": row["input_hash"], "arm": args.arm, "gpu": args.gpu, "status": "error", "stage": "record_serialization", "error": repr(exc), "correct": None}) + "\n")
                raise
            out.write(serialized + "\n")
            print(json.dumps({'arm': args.arm, 'request': row['request_id'], 'correct': record['correct']}), flush=True)
    clear()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True);p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True);p.add_argument('--gpu', type=int, choices=range(4), required=True)
    p.add_argument('--arm', choices=('full', 'strict_prefix', 'blend', 'blend_cow'), required=True)
    p.add_argument('--lmcache-config', type=Path);p.add_argument('--policy', choices=('topk', 'window', 'document'), default='topk')
    p.add_argument('--ratio', type=float, default=.05);p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-model-len', type=int, default=32768);p.add_argument('--max-output-tokens', type=int, default=32)
    p.add_argument('--limit', type=int, default=0);p.add_argument('--cold', action='store_true')
    p.add_argument('--capture-logprobs', action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.run-error.json').write_text(json.dumps({
            'status': 'failed', 'error': repr(exc), 'model': str(args.model.resolve()),
            'inputs': str(args.inputs.resolve()), 'arm': args.arm, 'gpu': args.gpu,
            'epoch': time.time(), 'is_request_accuracy_result': False}, indent=2))
        raise
