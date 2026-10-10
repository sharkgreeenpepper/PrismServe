"""Worker-local controls and telemetry; RPC uses method names and JSON payloads."""
STATE={'request':None,'telemetry':{},'force_full':False,'aliases':{},'provenance':{},'pending':{},'repair_applied':False}


def cache_candidate_allowed(start, end, write=False):
    q = request().get('query_span')
    if q is not None and start < q[1] and end > q[0]:
        STATE['telemetry']['excluded_query_chunks'] = STATE['telemetry'].get('excluded_query_chunks', 0) + 1
        return False
    return True


def request():
    return STATE['request'] or {}


def resolve_key(key,start,end,write=False):
    from dataclasses import replace
    base=key.to_string()
    mode=request().get('writeback','original_only')
    if not write and mode=='persistent_repair' and base in STATE['aliases']:
        resolved=STATE['aliases'][base]
    else:
        tags=dict(key.request_configs or {})
        tags['lmcache.tag.prism_namespace']='pristine'
        if write and STATE['repair_applied'] and mode=='persistent_repair':
            generation=STATE['telemetry'].get('max_source_generation',0)+1
            tags.update({'lmcache.tag.prism_namespace':'repaired',
                         'lmcache.tag.prism_generation':str(generation),
                         'lmcache.tag.prism_producer':request().get('request_id','canary'),
                         'lmcache.tag.prism_context':request().get('input_hash','canary')})
        resolved=replace(key,request_configs=tags)
    if write:
        generation=int((resolved.request_configs or {}).get('lmcache.tag.prism_generation',0))
        STATE['pending'].setdefault(base,(resolved,{'producer_request_id':request().get('request_id','canary'),
          'context_fingerprint':request().get('input_hash','canary'),'original_position':[start,end],
          'repair_generation':generation,'namespace':'repaired' if generation else 'pristine'}))
    return resolved


def commit_stores(engine):
    for base,(key,info) in list(STATE['pending'].items()):
        if all(engine.storage_manager.contains(k,engine.retrieve_locations) for k in key.split_layers(engine.num_layers)):
            STATE['provenance'].setdefault(key.to_string(),info)
            if info['repair_generation']:
                STATE['aliases'][base]=key
    STATE['pending']={}


def record_load(key,start,end):
    info=STATE['provenance'].get(key.to_string(),{})
    telemetry=STATE['telemetry']
    telemetry.setdefault('loaded_chunks',[]).append({'range':[start,end],**info})
    telemetry['max_source_generation']=max(telemetry.get('max_source_generation',0),info.get('repair_generation',0))


def process_qkv(self,q,k,v,residual,layer_id,attn_output,attn_metadata):
    import torch
    from repair_selectors import select_repair
    old_k,old_v=self.gpu_connector.get_kv(layer_id)
    rows=q.shape[0]
    if attn_output is None:attn_output=torch.empty_like(q)
    if self.metadata.positions is None:
        self.metadata.positions=torch.arange(rows,device=q.device,dtype=torch.int64)
    attn=self.layerwise_model.vllm_model.model.layers[layer_id].self_attn
    q,k=attn.rotary_emb(self.metadata.positions,q,k)
    if layer_id in self.common_metadata.check_layers:
        scores=torch.sum((k.float()-old_k.float())**2,dim=1)
        reusable=torch.ones(len(scores),dtype=torch.bool,device=scores.device)
        reusable[self.gpu_connector.current_gap_positions]=False
        spans=[(a,min(b,len(scores))) for a,b in request().get('document_spans',[(0,len(scores))]) if a<len(scores)]
        selection=select_repair(scores.detach().cpu().tolist(),reusable.cpu().tolist(),
            request().get('ratio',self.common_metadata.recomp_ratios[0]),request().get('policy','topk'),spans,
            request().get('edit_start'),request().get('dependency_end'))
        selected=sorted(set(selection.repair_positions)|set(selection.mandatory_positions))
        ranges=[]
        for pos in selected:
            if ranges and ranges[-1][1]==pos:ranges[-1]=(ranges[-1][0],pos+1)
            else:ranges.append((pos,pos+1))
        indices=torch.tensor(selected,
                             dtype=torch.int64,device=q.device)
        q,k,v,residual=q[indices],k[indices],v[indices],residual[indices]
        self.metadata.imp_indices=indices
        self.metadata.positions=self.metadata.positions[indices]
        attn_output=attn_output[:len(indices)]
        attn_metadata.update_from_top_indices(indices)
        attn_metadata.prism_positions=indices
        attn_metadata.prism_ranges=ranges
        STATE['telemetry'].update(repair_tokens=len(selection.repair_positions),
            mandatory_prefix_tokens=len(selection.mandatory_positions),reusable_tokens=selection.reusable_tokens,
            repair_budget=selection.budget,repair_positions=list(selection.repair_positions),
            fallback_full=selection.fallback_full,fallback_reason=selection.reason,
            loaded_span_tokens=len(scores),ratio_realized=len(selection.repair_positions)/max(1,selection.reusable_tokens))
    if self.metadata.imp_indices is None:
        # Before selection, all KV projections have already been computed.
        # Publish them, including uncached holes, to the native paged cache.
        old_k.copy_(k);old_v.copy_(v)
    else:
        old_k[self.metadata.imp_indices]=k;old_v[self.metadata.imp_indices]=v
    STATE['telemetry'].setdefault('layers',[]).append(
        {'layer':layer_id,'qkv_rows':rows,'attention_rows':len(q),'kv_rows':len(old_k)})
    return q,old_k,old_v,residual,attn_output,attn_metadata


def install_blend_controls():
    import os
    if os.environ.get('PRISMSERVE_CACHE_ROPE_TABLE') == '1':
        from lmcache.v1.compute.positional_encoding import FusedRope
        encode = FusedRope.fused_encode
        def cached_encode(self, old_positions, new_positions, k):
            # Retain the immutable table on its actual consumer device. The
            # existing fused kernel and dtype/positions remain unchanged.
            self.cos_sin_cache = self.cos_sin_cache.to(k.device)
            STATE['telemetry']['rope_table_resident'] = {
                'device': str(self.cos_sin_cache.device),
                'dtype': str(self.cos_sin_cache.dtype),
                'tensor_storage_bytes': self.cos_sin_cache.untyped_storage().nbytes()}
            return encode(self, old_positions, new_positions, k)
        FusedRope.fused_encode = cached_encode
    from lmcache.v1.compute.blend.blender import LMCBlender
    original=LMCBlender.blend
    LMCBlender.process_qkv=process_qkv
    def blend(self,*args,**kwargs):
        STATE['repair_applied']=True
        STATE['telemetry']['repair_invocations']=STATE['telemetry'].get('repair_invocations',0)+1
        if request().get('writeback','original_only')=='original_only':self.cache_engine.freeze(True)
        return original(self,*args,**kwargs)
    LMCBlender.blend=blend


def worker_control(worker,payload):
    if payload['action'].startswith('phase_'):
        from phase_hooks import control
        return control(worker,payload)
    import torch
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    engine=LMCacheEngineBuilder.get(ENGINE_NAME)
    action=payload['action']
    if action=='clear':
        torch.cuda.synchronize()
        engine.freeze(False)
        removed=engine.clear()
        STATE.update(request=None,telemetry={},force_full=False,aliases={},provenance={},pending={},repair_applied=False)
        return {'cleared_objects':removed}
    if action=='context':
        STATE['request']=payload['request']
        STATE['force_full']=payload.get('force_full',False)
        STATE['telemetry']={'repair_invocations':0,'layers':[]}
        STATE['repair_applied']=False
        STATE['pending']={}
        engine.freeze(payload.get('no_store',False))
        return {'request_id':STATE['request']['request_id']}
    if action=='telemetry':
        torch.cuda.synchronize()
        return STATE['telemetry']
    if action=='status':
        return {'healthy':engine.is_healthy(),'frozen':engine.is_frozen()}
    raise ValueError('Unknown worker action')
