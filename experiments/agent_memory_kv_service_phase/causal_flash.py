"""Selected causal queries through existing FA3 paged kernels, without KV duplication."""
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func


def causal_attention(q,k,v,positions=None,scale=None,out=None,ranges=None):
    n=k.shape[0]
    if positions is None:
        seq=torch.tensor([0,n],device=q.device,dtype=torch.int32)
        return flash_attn_varlen_func(q=q,k=k,v=v,out=out,cu_seqlens_q=seq,
                                     cu_seqlens_k=seq,max_seqlen_q=n,max_seqlen_k=n,
                                     causal=True,softmax_scale=scale,fa_version=3)
    count=len(positions)
    if ranges is not None:
        if not ranges or any(not 0<=a<b<=n for a,b in ranges):raise ValueError('Invalid contiguous query regions')
        lengths=[b-a for a,b in ranges]
        if sum(lengths)!=count:raise ValueError('Query region count mismatch')
        # Bottom-right causal alignment gives query a+j exactly keys 0..a+j
        # for a region [a,b): Q length b-a, K length b. Reuse existing FA3.
        if ranges==[(0,n)]:return causal_attention(q,k,v,None,scale,out)
    page=16
    extra=(-n)%page
    if extra:
        k=torch.cat((k,torch.zeros((extra,*k.shape[1:]),device=k.device,dtype=k.dtype)))
        v=torch.cat((v,torch.zeros((extra,*v.shape[1:]),device=v.device,dtype=v.dtype)))
    k=k.view(-1,page,*k.shape[1:]);v=v.view(-1,page,*v.shape[1:])
    groups=count if ranges is None else len(ranges)
    blocks=torch.arange(k.shape[0],device=q.device,dtype=torch.int32).repeat(groups,1)
    if ranges is None:
        seq_q=torch.arange(count+1,device=q.device,dtype=torch.int32)
        visible=(positions+1).to(dtype=torch.int32)
        max_q=1
    else:
        cumulative=[0]
        for length in lengths:cumulative.append(cumulative[-1]+length)
        seq_q=torch.tensor(cumulative,device=q.device,dtype=torch.int32)
        visible=torch.tensor([b for a,b in ranges],device=q.device,dtype=torch.int32)
        max_q=max(lengths)
    return flash_attn_varlen_func(q=q,k=k,v=v,out=out,cu_seqlens_q=seq_q,
                                 seqused_k=visible,max_seqlen_q=max_q,max_seqlen_k=n,
                                 block_table=blocks,causal=True,softmax_scale=scale,fa_version=3)


def install_causal_flash():
    from lmcache.v1.compute.attention.flash_attn import LMCFlashAttnBackend
    def forward(self,q,k,v,output,metadata,**kwargs):
        import os
        from engine_bridge import STATE
        grouped=os.environ.get('PRISMSERVE_GROUP_CONTIGUOUS_Q')=='1'
        ranges=getattr(metadata,'prism_ranges',None) if grouped else None
        STATE['telemetry']['attention_backend']='FA3 grouped contiguous causal regions' if grouped else 'FA3 original one-query paged sequences'
        if ranges is not None:
            STATE['telemetry'].setdefault('attention_query_regions',[]).append({'queries':len(q),'regions':len(ranges),'max_region_tokens':max(b-a for a,b in ranges)})
        return causal_attention(q,k,v,getattr(metadata,'prism_positions',None),
                                self.vllm_attn_impl.scale,output,ranges)
    LMCFlashAttnBackend.forward_contiguous=forward
