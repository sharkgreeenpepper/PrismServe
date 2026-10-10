"""Selected causal queries through existing FA3 paged kernels, without KV duplication."""
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func


def causal_attention(q,k,v,positions=None,scale=None,out=None):
    n=k.shape[0]
    if positions is None:
        seq=torch.tensor([0,n],device=q.device,dtype=torch.int32)
        return flash_attn_varlen_func(q=q,k=k,v=v,out=out,cu_seqlens_q=seq,
                                     cu_seqlens_k=seq,max_seqlen_q=n,max_seqlen_k=n,
                                     causal=True,softmax_scale=scale,fa_version=3)
    count=len(positions)
    page=16
    extra=(-n)%page
    if extra:
        k=torch.cat((k,torch.zeros((extra,*k.shape[1:]),device=k.device,dtype=k.dtype)))
        v=torch.cat((v,torch.zeros((extra,*v.shape[1:]),device=v.device,dtype=v.dtype)))
    k=k.view(-1,page,*k.shape[1:]);v=v.view(-1,page,*v.shape[1:])
    blocks=torch.arange(k.shape[0],device=q.device,dtype=torch.int32).repeat(count,1)
    seq_q=torch.arange(count+1,device=q.device,dtype=torch.int32)
    # A one-query sequence attends exactly positions [0, original_position].
    visible=(positions+1).to(dtype=torch.int32)
    return flash_attn_varlen_func(q=q,k=k,v=v,out=out,cu_seqlens_q=seq_q,
                                 seqused_k=visible,max_seqlen_q=1,max_seqlen_k=n,
                                 block_table=blocks,causal=True,softmax_scale=scale,fa_version=3)


def install_causal_flash():
    from lmcache.v1.compute.attention.flash_attn import LMCFlashAttnBackend
    def forward(self,q,k,v,output,metadata,**kwargs):
        from engine_bridge import STATE
        STATE['telemetry']['attention_backend']='FA3 dense/paged with original causal positions'
        return causal_attention(q,k,v,getattr(metadata,'prism_positions',None),
                                self.vllm_attn_impl.scale,output)
    LMCFlashAttnBackend.forward_contiguous=forward
