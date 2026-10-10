"""Request-range I/O adapter for vLLM fused K/V views; attention kernels stay native."""
import torch


def packed_transfer(lmc, kv, slots, direction, fmt, token_major=False):
    from lmcache import lmcache_native as native
    nhd=native.EngineKVFormat.NL_X_NB_BS_NH_CS
    hnd=native.EngineKVFormat.NL_X_NB_NH_BS_CS
    if fmt not in (nhd,hnd) or kv.ndim!=4:
        raise ValueError('Unsupported packed layout')
    if slots.numel()==0:return
    slots=slots.to(device=kv.device,dtype=torch.int64)
    if bool((slots<0).any()):raise ValueError('Invalid negative slot mapping')
    block_size=kv.shape[1] if fmt==nhd else kv.shape[2]
    heads=kv.shape[2] if fmt==nhd else kv.shape[1]
    head_size=kv.shape[-1]//2
    if kv.shape[-1]!=2*head_size or lmc.numel()!=2*slots.numel()*heads*head_size:
        raise ValueError('KV shape mismatch')
    blocks=slots//block_size;offsets=slots%block_size
    if bool((blocks>=kv.shape[0]).any()):raise ValueError('Slot mapping exceeds pool')
    separate=lmc.transpose(0,1) if token_major else lmc
    if direction==native.TransferDirection.D2H:
        packed=kv[blocks,offsets] if fmt==nhd else kv[blocks,:,offsets,:]
        separate.copy_(packed.view(-1,heads,2,head_size).permute(2,0,1,3).reshape(2,-1,heads*head_size))
    elif direction==native.TransferDirection.H2D:
        packed=separate.reshape(2,-1,heads,head_size).permute(1,2,0,3).reshape(-1,heads,2*head_size)
        if fmt==nhd:kv[blocks,offsets]=packed
        else:kv[blocks,:,offsets,:]=packed
    else:raise ValueError('Invalid transfer direction')


def install_packed_adapter():
    from lmcache import device_ops, lmcache_native as native
    device_ops.ensure_native()
    original=device_ops.single_layer_kv_transfer
    fused=(native.EngineKVFormat.NL_X_NB_BS_NH_CS,native.EngineKVFormat.NL_X_NB_NH_BS_CS)
    def transfer(lmc,kv,slots,direction,fmt,token_major=False):
        if kv.ndim==4 and fmt in fused:
            return packed_transfer(lmc,kv,slots,direction,fmt,token_major)
        return original(lmc,kv,slots,direction,fmt,token_major=token_major)
    device_ops.single_layer_kv_transfer=transfer
