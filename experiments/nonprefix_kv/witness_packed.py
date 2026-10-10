"""Seeded GPU round-trip validates exact K/V values and untouched slots."""
import json
import os
from pathlib import Path
import torch
from packed_kv import packed_transfer
from lmcache import lmcache_native as native

assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2','3')
torch.manual_seed(17)
report=[]
for layout in ('NHD','HND'):
    for major in (False,True):
        kv=torch.randn((5,16,4,256),device='cuda',dtype=torch.bfloat16)
        if layout=='HND':kv=kv.transpose(1,2).contiguous()
        original=kv.clone();slots=torch.tensor([0,15,16,31,69,79],device='cuda')
        fmt=getattr(native.EngineKVFormat,'NL_X_NB_BS_NH_CS' if layout=='NHD' else 'NL_X_NB_NH_BS_CS')
        lmc=torch.empty((len(slots),2,512) if major else (2,len(slots),512),device='cuda',dtype=torch.bfloat16)
        packed_transfer(lmc,kv,slots,native.TransferDirection.D2H,fmt,major)
        expected=original[slots//16,slots%16] if layout=='NHD' else original[slots//16,:,slots%16,:]
        split=lmc.transpose(0,1) if major else lmc
        assert torch.equal(split[0].reshape(-1,4,128),expected[...,:128])
        assert torch.equal(split[1].reshape(-1,4,128),expected[...,128:])
        packed_transfer(lmc,kv,slots,native.TransferDirection.H2D,fmt,major)
        assert torch.equal(kv,original)
        split.add_(1)
        packed_transfer(lmc,kv,slots,native.TransferDirection.H2D,fmt,major)
        modified=original.clone()
        if layout=='NHD':modified[slots//16,slots%16]+=1
        else:modified[slots//16,:,slots%16,:]+=1
        # Compare against separately rounded BF16 K/V updates, not float arithmetic.
        assert torch.equal(kv,modified)
        torch.cuda.synchronize()
        report.append({'layout':layout,'token_major':major,'roundtrip_exact':True,'untouched_slots_exact':True})
print('PACKED_KV_WITNESS',json.dumps(report),torch.cuda.get_device_name(0))
