import json
import os
import torch
from causal_flash import causal_attention

assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2','3')
torch.manual_seed(17)
q=torch.randn(513,28,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn(513,4,128,device='cuda',dtype=torch.bfloat16)
v=torch.randn_like(k)
indices=torch.tensor([0,1,17,31,32,127,255,512],device='cuda')
dense=causal_attention(q,k,v)
sparse=causal_attention(q[indices],k,v,indices)
error=(dense[indices].float()-sparse.float()).abs().max().item()
assert torch.allclose(dense[indices],sparse,atol=.02,rtol=.02),error
early=indices[indices<128]
old=causal_attention(q[early],k,v,early)
k[128:]*=3;v[128:]*=3
new=causal_attention(q[early],k,v,early)
assert torch.equal(old,new),'Future-token leakage'
print('CAUSAL_FA3_WITNESS',json.dumps({'selected_vs_dense_max_abs_error':error,'future_tokens_invisible':True,'kernel':'existing FA3 paged'}))
