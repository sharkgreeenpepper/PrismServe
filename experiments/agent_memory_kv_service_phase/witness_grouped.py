"""Real seeded FA3 causal alignment checks, not model quality evidence."""
import json
import os
import torch
from causal_flash import causal_attention

assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1', '2', '3')
torch.manual_seed(17)
cases = []
for n in (513, 2048):
    q = torch.randn(n, 32, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(n, 8, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dense = causal_attention(q, k, v)
    patterns = [[(0, 32), (127, 156), (n - 17, n)],
                [(127, 156), (0, 32), (n - 17, n)],
                [(7, n - 17)], [(0, n // 3), (n // 2, n - 1)],
                [(0, n)], [(7, 8), (31, 32), (127, 128)]]
    for ranges in patterns:
        selected = [i for a, b in ranges for i in range(a, b)]
        indices = torch.tensor(selected, device='cuda')
        original = causal_attention(q[indices], k, v, indices)
        grouped = causal_attention(q[indices], k, v, indices, ranges=ranges)
        delta = float((original.float() - grouped.float()).abs().max())
        dense_delta = float((dense[indices].float() - grouped.float()).abs().max())
        assert torch.allclose(original, grouped, atol=.02, rtol=.02)
        assert torch.allclose(dense[indices], grouped, atol=.02, rtol=.02)
        cases.append({'tokens': n, 'queries': len(selected), 'regions': len(ranges),
                      'max_abs_delta_single_query': delta, 'max_abs_delta_dense': dense_delta})
    indices = torch.arange(32, device='cuda')
    before = causal_attention(q[:32], k, v, indices, ranges=[(0, 32)])
    k[32:] *= 3;v[32:] *= 3
    after = causal_attention(q[:32], k, v, indices, ranges=[(0, 32)])
    assert torch.equal(before, after), 'Future token leakage'
torch.cuda.synchronize()
print(json.dumps({'checks': cases, 'future_invisible': True, 'GPU': torch.cuda.get_device_name(),
                  'scope': 'seeded FA3 primitive only; no task/performance claim'}))
