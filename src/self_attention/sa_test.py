import torch
from src.self_attention.self_attention_v1 import SelfAttention
from src.self_attention.self_attention_v2 import SelfAttention as SelfAttentionV2
from src.self_attention.self_attention_v3 import MultiHeadSelfAttention

# test 1: basic self-attention (v1)
x = torch.randn(2, 6, 16)
sa = SelfAttention(dim=16)
out = sa(x)
print("v1 output shape:", out.shape)

# test 2: self-attention with weights (v2)
x = torch.randn(1, 8, 32)
sa = SelfAttentionV2(dim=32)
out, weights = sa(x)
print("v2 output shape:", out.shape)
print("v2 weights shape:", weights.shape)

# test 3: multi-head self-attention (v3)
x = torch.randn(2, 10, 64)
sa = MultiHeadSelfAttention(dim=64, num_heads=8, dropout=0.1, causal=True)
out, _ = sa(x)
print("v3 output shape:", out.shape)

# test 4: output should differ for different inputs
x1 = torch.randn(2, 6, 32)
x2 = torch.randn(2, 6, 32)
sa = SelfAttentionV2(dim=32)
out1, _ = sa(x1)
out2, _ = sa(x2)
print("v2 outputs equal:", torch.allclose(out1, out2))