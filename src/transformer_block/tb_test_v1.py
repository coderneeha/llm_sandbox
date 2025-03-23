import torch
from src.transformer_block.transformer_block_v1 import TransformerBlock
from src.self_attention.self_attention_v3 import MultiHeadSelfAttention
from src.transformer_block.positional_encoding_v1 import get_positional_encoding

# test 1: basic forward pass
print("test 1: basic forward pass")
B, T, D = 2, 10, 32
x = torch.randn(B, T, D)
x = x + get_positional_encoding(T, D, device=x.device)
block = TransformerBlock(dim=D, num_heads=4, dropout=0.1, causal=True)
out = block(x)
print("output shape:", out.shape)

# test 2: longer sequence
print("\ntest 2: longer sequence")
B, T, D = 1, 50, 32
x = torch.randn(B, T, D)
x = x + get_positional_encoding(T, D, device=x.device)
block = TransformerBlock(dim=D, num_heads=4, dropout=0.1, causal=True)
out = block(x)
print("output shape:", out.shape)

# test 3: batch size of 1
print("\ntest 3: batch size of 1")
B, T, D = 1, 12, 32
x = torch.randn(B, T, D)
x = x + get_positional_encoding(T, D, device=x.device)
block = TransformerBlock(dim=D, num_heads=4, dropout=0.1, causal=True)
out = block(x)
print("output shape:", out.shape)

# test 4: no dropout (inference mode)
print("\ntest 4: no dropout (inference mode)")
B, T, D = 2, 10, 32
x = torch.randn(B, T, D)
x = x + get_positional_encoding(T, D, device=x.device)
block = TransformerBlock(dim=D, num_heads=4, dropout=0.0, causal=False)
block.eval()
out = block(x)
print("output shape:", out.shape)

# test 5: output should vary with different seeds
print("\ntest 5: output should vary with different seeds")
torch.manual_seed(42)
x1 = torch.randn(2, 10, 32)
x1 = x1 + get_positional_encoding(10, 32, device=x1.device)

torch.manual_seed(0)
x2 = torch.randn(2, 10, 32)
x2 = x2 + get_positional_encoding(10, 32, device=x2.device)

block = TransformerBlock(dim=32, num_heads=4, dropout=0.1, causal=True)
out1 = block(x1)
out2 = block(x2)

print("same shape:", out1.shape == out2.shape)
print("equal outputs:", torch.allclose(out1, out2))
