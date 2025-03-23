import torch
from src.transformer_block.transformer_block_v2 import TransformerBlock
from src.transformer_block.positional_encoding_v2 import LearnedPositionalEncoding
from src.self_attention.self_attention_v3 import MultiHeadSelfAttention

# test 1: basic forward pass
print("test 1: basic forward pass")
B, T, D = 2, 12, 64
x = torch.randn(B, T, D)

pos_enc = LearnedPositionalEncoding(max_len=T, dim=D)
x = pos_enc(x)

block = TransformerBlock(dim=D, num_heads=8, dropout=0.1, causal=True)
out, attn_weights = block(x)
print("output shape:", out.shape)
print("attention shape:", attn_weights.shape)


# test 2: longer sequence
print("\ntest 2: longer sequence")
B, T, D = 1, 50, 64
x = torch.randn(B, T, D)
x = LearnedPositionalEncoding(T, D)(x)
block = TransformerBlock(D, num_heads=8, dropout=0.1, causal=False)
out, attn_weights = block(x)
print("output shape:", out.shape)
print("attention shape:", attn_weights.shape)


# test 3: batch size of 1
print("\ntest 3: batch size of 1")
B, T, D = 1, 8, 64
x = torch.randn(B, T, D)
x = LearnedPositionalEncoding(T, D)(x)
block = TransformerBlock(D, num_heads=8, dropout=0.1, causal=True)
out, attn_weights = block(x)
print("output shape:", out.shape)
print("attention shape:", attn_weights.shape)


# test 4: no dropout (inference)
print("\ntest 4: no dropout (inference mode)")
B, T, D = 2, 10, 64
x = torch.randn(B, T, D)
x = LearnedPositionalEncoding(T, D)(x)
block = TransformerBlock(D, num_heads=8, dropout=0.0, causal=True)
block.eval()
out, attn_weights = block(x)
print("output shape:", out.shape)
print("attention shape:", attn_weights.shape)


# test 5: check that attention weights vary for different inputs
print("\ntest 5: attention weights should vary across different inputs")
T, D = 12, 64
x1 = torch.randn(1, T, D)
x2 = torch.randn(1, T, D)

x1 = LearnedPositionalEncoding(T, D)(x1)
x2 = LearnedPositionalEncoding(T, D)(x2)

block = TransformerBlock(D, num_heads=8, dropout=0.1, causal=True)
_, w1 = block(x1)
_, w2 = block(x2)

equal = torch.allclose(w1, w2)
print("attention weights equal:", equal)
