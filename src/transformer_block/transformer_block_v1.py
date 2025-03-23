import torch
import torch.nn as nn
import torch.nn.functional as F
from src.self_attention.self_attention_v3 import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, causal=False):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, causal)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + self.dropout(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)
        return x

