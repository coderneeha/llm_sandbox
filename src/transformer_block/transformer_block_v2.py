import torch
import torch.nn as nn
from src.self_attention.self_attention_v3 import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, causal=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, causal)
        self.dropout1 = nn.Dropout(dropout)

        # feedforward network with gelu and expansion
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # self-attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)

        # mlp block with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_out)

        return x, attn_weights
