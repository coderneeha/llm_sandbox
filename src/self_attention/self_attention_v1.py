import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        # basic self-attention mechanism for a single input sequence
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)

        # quick check of output shape
        # print("Output shape:", out.shape)

        return out
