import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        # attention over input tokens
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        sim = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = F.softmax(sim, dim=-1)
        out = torch.matmul(weights, v)

        return out, weights  # returning weights to visualize later
