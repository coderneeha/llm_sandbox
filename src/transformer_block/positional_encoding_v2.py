import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        # each position gets its own trainable embedding vector
        self.pos_embed = nn.Embedding(max_len, dim)

    def forward(self, x):
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pos_embed(positions)
