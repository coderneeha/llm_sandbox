import torch
import math

def get_positional_encoding(seq_len, dim, device='cpu'):
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    i = torch.arange(dim, device=device).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / dim)
    angle_rads = pos * angle_rates

    # sin for even indices, cos for odd
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pe
