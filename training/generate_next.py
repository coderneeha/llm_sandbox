# autoregressive generation using a trained next-token transformer model

import torch
from src.transformer_block.transformer_block_v2 import TransformerBlock
from src.transformer_block.positional_encoding_v2 import LearnedPositionalEncoding

# re-define the same model class used during training
class NextTokenModel(torch.nn.Module):
    def __init__(self, dim, vocab_size, num_heads):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.pos_enc = LearnedPositionalEncoding(max_len=100, dim=dim)
        self.transformer = TransformerBlock(dim=dim, num_heads=num_heads, dropout=0.0, causal=True)
        self.proj = torch.nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x, _ = self.transformer(x)
        out = self.proj(x)
        return out

# greedy autoregressive generation
def generate(model, start_seq, max_new_tokens, vocab_size):
    model.eval()
    seq = start_seq.clone()
    for _ in range(max_new_tokens):
        out = model(seq)
        next_token_logits = out[:, -1, :]  # logits for the last token
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        seq = torch.cat([seq, next_token], dim=1)
    return seq

# model config (should match training config)
vocab_size = 100
dim = 64
num_heads = 8

model = NextTokenModel(dim=dim, vocab_size=vocab_size, num_heads=num_heads)

# load the trained weights
model.load_state_dict(torch.load("model_weights.pth"))

# not saving/loading weights in this version — assumes you just trained it
# (could save/load with torch.save + torch.load if needed)

# seed input sequence
start_seq = torch.tensor([[10, 11, 12]])  # shape (1, T)
gen_seq = generate(model, start_seq, max_new_tokens=10, vocab_size=vocab_size)

print("seed:", start_seq[0].tolist())
print("generated:", gen_seq[0].tolist())
