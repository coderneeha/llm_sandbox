# generating character-level text from the model trained on wikitext-103

import torch
from src.transformer_block.transformer_block_v2 import TransformerBlock
from src.transformer_block.positional_encoding_v2 import LearnedPositionalEncoding

# vocab used during training (should match)
with open("input.txt", "r") as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join([itos[i] for i in indices])

# model definition
class CharModel(torch.nn.Module):
    def __init__(self, dim, vocab_size, num_heads):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.pos_enc = LearnedPositionalEncoding(max_len=1000, dim=dim)
        self.block = TransformerBlock(dim=dim, num_heads=num_heads, dropout=0.0, causal=True)
        self.proj = torch.nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x, _ = self.block(x)
        return self.proj(x)

# generation function
def generate(model, start_seq, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        out = model(start_seq)
        next_token = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
        start_seq = torch.cat([start_seq, next_token], dim=1)
    return start_seq

# model setup
dim = 128
num_heads = 8
model = CharModel(dim=dim, vocab_size=vocab_size, num_heads=num_heads)
model.load_state_dict(torch.load("real_model_weights.pth"))

# seed input (must exist in vocab)
seed = "The history of art "
start_ids = torch.tensor([encode(seed)], dtype=torch.long)

# generate
output_ids = generate(model, start_ids, max_new_tokens=200)
print(decode(output_ids[0].tolist()))
