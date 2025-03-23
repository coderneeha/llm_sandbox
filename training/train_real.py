# training your transformer on real wikipedia text using wikitext-103 

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from src.transformer_block.transformer_block_v2 import TransformerBlock
from src.transformer_block.positional_encoding_v2 import LearnedPositionalEncoding

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
lines = [line for line in dataset["train"]["text"] if line.strip() != ""]

with open("input.txt", "w") as f:
    f.write("\n".join(lines))

raw_text = "\n".join(lines)

# build vocab (char-level)
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join([itos[i] for i in indices])

# encode entire dataset as int tokens
data_ids = torch.tensor(encode(raw_text), dtype=torch.long)

# batching
seq_len = 64
batch_size = 32


def get_batch():
    ix = torch.randint(0, len(data_ids) - seq_len - 1, (batch_size,))
    x = torch.stack([data_ids[i:i+seq_len] for i in ix])
    y = torch.stack([data_ids[i+1:i+seq_len+1] for i in ix])
    return x, y

# model
dim = 128
num_heads = 8

class CharModel(nn.Module):
    def __init__(self, dim, vocab_size, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_enc = LearnedPositionalEncoding(max_len=1000, dim=dim)
        self.block = TransformerBlock(dim=dim, num_heads=num_heads, dropout=0.1, causal=True)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x, _ = self.block(x)
        return self.proj(x)

model = CharModel(dim=dim, vocab_size=vocab_size, num_heads=num_heads)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for _ in range(200):
        x, y = get_batch()
        out = model(x)
        loss = loss_fn(out.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / 200
    print(f"epoch {epoch+1:02d} - loss: {avg_loss:.4f}")

# save weights for generation
torch.save(model.state_dict(), "real_model_weights.pth")
