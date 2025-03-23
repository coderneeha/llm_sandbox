
# training a transformer to copy input sequences exactly

import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.transformer_block.transformer_block_v2 import TransformerBlock
from src.transformer_block.positional_encoding_v2 import LearnedPositionalEncoding

# generates batches of sequences like [1, 2, 3, 4] -> target: same as input
class CopyDataset:
    def __init__(self, seq_len, vocab_size, num_batches):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches

    def get_batch(self, batch_size):
        x = torch.randint(1, self.vocab_size, (batch_size, self.seq_len))
        y = x.clone()
        return x, y

# a simple transformer model that learns to copy input
class CopyModel(nn.Module):
    def __init__(self, dim, vocab_size, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_enc = LearnedPositionalEncoding(max_len=100, dim=dim)
        self.transformer = TransformerBlock(dim=dim, num_heads=num_heads, dropout=0.1, causal=False)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x, _ = self.transformer(x)
        return self.fc(x)

# training loop
vocab_size = 50
seq_len = 12
batch_size = 16
dim = 64
num_heads = 8
num_epochs = 50

model = CopyModel(dim=dim, vocab_size=vocab_size, num_heads=num_heads)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = CopyDataset(seq_len=seq_len, vocab_size=vocab_size, num_batches=200)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for _ in range(dataset.num_batches):
        x, y = dataset.get_batch(batch_size)
        out = model(x)  # (B, T, vocab_size)
        loss = loss_fn(out.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / dataset.num_batches
    print(f"epoch {epoch+1:02d} - loss: {avg_loss:.4f}")

# test on a single batch
model.eval()
x, _ = dataset.get_batch(1)
preds = model(x).argmax(dim=-1)
print("input:", x[0].tolist())
print("preds:", preds[0].tolist())
