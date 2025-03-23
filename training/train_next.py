
# training a transformer to predict the next token in a sequence

import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.transformer_block.transformer_block_v2 import TransformerBlock
from src.transformer_block.positional_encoding_v2 import LearnedPositionalEncoding

# generates input like [5, 10, 15] and target like [10, 15, 20]
class NextTokenDataset:
    def __init__(self, seq_len, vocab_size, num_batches):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches

    def get_batch(self, batch_size):
        x = torch.randint(1, self.vocab_size - self.seq_len, (batch_size, 1))
        x = x + torch.arange(self.seq_len).unsqueeze(0)  # increasing seq
        y = x + 1  # next-token target
        return x, y

# transformer model for predicting next tokens
class NextTokenModel(nn.Module):
    def __init__(self, dim, vocab_size, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_enc = LearnedPositionalEncoding(max_len=100, dim=dim)
        self.transformer = TransformerBlock(dim=dim, num_heads=num_heads, dropout=0.1, causal=True)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x, _ = self.transformer(x)
        out = self.proj(x)
        return out

# setup
vocab_size = 100
seq_len = 10
batch_size = 16
dim = 64
num_heads = 8
num_epochs = 30

dataset = NextTokenDataset(seq_len=seq_len, vocab_size=vocab_size, num_batches=150)
model = NextTokenModel(dim=dim, vocab_size=vocab_size, num_heads=num_heads)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for _ in range(dataset.num_batches):
        x, y = dataset.get_batch(batch_size)  # (B, T)
        out = model(x)  # (B, T, vocab)

        logits = out.view(-1, vocab_size)  # not used, but left for inspection/debugging
        loss = loss_fn(out.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch+1:02d} - loss: {total_loss / dataset.num_batches:.4f}")

# sample prediction
x, y = dataset.get_batch(1)
with torch.no_grad():
    out = model(x)
    preds = out.argmax(dim=-1)

print("input: ", x[0].tolist())
print("target:", y[0].tolist())
print("preds: ", preds[0].tolist())

# save the trained weights
torch.save(model.state_dict(), "model_weights.pth")
