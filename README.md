# LLM Playground: Building/Training Large Language Models from Scratch

A personal learning space where I'm teaching myself how to build LLMs from scratch, step by step. 

---

## Project Structure

```
llm_sandbox/
├── src/
│   ├── self_attention/         # self-attention implementations (v1–v3)
│   └── transformer_block/      # transformer blocks, positional encodings, test scripts
├── training/                   # training + generation scripts (toy + real data)
├── input.txt                   # (not committed) wikitext data generated locally
├── .gitignore
├── model_weights.pth           # weights for toy model
├── real_model_weights.pth      # weights for real-text model
└── README.md

```

---

### Self-Attention Implementations
- `self_attention_v1.py`: minimal single-head self-attention
- `self_attention_v2.py`: returns attention weights
- `self_attention_v3.py`: multi-head self-attention with masking

Run test script:
```bash
python -m src.self_attention.sa_test
```

---

### Transformer Block
- `transformer_block_v1.py`: transformer block with residuals, norm, dropout
- `transformer_block_v2.py`: adds attention weight return and learned positional embeddings

Run test script:
```bash
python -m src.transformer_block.tb_test_v1
python -m src.transformer_block.tb_test_v2
```

---

### Toy Training Tasks
#### Copying Task
Trains on synthetic input → output copying:
```bash
python -m training.train_copy
```

#### Next-Token Prediction 
```bash
python -m training.train_next
```

---

### Real-World Data
#### Using Wikitext-103 for character-level modeling

```

#### Train on real text:
```bash
python -m training.train_real
```
- input.txt is generated automatically in training.train_real

#### Generate new text from model:
```bash
python -m training.generate_real
```

---

## Notes
- `input.txt` is ignored by `.gitignore` (not tracked)
- Use a Python 3.9+ environment with `torch` and `datasets`
- For real training, Wikitext must be downloaded locally (automatically handled)

---


