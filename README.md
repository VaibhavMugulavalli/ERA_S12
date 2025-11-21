# GPT-2 Mini Training & Hugging Face Spaces Inference

*A lightweight GPT‑2 (124M) training + deployment pipeline with warmup, cosine annealing, and gradient clipping.*

---

## Overview

This repository contains a clean, from‑scratch GPT‑2 style implementation and a simple workflow to:

1. **Train the model on Google Colab** using a text corpus (`input.txt`).
2. **Export the trained weights** as a single file (`model.pt`).
3. **Visualize / run inference in Hugging Face Spaces** using a Gradio UI (**no training inside Spaces**).

This setup is ideal for educational experiments, small‑scale LLM training, and quick deployment demos.

---

## Key Features

### Model

* Multi‑head **causal self‑attention**
* Pre‑norm Transformer blocks (LayerNorm before attention/MLP)
* Feedforward MLP with GELU
* **Weight tying** between token embeddings and LM head

### Training Stability

* **AdamW** optimizer
* **Linear learning‑rate warmup**
* **Cosine annealing decay**
* **Gradient norm clipping** (prevents exploding gradients)
* Deterministic seeding for reproducibility

### Deployment

* Inference‑only Hugging Face Space
* Space loads **only `model.pt`**
* GPT‑2 tokenizer via `tiktoken`

---

## Training (Google Colab)

### 1) Prepare Dataset

Upload your corpus to Colab as:

```
input.txt
```

Examples: tiny‑Shakespeare, Wikipedia dump slices, custom domain text.

### 2) Run Training Script

Execute:

```python
!python train_colab.py
```

Default training settings:

* Steps: **5000**
* Batch size: **8**
* Sequence length: **256**
* LR: **3e‑4**
* Warmup: **500 steps**
* Scheduler: **cosine decay**
* Grad clip norm: **1.0**

### 3) Exported Files

After training finishes, the script saves:

```
hf_export/model.pt
hf_export/config.json
hf_export/meta.json
```

### 4) Download for Spaces

Zip and download:

```python
!zip -r hf_export.zip hf_export
```

Extract `model.pt` and upload it to your Hugging Face Space.

---

## Hugging Face Spaces (Inference Only)

<img width="1919" height="1073" alt="image" src="https://github.com/user-attachments/assets/46220c3c-b8d9-42b2-9148-5af0438f6048" />


Spaces automatically launches `app.py`. The UI allows you to:

* Enter a prompt
* Set max new tokens
* Choose top‑k sampling
* Generate text from your trained GPT model

---

## Model Configuration

The inference app hard‑codes the training config. These values **must match your Colab training run**:

| Parameter                 | Value |
| ------------------------- | ----: |
| Layers (`n_layer`)        |    12 |
| Heads (`n_head`)          |    12 |
| Embedding dim (`n_embd`)  |   768 |
| Block size (`block_size`) |  1024 |
| Vocab size (`vocab_size`) | 50257 |
| Params                    | ~124M |

---

## Example Inference (Local)

```python
from model_def import GPT, GPTConfig
import torch, tiktoken

config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Once upon a time")
x = torch.tensor(ids).unsqueeze(0)

logits = model(x)
```

---

## Training Logs (add your screenshots here)
```
Using device: cuda
loaded 338025 tokens
1 epoch = 165 batches
step 0/5000 | loss 10.9527 | lr 1.200000e-06
step 50/5000 | loss 8.3372 | lr 3.120000e-05
step 100/5000 | loss 6.3455 | lr 6.120000e-05
step 150/5000 | loss 6.0740 | lr 9.120000e-05
step 200/5000 | loss 5.2532 | lr 1.212000e-04
step 250/5000 | loss 5.3142 | lr 1.512000e-04
step 300/5000 | loss 4.7946 | lr 1.812000e-04
step 350/5000 | loss 5.0287 | lr 2.112000e-04
step 400/5000 | loss 4.6069 | lr 2.412000e-04
step 450/5000 | loss 4.8475 | lr 2.712000e-04
step 500/5000 | loss 4.9737 | lr 3.000000e-04
step 550/5000 | loss 4.7973 | lr 2.999049e-04
step 600/5000 | loss 4.9622 | lr 2.996273e-04
step 650/5000 | loss 3.9560 | lr 2.991673e-04
step 700/5000 | loss 4.4178 | lr 2.985256e-04
step 750/5000 | loss 4.0366 | lr 2.977029e-04
step 800/5000 | loss 3.6059 | lr 2.967003e-04
step 850/5000 | loss 3.8397 | lr 2.955190e-04
step 900/5000 | loss 4.1377 | lr 2.941604e-04
step 950/5000 | loss 4.0849 | lr 2.926261e-04
step 1000/5000 | loss 3.5316 | lr 2.909180e-04
step 1050/5000 | loss 3.5946 | lr 2.890383e-04
step 1100/5000 | loss 3.9135 | lr 2.869892e-04
step 1150/5000 | loss 3.7969 | lr 2.847732e-04
step 1200/5000 | loss 3.4448 | lr 2.823929e-04
step 1250/5000 | loss 3.5341 | lr 2.798514e-04
step 1300/5000 | loss 3.3212 | lr 2.771517e-04
step 1350/5000 | loss 3.1171 | lr 2.742970e-04
step 1400/5000 | loss 2.9183 | lr 2.712910e-04
step 1450/5000 | loss 2.8853 | lr 2.681371e-04
step 1500/5000 | loss 2.6886 | lr 2.648393e-04
step 1550/5000 | loss 2.9045 | lr 2.614016e-04
step 1600/5000 | loss 2.9311 | lr 2.578282e-04
step 1650/5000 | loss 2.9657 | lr 2.541234e-04
step 1700/5000 | loss 2.4980 | lr 2.502917e-04
step 1750/5000 | loss 2.2236 | lr 2.463379e-04
step 1800/5000 | loss 2.2177 | lr 2.422667e-04
step 1850/5000 | loss 1.9783 | lr 2.380830e-04
step 1900/5000 | loss 1.9466 | lr 2.337921e-04
step 1950/5000 | loss 1.8643 | lr 2.293991e-04
step 2000/5000 | loss 1.8911 | lr 2.249093e-04
step 2050/5000 | loss 1.7091 | lr 2.203283e-04
step 2100/5000 | loss 2.0025 | lr 2.156615e-04
step 2150/5000 | loss 1.5022 | lr 2.109148e-04
step 2200/5000 | loss 1.5125 | lr 2.060939e-04
step 2250/5000 | loss 1.4013 | lr 2.012046e-04
step 2300/5000 | loss 1.1199 | lr 1.962529e-04
step 2350/5000 | loss 1.1050 | lr 1.912449e-04
step 2400/5000 | loss 0.9787 | lr 1.861867e-04
step 2450/5000 | loss 1.0258 | lr 1.810843e-04
step 2500/5000 | loss 0.7873 | lr 1.759441e-04
step 2550/5000 | loss 0.7857 | lr 1.707723e-04
step 2600/5000 | loss 0.8596 | lr 1.655751e-04
step 2650/5000 | loss 0.6774 | lr 1.603590e-04
step 2700/5000 | loss 0.5623 | lr 1.551303e-04
step 2750/5000 | loss 0.6144 | lr 1.498953e-04
step 2800/5000 | loss 0.5100 | lr 1.446604e-04
step 2850/5000 | loss 0.3725 | lr 1.394321e-04
step 2900/5000 | loss 0.3978 | lr 1.342166e-04
step 2950/5000 | loss 0.3196 | lr 1.290203e-04
step 3000/5000 | loss 0.2752 | lr 1.238497e-04
step 3050/5000 | loss 0.1765 | lr 1.187108e-04
step 3100/5000 | loss 0.1818 | lr 1.136101e-04
step 3150/5000 | loss 0.1945 | lr 1.085537e-04
step 3200/5000 | loss 0.1810 | lr 1.035479e-04
step 3250/5000 | loss 0.1224 | lr 9.859859e-05
step 3300/5000 | loss 0.1454 | lr 9.371193e-05
step 3350/5000 | loss 0.0795 | lr 8.889385e-05
step 3400/5000 | loss 0.0502 | lr 8.415022e-05
step 3450/5000 | loss 0.0600 | lr 7.948682e-05
step 3500/5000 | loss 0.0542 | lr 7.490933e-05
step 3550/5000 | loss 0.1049 | lr 7.042332e-05
step 3600/5000 | loss 0.0620 | lr 6.603427e-05
step 3650/5000 | loss 0.0616 | lr 6.174751e-05
step 3700/5000 | loss 0.0267 | lr 5.756828e-05
step 3750/5000 | loss 0.0424 | lr 5.350166e-05
step 3800/5000 | loss 0.0357 | lr 4.955261e-05
step 3850/5000 | loss 0.0214 | lr 4.572594e-05
step 3900/5000 | loss 0.0226 | lr 4.202631e-05
step 3950/5000 | loss 0.0198 | lr 3.845823e-05
step 4000/5000 | loss 0.0135 | lr 3.502605e-05
step 4050/5000 | loss 0.0127 | lr 3.173394e-05
step 4100/5000 | loss 0.0161 | lr 2.858593e-05
step 4150/5000 | loss 0.0188 | lr 2.558584e-05
step 4200/5000 | loss 0.0172 | lr 2.273732e-05
step 4250/5000 | loss 0.0182 | lr 2.004386e-05
step 4300/5000 | loss 0.0130 | lr 1.750873e-05
step 4350/5000 | loss 0.0141 | lr 1.513502e-05
step 4400/5000 | loss 0.0190 | lr 1.292562e-05
step 4450/5000 | loss 0.0170 | lr 1.088323e-05
step 4500/5000 | loss 0.0181 | lr 9.010325e-06
step 4550/5000 | loss 0.0145 | lr 7.309197e-06
step 4600/5000 | loss 0.0096 | lr 5.781916e-06
step 4650/5000 | loss 0.0198 | lr 4.430343e-06
step 4700/5000 | loss 0.0152 | lr 3.256123e-06
step 4750/5000 | loss 0.0104 | lr 2.260689e-06
step 4800/5000 | loss 0.0202 | lr 1.445252e-06
step 4850/5000 | loss 0.0173 | lr 8.108059e-07
step 4900/5000 | loss 0.0132 | lr 3.581240e-07
step 4950/5000 | loss 0.0190 | lr 8.775781e-08
step 4999/5000 | loss 0.0109 | lr 0.000000e+00
final loss: 0.010906321927905083
Saved model + config to hf_export/
```

