# miniVLA v2

Upgrade from [miniVLA_v1](https://github.com/jimmy-h16/miniVLA_v1).

## What's New in v2

| Component | v1 | v2 |
|---|---|---|
| Image Input | `agentview_rgb` only | `agentview_rgb` + `eye_in_hand_rgb` (dual camera) |
| Fusion | `FusionMLP` (concat + MLP) | `TransformerFusion` (TransformerEncoder + modality embeddings) |
| Action Head | `ActionChunkHead` (flat MLP) | `ActionQueryDecoder` (TransformerDecoder + learnable queries) |
| Tokenizer | Char-level (vocab=1000) | CLIP tokenizer (vocab=49408) |

## Architecture

```
agentview_rgb  → ImageEncoder  ─┐
eye_in_hand_rgb→ WristEncoder  ─┤
state          → StateEncoder  ─┼→ stack(4 tokens) + modality_embed
text (CLIP)    → TextEncoder   ─┘
                                  ↓
                         TransformerEncoder  (cross-modal attention)
                                  ↓
                           memory_tokens  [B, 4, D]
                                  ↓
               16 learnable action queries
                        TransformerDecoder  (queries attend to memory)
                                  ↓
                         action_out linear
                                  ↓
                        [B, chunk_size=16, action_dim=7]
```

## Setup

```bash
conda env create -f environment.yml
conda activate mini_vla
pip install transformers  # for CLIP tokenizer
```

## Training

```bash
python train.py
```
