# tests/test_model.py
"""
Smoke tests — check that all modules run forward() without crashing
and output the expected tensor shapes.

Run with:  pytest tests/
"""
import pytest
import torch
from models.image_encoder import SmallImageEncoder
from models.state_encoder import StateEncoder
from models.text_encoder  import TextEncoder
from models.fusion        import TransformerFusion
from models.action_head   import ActionQueryDecoder
from models.mini_vla      import MiniVLA

B          = 2
D          = 256
H, W       = 64, 64
STATE_DIM  = 8
SEQ_LEN    = 77
CHUNK_SIZE = 16
ACTION_DIM = 7


def test_image_encoder():
    enc = SmallImageEncoder(embed_dim=D)
    x   = torch.randn(B, 3, H, W)
    out = enc(x)
    assert out.shape == (B, D), f"Expected ({B},{D}), got {out.shape}"


def test_state_encoder():
    enc = StateEncoder(state_dim=STATE_DIM, embed_dim=D)
    x   = torch.randn(B, STATE_DIM)
    out = enc(x)
    assert out.shape == (B, D)


def test_text_encoder():
    enc    = TextEncoder(vocab_size=49408, embed_dim=D)
    tokens = torch.randint(0, 49408, (B, SEQ_LEN))
    mask   = torch.ones(B, SEQ_LEN)
    out    = enc(tokens, mask)
    assert out.shape == (B, D)


def test_transformer_fusion():
    fusion = TransformerFusion(dim_model=D, nhead=4, num_layers=2, num_tokens=4)
    feats  = [torch.randn(B, D) for _ in range(4)]
    # TODO: update once you implement TransformerFusion.forward()
    # memory_tokens, fused_feat = fusion(*feats)
    # assert memory_tokens.shape == (B, 4, D)
    # assert fused_feat.shape    == (B, D)
    pytest.skip("TransformerFusion not yet implemented")


def test_action_query_decoder():
    decoder = ActionQueryDecoder(
        dim_model=D, chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM, nhead=4, num_layers=2
    )
    memory = torch.randn(B, 4, D)
    # TODO: update once you implement ActionQueryDecoder.forward()
    # actions = decoder(memory)
    # assert actions.shape == (B, CHUNK_SIZE, ACTION_DIM)
    pytest.skip("ActionQueryDecoder not yet implemented")


def test_mini_vla_shape():
    model  = MiniVLA()
    image  = torch.randn(B, 3, H, W)
    wrist  = torch.randn(B, 3, H, W)
    tokens = torch.randint(0, 49408, (B, SEQ_LEN))
    mask   = torch.ones(B, SEQ_LEN)
    state  = torch.randn(B, STATE_DIM)
    # TODO: update once you implement MiniVLA.forward()
    # out = model(image, wrist, tokens, mask, state)
    # assert out.shape == (B, CHUNK_SIZE, ACTION_DIM)
    pytest.skip("MiniVLA.forward() not yet implemented")
