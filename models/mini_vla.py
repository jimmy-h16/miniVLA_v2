import torch
import torch.nn as nn

from models.image_encoder import SmallImageEncoder
from models.text_encoder  import TextEncoder
from models.state_encoder import StateEncoder
from models.fusion        import TransformerFusion
from models.action_head   import ActionQueryDecoder


class MiniVLA(nn.Module):
    """
    Mini-VLA v2
    ===========
    Inputs:
        image       [B, 3, H, W]   agentview_rgb
        wrist_image [B, 3, H, W]   eye_in_hand_rgb  (NEW in v2)
        tokens      [B, seq_len]   CLIP token ids   (upgraded tokenizer)
        mask        [B, seq_len]   attention mask
        state       [B, state_dim] proprioceptive state

    Pipeline:
        image       → image_encoder  [B, D]
        wrist_image → wrist_encoder  [B, D]   (same class, independent weights)
        state       → state_encoder  [B, D]
        tokens+mask → text_encoder   [B, D]
                         ↓
              TransformerFusion  →  memory_tokens [B, 4, D]
                         ↓
             ActionQueryDecoder  →  [B, chunk_size, action_dim]
    """
    def __init__(
        self,
        dim_model:  int = 256,
        vocab_size: int = 49408,    # CLIP BPE vocab
        state_dim:  int = 8,
        chunk_size: int = 16,
        action_dim: int = 7,        # xyz + rpy + gripper
        nhead:      int = 4,
        fusion_layers:  int = 2,
        decoder_layers: int = 2,
    ):
        super().__init__()

        # --- Encoders ---
        # Two separate image encoders: same architecture, DIFFERENT weights.
        # agentview gives global scene context; eye_in_hand gives close-up wrist view.
        self.image_encoder = SmallImageEncoder(embed_dim=dim_model)
        self.wrist_encoder = SmallImageEncoder(embed_dim=dim_model)  # NEW

        self.state_encoder = StateEncoder(state_dim=state_dim, embed_dim=dim_model)
        self.text_encoder  = TextEncoder(vocab_size=vocab_size, embed_dim=dim_model)

        # --- Fusion (Transformer cross-modal attention, 4 tokens) ---
        self.fusion = TransformerFusion(
            dim_model=dim_model,
            nhead=nhead,
            num_layers=fusion_layers,
            num_tokens=4,           # img, wrist, state, text
        )

        # --- Action head (query-based decoder) ---
        self.action_head = ActionQueryDecoder(
            dim_model=dim_model,
            chunk_size=chunk_size,
            action_dim=action_dim,
            nhead=nhead,
            num_layers=decoder_layers,
        )

    def forward(
        self,
        image:       torch.Tensor,  # [B, 3, H, W]  agentview
        wrist_image: torch.Tensor,  # [B, 3, H, W]  eye_in_hand  (NEW)
        tokens:      torch.Tensor,  # [B, seq_len]  CLIP ids
        mask:        torch.Tensor,  # [B, seq_len]  attention mask
        state:       torch.Tensor,  # [B, state_dim]
    ) -> torch.Tensor:
        # TODO: implement forward
        # Step 1 — encode each modality
        # img_feat   = self.image_encoder(image)          # [B, D]
        # wrist_feat = self.wrist_encoder(wrist_image)    # [B, D]
        # state_feat = self.state_encoder(state)          # [B, D]
        # txt_feat   = self.text_encoder(tokens, mask)    # [B, D]

        # Step 2 — cross-modal fusion → memory tokens
        # memory_tokens, _ = self.fusion(img_feat, wrist_feat, state_feat, txt_feat)
        # memory_tokens shape: [B, 4, dim_model]

        # Step 3 — action query decoder
        # actions = self.action_head(memory_tokens)       # [B, chunk_size, action_dim]

        # return actions
        img_feat   = self.image_encoder(image)          # [B, D]
        wrist_feat = self.wrist_encoder(wrist_image)    # [B, D]
        state_feat = self.state_encoder(state)          # [B, D]
        txt_feat   = self.text_encoder(tokens, mask)    # [B, D]
        
        memory, _ = self.fusion(img_feat, wrist_feat, state_feat, txt_feat)
        actions = self.action_head(memory)
        
        return actions
        
        # raise NotImplementedError("MiniVLA.forward() — implement me!")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
