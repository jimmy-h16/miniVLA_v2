import torch
import torch.nn as nn


class TransformerFusion(nn.Module):
    """
    Replaces FusionMLP from v1.

    Instead of hard-concatenating [image; state; text] and squashing with an MLP,
    we treat each modality as ONE token and let them attend to each other via a
    TransformerEncoder (self-attention across tokens).

    Pipeline (mirrors the slide):
      1. Stack 4 modality vectors -> token sequence   [B, 4, dim_model]
      2. Add learnable modality_embedding             [B, 4, dim_model]
      3. TransformerEncoder (self-attention)          [B, 4, dim_model]
      4. Mean-pool across the 4 tokens                [B, dim_model]
      5. out_proj  (Linear + LayerNorm)               [B, dim_model]

    The output memory_tokens [B, 4, dim_model] is passed to ActionQueryDecoder.
    forward() returns BOTH: (memory_tokens, fused_feat).

    Args:
        dim_model  : embedding dimension (must match all encoders)
        nhead      : attention heads for TransformerEncoder
        num_layers : number of TransformerEncoder layers
        num_tokens : number of input modality tokens (4: img, wrist, state, text)
    """
    def __init__(
        self,
        dim_model:  int = 256,
        nhead:      int = 4,
        num_layers: int = 2,
        num_tokens: int = 4,
    ):
        super().__init__()
        
        self.modality_embedding = nn.Parameter(torch.randn([1,num_tokens,dim_model]))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = dim_model,
            nhead = nhead,
            dim_feedforward = 4*dim_model, #dim of middle layer MLP for feature extraction
            dropout=0.1,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers)
        
        self.out_proj = nn.Sequential(
            nn.Linear(dim_model,dim_model),
            nn.LayerNorm(dim_model)
        )
        pass

    def forward(
        self,
        img_feat:   torch.Tensor,   # [B, dim_model]  - agentview
        wrist_feat: torch.Tensor,   # [B, dim_model]  - eye_in_hand
        state_feat: torch.Tensor,   # [B, dim_model]
        txt_feat:   torch.Tensor,   # [B, dim_model]
    ):
        
                
        tokens = torch.stack([img_feat,wrist_feat,state_feat,txt_feat], dim = 1)
        tokens = tokens + self.modality_embedding
        memory = self.encoder(tokens)
        
        fused = memory.mean(dim=1)        
        fused_feat = self.out_proj(fused)
        
        # Two latent vector to choose from for cross attention in the decoder
        # one more expressive, one is the integration of all 
        return memory, fused_feat
