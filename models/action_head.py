import torch
import torch.nn as nn


class ActionQueryDecoder(nn.Module):
    """
    Replaces ActionChunkHead from v1.

    Instead of one flat MLP outputting chunk_size * action_dim numbers at once,
    we have chunk_size LEARNABLE queries - one per future timestep - that each
    attend to the fused observation memory and independently predict one action.

    Pipeline (mirrors the slide):
      1. query_embed  [1, chunk_size, dim_model]  - learnable, shared across batch
      2. expand to    [B, chunk_size, dim_model]
      3. TransformerDecoder: queries (tgt) attend to memory_tokens (memory)
      4. action_out Linear -> [B, chunk_size, action_dim]

    Args:
        dim_model  : must match TransformerFusion dim_model
        chunk_size : number of future action steps (16)
        action_dim : degrees of freedom per step (7: xyz + rpy + gripper)
        nhead      : attention heads
        num_layers : TransformerDecoder layers
    """
    def __init__(
        self,
        dim_model:  int = 256,
        chunk_size: int = 16,
        action_dim: int = 7,
        nhead:      int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # TODO: define self.query_embed
        # 16 learnable queries, one per future timestep.
        # Shape: [1, chunk_size, dim_model]  (broadcast over batch)
        # Hint: nn.Parameter(torch.randn(...))
        # self.query_embed = ...

        # TODO: define self.decoder
        # A TransformerDecoder where queries (tgt) cross-attend to memory_tokens.
        # Use nn.TransformerDecoderLayer + nn.TransformerDecoder.
        # Remember: batch_first=True.
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=dim_model, nhead=nhead,
        #     dim_feedforward=dim_model * 4, dropout=0.1, batch_first=True
        # )
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # TODO: define self.action_out
        # Projects each decoded query vector to one action.
        # Hint: nn.Linear(dim_model, action_dim)
        # self.action_out = ...
        
        self.queryEmbed = nn.Parameter(torch.randn([1,chunk_size,dim_model]))
        
        decoderLayer = nn.TransformerDecoderLayer(d_model=dim_model,nhead=nhead,dim_feedforward=4*dim_model,dropout=0.1,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderLayer,num_layers=num_layers)
        
        self.actionProj = nn.Linear(dim_model,action_dim)
        
        pass

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_tokens: [B, num_tokens, dim_model]  from TransformerFusion
        Returns:
            pred_action_chunk: [B, chunk_size, action_dim]
        """
        # TODO: Step 1 - expand learnable queries over batch
        # B = memory_tokens.shape[0]
        # queries = self.query_embed.expand(B, -1, -1)   # [B, 16, dim_model]

        # TODO: Step 2 - each query attends to observation memory
        # decoded_queries = self.decoder(
        #     tgt=queries,
        #     memory=memory_tokens,
        # )                                               # [B, 16, dim_model]

        # TODO: Step 3 - project each query to action space
        # pred_action_chunk = self.action_out(decoded_queries)  # [B, 16, action_dim]

        # return pred_action_chunk
        # raise NotImplementedError("ActionQueryDecoder.forward() - implement me!")
        
        B = memory_tokens.shape[0]
        queries = self.queryEmbed.expand(B,-1,-1)
        decodedQueries = self.decoder(tgt=queries,memory=memory_tokens)
        predActionChunk = self.actionProj(decodedQueries)
        return predActionChunk
