import torch
import torch.nn as nn


class SmallImageEncoder(nn.Module):
    """
    3-layer CNN + Global Average Pooling + Linear projection.
    Shared class — instantiated TWICE in MiniVLA:
      - self.image_encoder  for agentview_rgb
      - self.wrist_encoder  for eye_in_hand_rgb
    Each instance has its OWN weights (independent learning).

    Input:  [B, 3, H, W]
    Output: [B, embed_dim]
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),   # [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # [B, 64, H/4, W/4]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [B, 128, H/8, W/8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)  # [B, 128, 1, 1]
        self.flat = nn.Flatten()             # [B, 128]
        self.proj = nn.Linear(128, embed_dim) # [B, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.gap(x)
        x = self.flat(x)
        x = self.proj(x)
        return x
