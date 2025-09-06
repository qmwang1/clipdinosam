import torch
import torch.nn as nn


class TokenProjection(nn.Module):
    """Project DINO token features to CLIP text width for contrastive alignment or fusion."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenToMaskEmbedding(nn.Module):
    """Map token sequence to SAM's image embedding grid (C=256, H, W)."""

    def __init__(self, in_dim: int, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor, grid_hw) -> torch.Tensor:
        # tokens: (B, N, C_in), grid_hw = (h, w) such that N = h*w
        B, N, _ = tokens.shape
        h, w = grid_hw
        x = self.proj(tokens)  # (B, N, 256)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, h, w)
        return x

