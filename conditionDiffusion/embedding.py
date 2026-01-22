import torch
from torch import nn

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb


class MaskEmbedding(nn.Module):
    """Image-based conditional embedding.

    Takes an image-like mask tensor (B, C, H, W) and produces a global
    conditioning vector (B, cdim). This can be used instead of integer labels.
    """
    def __init__(self, in_ch:int = 1, cdim:int = 10, hidden:int = 32):
        super().__init__()
        # small CNN followed by global pooling and FC layers
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, hidden * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden * 2, cdim),
            nn.SiLU(),
            nn.Linear(cdim, cdim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward expects x shape (B, C, H, W) and returns (B, cdim)."""
        return self.net(x)
