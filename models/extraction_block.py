import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcentrationPipeline(nn.Module):
    """Single max-pool pipeline: pool → flatten → linear → embedding."""

    def __init__(self, in_channels: int, pool_size: int, embed_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(pool_size)
        flat_dim = in_channels * pool_size * pool_size
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.flatten(1)
        return self.mlp(x)


class ExtractionBlock(nn.Module):
    """
    Extraction block with parallel concentration pipelines.

    For early layers (spatial ≥ 8): 3 pipelines → pool sizes 8, 4, 2
    For later layers (spatial < 8):  2 pipelines → pool sizes 7, 3
    The output is an L2-normalized embedding used as u^(l) in the loss.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4,
                 dropout_p: float = 0.5, early_layer: bool = True,
                 embed_dim: int = 128):
        super().__init__()
        reduced = max(1, in_channels // reduction_ratio)
        self.conv1x1 = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduced)
        self.spatial_dropout = nn.Dropout2d(p=dropout_p)

        pool_sizes = [8, 4, 2] if early_layer else [7, 3]
        self.pipelines = nn.ModuleList([
            ConcentrationPipeline(reduced, ps, embed_dim) for ps in pool_sizes
        ])
        out_dim = embed_dim * len(pool_sizes)
        self.proj = nn.Linear(out_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv1x1(x)), inplace=True)
        x = self.spatial_dropout(x)
        parts = [p(x) for p in self.pipelines]
        out = torch.cat(parts, dim=1)
        out = self.proj(out)
        return F.normalize(out, dim=1)


if __name__ == "__main__":
    block = ExtractionBlock(64, early_layer=True)
    dummy = torch.randn(4, 64, 28, 28)
    emb = block(dummy)
    print(f"ExtractionBlock output: {emb.shape}")  # (4, 128)
