import torch
import torch.nn as nn


class ConcentrationPipeline(nn.Module):
    """
    Official-style concentration pipeline used by M2/M2-CL.

    The pipeline keeps the reduced channel axis separate, pools each channel's
    spatial map, and lets the MLP project every reduced channel independently.
    """

    def __init__(self, in_filters: int, p_comp: int, pool_size: int,
                 p_drop: float):
        super().__init__()
        self.in_filters = in_filters
        self.p_comp = p_comp
        self.pool_size = pool_size
        self.p_drop = p_drop
        self.compression_out_channels = in_filters // p_comp

        self.comp = nn.Conv2d(
            in_filters, self.compression_out_channels, kernel_size=(1, 1)
        )
        self.drop = nn.Dropout2d(p_drop)
        self.max = nn.MaxPool2d(pool_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.comp(x)
        x = self.drop(x)
        x = self.max(x)
        x = self.relu(x)
        return x.flatten(-2, -1)


class MLP(nn.Module):
    """Official projection MLP: pooled spatial dim -> 1024 -> 128."""

    def __init__(self, in_features: int, hidden_features: int = 1024,
                 out_features: int = 128, p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        energy = x.clone().flatten(-2, -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.flatten(-2, -1), energy


class ExtractionBlock(nn.Module):
    """
    Official-style M2 extraction block.

    For each pool size, a separate concentration pipeline is applied to the
    original feature map. The projected outputs are concatenated for the
    classifier. The hidden activations before GELU are returned for M2-CL loss.
    """

    def __init__(self, in_filters: int, p_comp: int, in_img_size: int,
                 pool_sizes: list[int], p_drop: float,
                 hidden_features: int = 1024, out_features: int = 128,
                 pipeline_type: str = "parallel"):
        super().__init__()
        if pipeline_type not in {"parallel", "cascading"}:
            raise ValueError("pipeline_type must be 'parallel' or 'cascading'")
        self.in_filters = in_filters
        self.p_comp = p_comp
        self.in_img_size = in_img_size
        self.pool_sizes = pool_sizes
        self.p_drop = p_drop
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.pipeline_type = pipeline_type
        self.reduced_channels = in_filters // p_comp

        self.pipelines = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.mlps = nn.ModuleList()
        if pipeline_type == "cascading":
            self.comp = nn.Conv2d(
                in_filters, self.reduced_channels, kernel_size=(1, 1)
            )
            self.drop = nn.Dropout2d(p_drop)
            self.relu = nn.ReLU(inplace=True)

        for pool_size in pool_sizes:
            if pipeline_type == "parallel":
                self.pipelines.append(
                    ConcentrationPipeline(in_filters, p_comp, pool_size, p_drop)
                )
            else:
                self.pools.append(nn.MaxPool2d(pool_size))
            dim = (in_img_size // pool_size) ** 2
            self.mlps.append(
                MLP(
                    in_features=dim,
                    hidden_features=hidden_features,
                    out_features=out_features,
                )
            )

    @property
    def output_dim(self) -> int:
        return self.reduced_channels * self.out_features * len(self.pool_sizes)

    def forward(self, x: torch.Tensor,
                return_energy: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        x_init = x.clone()
        outputs = []
        energies = []
        if self.pipeline_type == "parallel":
            for pipeline, mlp in zip(self.pipelines, self.mlps):
                projected, energy = mlp(pipeline(x_init))
                outputs.append(projected)
                energies.append(energy)
        else:
            x_shared = self.drop(self.comp(x_init))
            for pool, mlp in zip(self.pools, self.mlps):
                pooled = self.relu(pool(x_shared)).flatten(-2, -1)
                projected, energy = mlp(pooled)
                outputs.append(projected)
                energies.append(energy)

        output = torch.cat(outputs, dim=1)
        if return_energy:
            return output, energies
        return output


if __name__ == "__main__":
    block = ExtractionBlock(
        in_filters=64,
        p_comp=4,
        in_img_size=56,
        pool_sizes=[3, 7, 14],
        p_drop=0.3,
    )
    dummy = torch.randn(2, 64, 56, 56)
    out, energies = block(dummy)
    print(f"ExtractionBlock output: {tuple(out.shape)}")
    print(f"Energy tensors: {[tuple(item.shape) for item in energies]}")
