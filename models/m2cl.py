import torch
import torch.nn as nn
from torchvision import models

try:
    from .extraction_block import ExtractionBlock
except ImportError:  # Allows: python models/m2cl.py
    from extraction_block import ExtractionBlock


RESNET18_OFFICIAL_SPECS = [
    ("layer1.0.conv1", 64, 56, [3, 7, 14], 0.3),
    ("layer1.0.conv2", 64, 56, [3, 7, 14], 0.3),
    ("layer1.1.conv1", 64, 56, [3, 7, 14], 0.3),
    ("layer2.0.conv1", 128, 28, [3], 0.3),
    ("layer2.1.conv2", 128, 28, [3], 0.3),
    ("layer3.1.conv1", 256, 14, [2], 0.3),
    ("layer3.1.conv2", 256, 14, [2], 0.3),
    ("layer4.0.conv2", 512, 7, [1], 0.3),
    ("layer4.1.conv1", 512, 7, [1], 0.3),
    ("layer4.1.conv2", 512, 7, [1], 0.3),
    ("layer3.1", 256, 14, [2, 4, 7], 0.3),
    ("layer4.0", 512, 7, [1], 0.3),
    ("layer4.1", 512, 7, [1], 0.3),
]


RESNET50_OFFICIAL_SPECS = (
    [(f"layer1.{i}", 256, 56, [3, 7, 14], 0.3) for i in range(3)] +
    [(f"layer2.{i}", 512, 28, [2, 4, 7], 0.3) for i in range(4)] +
    [(f"layer3.{i}", 1024, 14, [2, 3], 0.3) for i in range(6)] +
    [(f"layer4.{i}", 2048, 7, [1], 0.3) for i in range(3)]
)


def _resnet_backbone(backbone: str, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
    else:
        raise ValueError("backbone must be resnet18 or resnet50")
    return base


def _get_submodule(module: nn.Module, name: str) -> nn.Module:
    current = module
    for part in name.split("."):
        current = getattr(current, part)
    return current


class ERMResNet(nn.Module):
    """Plain ResNet baseline with the usual final classifier."""

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.network = _resnet_backbone(backbone, pretrained)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        return self.network(x), []


class OfficialM2(nn.Module):
    """
    M2/M2-CL architecture following the official implementation.

    ResNet-18 uses the same selected conv/addition extraction points as
    `domainbed/mymodels/m2cl_18.py`. ResNet-50 extracts from every bottleneck
    and selects the same subset for the classification hypercolumn as
    `domainbed/mymodels/m2cl_50.py`.
    """

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, reduction_ratio: int = 4,
                 dropout_p: float | None = None, embed_dim: int = 128,
                 pipeline_type: str = "parallel",
                 return_energies: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.return_energies = return_energies
        self.reduction_ratio = reduction_ratio
        self.embed_dim = embed_dim
        self.pipeline_type = pipeline_type

        self.backbone = _resnet_backbone(backbone, pretrained)
        self.backbone.fc = nn.Identity()
        self.specs = (
            RESNET18_OFFICIAL_SPECS if backbone == "resnet18"
            else RESNET50_OFFICIAL_SPECS
        )

        self.extraction_blocks = nn.ModuleList()
        for _, channels, image_size, pool_sizes, official_drop in self.specs:
            self.extraction_blocks.append(
                ExtractionBlock(
                    in_filters=channels,
                    p_comp=reduction_ratio,
                    in_img_size=image_size,
                    pool_sizes=pool_sizes,
                    p_drop=official_drop if dropout_p is None else dropout_p,
                    out_features=embed_dim,
                    pipeline_type=pipeline_type,
                )
            )

        if backbone == "resnet50":
            self.resnet50_selected_outputs = [0, 4, 8, len(self.specs) - 1]
            classifier_dim = sum(
                self.extraction_blocks[i].output_dim
                for i in self.resnet50_selected_outputs
            )
        else:
            classifier_dim = sum(block.output_dim for block in self.extraction_blocks)

        self.fc_hc = nn.Linear(classifier_dim, num_classes)
        self._features = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for idx, (name, _, _, _, _) in enumerate(self.specs):
            handle = _get_submodule(self.backbone, name).register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(handle)

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            self._features[idx] = output
        return hook

    def _selected_resnet50_energies(self, all_energies: list[torch.Tensor]):
        indices = list(range(0, len(all_energies), 3))
        if all_energies and (len(all_energies) - 1) not in indices:
            indices[-1] = len(all_energies) - 1
        return [all_energies[i] for i in indices]

    def forward(self, x: torch.Tensor):
        self._features.clear()
        _ = self.backbone(x)

        outputs = []
        energies = []
        for idx, block in enumerate(self.extraction_blocks):
            if self.return_energies:
                out, block_energies = block(self._features[idx], return_energy=True)
            else:
                out = block(self._features[idx], return_energy=False)
                block_energies = []
            outputs.append(out)
            if self.return_energies:
                energies.extend(block_energies)

        if self.backbone_name == "resnet50":
            outputs = [outputs[i] for i in self.resnet50_selected_outputs]
            if self.return_energies:
                energies = self._selected_resnet50_energies(energies)

        hc = torch.cat(outputs, dim=1)
        logits = self.fc_hc(hc)
        return logits, energies if self.return_energies else []

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class M2CL(OfficialM2):
    """Backward-compatible name for the M2-CL architecture."""

    def __init__(self, num_classes: int, **kwargs):
        kwargs.setdefault("return_energies", True)
        super().__init__(num_classes=num_classes, **kwargs)


def build_model(num_classes: int, method: str = "m2cl", **kwargs) -> nn.Module:
    if method not in {"erm", "m2", "m2cl"}:
        raise ValueError("method must be one of: erm, m2, m2cl")

    accepted = {
        "backbone", "pretrained", "reduction_ratio", "dropout_p", "embed_dim",
        "pipeline_type",
    }
    model_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    model_kwargs.setdefault("backbone", "resnet18")
    model_kwargs.setdefault("pretrained", True)

    if method == "erm":
        return ERMResNet(
            num_classes=num_classes,
            backbone=model_kwargs["backbone"],
            pretrained=model_kwargs["pretrained"],
        )

    return OfficialM2(
        num_classes=num_classes,
        return_energies=(method == "m2cl"),
        **model_kwargs,
    )


if __name__ == "__main__":
    for method_name in ("erm", "m2", "m2cl"):
        model = build_model(
            num_classes=7,
            method=method_name,
            backbone="resnet18",
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        logits, energies = model(x)
        print(
            f"{method_name}: logits={tuple(logits.shape)}, "
            f"energies={len(energies)}"
        )

    model = build_model(
        num_classes=7,
        method="m2cl",
        backbone="resnet50",
        pretrained=False,
    )
    logits, energies = model(torch.randn(1, 3, 224, 224))
    print(f"resnet50 m2cl: logits={tuple(logits.shape)}, energies={len(energies)}")
