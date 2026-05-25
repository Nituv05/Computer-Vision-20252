import torch
import torch.nn as nn
from torchvision import models
from .extraction_block import ExtractionBlock


# ResNet-18 intermediate conv layer names to hook (13 layers)
RESNET18_HOOK_LAYERS = [
    "layer1.0.conv1", "layer1.0.conv2",
    "layer1.1.conv1", "layer1.1.conv2",
    "layer2.0.conv1", "layer2.0.conv2",
    "layer2.1.conv1", "layer2.1.conv2",
    "layer3.0.conv1", "layer3.0.conv2",
    "layer3.1.conv1", "layer3.1.conv2",
    "layer4.0.conv1",
]

# ResNet-18 channel counts for each hooked layer
RESNET18_CHANNELS = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512]

# Layers 0-7 have spatial ≥ 8 (early); layers 8+ are later
RESNET18_EARLY = [True] * 8 + [False] * 5


class M2CL(nn.Module):
    """
    M²-CL: Multi-Scale and Multi-Layer Contrastive Learning.

    Wraps a ResNet-18 backbone with ExtractionBlocks on 13 intermediate layers.
    Forward returns (logits, list_of_embeddings).
    """

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 reduction_ratio: int = 4, dropout_p: float = 0.5,
                 embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        assert backbone == "resnet18", "Only resnet18 supported in this reproduction."

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        # Replace final FC with task-specific classifier
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.backbone = base

        # Build extraction blocks
        self.extraction_blocks = nn.ModuleList()
        for ch, early in zip(RESNET18_CHANNELS, RESNET18_EARLY):
            self.extraction_blocks.append(
                ExtractionBlock(ch, reduction_ratio, dropout_p, early, embed_dim)
            )

        self._hooks = []
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        for idx, name in enumerate(RESNET18_HOOK_LAYERS):
            module = self._get_submodule(name)
            handle = module.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _get_submodule(self, name: str) -> nn.Module:
        parts = name.split(".")
        mod = self.backbone
        for p in parts:
            mod = getattr(mod, p)
        return mod

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._features[idx] = output
        return hook

    def forward(self, x: torch.Tensor):
        self._features.clear()
        logits = self.backbone(x)

        embeddings = []
        for idx, block in enumerate(self.extraction_blocks):
            feat = self._features.get(idx)
            if feat is not None:
                embeddings.append(block(feat))

        return logits, embeddings

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


if __name__ == "__main__":
    model = M2CL(num_classes=7, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits, embs = model(x)
    print(f"Logits: {logits.shape}")
    print(f"Embeddings: {len(embs)} layers, each {embs[0].shape}")
