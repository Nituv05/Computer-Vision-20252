"""
Explicit architecture/model baselines for the M2/M2-CL paper.

This file is intentionally separate from the experiment runner. Each class
below is a concrete model variant used in the architecture ablation:
plain ResNet, M2 with cascading/parallel concentration pipelines, different
reduction ratios, dropout on/off, and the final M2-CL architecture.
"""
import torch
import torch.nn as nn

from architecture_specs import ARCHITECTURE_SPECS, ArchitectureSpec
from .m2cl import ERMResNet, OfficialM2


class ResNetERMBaseline(ERMResNet):
    """Plain ResNet classifier baseline."""

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, **kwargs):
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
        )


class M2ArchitectureBaseline(OfficialM2):
    """Base class for M2 architecture-only baselines without contrastive loss."""

    pipeline_type = "parallel"
    reduction_ratio = 4
    dropout_p = 0.3

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, embed_dim: int = 128, **kwargs):
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            reduction_ratio=self.reduction_ratio,
            dropout_p=self.dropout_p,
            embed_dim=embed_dim,
            pipeline_type=self.pipeline_type,
            return_energies=False,
        )


class M2CascadingR2NoDropout(M2ArchitectureBaseline):
    pipeline_type = "cascading"
    reduction_ratio = 2
    dropout_p = 0.0


class M2CascadingR4NoDropout(M2ArchitectureBaseline):
    pipeline_type = "cascading"
    reduction_ratio = 4
    dropout_p = 0.0


class M2CascadingR6NoDropout(M2ArchitectureBaseline):
    pipeline_type = "cascading"
    reduction_ratio = 6
    dropout_p = 0.0


class M2ParallelR2NoDropout(M2ArchitectureBaseline):
    pipeline_type = "parallel"
    reduction_ratio = 2
    dropout_p = 0.0


class M2ParallelR4NoDropout(M2ArchitectureBaseline):
    pipeline_type = "parallel"
    reduction_ratio = 4
    dropout_p = 0.0


class M2ParallelR6NoDropout(M2ArchitectureBaseline):
    pipeline_type = "parallel"
    reduction_ratio = 6
    dropout_p = 0.0


class M2ParallelR2Dropout(M2ArchitectureBaseline):
    pipeline_type = "parallel"
    reduction_ratio = 2
    dropout_p = 0.3


class M2ParallelR4Dropout(M2ArchitectureBaseline):
    pipeline_type = "parallel"
    reduction_ratio = 4
    dropout_p = 0.3


class M2ParallelR6Dropout(M2ArchitectureBaseline):
    pipeline_type = "parallel"
    reduction_ratio = 6
    dropout_p = 0.3


class M2CLParallelR4Dropout(OfficialM2):
    """Full paper model: M2 architecture plus M2-CL contrastive outputs."""

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, embed_dim: int = 128, **kwargs):
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            reduction_ratio=4,
            dropout_p=0.3,
            embed_dim=embed_dim,
            pipeline_type="parallel",
            return_energies=True,
        )


ARCHITECTURE_CLASSES = {
    "resnet_erm": ResNetERMBaseline,
    "m2_cascading_r2_no_dropout": M2CascadingR2NoDropout,
    "m2_cascading_r4_no_dropout": M2CascadingR4NoDropout,
    "m2_cascading_r6_no_dropout": M2CascadingR6NoDropout,
    "m2_parallel_r2_no_dropout": M2ParallelR2NoDropout,
    "m2_parallel_r4_no_dropout": M2ParallelR4NoDropout,
    "m2_parallel_r6_no_dropout": M2ParallelR6NoDropout,
    "m2_parallel_r2_dropout": M2ParallelR2Dropout,
    "m2_parallel_r4_dropout": M2ParallelR4Dropout,
    "m2_parallel_r6_dropout": M2ParallelR6Dropout,
    "m2cl_parallel_r4_dropout": M2CLParallelR4Dropout,
}


def get_architecture_spec(tag: str) -> ArchitectureSpec:
    for spec in ARCHITECTURE_SPECS:
        if spec.tag == tag:
            return spec
    raise KeyError(f"Unknown architecture baseline: {tag}")


def build_architecture_baseline(tag: str, num_classes: int,
                                backbone: str = "resnet18",
                                pretrained: bool = True,
                                embed_dim: int = 128) -> nn.Module:
    try:
        model_class = ARCHITECTURE_CLASSES[tag]
    except KeyError as exc:
        raise KeyError(f"Unknown architecture baseline: {tag}") from exc
    return model_class(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        embed_dim=embed_dim,
    )


if __name__ == "__main__":
    for spec in ARCHITECTURE_SPECS:
        model = build_architecture_baseline(
            spec.tag,
            num_classes=7,
            backbone="resnet18",
            pretrained=False,
        )
        logits, energies = model(torch.randn(1, 3, 224, 224))
        print(
            f"{spec.tag}: class={spec.cls_name}, "
            f"logits={tuple(logits.shape)}, energies={len(energies)}"
        )
