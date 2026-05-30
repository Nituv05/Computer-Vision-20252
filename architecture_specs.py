"""Architecture baseline metadata shared by model code and experiment runners."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureSpec:
    tag: str
    cls_name: str
    method: str
    description: str
    pipeline_type: str | None = None
    reduction_ratio: int | None = None
    dropout_p: float | None = None

    def train_flags(self) -> list[str]:
        flags = []
        if self.pipeline_type is not None:
            flags.extend(["--pipeline_type", self.pipeline_type])
        if self.reduction_ratio is not None:
            flags.extend(["--reduction_ratio", str(self.reduction_ratio)])
        if self.dropout_p is not None:
            flags.extend(["--dropout_p", str(self.dropout_p)])
        return flags


ARCHITECTURE_SPECS = [
    ArchitectureSpec(
        tag="resnet_erm",
        cls_name="ResNetERMBaseline",
        method="erm",
        description="Plain ResNet classifier baseline",
    ),
    ArchitectureSpec(
        tag="m2_cascading_r2_no_dropout",
        cls_name="M2CascadingR2NoDropout",
        method="m2",
        description="M2 cascading concentration pipeline, r=2, no dropout",
        pipeline_type="cascading",
        reduction_ratio=2,
        dropout_p=0.0,
    ),
    ArchitectureSpec(
        tag="m2_cascading_r4_no_dropout",
        cls_name="M2CascadingR4NoDropout",
        method="m2",
        description="M2 cascading concentration pipeline, r=4, no dropout",
        pipeline_type="cascading",
        reduction_ratio=4,
        dropout_p=0.0,
    ),
    ArchitectureSpec(
        tag="m2_cascading_r6_no_dropout",
        cls_name="M2CascadingR6NoDropout",
        method="m2",
        description="M2 cascading concentration pipeline, r=6, no dropout",
        pipeline_type="cascading",
        reduction_ratio=6,
        dropout_p=0.0,
    ),
    ArchitectureSpec(
        tag="m2_parallel_r2_no_dropout",
        cls_name="M2ParallelR2NoDropout",
        method="m2",
        description="M2 parallel concentration pipeline, r=2, no dropout",
        pipeline_type="parallel",
        reduction_ratio=2,
        dropout_p=0.0,
    ),
    ArchitectureSpec(
        tag="m2_parallel_r4_no_dropout",
        cls_name="M2ParallelR4NoDropout",
        method="m2",
        description="M2 parallel concentration pipeline, r=4, no dropout",
        pipeline_type="parallel",
        reduction_ratio=4,
        dropout_p=0.0,
    ),
    ArchitectureSpec(
        tag="m2_parallel_r6_no_dropout",
        cls_name="M2ParallelR6NoDropout",
        method="m2",
        description="M2 parallel concentration pipeline, r=6, no dropout",
        pipeline_type="parallel",
        reduction_ratio=6,
        dropout_p=0.0,
    ),
    ArchitectureSpec(
        tag="m2_parallel_r2_dropout",
        cls_name="M2ParallelR2Dropout",
        method="m2",
        description="M2 parallel concentration pipeline, r=2, spatial dropout",
        pipeline_type="parallel",
        reduction_ratio=2,
        dropout_p=0.3,
    ),
    ArchitectureSpec(
        tag="m2_parallel_r4_dropout",
        cls_name="M2ParallelR4Dropout",
        method="m2",
        description="M2 parallel concentration pipeline, r=4, spatial dropout",
        pipeline_type="parallel",
        reduction_ratio=4,
        dropout_p=0.3,
    ),
    ArchitectureSpec(
        tag="m2_parallel_r6_dropout",
        cls_name="M2ParallelR6Dropout",
        method="m2",
        description="M2 parallel concentration pipeline, r=6, spatial dropout",
        pipeline_type="parallel",
        reduction_ratio=6,
        dropout_p=0.3,
    ),
    ArchitectureSpec(
        tag="m2cl_parallel_r4_dropout",
        cls_name="M2CLParallelR4Dropout",
        method="m2cl",
        description="Full paper model: M2 + contrastive loss",
        pipeline_type="parallel",
        reduction_ratio=4,
        dropout_p=0.3,
    ),
]
