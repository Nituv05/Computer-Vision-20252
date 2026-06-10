import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.new_tensor(0.0)


class LayerContrastiveLoss(nn.Module):
    """
    Official M2-CL layer loss.

    This follows `domainbed/lib/myloss.py`: activations are L2-normalized,
    exponentiated pair similarities are summed over the full batch, and
    positive same-class pairs contribute log(pos_energy / denominator).
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def _pairwise_energy(self, activations: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(activations, dim=1)
        similarity = torch.matmul(normalized, normalized.t())
        return torch.exp(similarity)

    def _positive_energy_for_class(
        self,
        all_energy: torch.Tensor,
        labels: torch.Tensor,
        class_label: torch.Tensor,
    ) -> torch.Tensor | None:
        class_indices = torch.nonzero(labels == class_label, as_tuple=False).flatten()
        if class_indices.numel() < 2:
            return None

        pair_indices = torch.combinations(class_indices, r=2)
        positive_energy = all_energy[pair_indices[:, 0], pair_indices[:, 1]].sum()
        return (positive_energy * 2.0) / self.temperature

    def forward(self, activations: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        if activations.size(0) != labels.size(0):
            raise ValueError("activations and labels must have the same batch size")
        if activations.size(0) < 2:
            return _zero_like(activations)

        labels = labels.view(-1)
        all_energy = self._pairwise_energy(activations)
        denominator = all_energy.sum()
        if denominator <= 0:
            return _zero_like(activations)

        layer_score = _zero_like(activations)
        for class_label in labels.unique(sorted=True):
            positive_energy = self._positive_energy_for_class(
                all_energy,
                labels,
                class_label,
            )
            if positive_energy is None or positive_energy <= 0:
                continue
            layer_score = layer_score + torch.log(positive_energy / denominator)
        return layer_score


class MultiLayerContrastiveLoss(nn.Module):
    """Official-style objective: CE - alpha * sum_l layer_score_l."""

    def __init__(self, alpha: float = 0.01, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.layer_loss = LayerContrastiveLoss(temperature)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                activations: list[torch.Tensor]) -> torch.Tensor:
        loss = self.ce(logits, labels)
        if self.alpha == 0 or not activations:
            return loss

        custom_score = _zero_like(logits)
        for activation in activations:
            custom_score = custom_score + self.layer_loss(activation, labels)
        return loss - self.alpha * custom_score


if __name__ == "__main__":
    criterion = MultiLayerContrastiveLoss()
    logits = torch.randn(8, 7)
    labels = torch.randint(0, 7, (8,))
    activations = [torch.randn(8, 4096) for _ in range(3)]
    loss = criterion(logits, labels, activations)
    print(f"Loss: {loss.item():.4f}")
