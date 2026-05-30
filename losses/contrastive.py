import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, activations: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        if activations.size(0) < 2:
            return activations.new_tensor(0.0)

        normalized = F.normalize(activations, dim=1)
        all_energy = torch.exp(torch.matmul(normalized, normalized.t()))
        denominator = all_energy.sum()
        if denominator <= 0:
            return activations.new_tensor(0.0)

        layer_score = activations.new_tensor(0.0)
        for cls in labels.unique(sorted=True):
            indices = torch.nonzero(labels == cls, as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            pairs = torch.combinations(indices, r=2)
            pos_energy = (
                all_energy[pairs[:, 0], pairs[:, 1]].sum() * 2.0
            ) / self.temperature
            if pos_energy > 0:
                layer_score = layer_score + torch.log(pos_energy / denominator)
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

        custom_score = logits.new_tensor(0.0)
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
