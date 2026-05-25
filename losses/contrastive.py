import torch
import torch.nn as nn


class LayerContrastiveLoss(nn.Module):
    """
    Contrastive loss for a single layer l.

    p^(l)(c) = Σ_{i,j: y_i=y_j=c} exp(u_i^T u_j / τ)
               ─────────────────────────────────────────
               Σ_{k,m: k≠m} exp(u_k^T u_m / τ)

    L^(l) = -Σ_c log p^(l)(c)
    Embeddings u^(l) must already be L2-normalized.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.tau = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (N, D) already L2-normalized
        # labels:     (N,)
        sim = torch.mm(embeddings, embeddings.t()) / self.tau  # (N, N)
        exp_sim = torch.exp(sim)

        # mask for same class pairs (excluding diagonal)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        eye = torch.eye(len(labels), device=labels.device).bool()
        same_class = label_eq & ~eye

        # denominator: all off-diagonal pairs
        off_diag = ~eye
        denom = exp_sim[off_diag].sum()

        classes = labels.unique()
        loss = torch.tensor(0.0, device=embeddings.device)
        for c in classes:
            pairs = same_class[labels == c][:, labels == c]
            if pairs.sum() == 0:
                continue
            numerator = exp_sim[same_class & (labels.unsqueeze(0) == c)].sum()
            if numerator > 0 and denom > 0:
                loss = loss - torch.log(numerator / denom)
        return loss


class MultiLayerContrastiveLoss(nn.Module):
    """
    Combined loss: L = L_CE + α * Σ_l L^(l)
    """

    def __init__(self, alpha: float = 0.01, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.layer_loss = LayerContrastiveLoss(temperature)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                embeddings: list) -> torch.Tensor:
        loss = self.ce(logits, labels)
        for emb in embeddings:
            loss = loss + self.alpha * self.layer_loss(emb, labels)
        return loss


if __name__ == "__main__":
    criterion = MultiLayerContrastiveLoss()
    logits = torch.randn(8, 7)
    labels = torch.randint(0, 7, (8,))
    embs = [torch.randn(8, 128) for _ in range(3)]
    loss = criterion(logits, labels, embs)
    print(f"Loss: {loss.item():.4f}")
