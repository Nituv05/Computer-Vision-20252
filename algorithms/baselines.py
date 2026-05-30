import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchvision import models

from losses import MultiLayerContrastiveLoss
from models import (
    ARCHITECTURE_CLASSES,
    build_architecture_baseline,
    build_model as build_m2_model,
)


BASELINE_METHODS = [
    "erm", "rsc", "mixup", "coral", "mmd", "sagnet",
    "selfreg", "arm", "eqrm", "sagm",
]
M2_METHODS = ["m2", "m2cl"]
METHODS = BASELINE_METHODS + M2_METHODS


@dataclass
class AlgorithmConfig:
    num_classes: int
    method: str
    backbone: str = "resnet18"
    pretrained: bool = True
    lr: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    alpha: float = 0.01
    temperature: float = 1.0
    reduction_ratio: int = 4
    dropout_p: float = 0.3
    embed_dim: int = 128
    pipeline_type: str = "parallel"
    batch_size: int = 128
    mixup_alpha: float = 0.2
    penalty_weight: float = 1.0
    sag_w_adv: float = 0.1
    rsc_f_drop_factor: float = 1.0 / 3.0
    rsc_b_drop_factor: float = 1.0 / 3.0
    eqrm_quantile: float = 0.75
    eqrm_burnin_iters: int = 100
    sam_rho: float = 0.05
    sagm_gamma: float = 0.1
    architecture_tag: str | None = None


def _resnet(backbone: str, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=weights)
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet50(weights=weights)
    raise ValueError(f"Unsupported backbone: {backbone}")


def _adapt_first_conv(model: nn.Module, in_channels: int) -> None:
    if in_channels == 3:
        return
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        for channel in range(in_channels):
            new_conv.weight[:, channel] = old_conv.weight[:, channel % 3]
    model.conv1 = new_conv


class ResNetFeaturizer(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, in_channels: int = 3,
                 dropout_p: float = 0.0):
        super().__init__()
        network = _resnet(backbone, pretrained)
        _adapt_first_conv(network, in_channels)
        self.n_outputs = network.fc.in_features
        network.fc = nn.Identity()
        self.network = network
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.network(x))


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class ContextNet(nn.Module):
    """DomainBed ARM context network."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        padding = 2
        self.context_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_net(x)


def parse_batch(batch):
    if len(batch) == 3:
        return batch
    x, y = batch
    domains = torch.zeros_like(y)
    return x, y, domains


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def split_by_domain(tensor: torch.Tensor, labels: torch.Tensor,
                    domains: torch.Tensor):
    groups = []
    for domain in domains.unique(sorted=True):
        idx = torch.nonzero(domains == domain, as_tuple=False).flatten()
        if idx.numel() > 0:
            groups.append((tensor[idx], labels[idx]))
    return groups


def pairwise_mean_penalty(groups, penalty_fn):
    if len(groups) < 2:
        return groups[0][0].new_tensor(0.0) if groups else torch.tensor(0.0)
    penalties = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if len(groups[i][0]) > 1 and len(groups[j][0]) > 1:
                penalties.append(penalty_fn(groups[i][0], groups[j][0]))
    if not penalties:
        return groups[0][0].new_tensor(0.0)
    return torch.stack(penalties).mean()


def coral_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cov_x = cent_x.t().matmul(cent_x) / max(1, x.size(0) - 1)
    cov_y = cent_y.t().matmul(cent_y) / max(1, y.size(0) - 1)
    return (mean_x - mean_y).pow(2).mean() + (cov_x - cov_y).pow(2).mean()


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x.pow(2).sum(dim=1, keepdim=True)
    y_norm = y.pow(2).sum(dim=1, keepdim=True)
    dist = x_norm + y_norm.t() - 2.0 * x.matmul(y.t())
    dist = dist.clamp_min(1e-30)
    kernel = torch.zeros_like(dist)
    for gamma in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        kernel = kernel + torch.exp(-gamma * dist)
    return kernel


def mmd_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        gaussian_kernel(x, x).mean()
        + gaussian_kernel(y, y).mean()
        - 2.0 * gaussian_kernel(x, y).mean()
    )


class Algorithm(nn.Module):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__()
        self.cfg = cfg
        self.optimizer = None

    def configure_optimizer(self, parameters):
        self.optimizer = torch.optim.SGD(
            parameters,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

    def optimizers(self):
        return [self.optimizer] if self.optimizer is not None else []

    def update(self, batch):
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class M2Algorithm(Algorithm):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__(cfg)
        if cfg.architecture_tag in ARCHITECTURE_CLASSES:
            self.model = build_architecture_baseline(
                tag=cfg.architecture_tag,
                num_classes=cfg.num_classes,
                backbone=cfg.backbone,
                pretrained=cfg.pretrained,
                embed_dim=cfg.embed_dim,
            )
        else:
            self.model = build_m2_model(
                num_classes=cfg.num_classes,
                method=cfg.method,
                backbone=cfg.backbone,
                reduction_ratio=cfg.reduction_ratio,
                dropout_p=cfg.dropout_p,
                embed_dim=cfg.embed_dim,
                pretrained=cfg.pretrained,
                pipeline_type=cfg.pipeline_type,
            )
        alpha = cfg.alpha if cfg.method == "m2cl" else 0.0
        self.criterion = MultiLayerContrastiveLoss(
            alpha=alpha,
            temperature=cfg.temperature,
        )
        self.configure_optimizer(self.model.parameters())

    def update(self, batch):
        x, y, _ = parse_batch(batch)
        logits, embeddings = self.model(x)
        loss = self.criterion(logits, y, embeddings)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "acc": accuracy_from_logits(logits, y)}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits


class ResNetAlgorithm(Algorithm):
    def __init__(self, cfg: AlgorithmConfig, in_channels: int = 3):
        super().__init__(cfg)
        self.featurizer = ResNetFeaturizer(
            cfg.backbone, cfg.pretrained, in_channels=in_channels
        )
        self.classifier = LinearClassifier(
            self.featurizer.n_outputs, cfg.num_classes
        )
        self.configure_optimizer(self.parameters())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.featurizer(x)

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_features(self.forward_features(x))

    def objective(self, batch):
        x, y, _ = parse_batch(batch)
        logits = self.predict(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits, {"loss": loss.item()}

    def update(self, batch):
        x, y, _ = parse_batch(batch)
        loss, logits, metrics = self.objective(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        metrics["acc"] = accuracy_from_logits(logits, y)
        return metrics


class MixupAlgorithm(ResNetAlgorithm):
    def objective(self, batch):
        x, y, domains = parse_batch(batch)
        alpha = self.cfg.mixup_alpha
        if alpha <= 0:
            logits = self.predict(x)
            loss = F.cross_entropy(logits, y)
            return loss, logits, {"loss": loss.item()}

        groups = split_by_domain(x, y, domains)
        if len(groups) < 2:
            perm = torch.randperm(x.size(0), device=x.device)
            lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
            mixed_x = lam * x + (1.0 - lam) * x[perm]
            logits = self.predict(mixed_x)
            loss = lam * F.cross_entropy(logits, y)
            loss = loss + (1.0 - lam) * F.cross_entropy(logits, y[perm])
            return loss, self.predict(x), {"loss": loss.item()}

        order = torch.randperm(len(groups), device=x.device).tolist()
        losses = []
        for i, j in zip(order, order[1:] + order[:1]):
            xi, yi = groups[i]
            xj, yj = groups[j]
            n = min(xi.size(0), xj.size(0))
            if n == 0:
                continue
            lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
            mixed_x = lam * xi[:n] + (1.0 - lam) * xj[:n]
            mixed_logits = self.predict(mixed_x)
            losses.append(
                lam * F.cross_entropy(mixed_logits, yi[:n])
                + (1.0 - lam) * F.cross_entropy(mixed_logits, yj[:n])
            )
        loss = torch.stack(losses).mean() if losses else F.cross_entropy(
            self.predict(x), y
        )
        return loss, self.predict(x), {"loss": loss.item()}


class DistributionMatchingAlgorithm(ResNetAlgorithm):
    penalty = staticmethod(coral_penalty)

    def objective(self, batch):
        x, y, domains = parse_batch(batch)
        features = self.forward_features(x)
        logits = self.logits_from_features(features)
        objective = F.cross_entropy(logits, y)
        groups = split_by_domain(features, y, domains)
        penalty = pairwise_mean_penalty(groups, self.penalty)
        loss = objective + self.cfg.penalty_weight * penalty
        return loss, logits, {
            "loss": loss.item(),
            "ce": objective.item(),
            "penalty": penalty.item(),
        }


class CORALAlgorithm(DistributionMatchingAlgorithm):
    penalty = staticmethod(coral_penalty)


class MMDAlgorithm(DistributionMatchingAlgorithm):
    penalty = staticmethod(mmd_penalty)


class RSCAlgorithm(ResNetAlgorithm):
    def objective(self, batch):
        x, y, _ = parse_batch(batch)
        features = self.forward_features(x)
        features.requires_grad_(True)
        logits = self.logits_from_features(features)
        one_hot = F.one_hot(y, self.cfg.num_classes).float()
        grads = autograd.grad(
            (logits * one_hot).sum(), features, create_graph=True
        )[0]

        drop_f = 1.0 - self.cfg.rsc_f_drop_factor
        percentile_f = torch.quantile(grads.detach(), drop_f, dim=1, keepdim=True)
        mask_f = grads.lt(percentile_f).float()
        muted_logits = self.logits_from_features(features * mask_f)

        probs = F.softmax(logits, dim=1)
        muted_probs = F.softmax(muted_logits, dim=1)
        changes = (probs * one_hot).sum(1) - (muted_probs * one_hot).sum(1)
        drop_b = 1.0 - self.cfg.rsc_b_drop_factor
        percentile_b = torch.quantile(changes.detach(), drop_b)
        mask_b = changes.lt(percentile_b).float().view(-1, 1)
        mask = torch.logical_or(mask_f.bool(), mask_b.bool()).float()

        final_logits = self.logits_from_features(features * mask)
        loss = F.cross_entropy(final_logits, y)
        return loss, final_logits, {"loss": loss.item()}


class SelfRegAlgorithm(ResNetAlgorithm):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__(cfg)
        input_dim = self.featurizer.n_outputs
        hidden_dim = input_dim if input_dim == 2048 else input_dim * 2
        self.cdpl = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
        self.configure_optimizer(self.parameters())

    def objective(self, batch):
        x, y, _ = parse_batch(batch)
        if x.size(0) < 2:
            logits = self.predict(x)
            loss = F.cross_entropy(logits, y)
            return loss, logits, {"loss": loss.item()}

        sorted_y, indices = torch.sort(y)
        x = x[indices]
        y = sorted_y
        features = self.forward_features(x)
        projected = self.cdpl(features)
        logits = self.logits_from_features(features)

        logits_2 = torch.zeros_like(logits)
        logits_3 = torch.zeros_like(logits)
        feat_2 = torch.zeros_like(projected)
        feat_3 = torch.zeros_like(projected)

        boundaries = []
        start = 0
        for idx in range(1, y.size(0)):
            if y[idx] != y[idx - 1]:
                boundaries.append((start, idx))
                start = idx
        boundaries.append((start, y.size(0)))

        for start, end in boundaries:
            count = end - start
            perm_1 = torch.randperm(count, device=x.device) + start
            perm_2 = torch.randperm(count, device=x.device) + start
            logits_2[start:end] = logits[perm_1]
            logits_3[start:end] = logits[perm_2]
            feat_2[start:end] = projected[perm_1]
            feat_3[start:end] = projected[perm_2]

        lam = torch.distributions.Beta(0.5, 0.5).sample().to(x.device)
        logits_mix = lam * logits_2 + (1.0 - lam) * logits_3
        feat_mix = lam * feat_2 + (1.0 - lam) * feat_3

        ce = F.cross_entropy(logits, y)
        scale = min(float(ce.detach().item()), 1.0)
        mse = nn.MSELoss()
        loss_ind = mse(logits, logits_2) + 0.3 * mse(features, feat_2)
        loss_hdl = mse(logits, logits_mix) + 0.3 * mse(features, feat_mix)
        loss = ce + scale * (lam * loss_ind + (1.0 - lam) * loss_hdl)
        return loss, logits, {"loss": loss.item(), "ce": ce.item()}


class ARMAlgorithm(ResNetAlgorithm):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__(cfg, in_channels=4)
        self.context_net = ContextNet(in_channels=3)
        self.support_size = cfg.batch_size
        self.configure_optimizer(self.parameters())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size = 1
            support_size = batch_size

        context = self.context_net(x)
        context = context.reshape(meta_batch_size, support_size, 1, height, width)
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x_context = torch.cat([x, context], dim=1)
        return self.logits_from_features(self.forward_features(x_context))


class SagNetAlgorithm(Algorithm):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__(cfg)
        self.network_f = ResNetFeaturizer(cfg.backbone, cfg.pretrained)
        self.network_c = LinearClassifier(self.network_f.n_outputs, cfg.num_classes)
        self.network_s = LinearClassifier(self.network_f.n_outputs, cfg.num_classes)
        self.optimizer_f = torch.optim.SGD(
            self.network_f.parameters(), lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        self.optimizer_c = torch.optim.SGD(
            self.network_c.parameters(), lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        self.optimizer_s = torch.optim.SGD(
            self.network_s.parameters(), lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )

    def optimizers(self):
        return [self.optimizer_f, self.optimizer_c, self.optimizer_s]

    def randomize(self, x: torch.Tensor, what: str = "style",
                  eps: float = 1e-5) -> torch.Tensor:
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1, device=x.device)
        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        normalized = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0], device=x.device)
        if what == "style":
            mean = alpha * mean + (1.0 - alpha) * mean[idx_swap]
            var = alpha * var + (1.0 - alpha) * var[idx_swap]
        else:
            normalized = normalized[idx_swap].detach()

        out = normalized * (var + eps).sqrt() + mean
        return out.view(*sizes)

    def forward_c(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def update(self, batch):
        x, y, _ = parse_batch(batch)

        self.optimizer_f.zero_grad(set_to_none=True)
        self.optimizer_c.zero_grad(set_to_none=True)
        loss_c = F.cross_entropy(self.forward_c(x), y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        self.optimizer_s.zero_grad(set_to_none=True)
        loss_s = F.cross_entropy(self.forward_s(x), y)
        loss_s.backward()
        self.optimizer_s.step()

        self.optimizer_f.zero_grad(set_to_none=True)
        loss_adv = -F.log_softmax(self.forward_s(x), dim=1).mean(1).mean()
        loss_adv = self.cfg.sag_w_adv * loss_adv
        loss_adv.backward()
        self.optimizer_f.step()

        with torch.no_grad():
            logits = self.predict(x)
        return {
            "loss": loss_c.item(),
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
            "acc": accuracy_from_logits(logits, y),
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_c(self.network_f(x))


class EQRMAlgorithm(ResNetAlgorithm):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__(cfg)
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

    def objective(self, batch):
        x, y, domains = parse_batch(batch)
        logits = self.predict(x)
        losses = []
        for domain in domains.unique(sorted=True):
            idx = torch.nonzero(domains == domain, as_tuple=False).flatten()
            if idx.numel() > 0:
                losses.append(F.cross_entropy(logits[idx], y[idx]))
        env_risks = torch.stack(losses) if losses else F.cross_entropy(
            logits, y
        ).reshape(1)

        if int(self.update_count.item()) < self.cfg.eqrm_burnin_iters:
            loss = env_risks.mean()
        else:
            loss = torch.quantile(env_risks, self.cfg.eqrm_quantile)
        return loss, logits, {
            "loss": loss.item(),
            "risk_mean": env_risks.mean().item(),
        }

    def update(self, batch):
        metrics = super().update(batch)
        self.update_count += 1
        return metrics


class SAGMAlgorithm(ResNetAlgorithm):
    def gradient_matching_penalty(self, features, logits, labels, domains):
        grads = []
        for domain in domains.unique(sorted=True):
            idx = torch.nonzero(domains == domain, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            loss_i = F.cross_entropy(logits[idx], labels[idx])
            grad_i = autograd.grad(
                loss_i,
                features,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad_i is not None:
                grads.append(grad_i[idx].mean(0))
        if len(grads) < 2:
            return features.new_tensor(0.0)

        penalties = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                penalties.append(
                    1.0 - F.cosine_similarity(
                        grads[i].flatten(), grads[j].flatten(), dim=0
                    )
                )
        return torch.stack(penalties).mean()

    def objective(self, batch):
        x, y, domains = parse_batch(batch)
        features = self.forward_features(x)
        logits = self.logits_from_features(features)
        ce = F.cross_entropy(logits, y)
        penalty = self.gradient_matching_penalty(features, logits, y, domains)
        loss = ce + self.cfg.sagm_gamma * penalty
        return loss, logits, {
            "loss": loss.item(),
            "ce": ce.item(),
            "gm_penalty": penalty.item(),
        }

    def _grad_norm(self) -> torch.Tensor:
        norms = []
        for param in self.parameters():
            if param.grad is not None:
                norms.append(param.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)

    def _add_sam_perturbation(self, scale: torch.Tensor):
        perturbations = []
        for param in self.parameters():
            if param.grad is None:
                perturbations.append(None)
                continue
            perturb = param.grad * scale.to(param.device)
            param.data.add_(perturb)
            perturbations.append(perturb)
        return perturbations

    def _remove_sam_perturbation(self, perturbations):
        for param, perturb in zip(self.parameters(), perturbations):
            if perturb is not None:
                param.data.sub_(perturb)

    def update(self, batch):
        x, y, _ = parse_batch(batch)
        loss, logits, metrics = self.objective(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._grad_norm()
        scale = self.cfg.sam_rho / (grad_norm + 1e-12)
        perturbations = self._add_sam_perturbation(scale)

        sharp_loss, sharp_logits, sharp_metrics = self.objective(batch)
        self.optimizer.zero_grad(set_to_none=True)
        sharp_loss.backward()
        self._remove_sam_perturbation(perturbations)
        self.optimizer.step()

        metrics["sharp_loss"] = sharp_metrics["loss"]
        metrics["acc"] = accuracy_from_logits(sharp_logits.detach(), y)
        return metrics


def build_algorithm(**kwargs) -> Algorithm:
    cfg = AlgorithmConfig(**kwargs)
    method = cfg.method.lower()
    if cfg.architecture_tag in ARCHITECTURE_CLASSES:
        return M2Algorithm(cfg)
    if method in M2_METHODS:
        return M2Algorithm(cfg)
    if method == "erm":
        return ResNetAlgorithm(cfg)
    if method == "mixup":
        return MixupAlgorithm(cfg)
    if method == "coral":
        return CORALAlgorithm(cfg)
    if method == "mmd":
        return MMDAlgorithm(cfg)
    if method == "rsc":
        return RSCAlgorithm(cfg)
    if method == "selfreg":
        return SelfRegAlgorithm(cfg)
    if method == "arm":
        return ARMAlgorithm(cfg)
    if method == "sagnet":
        return SagNetAlgorithm(cfg)
    if method == "eqrm":
        return EQRMAlgorithm(cfg)
    if method == "sagm":
        return SAGMAlgorithm(cfg)
    raise ValueError(f"Unknown method: {cfg.method}")
