import math
from dataclasses import dataclass

import numpy as np
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
    optimizer: str = "adam"
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
    eqrm_lr: float = 1e-6
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


def domain_minibatches(batch):
    x, y, domains = parse_batch(batch)
    minibatches = []
    for domain in domains.unique(sorted=True):
        idx = torch.nonzero(domains == domain, as_tuple=False).flatten()
        if idx.numel() > 0:
            minibatches.append((x[idx], y[idx]))
    if not minibatches:
        minibatches.append((x, y))
    return minibatches


def cat_minibatches(minibatches):
    return (
        torch.cat([x for x, _ in minibatches]),
        torch.cat([y for _, y in minibatches]),
    )


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []
    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        xi, yi = minibatches[perm[i]]
        xj, yj = minibatches[perm[j]]
        n = min(len(xi), len(xj))
        if n > 0:
            pairs.append(((xi[:n], yi[:n]), (xj[:n], yj[:n])))
    return pairs


class Kernel(nn.Module):
    def __init__(self, bw=None):
        super().__init__()
        self.bw = 0.05 if bw is None else bw

    def _diffs(self, test_xs, train_xs):
        test_xs = test_xs.view(test_xs.shape[0], 1, *test_xs.shape[1:])
        train_xs = train_xs.view(1, train_xs.shape[0], *train_xs.shape[1:])
        return test_xs - train_xs


class GaussianKernel(Kernel):
    def forward(self, test_xs, train_xs):
        diffs = self._diffs(test_xs, train_xs)
        dims = tuple(range(len(diffs.shape))[2:])
        x_sq = diffs ** 2 if dims == () else torch.norm(diffs, p=2, dim=dims) ** 2
        var = self.bw ** 2
        coef = 1.0 / torch.sqrt(2 * np.pi * var)
        return (coef * torch.exp(-x_sq / (2 * var))).mean(dim=1)

    def sample(self, train_xs):
        noise = torch.randn(train_xs.shape, device=train_xs.device) * self.bw
        return train_xs + noise

    def cdf(self, test_xs, train_xs):
        mus = train_xs
        sigmas = torch.ones(len(mus), device=test_xs.device) * self.bw
        x = test_xs.repeat(len(mus), 1).T
        return torch.mean(torch.distributions.Normal(mus, sigmas).cdf(x))


def estimate_bandwidth(x, method="silverman"):
    x, _ = torch.sort(x)
    n = len(x)
    if n < 2:
        return torch.ones((), device=x.device, dtype=x.dtype) * 1e-6
    sample_std = torch.std(x, unbiased=True).clamp_min(1e-12)
    method = method.lower()
    if method == "silverman":
        iqr = torch.quantile(x, 0.75) - torch.quantile(x, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)
    elif method == "gauss-optimal":
        bandwidth = 1.06 * sample_std * (n ** -0.2)
    else:
        raise ValueError(f"Invalid bandwidth method: {method}")
    return bandwidth.clamp_min(1e-6)


class KernelDensityEstimator(nn.Module):
    def __init__(self, train_xs, kernel="gaussian", bw_select="Gauss-optimal"):
        super().__init__()
        self.train_xs = train_xs
        self._n_kernels = len(train_xs)
        self.bw = (
            estimate_bandwidth(self.train_xs, bw_select)
            if bw_select is not None else None
        )
        if kernel.lower() != "gaussian":
            raise NotImplementedError(f"'{kernel}' kernel not implemented.")
        self.kernel = GaussianKernel(self.bw)

    def forward(self, x):
        return self.kernel(x, self.train_xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(self._n_kernels), size=n_samples)
        return self.kernel.sample(self.train_xs[idxs])

    def cdf(self, x):
        return self.kernel.cdf(x, self.train_xs)


def continuous_bisect_fun_left(f, value, lo, hi, n_steps=32):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for _ in range(n_steps):
        val_range[int((f(k) > value).detach().cpu().item())] = k
        next_k = 0.5 * sum(val_range)
        if bool((next_k == k).detach().cpu().item()):
            break
        k = next_k
    return k


class Nonparametric:
    def __init__(self, use_kde=True, bw_select="Gauss-optimal"):
        self.use_kde = use_kde
        self.bw_select = bw_select
        self.bw = None
        self.data = None
        self.kde = None

    def estimate_parameters(self, x):
        self.data, _ = torch.sort(x)
        if self.use_kde:
            self.kde = KernelDensityEstimator(self.data, bw_select=self.bw_select)
            self.bw = torch.ones(1, device=self.data.device) * self.kde.bw

    def _empirical_icdf(self, q_value):
        if self.data.numel() == 1:
            return self.data[0]
        rank = q_value * (self.data.numel() - 1)
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        weight = rank - low
        return self.data[low] * (1.0 - weight) + self.data[high] * weight

    def icdf(self, q):
        if torch.is_tensor(q):
            q_value = float(q.detach().cpu().item())
            q_tensor = q.to(device=self.data.device, dtype=self.data.dtype)
        else:
            q_value = float(q)
            q_tensor = torch.tensor(q_value, device=self.data.device,
                                    dtype=self.data.dtype)
        if not self.use_kde or self.data.numel() < 2:
            return self._empirical_icdf(q_value)
        if q_value >= 0:
            lo = torch.distributions.Normal(self.data[0], self.bw[0]).icdf(q_tensor)
            hi = torch.distributions.Normal(self.data[-1], self.bw[-1]).icdf(q_tensor)
            return continuous_bisect_fun_left(self.kde.cdf, q_tensor, lo, hi)
        log_y = q_value
        return torch.mean(self.data + self.bw * math.sqrt(-2 * log_y))


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
        if self.cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def optimizers(self):
        return [self.optimizer] if self.optimizer is not None else []

    def update(self, batch):
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum
    model.apply(_enable)


class ConstantScheduler:
    def __init__(self, value: float):
        self.value = value

    def step(self):
        return self.value


class SAGMOptimizer(torch.optim.Optimizer):
    """Minimal single-process port of the official M2CL SAGM optimizer."""

    def __init__(self, params, base_optimizer, model, alpha: float,
                 rho_scheduler, adaptive: bool = False,
                 perturb_eps: float = 1e-12):
        super().__init__(params, dict(adaptive=adaptive))
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.rho_scheduler = rho_scheduler
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.update_rho_t()

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def _grad_norm(self):
        norms = [
            ((torch.abs(p.data) if self.adaptive else 1.0) * p.grad).norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def perturb_weights(self, rho: float):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps) - self.alpha
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["old_g"] = param.grad.data.clone()
                perturb = param.grad * scale.to(param)
                if self.adaptive:
                    perturb *= torch.pow(param, 2)
                param.add_(perturb)
                self.state[param]["e_w"] = perturb

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for param in group["params"]:
                if "e_w" in self.state[param]:
                    param.data.sub_(self.state[param]["e_w"])

    @torch.no_grad()
    def gradient_decompose(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                sam_grad = self.state[param]["old_g"] * 0.5 - param.grad * 0.5
                param.grad.data.add_(sam_grad)

    def set_closure(self, loss_fn, inputs, targets):
        def get_grad():
            self.base_optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
            loss_value = loss.detach().clone()
            loss.backward()
            return outputs, loss_value
        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):
        get_grad = closure if closure is not None else self.forward_backward_func
        outputs, loss_value = get_grad()
        self.perturb_weights(rho=self.rho_t)
        disable_running_stats(self.model)
        get_grad()
        self.gradient_decompose()
        self.unperturb()
        self.base_optimizer.step()
        enable_running_stats(self.model)
        return outputs, loss_value


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
        minibatches = domain_minibatches(batch)
        x, y = cat_minibatches(minibatches)
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
        minibatches = domain_minibatches(batch)
        x, y = cat_minibatches(minibatches)
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
        minibatches = domain_minibatches(batch)
        x, y = cat_minibatches(minibatches)
        alpha = self.cfg.mixup_alpha
        if alpha <= 0:
            logits = self.predict(x)
            loss = F.cross_entropy(logits, y)
            return loss, logits, {"loss": loss.item()}

        if len(minibatches) < 2:
            perm = torch.randperm(x.size(0), device=x.device)
            lam = float(np.random.beta(alpha, alpha))
            mixed_x = lam * x + (1.0 - lam) * x[perm]
            logits = self.predict(mixed_x)
            loss = lam * F.cross_entropy(logits, y)
            loss = loss + (1.0 - lam) * F.cross_entropy(logits, y[perm])
            return loss, self.predict(x), {"loss": loss.item()}

        objective = x.new_tensor(0.0)
        pairs = random_pairs_of_minibatches(minibatches)
        for (xi, yi), (xj, yj) in pairs:
            lam = float(np.random.beta(alpha, alpha))
            mixed_x = lam * xi + (1.0 - lam) * xj
            mixed_logits = self.predict(mixed_x)
            objective = objective + lam * F.cross_entropy(mixed_logits, yi)
            objective = objective + (1.0 - lam) * F.cross_entropy(
                mixed_logits, yj
            )
        loss = objective / max(1, len(minibatches))
        return loss, self.predict(x), {"loss": loss.item()}


class DistributionMatchingAlgorithm(ResNetAlgorithm):
    penalty = staticmethod(coral_penalty)

    def objective(self, batch):
        minibatches = domain_minibatches(batch)
        features = [self.forward_features(xi) for xi, _ in minibatches]
        logits_by_domain = [
            self.logits_from_features(feat) for feat in features
        ]
        targets = [yi for _, yi in minibatches]
        objective = sum(
            F.cross_entropy(logits, target)
            for logits, target in zip(logits_by_domain, targets)
        ) / len(minibatches)
        penalty = objective.new_tensor(0.0)
        for i in range(len(minibatches)):
            for j in range(i + 1, len(minibatches)):
                if len(features[i]) > 1 and len(features[j]) > 1:
                    penalty = penalty + self.penalty(features[i], features[j])
        if len(minibatches) > 1:
            penalty = penalty / (len(minibatches) * (len(minibatches) - 1) / 2)
        loss = objective + self.cfg.penalty_weight * penalty
        logits = torch.cat(logits_by_domain)
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
        minibatches = domain_minibatches(batch)
        x, y = cat_minibatches(minibatches)
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

    def objective(self, batch):
        minibatches = domain_minibatches(batch)
        x, y = cat_minibatches(minibatches)
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

        lam = float(np.random.beta(0.5, 0.5))
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
        self.optimizer_f = torch.optim.Adam(
            self.network_f.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.optimizer_c = torch.optim.Adam(
            self.network_c.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.optimizer_s = torch.optim.Adam(
            self.network_s.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
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
        minibatches = domain_minibatches(batch)
        x, y = cat_minibatches(minibatches)

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
        self.register_buffer(
            "eqrm_alpha", torch.tensor(cfg.eqrm_quantile, dtype=torch.float32)
        )
        self.dist = Nonparametric()

    def objective(self, batch):
        minibatches = domain_minibatches(batch)
        env_risks = torch.cat([
            F.cross_entropy(self.predict(xi), yi).reshape(1)
            for xi, yi in minibatches
        ])
        x, _ = cat_minibatches(minibatches)
        logits = self.predict(x)

        if int(self.update_count.item()) < self.cfg.eqrm_burnin_iters:
            loss = env_risks.mean()
        else:
            self.dist.estimate_parameters(env_risks)
            loss = self.dist.icdf(self.eqrm_alpha)
        return loss, logits, {
            "loss": loss.item(),
            "risk_mean": env_risks.mean().item(),
        }

    def update(self, batch):
        if int(self.update_count.item()) == self.cfg.eqrm_burnin_iters:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.eqrm_lr,
                weight_decay=self.cfg.weight_decay,
            )
        loss, logits, metrics = self.objective(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        minibatches = domain_minibatches(batch)
        _, y = cat_minibatches(minibatches)
        metrics["acc"] = accuracy_from_logits(logits, y)
        self.update_count += 1
        return metrics


class SAGMAlgorithm(ResNetAlgorithm):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__(cfg)
        self.base_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.sagm_optimizer = SAGMOptimizer(
            params=self.parameters(),
            base_optimizer=self.base_optimizer,
            model=self,
            alpha=cfg.sagm_gamma,
            rho_scheduler=ConstantScheduler(cfg.sam_rho),
            adaptive=False,
        )

    def optimizers(self):
        return [self.base_optimizer]

    def update(self, batch):
        minibatches = domain_minibatches(batch)
        x, y = cat_minibatches(minibatches)

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.sagm_optimizer.set_closure(loss_fn, x, y)
        logits, loss = self.sagm_optimizer.step()
        self.sagm_optimizer.update_rho_t()
        return {
            "loss": loss.item(),
            "acc": accuracy_from_logits(logits.detach(), y),
        }


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
