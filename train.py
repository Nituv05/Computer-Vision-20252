"""
Train paper baselines, M2 or M2-CL on a domain generalization split.

Examples:
  python train.py --dataset pacs --test_domain photo --method m2cl
  python train.py --dataset nico --n_heldout 7 --method m2cl
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from algorithms import METHODS, build_algorithm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


def move_batch_to_device(batch, device):
    return tuple(item.to(device) for item in batch)


def train_one_epoch(algorithm, loader, device):
    algorithm.train()
    total_loss, total_acc = 0.0, 0.0
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        metrics = algorithm.update(batch)
        total_loss += metrics.get("loss", 0.0)
        total_acc += metrics.get("acc", 0.0)
    n = max(1, len(loader))
    return total_loss / n, total_acc / n


def train_one_step(algorithm, batch, device):
    algorithm.train()
    batch = move_batch_to_device(batch, device)
    metrics = algorithm.update(batch)
    return metrics.get("loss", 0.0), metrics.get("acc", 0.0)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(algorithm, loader, device):
    algorithm.eval()
    correct, total = 0, 0
    for batch in loader:
        imgs, labels = batch[:2]
        imgs, labels = imgs.to(device), labels.to(device)
        logits = algorithm.predict(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


class SplitDataset(Dataset):
    """DomainBed-style dataset split wrapper."""

    def __init__(self, underlying_dataset: Dataset, keys: list[int]):
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, index):
        return self.underlying_dataset[self.keys[index]]

    def __len__(self):
        return len(self.keys)


class DomainDataset(Dataset):
    """Attach a source-environment id to each sample for DG baselines."""

    def __init__(self, underlying_dataset: Dataset, domain_id: int):
        self.underlying_dataset = underlying_dataset
        self.domain_id = domain_id

    def __getitem__(self, index):
        image, label = self.underlying_dataset[index]
        return image, label, self.domain_id

    def __len__(self):
        return len(self.underlying_dataset)


class _CombinedDatasetView:
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)


class MultiDomainLoader:
    """Yield one minibatch per source environment, matching DomainBed style."""

    def __init__(self, datasets: list[Dataset], batch_size: int,
                 shuffle: bool, num_workers: int, pin_memory: bool):
        self.loaders = [
            DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin_memory
            )
            for dataset in datasets
        ]
        self.dataset = _CombinedDatasetView(datasets)

    def __iter__(self):
        for batches in zip(*self.loaders):
            items = list(zip(*batches))
            yield tuple(torch.cat(list(values), dim=0) for values in items)

    def __len__(self):
        return min(len(loader) for loader in self.loaders)


def seed_hash(*args) -> int:
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def split_dataset(dataset: Dataset, n: int, seed: int = 0):
    if n > len(dataset):
        raise ValueError(f"Cannot split {n} samples from dataset of size {len(dataset)}")
    keys = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(keys)
    return SplitDataset(dataset, keys[:n]), SplitDataset(dataset, keys[n:])


def _concat(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("Expected at least one dataset")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _assert_nonempty(name: str, dataset: Dataset) -> None:
    if len(dataset) == 0:
        raise ValueError(
            f"{name} has 0 samples. Check --data_root and expected folder layout."
        )


def split_source_environments(source_envs: list[Dataset], target_dataset: Dataset,
                              batch_size: int, num_workers: int,
                              holdout_fraction: float, seed: int,
                              domainbed_batching: bool = False):
    """
    Match DomainBed's source-domain split style: each source environment is
    split independently into in/out subsets using seed_hash(trial_seed, env_i).
    """
    pin_memory = torch.cuda.is_available()
    for env_i, env in enumerate(source_envs):
        _assert_nonempty(f"source_env_{env_i}", env)
    _assert_nonempty("target_env", target_dataset)

    if holdout_fraction <= 0:
        train_parts = [
            DomainDataset(env, env_i) for env_i, env in enumerate(source_envs)
        ]
        if domainbed_batching:
            train_loader = MultiDomainLoader(
                train_parts, batch_size, True, num_workers, pin_memory
            )
        else:
            train_dataset = _concat(train_parts)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory
            )
        test_loader = DataLoader(
            target_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        return train_loader, None, test_loader

    train_parts = []
    val_parts = []
    for env_i, env in enumerate(source_envs):
        n_out = int(len(env) * holdout_fraction)
        n_out = min(max(1, n_out), len(env) - 1)
        val_env, train_env = split_dataset(env, n_out, seed_hash(seed, env_i))
        train_parts.append(DomainDataset(train_env, env_i))
        val_parts.append(val_env)

    train_dataset = _concat(train_parts)
    val_dataset = _concat(val_parts)

    if domainbed_batching:
        train_loader = MultiDomainLoader(
            train_parts, batch_size, True, num_workers, pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader


def load_config(dataset: str):
    cfg_path = Path(__file__).parent / "configs" / f"{dataset}.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_datasets(dataset: str, data_root: str, test_domain: str | None,
                   n_heldout: int | None, seed: int):
    if dataset == "pacs":
        from data.pacs import DOMAINS, PACSDataset
        if test_domain is None:
            raise ValueError("--test_domain is required for PACS")
        source_envs = [
            PACSDataset(data_root, domain, "train")
            for domain in DOMAINS if domain != test_domain
        ]
        target_dataset = PACSDataset(data_root, test_domain, "test")
        split_name = test_domain
    elif dataset == "vlcs":
        from data.vlcs import DOMAINS, VLCSDataset
        if test_domain is None:
            raise ValueError("--test_domain is required for VLCS")
        source_envs = [
            VLCSDataset(data_root, domain, "train")
            for domain in DOMAINS if domain != test_domain
        ]
        target_dataset = VLCSDataset(data_root, test_domain, "test")
        split_name = test_domain
    elif dataset == "office_home":
        from data.office_home import DOMAINS, OfficeHomeDataset
        if test_domain is None:
            raise ValueError("--test_domain is required for Office-Home")
        source_envs = [
            OfficeHomeDataset(data_root, domain, "train")
            for domain in DOMAINS if domain != test_domain
        ]
        target_dataset = OfficeHomeDataset(data_root, test_domain, "test")
        split_name = test_domain
    elif dataset == "nico":
        from data.nico import NICODataset, build_context_split
        if n_heldout is None:
            raise ValueError("--n_heldout is required for NICO")
        train_contexts, test_contexts = build_context_split(
            data_root, n_heldout, seed
        )
        source_envs = [NICODataset(data_root, train_contexts, "train")]
        target_dataset = NICODataset(data_root, test_contexts, "test")
        split_name = f"N{n_heldout}"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return source_envs, target_dataset, split_name


def build_raw_loaders(dataset: str, data_root: str, test_domain: str | None,
                      n_heldout: int | None, batch_size: int,
                      num_workers: int, seed: int):
    source_envs, target_dataset, split_name = build_datasets(
        dataset, data_root, test_domain, n_heldout, seed
    )
    train_dataset = _concat(source_envs)
    return train_dataset, target_dataset, split_name


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def checkpoint_path(save_dir: str, dataset: str, split_name: str,
                    method: str, backbone: str, seed: int,
                    tag: str | None = None) -> Path:
    method_name = method if tag is None else f"{method}_{safe_name(tag)}"
    filename = (
        f"{dataset}_{safe_name(split_name)}_{method_name}_{backbone}_seed{seed}_best.pth"
    )
    return Path(save_dir) / filename


def init_wandb(args, cfg: dict, resolved: dict, split_name: str,
               checkpoint_file: Path):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested. Install it with `pip install wandb` "
            "or run `pip install -r requirements.txt`."
        ) from exc

    run_name = args.wandb_name or (
        f"{args.dataset}-{split_name}-{args.method}-"
        f"{resolved['backbone']}-seed{resolved['seed']}"
    )
    run_group = args.wandb_group or (
        f"{args.dataset}-{args.method}-{resolved['backbone']}"
    )
    wandb_config = {
        **resolved,
        "dataset": args.dataset,
        "split_name": split_name,
        "method": args.method,
        "tag": args.tag,
        "num_classes": cfg["num_classes"],
        "checkpoint": str(checkpoint_file),
        "source": "train.py",
    }
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": run_name,
        "group": run_group,
        "tags": args.wandb_tags,
        "config": wandb_config,
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    init_kwargs = {key: value for key, value in init_kwargs.items()
                   if value is not None}
    run = wandb.init(**init_kwargs)
    run.define_metric("epoch")
    run.define_metric("step")
    run.define_metric("train/*", step_metric="step")
    run.define_metric("val/*", step_metric="step")
    run.define_metric("test/*", step_metric="step")
    run.define_metric("best/*", step_metric="step")
    run.define_metric("optim/*", step_metric="step")
    return run


def resolve_default_hparams(args, cfg: dict):
    method = args.method.lower()
    is_m2 = method in {"m2", "m2cl"}
    profile = args.hparams_profile

    if profile in {"paper", "project"}:
        lr = args.lr if args.lr is not None else cfg["lr"]
        batch_size = (
            args.batch_size if args.batch_size is not None else cfg["batch_size"]
        )
        if profile == "paper" and not is_m2:
            lr = args.lr if args.lr is not None else 5e-5
            if args.batch_size is None:
                batch_size = 8 if method == "arm" else 32
            weight_decay_default = 0.0
            optimizer_default = "adam"
            eqrm_default = 2500
        else:
            weight_decay_default = 5e-4
            optimizer_default = "sgd"
            eqrm_default = 100
        weight_decay = args.weight_decay if args.weight_decay is not None else weight_decay_default
        optimizer = args.optimizer or optimizer_default
        eqrm_burnin_iters = args.eqrm_burnin_iters if args.eqrm_burnin_iters is not None else eqrm_default
    else:
        lr = args.lr if args.lr is not None else 5e-5
        if args.batch_size is not None:
            batch_size = args.batch_size
        elif method == "arm":
            batch_size = 8
        else:
            batch_size = 32
        weight_decay_default = 5e-4 if is_m2 else 0.0
        weight_decay = (
            args.weight_decay if args.weight_decay is not None
            else weight_decay_default
        )
        optimizer = args.optimizer or ("sgd" if is_m2 else "adam")
        eqrm_burnin_iters = (
            args.eqrm_burnin_iters
            if args.eqrm_burnin_iters is not None else 2500
        )

    return {
        "profile": profile,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "eqrm_burnin_iters": eqrm_burnin_iters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["pacs", "vlcs", "office_home", "nico"])
    parser.add_argument("--test_domain", default=None)
    parser.add_argument("--n_heldout", type=int, default=None,
                        help="NICO held-out contexts per class: 3, 5 or 7")
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--method", choices=METHODS, default="m2cl")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default=None)
    parser.add_argument("--pipeline_type", choices=["parallel", "cascading"],
                        default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None,
                        help="Use DomainBed-style update count instead of epochs")
    parser.add_argument("--checkpoint_freq", type=int, default=None,
                        help="Validation/checkpoint interval for --steps")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hparams_profile",
                        choices=["paper", "domainbed", "project"],
                        default="paper",
                        help="paper matches the M2-CL paper setup")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reduction_ratio", type=int, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--holdout_fraction", type=float, default=None,
                        help="DomainBed-style source validation holdout fraction")
    parser.add_argument("--val_fraction", dest="holdout_fraction",
                        type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--penalty_weight", type=float, default=1.0,
                        help="CORAL/MMD distribution penalty weight")
    parser.add_argument("--sag_w_adv", type=float, default=0.1)
    parser.add_argument("--rsc_f_drop_factor", type=float, default=1.0 / 3.0)
    parser.add_argument("--rsc_b_drop_factor", type=float, default=1.0 / 3.0)
    parser.add_argument("--eqrm_quantile", type=float, default=0.75)
    parser.add_argument("--eqrm_burnin_iters", type=int, default=None)
    parser.add_argument("--eqrm_lr", type=float, default=1e-6)
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--sagm_gamma", type=float, default=0.1)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--save_dir", default="./outputs/checkpoints")
    parser.add_argument("--tag", default=None,
                        help="Optional variant label for ablations/baselines")
    parser.add_argument("--wandb", action="store_true",
                        help="Log train/validation/test metrics to W&B")
    parser.add_argument("--wandb_project", default="m2cl-domain-generalization")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                        default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    set_seed(args.seed if args.seed is not None else cfg.get("seed", 0))

    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    resolved_hparams = resolve_default_hparams(args, cfg)
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    steps = args.steps
    checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq is not None else 300
    if steps is not None and steps <= 0:
        raise ValueError("--steps must be a positive integer")
    if steps is not None and checkpoint_freq <= 0:
        raise ValueError("--checkpoint_freq must be a positive integer")
    batch_size = resolved_hparams["batch_size"]
    lr = resolved_hparams["lr"]
    optimizer_name = resolved_hparams["optimizer"]
    weight_decay = resolved_hparams["weight_decay"]
    eqrm_burnin_iters = resolved_hparams["eqrm_burnin_iters"]
    alpha = args.alpha if args.alpha is not None else cfg["alpha"]
    temperature = (
        args.temperature if args.temperature is not None else cfg["temperature"]
    )
    backbone = args.backbone or cfg.get("backbone", "resnet18")
    pipeline_type = args.pipeline_type or cfg.get("pipeline_type", "parallel")
    reduction_ratio = (
        args.reduction_ratio if args.reduction_ratio is not None
        else cfg["reduction_ratio"]
    )
    dropout_p = args.dropout_p if args.dropout_p is not None else cfg["dropout_p"]
    embed_dim = args.embed_dim if args.embed_dim is not None else cfg["embed_dim"]
    holdout_fraction = (
        args.holdout_fraction if args.holdout_fraction is not None
        else cfg.get("holdout_fraction", cfg.get("val_fraction", 0.2))
    )
    num_workers = (
        args.num_workers if args.num_workers is not None
        else cfg.get("num_workers", 4)
    )
    pretrained = (
        args.pretrained if args.pretrained is not None
        else cfg.get("pretrained", True)
    )
    scheduler_name = args.scheduler or cfg.get("scheduler", "none")

    if args.method in {"m2"}:
        alpha = 0.0

    source_envs, target_set, split_name = build_datasets(
        args.dataset, args.data_root, args.test_domain, args.n_heldout, seed
    )
    use_domainbed_batching = (
        args.hparams_profile == "domainbed"
        or (args.hparams_profile == "paper" and args.method not in {"m2", "m2cl"})
    )
    train_loader, val_loader, test_loader = split_source_environments(
        source_envs, target_set, batch_size, num_workers, holdout_fraction,
        seed, domainbed_batching=use_domainbed_batching
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Dataset={args.dataset} split={split_name} method={args.method} "
        f"backbone={backbone} train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset) if val_loader else 0} "
        f"test={len(test_loader.dataset)} holdout={holdout_fraction} "
        f"profile={args.hparams_profile} "
        f"domainbed_batching={use_domainbed_batching} "
        f"steps={steps if steps is not None else 'epoch-mode'}"
    )

    algorithm_args = {
        "num_classes": cfg["num_classes"],
        "method": args.method,
        "backbone": backbone,
        "reduction_ratio": reduction_ratio,
        "dropout_p": dropout_p,
        "embed_dim": embed_dim,
        "pretrained": pretrained,
        "pipeline_type": pipeline_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "alpha": alpha,
        "temperature": temperature,
        "batch_size": batch_size,
        "mixup_alpha": args.mixup_alpha,
        "penalty_weight": args.penalty_weight,
        "sag_w_adv": args.sag_w_adv,
        "rsc_f_drop_factor": args.rsc_f_drop_factor,
        "rsc_b_drop_factor": args.rsc_b_drop_factor,
        "eqrm_quantile": args.eqrm_quantile,
        "eqrm_burnin_iters": eqrm_burnin_iters,
        "eqrm_lr": args.eqrm_lr,
        "sam_rho": args.sam_rho,
        "sagm_gamma": args.sagm_gamma,
        "architecture_tag": args.tag,
    }
    algorithm = build_algorithm(
        **algorithm_args,
    ).to(device)
    for optimizer in algorithm.optimizers():
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = weight_decay
    schedulers = []
    if scheduler_name == "cosine":
        schedulers = [
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps if steps is not None else epochs
            )
            for optimizer in algorithm.optimizers()
        ]
    elif scheduler_name != "none":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # Store reconstruction args with pretrained disabled for checkpoint reloads.
    checkpoint_algorithm_args = dict(algorithm_args)
    checkpoint_algorithm_args["pretrained"] = False

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = checkpoint_path(
        args.save_dir, args.dataset, split_name, args.method, backbone, seed,
        args.tag,
    )
    resolved_args = {
        "seed": seed,
        "epochs": epochs,
        "steps": steps,
        "checkpoint_freq": checkpoint_freq,
        "batch_size": batch_size,
        "lr": lr,
        "alpha": alpha,
        "temperature": temperature,
        "backbone": backbone,
        "pipeline_type": pipeline_type,
        "reduction_ratio": reduction_ratio,
        "dropout_p": dropout_p,
        "embed_dim": embed_dim,
        "holdout_fraction": holdout_fraction,
        "num_workers": num_workers,
        "pretrained": pretrained,
        "scheduler": scheduler_name,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "hparams_profile": args.hparams_profile,
        "domainbed_batching": use_domainbed_batching,
        "mixup_alpha": args.mixup_alpha,
        "penalty_weight": args.penalty_weight,
        "sag_w_adv": args.sag_w_adv,
        "rsc_f_drop_factor": args.rsc_f_drop_factor,
        "rsc_b_drop_factor": args.rsc_b_drop_factor,
        "eqrm_quantile": args.eqrm_quantile,
        "eqrm_burnin_iters": eqrm_burnin_iters,
        "eqrm_lr": args.eqrm_lr,
        "sam_rho": args.sam_rho,
        "sagm_gamma": args.sagm_gamma,
    }
    wandb_run = init_wandb(args, cfg, resolved_args, split_name, ckpt_path)
    best_score = -1.0
    best_progress = 0
    progress_name = "step" if steps is not None else "epoch"
    score_name = "val" if val_loader is not None else "test"
    score_loader = val_loader if val_loader is not None else test_loader

    def save_best_checkpoint(progress_value: int):
        torch.save({
            "epoch": progress_value if progress_name == "epoch" else None,
            "step": progress_value if progress_name == "step" else None,
            "progress_name": progress_name,
            "progress": progress_value,
            "state_dict": algorithm.state_dict(),
            "score": best_score,
            "score_name": score_name,
            "dataset": args.dataset,
            "split_name": split_name,
            "method": args.method,
            "tag": args.tag,
            "backbone": backbone,
            "num_classes": cfg["num_classes"],
            "model_args": checkpoint_algorithm_args,
            "train_args": {
                "holdout_fraction": holdout_fraction,
                "scheduler": scheduler_name,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "steps": steps,
                "checkpoint_freq": checkpoint_freq,
                "weight_decay": weight_decay,
                    "hparams_profile": args.hparams_profile,
                    "optimizer": optimizer_name,
                    "domainbed_batching": use_domainbed_batching,
                },
            }, ckpt_path)

    if steps is None:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                algorithm, train_loader, device
            )
            score_acc = evaluate(algorithm, score_loader, device)
            for scheduler in schedulers:
                scheduler.step()

            print(
                f"Epoch {epoch:03d}/{epochs} | loss={train_loss:.4f} "
                f"| train_acc={train_acc:.4f} | {score_name}_acc={score_acc:.4f}"
            )

            if score_acc > best_score:
                best_score = score_acc
                best_progress = epoch
                save_best_checkpoint(epoch)

            if wandb_run is not None:
                current_lr = algorithm.optimizers()[0].param_groups[0]["lr"]
                wandb_run.log({
                    "epoch": epoch,
                    "step": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    f"{score_name}/acc": score_acc,
                    "best/score": best_score,
                    "best/epoch": best_progress,
                    "optim/lr": current_lr,
                }, step=epoch)
    else:
        train_iter = infinite_loader(train_loader)
        running_loss, running_acc, running_count = 0.0, 0.0, 0
        for step in tqdm(range(1, steps + 1), desc="Train steps"):
            train_loss, train_acc = train_one_step(
                algorithm, next(train_iter), device
            )
            running_loss += train_loss
            running_acc += train_acc
            running_count += 1
            for scheduler in schedulers:
                scheduler.step()

            should_eval = step % checkpoint_freq == 0 or step == steps
            if not should_eval:
                continue

            avg_loss = running_loss / max(1, running_count)
            avg_acc = running_acc / max(1, running_count)
            score_acc = evaluate(algorithm, score_loader, device)
            running_loss, running_acc, running_count = 0.0, 0.0, 0

            print(
                f"Step {step:05d}/{steps} | loss={avg_loss:.4f} "
                f"| train_acc={avg_acc:.4f} | {score_name}_acc={score_acc:.4f}"
            )

            if score_acc > best_score:
                best_score = score_acc
                best_progress = step
                save_best_checkpoint(step)

            if wandb_run is not None:
                current_lr = algorithm.optimizers()[0].param_groups[0]["lr"]
                wandb_run.log({
                    "step": step,
                    "train/loss": avg_loss,
                    "train/acc": avg_acc,
                    f"{score_name}/acc": score_acc,
                    "best/score": best_score,
                    "best/step": best_progress,
                    "optim/lr": current_lr,
                }, step=step)

    ckpt = torch.load(ckpt_path, map_location=device)
    algorithm.load_state_dict(ckpt["state_dict"])
    test_acc = evaluate(algorithm, test_loader, device)

    metrics = {
        "dataset": args.dataset,
        "split": split_name,
        "method": args.method,
        "tag": args.tag,
        "backbone": backbone,
        "seed": seed,
        "best_epoch": best_progress if progress_name == "epoch" else None,
        "best_step": best_progress if progress_name == "step" else None,
        "progress_name": progress_name,
        f"best_{score_name}_acc": best_score,
        "test_acc": test_acc,
        "holdout_fraction": holdout_fraction,
        "scheduler": scheduler_name,
        "lr": lr,
        "optimizer": optimizer_name,
        "hparams_profile": args.hparams_profile,
        "domainbed_batching": use_domainbed_batching,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps": steps,
        "checkpoint_freq": checkpoint_freq,
        "weight_decay": weight_decay,
        "checkpoint": str(ckpt_path),
    }
    metrics_path = ckpt_path.with_suffix(".json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if wandb_run is not None:
        final_step = steps + 1 if steps is not None else epochs + 1
        final_log = {
            "step": final_step,
            "test/acc": test_acc,
            "best/final_score": best_score,
        }
        if progress_name == "epoch":
            final_log["epoch"] = epochs + 1
            final_log["best/final_epoch"] = best_progress
        else:
            final_log["best/final_step"] = best_progress
        wandb_run.log({
            **final_log,
        }, step=final_step)
        wandb_run.summary["test_acc"] = test_acc
        wandb_run.summary[f"best_{score_name}_acc"] = best_score
        wandb_run.summary[f"best_{progress_name}"] = best_progress
        wandb_run.summary["checkpoint"] = str(ckpt_path)
        wandb_run.summary["metrics_json"] = str(metrics_path)
        wandb_run.finish()

    print(f"\nBest {progress_name}: {best_progress} ({score_name}_acc={best_score:.4f})")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
