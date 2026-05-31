"""
Run command grids for baseline and main M2/M2-CL experiments.

This script only orchestrates training jobs. It does not hide any training
logic; every printed command is a normal train.py invocation.
"""
import argparse
import subprocess
import sys

from algorithms import BASELINE_METHODS, METHODS
from train import checkpoint_path, load_config


def dataset_splits(dataset, cfg, nico_values):
    if dataset == "nico":
        return [("NICO", str(n)) for n in nico_values]
    return [("DOMAIN", domain) for domain in cfg["domains"]]


def append_if_not_none(command, flag, value):
    if value is not None:
        command.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["pacs", "vlcs", "office_home", "nico"])
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--methods", nargs="+",
                        choices=[*METHODS, "paper_baselines", "all"],
                        default=["all"])
    parser.add_argument("--backbones", nargs="+", choices=["resnet18", "resnet50"],
                        default=["resnet18"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--nico_values", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint_freq", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hparams_profile",
                        choices=["paper", "domainbed", "project"],
                        default="paper")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--holdout_fraction", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--penalty_weight", type=float, default=None)
    parser.add_argument("--mixup_alpha", type=float, default=None)
    parser.add_argument("--sam_rho", type=float, default=None)
    parser.add_argument("--sagm_gamma", type=float, default=None)
    parser.add_argument("--eqrm_lr", type=float, default=None)
    parser.add_argument("--save_dir", default="./outputs/checkpoints")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--wandb", action="store_true",
                        help="Forward W&B logging flags to each train.py run")
    parser.add_argument("--wandb_project", default="m2cl-domain-generalization")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                        default=None)
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    methods = args.methods
    if "all" in methods:
        methods = METHODS
    elif "paper_baselines" in methods:
        methods = BASELINE_METHODS

    commands = []
    for method in methods:
        for backbone in args.backbones:
            for seed in args.seeds:
                for split_type, split_value in dataset_splits(
                    args.dataset, cfg, args.nico_values
                ):
                    command = [
                        sys.executable,
                        "train.py",
                        "--dataset", args.dataset,
                        "--data_root", args.data_root,
                        "--method", method,
                        "--backbone", backbone,
                        "--seed", str(seed),
                        "--save_dir", args.save_dir,
                    ]
                    if split_type == "NICO":
                        command.extend(["--n_heldout", split_value])
                    else:
                        command.extend(["--test_domain", split_value])

                    append_if_not_none(command, "--epochs", args.epochs)
                    append_if_not_none(command, "--steps", args.steps)
                    append_if_not_none(
                        command, "--checkpoint_freq", args.checkpoint_freq
                    )
                    append_if_not_none(command, "--batch_size", args.batch_size)
                    append_if_not_none(command, "--lr", args.lr)
                    append_if_not_none(
                        command, "--hparams_profile", args.hparams_profile
                    )
                    append_if_not_none(command, "--optimizer", args.optimizer)
                    append_if_not_none(
                        command, "--weight_decay", args.weight_decay
                    )
                    append_if_not_none(
                        command, "--holdout_fraction", args.holdout_fraction
                    )
                    append_if_not_none(command, "--num_workers", args.num_workers)
                    append_if_not_none(command, "--scheduler", args.scheduler)
                    append_if_not_none(command, "--penalty_weight", args.penalty_weight)
                    append_if_not_none(command, "--mixup_alpha", args.mixup_alpha)
                    append_if_not_none(command, "--sam_rho", args.sam_rho)
                    append_if_not_none(command, "--sagm_gamma", args.sagm_gamma)
                    append_if_not_none(command, "--eqrm_lr", args.eqrm_lr)
                    if args.no_pretrained:
                        command.append("--no_pretrained")
                    if args.wandb:
                        command.append("--wandb")
                        append_if_not_none(
                            command, "--wandb_project", args.wandb_project
                        )
                        append_if_not_none(
                            command, "--wandb_entity", args.wandb_entity
                        )
                        append_if_not_none(
                            command, "--wandb_group", args.wandb_group
                        )
                        append_if_not_none(
                            command, "--wandb_mode", args.wandb_mode
                        )
                        if args.wandb_tags:
                            command.append("--wandb_tags")
                            command.extend(args.wandb_tags)
                    split_name = f"N{split_value}" if split_type == "NICO" else split_value
                    expected_metrics = checkpoint_path(
                        args.save_dir, args.dataset, split_name,
                        method, backbone, seed
                    ).with_suffix(".json")
                    commands.append((command, expected_metrics))

    print(f"Planned runs: {len(commands)}")
    for command, expected_metrics in commands:
        if args.skip_existing and expected_metrics.exists():
            print(f"[SKIP] {expected_metrics}")
            continue
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
