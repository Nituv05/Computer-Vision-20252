"""
Run the ablation and sensitivity studies used by the slide/report tables.

This is only an experiment orchestrator. Every planned job is printed as a
normal tools/train.py command, and each command writes a regular checkpoint plus a
metrics JSON file. The table modes correspond to:

  Table 5: architecture/loss ablation on PACS and VLCS.
  Table 6: temperature sensitivity for M2-CL.
  Table 7: contrastive-loss weight sensitivity for M2-CL.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from models.architecture_specs import ARCHITECTURE_SPECS

TABLE6_TAU_GRID = [
    0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0, 10.0, 100.0,
]

TABLE7_ALPHA_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

TABLE_TO_STUDY = {
    "5": "architecture",
    "6": "tau",
    "7": "alpha",
}


def load_config(dataset: str) -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / f"{dataset}.yaml"
    cfg = {}
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "domains":
            cfg["domains"] = [
                item.strip().strip("'\"")
                for item in value.strip("[]").split(",")
                if item.strip()
            ]
    if "domains" not in cfg:
        raise ValueError(f"Missing domains in {cfg_path}")
    return cfg


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


def safe_tag_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def append_if_not_none(command: list[str], flag: str, value):
    if value is not None:
        command.extend([flag, str(value)])


def selected_studies(args) -> list[str]:
    if args.tables:
        tables = args.tables
        if "all" in tables:
            tables = ["5", "6", "7"]
        return [TABLE_TO_STUDY[table] for table in tables]
    if args.study == "all":
        return ["architecture", "tau", "alpha"]
    return [args.study]


def split_values(dataset: str, cfg: dict, args, table_mode: bool):
    if table_mode or args.all_domains:
        return [("DOMAIN", domain) for domain in cfg["domains"]]
    return [("DOMAIN", args.test_domain)]


def add_common_train_flags(command: list[str], args):
    append_if_not_none(command, "--epochs", args.epochs)
    append_if_not_none(command, "--steps", args.steps)
    append_if_not_none(command, "--checkpoint_freq", args.checkpoint_freq)
    append_if_not_none(command, "--batch_size", args.batch_size)
    append_if_not_none(command, "--lr", args.lr)
    append_if_not_none(command, "--hparams_profile", args.hparams_profile)
    append_if_not_none(command, "--optimizer", args.optimizer)
    append_if_not_none(command, "--weight_decay", args.weight_decay)
    append_if_not_none(command, "--holdout_fraction", args.holdout_fraction)
    append_if_not_none(command, "--num_workers", args.num_workers)
    append_if_not_none(command, "--scheduler", args.scheduler)
    if args.no_pretrained:
        command.append("--no_pretrained")


def add_wandb_flags(command: list[str], args, dataset: str, split_name: str,
                    tag: str, study: str, seed: int):
    if not args.wandb:
        return
    command.append("--wandb")
    append_if_not_none(command, "--wandb_project", args.wandb_project)
    append_if_not_none(command, "--wandb_entity", args.wandb_entity)
    append_if_not_none(command, "--wandb_mode", args.wandb_mode)
    group = args.wandb_group or f"{dataset}-{study}-{args.backbone}"
    command.extend(["--wandb_group", group])
    command.extend([
        "--wandb_name",
        f"{dataset}-{split_name}-{tag}-{args.backbone}-seed{seed}",
    ])
    tags = list(args.wandb_tags or [])
    tags.extend([study, tag, args.backbone])
    command.append("--wandb_tags")
    command.extend(tags)


def base_command(args, dataset: str, split_type: str, split_value: str,
                 method: str, seed: int, tag: str) -> tuple[list[str], str]:
    command = [
        sys.executable,
        "tools/train.py",
        "--dataset", dataset,
        "--data_root", args.data_root,
        "--method", method,
        "--tag", tag,
        "--backbone", args.backbone,
        "--seed", str(seed),
        "--save_dir", args.save_dir,
    ]
    split_name = split_value
    command.extend(["--test_domain", split_value])
    add_common_train_flags(command, args)
    return command, split_name


def architecture_commands(args, dataset: str, cfg: dict, seeds: list[int],
                          table_mode: bool):
    commands = []
    for spec in ARCHITECTURE_SPECS:
        for seed in seeds:
            for split_type, split_value in split_values(dataset, cfg, args, table_mode):
                tag = spec.tag
                command, split_name = base_command(
                    args, dataset, split_type, split_value,
                    spec.method, seed, tag,
                )
                command.extend(spec.train_flags())
                if spec.method == "m2cl":
                    command.extend(["--alpha", str(args.alpha)])
                    command.extend(["--temperature", str(args.temperature)])
                add_wandb_flags(
                    command, args, dataset, split_name, tag, "table5", seed
                )
                commands.append((command, dataset, split_name, spec.method, seed, tag))
    return commands


def tau_commands(args, dataset: str, cfg: dict, seeds: list[int],
                 table_mode: bool):
    commands = []
    for tau in TABLE6_TAU_GRID:
        tag = f"table6_tau_{safe_tag_float(tau)}"
        for seed in seeds:
            for split_type, split_value in split_values(dataset, cfg, args, table_mode):
                command, split_name = base_command(
                    args, dataset, split_type, split_value, "m2cl", seed, tag
                )
                command.extend([
                    "--pipeline_type", "parallel",
                    "--reduction_ratio", "4",
                    "--dropout_p", "0.3",
                    "--alpha", str(args.alpha),
                    "--temperature", str(tau),
                ])
                add_wandb_flags(
                    command, args, dataset, split_name, tag, "table6", seed
                )
                commands.append((command, dataset, split_name, "m2cl", seed, tag))
    return commands


def alpha_commands(args, dataset: str, cfg: dict, seeds: list[int],
                   table_mode: bool):
    commands = []
    for alpha in TABLE7_ALPHA_GRID:
        tag = f"table7_alpha_{safe_tag_float(alpha)}"
        for seed in seeds:
            for split_type, split_value in split_values(dataset, cfg, args, table_mode):
                command, split_name = base_command(
                    args, dataset, split_type, split_value, "m2cl", seed, tag
                )
                command.extend([
                    "--pipeline_type", "parallel",
                    "--reduction_ratio", "4",
                    "--dropout_p", "0.3",
                    "--alpha", str(alpha),
                    "--temperature", str(args.temperature),
                ])
                add_wandb_flags(
                    command, args, dataset, split_name, tag, "table7", seed
                )
                commands.append((command, dataset, split_name, "m2cl", seed, tag))
    return commands


def planned_commands(args):
    studies = selected_studies(args)
    datasets = args.datasets or [args.dataset]
    seeds = args.seeds if args.seeds is not None else [args.seed]
    table_mode = bool(args.tables or args.datasets or args.all_domains)

    commands = []
    for dataset in datasets:
        cfg = load_config(dataset)
        if "architecture" in studies:
            commands.extend(architecture_commands(args, dataset, cfg, seeds, table_mode))
        if "tau" in studies:
            commands.extend(tau_commands(args, dataset, cfg, seeds, table_mode))
        if "alpha" in studies:
            commands.extend(alpha_commands(args, dataset, cfg, seeds, table_mode))
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pacs",
                        choices=["pacs", "vlcs", "office_home"])
    parser.add_argument("--datasets", nargs="+",
                        choices=["pacs", "vlcs", "office_home"],
                        default=None,
                        help="Table mode: run the selected study on all splits "
                             "of each listed dataset.")
    parser.add_argument("--test_domain", default="photo")
    parser.add_argument("--all_domains", action="store_true",
                        help="Run all domains for --dataset.")
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"],
                        default="resnet18")
    parser.add_argument("--study",
                        choices=["architecture", "tau", "alpha", "all"],
                        default="architecture")
    parser.add_argument("--tables", nargs="+",
                        choices=["5", "6", "7", "all"],
                        default=None,
                        help="Run the exact studies needed for slide tables "
                             "5, 6, and/or 7.")
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
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Fixed M2-CL loss weight for Tables 5 and 6.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Fixed M2-CL temperature for Tables 5 and 7.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--save_dir", default="./outputs/ablation_tables")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--wandb", action="store_true",
                        help="Forward W&B logging flags to each tools/train.py run.")
    parser.add_argument("--wandb_project", default="m2cl-ablation-tables")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                        default=None)
    args = parser.parse_args()

    commands = planned_commands(args)
    print(f"Planned ablation/sensitivity runs: {len(commands)}")

    for command, dataset, split_name, method, seed, tag in commands:
        metrics_path = checkpoint_path(
            args.save_dir, dataset, split_name, method,
            args.backbone, seed, tag,
        ).with_suffix(".json")
        if args.skip_existing and metrics_path.exists():
            print(f"[SKIP] {metrics_path}")
            continue
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
