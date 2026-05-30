"""
Run architecture/model baselines for the M2/M2-CL paper comparison.

These are not the DomainBed algorithm baselines. They are the model variants
used to isolate what the M2 architecture contributes:
plain ResNet, M2 with different pipeline layouts/reduction ratios/dropout, and
full M2-CL.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from architecture_specs import ARCHITECTURE_SPECS


def load_config(dataset: str) -> dict:
    """Load only the config fields needed for planning runs."""
    cfg_path = Path(__file__).parent / "configs" / f"{dataset}.yaml"
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
    if dataset != "nico" and "domains" not in cfg:
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
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"],
                        default="resnet18")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--nico_values", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--holdout_fraction", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--save_dir", default="./outputs/architecture")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    commands = []
    for variant in ARCHITECTURE_SPECS:
        for seed in args.seeds:
            for split_type, split_value in dataset_splits(
                args.dataset, cfg, args.nico_values
            ):
                command = [
                    sys.executable,
                    "train.py",
                    "--dataset", args.dataset,
                    "--data_root", args.data_root,
                    "--method", variant.method,
                    "--tag", variant.tag,
                    "--backbone", args.backbone,
                    "--seed", str(seed),
                    "--save_dir", args.save_dir,
                ]
                if split_type == "NICO":
                    command.extend(["--n_heldout", split_value])
                    split_name = f"N{split_value}"
                else:
                    command.extend(["--test_domain", split_value])
                    split_name = split_value

                command.extend(variant.train_flags())
                append_if_not_none(command, "--epochs", args.epochs)
                append_if_not_none(command, "--batch_size", args.batch_size)
                append_if_not_none(command, "--lr", args.lr)
                append_if_not_none(
                    command, "--holdout_fraction", args.holdout_fraction
                )
                append_if_not_none(command, "--num_workers", args.num_workers)
                append_if_not_none(command, "--scheduler", args.scheduler)
                if args.no_pretrained:
                    command.append("--no_pretrained")

                expected_metrics = checkpoint_path(
                    args.save_dir, args.dataset, split_name,
                    variant.method, args.backbone, seed, variant.tag,
                ).with_suffix(".json")
                commands.append((command, expected_metrics, variant))

    print(f"Planned architecture runs: {len(commands)}")
    for command, expected_metrics, variant in commands:
        if args.skip_existing and expected_metrics.exists():
            print(f"[SKIP] {expected_metrics}")
            continue
        print(f"# {variant.tag} ({variant.cls_name}): {variant.description}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
