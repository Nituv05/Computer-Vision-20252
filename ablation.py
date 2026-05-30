"""
Run the project ablation studies used in the report.

This script is a thin orchestrator around train.py. It intentionally keeps
each run as a normal checkpoint-producing training job.
"""
import argparse
import os
import subprocess
import sys


ARCHITECTURE_GRID = [
    {"pipeline_type": "cascading", "reduction_ratio": 2, "dropout_p": 0.0, "method": "m2"},
    {"pipeline_type": "cascading", "reduction_ratio": 4, "dropout_p": 0.0, "method": "m2"},
    {"pipeline_type": "cascading", "reduction_ratio": 6, "dropout_p": 0.0, "method": "m2"},
    {"pipeline_type": "parallel", "reduction_ratio": 2, "dropout_p": 0.0, "method": "m2"},
    {"pipeline_type": "parallel", "reduction_ratio": 4, "dropout_p": 0.0, "method": "m2"},
    {"pipeline_type": "parallel", "reduction_ratio": 6, "dropout_p": 0.0, "method": "m2"},
    {"pipeline_type": "parallel", "reduction_ratio": 2, "dropout_p": 0.3, "method": "m2"},
    {"pipeline_type": "parallel", "reduction_ratio": 4, "dropout_p": 0.3, "method": "m2"},
    {"pipeline_type": "parallel", "reduction_ratio": 6, "dropout_p": 0.3, "method": "m2"},
    {"pipeline_type": "parallel", "reduction_ratio": 4, "dropout_p": 0.3, "method": "m2cl"},
]

TAU_GRID = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 10.0, 100.0]
ALPHA_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def base_command(args, method, save_suffix=None):
    save_dir = args.save_dir if save_suffix is None else os.path.join(
        args.save_dir, save_suffix
    )
    command = [
        sys.executable,
        "train.py",
        "--dataset", args.dataset,
        "--data_root", args.data_root,
        "--method", method,
        "--backbone", args.backbone,
        "--seed", str(args.seed),
        "--save_dir", save_dir,
    ]
    if args.dataset == "nico":
        command.extend(["--n_heldout", str(args.n_heldout)])
    else:
        command.extend(["--test_domain", args.test_domain])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.no_pretrained:
        command.append("--no_pretrained")
    return command


def run_or_print(command, dry_run):
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pacs",
                        choices=["pacs", "vlcs", "office_home", "nico"])
    parser.add_argument("--test_domain", default="photo")
    parser.add_argument("--n_heldout", type=int, default=7)
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"],
                        default="resnet18")
    parser.add_argument("--study", choices=["architecture", "tau", "alpha"],
                        default="architecture")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default="./outputs/ablation")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    if args.study == "architecture":
        for item in ARCHITECTURE_GRID:
            suffix = (
                f"{item['method']}_{item['pipeline_type']}_"
                f"r{item['reduction_ratio']}_drop{item['dropout_p']}"
            )
            command = base_command(args, item["method"], suffix)
            command.extend([
                "--pipeline_type", item["pipeline_type"],
                "--reduction_ratio", str(item["reduction_ratio"]),
                "--dropout_p", str(item["dropout_p"]),
            ])
            run_or_print(command, args.dry_run)
    elif args.study == "tau":
        for tau in TAU_GRID:
            command = base_command(args, "m2cl", f"tau_{tau}")
            command.extend(["--temperature", str(tau)])
            run_or_print(command, args.dry_run)
    else:
        for alpha in ALPHA_GRID:
            command = base_command(args, "m2cl", f"alpha_{alpha}")
            command.extend(["--alpha", str(alpha)])
            run_or_print(command, args.dry_run)


if __name__ == "__main__":
    main()
