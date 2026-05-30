"""
Evaluate saved checkpoints on leave-one-domain/context-out splits.

Examples:
  python evaluate.py --dataset pacs --method m2cl
  python evaluate.py --dataset nico --method m2cl
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from algorithms import METHODS, build_algorithm
from train import build_raw_loaders, checkpoint_path, evaluate, load_config


PAPER_AVERAGES = {
    "pacs": {
        "resnet18": {"ERM": 80.53, "M2": 82.83, "M2-CL": 83.54},
        "resnet50": {"ERM": 83.33, "M2": 84.26, "M2-CL": 85.97},
    },
    "vlcs": {
        "resnet18": {"ERM": 69.81, "M2": 76.11, "M2-CL": 77.78},
        "resnet50": {"ERM": 76.83, "M2": 76.81, "M2-CL": 78.36},
    },
    "office_home": {
        "resnet18": {"ERM": 57.10, "M2": 61.29, "M2-CL": 63.27},
        "resnet50": {"ERM": 67.27, "M2": 69.64, "M2-CL": 71.07},
    },
    "nico": {
        "resnet18": {
            "M2-CL N3": 87.93,
            "M2-CL N5": 84.10,
            "M2-CL N7": 82.14,
        },
        "resnet50": {
            "M2-CL N3": 89.30,
            "M2-CL N5": 87.68,
            "M2-CL N7": 86.90,
        },
    },
}


def evaluate_checkpoint(args, cfg, split_name, test_domain=None, n_heldout=None):
    ckpt_path = checkpoint_path(
        args.checkpoint_dir,
        args.dataset,
        split_name,
        args.method,
        args.backbone,
        args.seed,
        args.tag,
    )
    if not ckpt_path.exists():
        print(f"[SKIP] Missing checkpoint: {ckpt_path}")
        return None

    _, test_set, _ = build_raw_loaders(
        args.dataset,
        args.data_root,
        test_domain,
        n_heldout,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    algorithm_args = ckpt.get("model_args", {})
    algorithm_args.setdefault("num_classes", cfg["num_classes"])
    algorithm_args.setdefault("method", args.method)
    algorithm_args.setdefault("backbone", args.backbone)
    algorithm_args.setdefault("architecture_tag", args.tag)
    algorithm_args["pretrained"] = False
    algorithm = build_algorithm(
        **algorithm_args,
    ).to(device)
    algorithm.load_state_dict(ckpt["state_dict"])
    acc = evaluate(algorithm, test_loader, device) * 100.0
    print(f"  {split_name:20s}: {acc:6.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["pacs", "vlcs", "office_home", "nico"])
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--checkpoint_dir", default="./outputs/checkpoints")
    parser.add_argument("--method", choices=METHODS, default="m2cl")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"],
                        default="resnet18")
    parser.add_argument("--test_domain", default=None)
    parser.add_argument("--n_heldout", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = load_config(args.dataset)

    print(
        f"Evaluating dataset={args.dataset} method={args.method} "
        f"backbone={args.backbone} seed={args.seed}"
    )

    results = {}
    if args.dataset == "nico":
        heldout_values = [args.n_heldout] if args.n_heldout else cfg.get(
            "heldout_contexts", [3, 5, 7]
        )
        for n_heldout in heldout_values:
            split_name = f"N{n_heldout}"
            acc = evaluate_checkpoint(
                args, cfg, split_name, n_heldout=n_heldout
            )
            if acc is not None:
                results[split_name] = acc
    else:
        domains = [args.test_domain] if args.test_domain else cfg["domains"]
        for domain in domains:
            acc = evaluate_checkpoint(
                args, cfg, domain, test_domain=domain
            )
            if acc is not None:
                results[domain] = acc

    if results:
        avg = sum(results.values()) / len(results)
        print(f"\n  {'Average':20s}: {avg:6.2f}%")

    paper = PAPER_AVERAGES.get(args.dataset, {}).get(args.backbone, {})
    if paper:
        print(f"\n--- Paper reference averages ({args.backbone}) ---")
        for method, value in paper.items():
            print(f"  {method:20s}: {value:6.2f}%")


if __name__ == "__main__":
    main()
