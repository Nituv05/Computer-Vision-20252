"""
Leave-one-domain-out evaluation across all domains.

Usage:
  python evaluate.py --dataset pacs --data_root ./data_root
  python evaluate.py --dataset vlcs --data_root ./data_root
"""
import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import M2CL
from train import evaluate

DATASET_LOADERS = {
    "pacs": ("data", "get_pacs_loaders"),
    "vlcs": ("data", "get_vlcs_loaders"),
    "office_home": ("data", "get_office_home_loaders"),
}

PAPER_BASELINES = {
    "pacs": {
        "ERM": 84.51,
        "CORAL": 84.29,
        "MIXUP": 84.13,
        "SagNet": 83.58,
        "SAGM": 82.68,
        "M2-CL (paper)": 83.54,
    },
    "vlcs": {
        "ERM": 74.71,
        "CORAL": 74.83,
        "SAGM": 75.17,
        "M2-CL (paper)": 77.78,
    },
    "office_home": {
        "ERM": 58.89,
        "CORAL": 60.09,
        "SAGM": 59.66,
        "M2-CL (paper)": 63.27,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["pacs", "vlcs", "office_home"])
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--checkpoint_dir", default="./outputs/checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "configs", f"{args.dataset}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domains = cfg["domains"]
    num_classes = cfg["num_classes"]

    import importlib
    mod_name, fn_name = DATASET_LOADERS[args.dataset]
    loader_fn = getattr(importlib.import_module(mod_name), fn_name)

    results = {}
    for test_domain in domains:
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"{args.dataset}_{test_domain}_best.pth"
        )
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] No checkpoint for {test_domain} at {ckpt_path}")
            continue

        _, test_loader = loader_fn(args.data_root, test_domain, args.batch_size)
        model = M2CL(num_classes=num_classes).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

        acc = evaluate(model, test_loader, device)
        results[test_domain] = acc * 100
        print(f"  {test_domain:20s}: {acc * 100:.2f}%")

    if results:
        avg = sum(results.values()) / len(results)
        results["Average"] = avg
        print(f"\n  {'Average':20s}: {avg:.2f}%")

        print(f"\n--- Comparison with paper baselines ({args.dataset.upper()}) ---")
        baselines = PAPER_BASELINES.get(args.dataset, {})
        for method, acc in baselines.items():
            print(f"  {method:25s}: {acc:.2f}%")
        print(f"  {'Ours (reproduced)':25s}: {avg:.2f}%")


if __name__ == "__main__":
    main()
