"""
Summarize metrics JSON files produced by tools/train.py.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def std(values):
    return stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="./outputs/checkpoints")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    paths = sorted(Path(args.metrics_dir).rglob("*.json"))
    if not paths:
        print(f"No metrics JSON files found under {args.metrics_dir}")
        return

    split_groups = defaultdict(list)
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        acc = row["test_acc"] * 100.0
        method_label = row.get("tag") or row["method"]
        key = (
            row["dataset"],
            method_label,
            row["backbone"],
            row["split"],
        )
        split_groups[key].append(acc)
        rows.append({
            "dataset": row["dataset"],
            "method": row["method"],
            "tag": row.get("tag", ""),
            "method_label": method_label,
            "backbone": row["backbone"],
            "split": row["split"],
            "seed": row["seed"],
            "test_acc": acc,
            "best_epoch": row.get("best_epoch", ""),
            "alpha": row.get("alpha", ""),
            "temperature": row.get("temperature", ""),
            "pipeline_type": row.get("pipeline_type", ""),
            "reduction_ratio": row.get("reduction_ratio", ""),
            "dropout_p": row.get("dropout_p", ""),
            "checkpoint": row.get("checkpoint", ""),
        })

    print("Per-split results")
    print("dataset,method,backbone,split,runs,mean,std")
    split_summary = []
    for key, values in sorted(split_groups.items()):
        summary = {
            "dataset": key[0],
            "method": key[1],
            "backbone": key[2],
            "split": key[3],
            "runs": len(values),
            "mean": mean(values),
            "std": std(values),
        }
        split_summary.append(summary)
        print(
            f"{key[0]},{key[1]},{key[2]},{key[3]},"
            f"{len(values)},{summary['mean']:.2f},{summary['std']:.2f}"
        )

    dataset_groups = defaultdict(list)
    for summary in split_summary:
        key = (summary["dataset"], summary["method"], summary["backbone"])
        dataset_groups[key].append(summary["mean"])

    print("\nDataset averages across splits")
    print("dataset,method,backbone,num_splits,avg")
    for key, values in sorted(dataset_groups.items()):
        print(f"{key[0]},{key[1]},{key[2]},{len(values)},{mean(values):.2f}")

    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset", "method", "backbone", "split", "seed",
                    "tag", "method_label", "test_acc", "best_epoch",
                    "alpha", "temperature", "pipeline_type",
                    "reduction_ratio", "dropout_p",
                    "checkpoint",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote raw run CSV: {output}")


if __name__ == "__main__":
    main()
