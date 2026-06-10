import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


METHODS = [
    "erm", "rsc", "coral", "mixup", "mmd", "sagnet",
    "selfreg", "arm", "eqrm", "sagm", "m2", "m2cl",
]

LABEL = {
    "erm": "ERM",
    "rsc": "RSC",
    "coral": "CORAL",
    "mixup": "MIXUP",
    "mmd": "MMD",
    "sagnet": "SagNet",
    "selfreg": "SelfReg",
    "arm": "ARM",
    "eqrm": "EQRM",
    "sagm": "SAGM",
    "m2": "M2",
    "m2cl": "M2-CL",
}

DOMAINS = {
    "pacs": ["art_painting", "cartoon", "photo", "sketch"],
    "vlcs": ["CALTECH", "LABELME", "SUN", "PASCAL"],
    "office_home": ["Art", "Clipart", "Product", "RealWorld"],
    "nico": ["N=3", "N=5", "N=7"],
}

DOMAIN_LABELS = {
    "art_painting": "Art",
    "cartoon": "Cartoon",
    "photo": "Photo",
    "sketch": "Sketch",
    "CALTECH": "Caltech",
    "LABELME": "Labelme",
    "SUN": "Sun",
    "PASCAL": "Voc",
    "Art": "Art",
    "Clipart": "Clipart",
    "Product": "Product",
    "RealWorld": "Real-World",
    "N=3": "N=3",
    "N=5": "N=5",
    "N=7": "N=7",
}


PAPER = {
    "resnet18": {
        "pacs": {
            "erm": [79.68, 77.76, 88.91, 75.72],
            "rsc": [78.21, 74.30, 92.51, 78.75],
            "coral": [76.03, 75.60, 93.23, 78.78],
            "mixup": [79.19, 74.09, 95.36, 73.95],
            "mmd": [75.71, 66.47, 94.08, 69.59],
            "sagnet": [77.24, 77.10, 93.42, 73.28],
            "selfreg": [79.68, 77.91, 94.61, 74.64],
            "arm": [79.21, 75.74, 92.82, 75.09],
            "eqrm": [81.39, 74.57, 93.41, 75.60],
            "sagm": [79.21, 77.35, 94.61, 79.58],
            "m2": [79.86, 77.91, 95.65, 77.90],
            "m2cl": [81.66, 78.42, 97.00, 77.07],
        },
        "vlcs": {
            "erm": [84.80, 59.05, 65.27, 70.12],
            "rsc": [88.69, 62.91, 68.24, 68.95],
            "coral": [95.67, 60.32, 69.95, 72.82],
            "mixup": [96.20, 61.77, 69.76, 72.93],
            "mmd": [95.67, 58.58, 60.70, 68.04],
            "sagnet": [89.39, 61.11, 68.20, 73.86],
            "selfreg": [96.11, 60.51, 68.12, 74.51],
            "arm": [96.64, 59.88, 68.10, 71.31],
            "eqrm": [96.11, 61.12, 66.56, 71.40],
            "sagm": [96.81, 60.07, 69.15, 73.04],
            "m2": [97.61, 60.18, 70.75, 75.89],
            "m2cl": [98.23, 63.85, 72.35, 76.67],
        },
        "office_home": {
            "erm": [47.88, 45.64, 67.34, 67.52],
            "rsc": [54.43, 46.89, 67.84, 70.53],
            "coral": [53.86, 46.96, 68.88, 71.87],
            "mixup": [53.35, 48.13, 68.35, 70.51],
            "mmd": [48.86, 44.67, 66.83, 66.37],
            "sagnet": [52.52, 47.99, 67.30, 70.83],
            "selfreg": [54.22, 48.11, 68.43, 71.83],
            "arm": [49.48, 45.19, 63.73, 68.36],
            "eqrm": [55.67, 47.85, 69.76, 71.29],
            "sagm": [53.14, 48.22, 70.74, 71.25],
            "m2": [56.91, 47.77, 68.33, 72.21],
            "m2cl": [58.55, 49.57, 71.53, 73.43],
        },
        "nico": {
            "erm": [83.77, 79.53, 74.53],
            "rsc": [81.36, 77.30, 74.52],
            "coral": [84.15, 79.86, 76.31],
            "mixup": [84.01, 80.20, 78.90],
            "mmd": [84.33, 79.23, 77.38],
            "sagnet": [85.09, 82.23, 79.05],
            "selfreg": [86.27, 83.23, 81.40],
            "arm": [84.54, 80.76, 77.43],
            "eqrm": [84.93, 80.81, 76.32],
            "sagm": [85.02, 81.94, 79.33],
            "m2": [86.02, 82.81, 81.64],
            "m2cl": [87.93, 84.10, 82.14],
        },
    },
    "resnet50": {
        "pacs": {
            "erm": [83.90, 78.60, 97.30, 73.50],
            "rsc": [81.66, 80.86, 96.18, 75.79],
            "coral": [86.00, 75.50, 96.20, 76.60],
            "mixup": [88.00, 74.30, 97.20, 75.30],
            "mmd": [85.90, 78.10, 96.20, 71.10],
            "sagnet": [85.47, 80.09, 94.46, 77.83],
            "selfreg": [83.61, 79.15, 95.80, 78.10],
            "arm": [83.22, 80.01, 95.13, 76.75],
            "eqrm": [86.82, 79.85, 94.91, 78.09],
            "sagm": [85.33, 80.55, 95.88, 80.06],
            "m2": [86.29, 77.56, 97.79, 75.41],
            "m2cl": [87.25, 81.84, 98.50, 76.30],
        },
        "vlcs": {
            "erm": [97.80, 63.30, 70.30, 75.90],
            "rsc": [95.40, 64.65, 70.45, 73.33],
            "coral": [97.50, 64.00, 69.70, 76.70],
            "mixup": [97.90, 65.50, 73.30, 77.80],
            "mmd": [97.70, 63.10, 68.60, 77.40],
            "sagnet": [97.17, 64.00, 70.50, 73.48],
            "selfreg": [94.69, 64.47, 68.88, 73.82],
            "arm": [96.55, 65.16, 70.18, 73.63],
            "eqrm": [97.17, 63.38, 69.53, 74.37],
            "sagm": [97.87, 64.97, 70.57, 76.29],
            "m2": [97.79, 64.33, 70.33, 74.78],
            "m2cl": [98.23, 65.50, 72.25, 77.44],
        },
        "office_home": {
            "erm": [62.30, 54.10, 75.30, 77.40],
            "rsc": [61.30, 52.50, 74.38, 76.24],
            "coral": [64.40, 55.40, 76.20, 78.40],
            "mixup": [63.80, 52.90, 77.30, 78.70],
            "mmd": [62.20, 52.70, 75.50, 78.10],
            "sagnet": [63.50, 52.89, 74.07, 75.21],
            "selfreg": [60.60, 53.32, 72.07, 76.92],
            "arm": [56.18, 52.23, 72.21, 73.36],
            "eqrm": [65.15, 55.49, 76.89, 78.25],
            "sagm": [63.29, 54.26, 76.37, 77.84],
            "m2": [65.36, 55.55, 78.15, 79.50],
            "m2cl": [68.45, 57.04, 78.66, 80.14],
        },
        "nico": {
            "erm": [87.02, 83.66, 82.04],
            "rsc": [87.78, 83.40, 80.01],
            "coral": [87.15, 84.91, 83.64],
            "mixup": [88.15, 83.28, 82.03],
            "mmd": [88.38, 83.69, 83.04],
            "sagnet": [85.96, 83.62, 81.72],
            "selfreg": [86.78, 85.67, 84.59],
            "arm": [87.32, 82.67, 82.38],
            "eqrm": [89.55, 85.92, 83.07],
            "sagm": [87.79, 85.27, 84.32],
            "m2": [88.78, 86.88, 85.84],
            "m2cl": [89.30, 87.68, 86.90],
        },
    },
}

TABLE3_ROWS = [
    ("c", "2", "", "", [75.98, 73.68, 94.39, 71.27, 78.83, 96.07, 61.26, 69.05, 65.38, 72.94]),
    ("c", "4", "", "", [77.50, 73.96, 94.03, 70.23, 78.93, 95.59, 59.47, 70.55, 66.34, 72.99]),
    ("c", "6", "", "", [77.37, 74.09, 94.71, 73.43, 79.90, 95.91, 58.55, 69.35, 66.38, 72.55]),
    ("c", "2", "yes", "", [76.30, 73.94, 94.75, 76.05, 80.26, 97.53, 58.87, 68.77, 66.17, 72.84]),
    ("c", "4", "yes", "", [77.86, 74.96, 94.24, 76.51, 80.89, 96.08, 59.25, 70.47, 67.22, 66.50]),
    ("c", "6", "yes", "", [78.13, 74.41, 94.24, 75.06, 80.46, 96.08, 60.19, 69.85, 67.91, 73.51]),
    ("p", "2", "", "", [78.70, 59.59, 69.52, 64.84, 68.16, 96.57, 59.51, 69.03, 74.62, 74.93]),
    ("p", "4", "", "", [76.93, 74.75, 94.42, 73.54, 79.92, 95.95, 60.16, 69.55, 65.89, 72.89]),
    ("p", "6", "", "", [76.45, 73.63, 94.19, 72.59, 79.21, 95.43, 60.22, 69.73, 66.10, 72.87]),
    ("p", "2", "yes", "", [77.37, 75.84, 87.37, 77.68, 79.56, 97.28, 59.99, 70.68, 65.90, 73.46]),
    ("p", "6", "yes", "", [77.37, 74.83, 94.83, 76.56, 80.89, 96.21, 59.69, 69.80, 67.02, 73.18]),
    ("p", "4", "yes", "", [79.86, 77.91, 95.65, 77.90, 82.83, 97.61, 60.18, 70.75, 75.89, 76.11]),
    ("p", "4", "yes", "yes", [81.66, 78.42, 97.00, 77.07, 83.54, 98.23, 63.85, 72.35, 76.67, 77.78]),
]

TABLE4_TAU = [
    ("0.01", 81.88, 77.57),
    ("0.1", 82.20, 77.48),
    ("0.2", 83.82, 77.58),
    ("0.4", 85.38, 75.31),
    ("0.6", 84.97, 75.88),
    ("0.8", 85.30, 77.71),
    ("1.0", 83.54, 77.78),
    ("1.2", 84.75, 77.25),
    ("1.4", 85.07, 77.54),
    ("1.6", 85.41, 77.80),
    ("1.8", 83.67, 78.10),
    ("2.0", 82.17, 77.29),
    ("10.0", 83.31, 77.32),
    ("100.0", 82.04, 76.69),
]

TABLE4_ALPHA = [
    ("0.00", 82.83, 76.11),
    ("10e-5", 82.35, 76.43),
    ("10e-4", 83.12, 77.59),
    ("10e-3", 83.54, 77.78),
    ("10e-2", 80.52, 77.01),
    ("10e-1", 67.26, 56.92),
]


def synthetic_value(backbone, dataset, method, domain):
    idx = DOMAINS[dataset].index(domain)
    paper_value = PAPER[backbone][dataset][method][idx]
    token = f"20260610|{backbone}|{dataset}|{method}|{domain}"
    seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
    drop = random.Random(seed).uniform(0.0, 4.0)
    return paper_value - drop, drop


def synthetic_from_paper(tag, paper_value, audit):
    seed = int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16], 16)
    drop = random.Random(seed).uniform(0.0, 4.0)
    value = paper_value - drop
    parts = tag.split("|")
    audit.append({
        "backbone": "resnet18",
        "dataset": parts[1],
        "method": parts[2] if len(parts) > 2 else "synthetic",
        "domain": parts[-1],
        "paper": paper_value,
        "drop": drop,
        "synthetic": value,
    })
    return value


def collect_real(repo_root):
    values = defaultdict(list)
    search_roots = [
        repo_root / "m2cl-main" / "outputs",
        repo_root / "m2cl-duck" / "server-backup" / "outputs",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            dataset = row.get("dataset")
            backbone = row.get("backbone")
            method = row.get("tag") or row.get("method")
            split = row.get("split")
            if dataset not in ("pacs", "vlcs", "office_home"):
                continue
            if backbone not in ("resnet18", "resnet50"):
                continue
            if method not in METHODS:
                continue
            if split not in DOMAINS[dataset]:
                continue
            test_acc = row.get("test_acc")
            if test_acc is None:
                continue
            values[(backbone, dataset, method, split)].append(
                (int(row.get("seed", -1)), float(test_acc) * 100.0, str(path))
            )
    return values


def cell(values, backbone, dataset, method, domain, audit):
    real = values.get((backbone, dataset, method, domain), [])
    if real:
        by_seed = {}
        for seed, value, path in real:
            by_seed.setdefault(seed, []).append(value)
        seed_values = [sum(vals) / len(vals) for _, vals in sorted(by_seed.items())]
        value = sum(seed_values) / len(seed_values)
        suffix = "" if len(seed_values) >= 3 else "[1s]"
        source = "real-3seed-mean" if len(seed_values) >= 3 else "real-seed0-as-mean"
        return value, suffix, source
    value, drop = synthetic_value(backbone, dataset, method, domain)
    audit.append({
        "backbone": backbone,
        "dataset": dataset,
        "method": method,
        "domain": domain,
        "paper": PAPER[backbone][dataset][method][DOMAINS[dataset].index(domain)],
        "drop": drop,
        "synthetic": value,
    })
    return value, "[S]", "synthetic"


def avg(cells):
    values = [item[0] for item in cells]
    sources = [item[2] for item in cells]
    value = sum(values) / len(values)
    if any(src == "synthetic" for src in sources):
        suffix = "[S]"
    elif any(src == "real-seed0-as-mean" for src in sources):
        suffix = "[1s]"
    else:
        suffix = ""
    return value, suffix


def fmt(value, suffix):
    return f"{value:.2f}{suffix}"


def combined_table(values, backbone, left_dataset, right_dataset, audit):
    left_domains = DOMAINS[left_dataset]
    right_domains = DOMAINS[right_dataset]
    headers = (
        ["Method"]
        + [DOMAIN_LABELS[d] for d in left_domains]
        + (["Avg"] if left_dataset != "nico" else [])
        + [DOMAIN_LABELS[d] for d in right_domains]
        + (["Avg"] if right_dataset != "nico" else [])
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for method in METHODS:
        left_cells = [cell(values, backbone, left_dataset, method, d, audit) for d in left_domains]
        right_cells = [cell(values, backbone, right_dataset, method, d, audit) for d in right_domains]
        row = [LABEL[method]]
        row.extend(fmt(v, s) for v, s, _ in left_cells)
        if left_dataset != "nico":
            row.append(fmt(*avg(left_cells)))
        row.extend(fmt(v, s) for v, s, _ in right_cells)
        if right_dataset != "nico":
            row.append(fmt(*avg(right_cells)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def audit_summary(audit):
    grouped = defaultdict(list)
    for row in audit:
        grouped[(row["backbone"], row["dataset"])].append(row["drop"])
    lines = [
        "| Backbone | Dataset | Synthetic cells | Min drop | Mean drop | Max drop |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (backbone, dataset), drops in sorted(grouped.items()):
        lines.append(
            f"| {backbone} | {dataset} | {len(drops)} | "
            f"{min(drops):.2f} | {sum(drops) / len(drops):.2f} | {max(drops):.2f} |"
        )
    return "\n".join(lines)


def table3(audit):
    headers = [
        "pipe", "r", "drop", "loss",
        "A", "C", "P", "S", "PACS Avg",
        "C", "L", "S", "V", "VLCS Avg",
    ]
    lines = [
        "## Table III - Ablation Study",
        "",
        "All cells in this table are synthetic placeholders because ablation runs were not present in the local JSON outputs.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 4 + ["---:"] * 10) + " |",
    ]
    for idx, (pipe, r, drop, loss, vals) in enumerate(TABLE3_ROWS):
        cells = [pipe, r, drop or "-", loss or "-"]
        value_cells = []
        for col_idx, paper_value in enumerate(vals):
            tag = f"20260610|table3_ablation|row{idx}|col{col_idx}"
            value_cells.append(f"{synthetic_from_paper(tag, paper_value, audit):.2f}[S]")
        cells.extend(value_cells)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table4(audit):
    lines = [
        "## Table IV - Sensitivity Analysis",
        "",
        "All cells in this table are synthetic placeholders because sensitivity runs were not present in the local JSON outputs.",
        "",
        "### Tau",
        "",
        "| tau | PACS | VLCS |",
        "|---|---:|---:|",
    ]
    for idx, (tau, pacs, vlcs) in enumerate(TABLE4_TAU):
        pacs_value = synthetic_from_paper(f"20260610|table4_tau|row{idx}|PACS", pacs, audit)
        vlcs_value = synthetic_from_paper(f"20260610|table4_tau|row{idx}|VLCS", vlcs, audit)
        lines.append(f"| {tau} | {pacs_value:.2f}[S] | {vlcs_value:.2f}[S] |")
    lines.extend([
        "",
        "### Alpha",
        "",
        "| alpha | PACS | VLCS |",
        "|---|---:|---:|",
    ])
    for idx, (alpha, pacs, vlcs) in enumerate(TABLE4_ALPHA):
        pacs_value = synthetic_from_paper(f"20260610|table4_alpha|row{idx}|PACS", pacs, audit)
        vlcs_value = synthetic_from_paper(f"20260610|table4_alpha|row{idx}|VLCS", vlcs, audit)
        lines.append(f"| {alpha} | {pacs_value:.2f}[S] | {vlcs_value:.2f}[S] |")
    return "\n".join(lines)


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    out_dir = script_path.parent
    values = collect_real(repo_root)
    audit = []

    sections = [
        "# Final Paper-Style Result Tables",
        "",
        "Legend:",
        "",
        "- no suffix: real JSON result averaged over 3 seeds.",
        "- `[1s]`: real JSON result from one seed, treated as mean because no extra seeds are available.",
        "- `[S]`: synthetic placeholder, generated as original paper value minus `U(0, 4)` with a fixed seed.",
        "- NICO was not trained in this project, so all NICO cells are synthetic placeholders.",
        "",
        "## Table I - PACS and VLCS",
        "",
        "### ResNet-18",
        "",
        combined_table(values, "resnet18", "pacs", "vlcs", audit),
        "",
        "### ResNet-50",
        "",
        combined_table(values, "resnet50", "pacs", "vlcs", audit),
        "",
        "## Table II - Office-Home and NICO",
        "",
        "### ResNet-18",
        "",
        combined_table(values, "resnet18", "office_home", "nico", audit),
        "",
        "### ResNet-50",
        "",
        combined_table(values, "resnet50", "office_home", "nico", audit),
        "",
        table3(audit),
        "",
        table4(audit),
        "",
        "## Synthetic Drop Audit",
        "",
        audit_summary(audit),
        "",
    ]

    output = out_dir / "final_paper_style_tables.md"
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
