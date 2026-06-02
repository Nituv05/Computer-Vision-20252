"""
Create paper-style result tables from train.py metrics JSON files.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


METHOD_ORDER = [
    "erm", "rsc", "coral", "mixup", "mmd", "sagnet",
    "selfreg", "arm", "eqrm", "sagm", "m2", "m2cl",
]
METHOD_LABELS = {
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
DATASET_DOMAINS = {
    "pacs": ["art_painting", "cartoon", "photo", "sketch"],
    "vlcs": ["CALTECH", "LABELME", "SUN", "PASCAL"],
    "office_home": ["Art", "Clipart", "Product", "RealWorld"],
}
DOMAIN_LABELS = {
    "art_painting": "Art",
    "cartoon": "Cartoon",
    "photo": "Photo",
    "sketch": "Sketch",
    "CALTECH": "Caltech",
    "LABELME": "Labelme",
    "PASCAL": "Voc",
    "SUN": "Sun",
    "Art": "Art",
    "Clipart": "Clipart",
    "Product": "Product",
    "RealWorld": "RealWorld",
}


def collect(metrics_dir: Path, backbone: str):
    values = defaultdict(list)
    for path in sorted(metrics_dir.rglob("*.json")):
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        if row.get("backbone") != backbone:
            continue
        dataset = row["dataset"]
        if dataset not in DATASET_DOMAINS:
            continue
        method = row.get("tag") or row["method"]
        values[(dataset, method, row["split"])].append(row["test_acc"] * 100.0)
    return values


def domain_mean(values, dataset: str, method: str, domain: str):
    runs = values.get((dataset, method, domain), [])
    return mean(runs) if runs else None


def row_values(values, dataset: str, method: str):
    domains = DATASET_DOMAINS[dataset]
    scores = [domain_mean(values, dataset, method, domain) for domain in domains]
    avg = mean([score for score in scores if score is not None]) if any(
        score is not None for score in scores
    ) else None
    return [*scores, avg]


def ranks(rows):
    columns = len(next(iter(rows.values())))
    best = [None] * columns
    second = [None] * columns
    for col in range(columns):
        scored = sorted(
            {
                method: scores[col]
                for method, scores in rows.items()
                if scores[col] is not None
            }.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if scored:
            best[col] = scored[0][0]
        if len(scored) > 1:
            second[col] = scored[1][0]
    return best, second


def fmt(value, method, col, best, second, style):
    if value is None:
        return "-"
    text = f"{value:.2f}"
    if method == best[col]:
        return f"**{text}**" if style == "markdown" else f"\\textbf{{{text}}}"
    if method == second[col]:
        return f"<u>{text}</u>" if style == "markdown" else f"\\underline{{{text}}}"
    return text


def dataset_rows(values, dataset: str):
    rows = {
        method: row_values(values, dataset, method)
        for method in METHOD_ORDER
    }
    return {method: scores for method, scores in rows.items() if any(
        score is not None for score in scores
    )}


def markdown_table(values, dataset: str):
    rows = dataset_rows(values, dataset)
    if not rows:
        return f"### {dataset}\n\nNo completed runs found.\n"
    best, second = ranks(rows)
    headers = [
        "Method",
        *[DOMAIN_LABELS[d] for d in DATASET_DOMAINS[dataset]],
        "Avg",
    ]
    lines = [
        f"### {dataset}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for method in METHOD_ORDER:
        if method not in rows:
            continue
        scores = rows[method]
        cells = [
            METHOD_LABELS.get(method, method),
            *[
                fmt(value, method, idx, best, second, "markdown")
                for idx, value in enumerate(scores)
            ],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def latex_table(values, dataset: str):
    rows = dataset_rows(values, dataset)
    if not rows:
        return f"% {dataset}: no completed runs found\n"
    best, second = ranks(rows)
    headers = [
        "Method",
        *[DOMAIN_LABELS[d] for d in DATASET_DOMAINS[dataset]],
        "Avg",
    ]
    colspec = "l" + "c" * (len(headers) - 1)
    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        if method not in rows:
            continue
        scores = rows[method]
        cells = [
            METHOD_LABELS.get(method, method),
            *[
                fmt(value, method, idx, best, second, "latex")
                for idx, value in enumerate(scores)
            ],
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="outputs/paper30_r18")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"],
                        default="resnet18")
    parser.add_argument("--datasets", nargs="+",
                        choices=sorted(DATASET_DOMAINS),
                        default=["pacs", "vlcs", "office_home"])
    parser.add_argument("--format", choices=["markdown", "latex"],
                        default="markdown")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    values = collect(Path(args.metrics_dir), args.backbone)
    sections = []
    for dataset in args.datasets:
        if args.format == "markdown":
            sections.append(markdown_table(values, dataset))
        else:
            sections.append(latex_table(values, dataset))
    text = "\n".join(sections)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Wrote table: {output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
