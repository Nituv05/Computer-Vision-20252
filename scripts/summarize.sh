#!/usr/bin/env bash
set -euo pipefail

METRICS_DIR="${METRICS_DIR:-outputs/checkpoints}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/results.csv}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

python summarize_results.py \
  --metrics_dir "${METRICS_DIR}" \
  --output_csv "${OUTPUT_CSV}"

echo "Summary CSV: ${OUTPUT_CSV}"
