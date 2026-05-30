#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${BACKBONE:-resnet18}" bash scripts/run_main.sh
