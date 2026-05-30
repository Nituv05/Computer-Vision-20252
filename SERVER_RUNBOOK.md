# Server Runbook

This file is the short instruction sheet for running the project on a GPU
server. The repo contains code only. Datasets and checkpoints must stay local
on the server.

## 1. Clone

```bash
git clone https://github.com/Nituv05/Computer-Vision-20252.git
cd Computer-Vision-20252
```

## 2. Environment

Use Python 3.10 or 3.11. Install the CUDA build of PyTorch that matches the
server, then install the remaining requirements.

Example for CUDA 12.1:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If PyTorch is already installed on the server, just run:

```bash
pip install -r requirements.txt
```

## 3. Dataset Layout

Put the data under `data_root/`:

```text
data_root/
  pacs/
  vlcs/
  office_home/
```

NICO is optional. Do not run `--dataset all` unless `data_root/nico/` exists.

## 4. Check Server And Data

```bash
bash scripts/check_server_ready.sh
```

If the data is not under `data_root`, pass another path:

```bash
DATA_ROOT=/path/to/data_root bash scripts/check_server_ready.sh
```

## 5. Smoke Test

Run this before launching long jobs:

```bash
bash scripts/smoke_test.sh
```

If GPU memory is small:

```bash
BATCH_SIZE=32 bash scripts/smoke_test.sh
```

## 6. Main Experiments

Default: PACS, VLCS, Office-Home; methods ERM, M2, M2-CL; ResNet-18; seeds
0, 1, 2; 30 epochs from config.

```bash
bash scripts/run_main_resnet18.sh
```

For a faster first pass:

```bash
SEEDS="0" EPOCHS=10 bash scripts/run_main_resnet18.sh
```

If CUDA runs out of memory:

```bash
BATCH_SIZE=64 bash scripts/run_main_resnet18.sh
```

Resume an interrupted run with the same command. The script passes
`--skip_existing`, so completed metric files are skipped.

## 7. Optional Architecture Ablations

Run this only after the main experiments finish:

```bash
bash scripts/run_architecture_ablations.sh
```

For a smaller ablation:

```bash
SEEDS="0" DATASETS="pacs" bash scripts/run_architecture_ablations.sh
```

## 8. Outputs

Main checkpoints and metrics:

```text
outputs/checkpoints/
```

Main summary:

```text
outputs/main_results.csv
```

Architecture ablation summary:

```text
outputs/architecture_results.csv
```

`outputs/` is ignored by git. Copy the CSV files and selected checkpoints back
manually if needed.
