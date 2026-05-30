# Server Runbook

Use this as the command template for running the project on a GPU server. The
repo contains code only. Put datasets and generated checkpoints on the server,
not in git.

## 1. Clone

```bash
git clone https://github.com/Nituv05/Computer-Vision-20252.git
cd Computer-Vision-20252
```

## 2. Environment

Use Python 3.10 or 3.11.

Example for CUDA 12.1:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If PyTorch CUDA is already installed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Dataset Layout

Default dataset root:

```text
data_root/
  pacs/
  vlcs/
  office_home/
```

NICO is optional and is not required for the current run. Do not run
`--dataset all` unless `data_root/nico/` exists.

If data is elsewhere, replace `data_root` in commands with that path.

## 4. Check Server And Data

Default:

```bash
bash scripts/check_server_ready.sh
```

Custom data path:

```bash
DATA_ROOT=/path/to/data_root bash scripts/check_server_ready.sh
```

## 5. Smoke Test

This is only a sanity check for CUDA, dependency, dataset path and checkpoint
writing. Do not use smoke-test numbers in the report.

```bash
bash scripts/smoke_test.sh
```

If the data path is custom:

```bash
DATA_ROOT=/path/to/data_root bash scripts/smoke_test.sh
```

If CUDA runs out of memory:

```bash
BATCH_SIZE=64 bash scripts/smoke_test.sh
BATCH_SIZE=32 bash scripts/smoke_test.sh
```

## 6. General Run Form

Use this form when launching experiments manually:

```bash
python run_experiments.py \
  --dataset <pacs|vlcs|office_home> \
  --data_root <data_root_path> \
  --methods <method_1> <method_2> ... \
  --backbones <resnet18|resnet50> \
  --seeds <seed_1> <seed_2> ... \
  --batch_size <batch_size> \
  --num_workers <num_workers> \
  --save_dir outputs/checkpoints \
  --skip_existing
```

Available method values:

```text
erm rsc mixup coral mmd sagnet selfreg arm eqrm sagm m2 m2cl
paper_baselines
all
```

Meaning:

```text
paper_baselines = ERM, RSC, Mixup, CORAL, MMD, SagNet, SelfReg, ARM, EQRM, SAGM
all = paper_baselines + M2 + M2CL
```

## 7. Sequential Full Run

Run in this order. Each command is independent and resumable because
`--skip_existing` is used.

### ResNet-18, PACS

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods all \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --batch_size 128 \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

### ResNet-18, VLCS

```bash
python run_experiments.py \
  --dataset vlcs \
  --data_root data_root \
  --methods all \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --batch_size 128 \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

### ResNet-18, Office-Home

```bash
python run_experiments.py \
  --dataset office_home \
  --data_root data_root \
  --methods all \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --batch_size 128 \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

## 8. Optional ResNet-50 Runs

Run ResNet-50 after ResNet-18. Start with smaller batch size because M2/M2CL
with ResNet-50 uses much more VRAM.

### ResNet-50, PACS

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods all \
  --backbones resnet50 \
  --seeds 0 1 2 \
  --batch_size 32 \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

Use the same form for `vlcs` and `office_home` if there is enough GPU time.

## 9. Running Only Selected Baselines

Example: run 4 baselines only:

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods erm rsc mixup coral \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --batch_size 128 \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

Example: run only M2CL:

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods m2cl \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --batch_size 128 \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

## 10. Wrapper Script Alternative

The wrapper is equivalent to launching the three ResNet-18 dataset commands
above:

```bash
bash scripts/run_main.sh
```

Override options like this:

```bash
BACKBONE=resnet50 BATCH_SIZE=32 METHODS="m2cl" DATASETS="pacs" bash scripts/run_main.sh
METHODS="erm rsc mixup coral" DATASETS="pacs vlcs" bash scripts/run_main.sh
```

## 11. Results

Checkpoints and per-run metrics:

```text
outputs/checkpoints/
```

Summarize:

```bash
python summarize_results.py \
  --metrics_dir outputs/checkpoints \
  --output_csv outputs/main_results.csv
```

Main CSV:

```text
outputs/main_results.csv
```

`outputs/` is ignored by git. Copy CSV files and selected checkpoints back
manually if needed.
