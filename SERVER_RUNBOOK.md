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
  --epochs 30 \
  --holdout_fraction <source_val_fraction> \
  --scheduler <none|cosine> \
  --hparams_profile paper \
  --num_workers <num_workers> \
  --save_dir outputs/checkpoints \
  --wandb \
  --wandb_project <wandb_project_name> \
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

Default values with `--hparams_profile paper`:

```text
--methods all
--backbones resnet18
--seeds 0 1 2
--epochs 30
--holdout_fraction 0.2
--scheduler none
--num_workers 4
--save_dir outputs/checkpoints
pretrained ImageNet ResNet: enabled
M2/M2CL optimizer: SGD with momentum 0.9
M2/M2CL lr: 0.001
M2/M2CL batch_size: 128
M2/M2CL weight_decay: 5e-4
baselines: DomainBed-style/recommended baseline hyperparameters
M2/M2CL alpha: 0.01
M2/M2CL temperature: 1.0
M2/M2CL reduction_ratio: 4
M2/M2CL dropout_p: 0.3
M2/M2CL embed_dim: 128
M2/M2CL pipeline_type: parallel
mixup_alpha: 0.2
penalty_weight for CORAL/MMD: 1.0
```

Do not pass `--batch_size`, `--lr`, `--weight_decay`, or `--optimizer` unless
you intentionally want to override the paper profile.

Use a fresh `--save_dir` when switching away from older runs; otherwise
`--skip_existing` may skip metrics from an older protocol because filenames
are intentionally stable.

Use `--no_pretrained` only for debugging when ImageNet weights cannot be
downloaded. Do not use `--no_pretrained` for final paper-comparison results.

## 7. W&B Logging

Login once on the server:

```bash
wandb login
```

Direct `run_experiments.py` form:

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods erm mixup coral rsc sagm m2 m2cl \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --wandb \
  --wandb_project cv20252-m2cl \
  --wandb_tags resnet18 priority \
  --skip_existing
```

Wrapper form:

```bash
WANDB=1 WANDB_PROJECT=cv20252-m2cl WANDB_TAGS="resnet18 priority" \
METHODS="erm mixup coral rsc sagm m2 m2cl" bash scripts/run_main.sh
```

If the server has no stable internet during training, use offline mode:

```bash
WANDB=1 WANDB_MODE=offline WANDB_PROJECT=cv20252-m2cl bash scripts/run_main.sh
```

Then sync later:

```bash
wandb sync wandb/offline-run-*
```

## 8. Sequential Full Run

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
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
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
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
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
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

## 9. Optional ResNet-50 Runs

Run ResNet-50 after ResNet-18. Keep the paper batch size unless the server
actually runs out of memory; reducing batch size means the run is no longer
the exact paper setting.

### ResNet-50, PACS

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods all \
  --backbones resnet50 \
  --seeds 0 1 2 \
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

Use the same form for `vlcs` and `office_home` if there is enough GPU time.

## 10. Running Only Selected Baselines

Example: run 4 baselines only:

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods erm rsc mixup coral \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

Recommended priority set for limited GPU time:

```text
ERM      standard lower-bound baseline
Mixup    strong augmentation/domain-mixing baseline
CORAL    distribution-alignment baseline
RSC      representation self-challenging baseline
SAGM     recent/sharpness-aware DG baseline
M2       proposed architecture without contrastive loss
M2CL     full proposed method
```

Command:

```bash
python run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods erm mixup coral rsc sagm m2 m2cl \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --wandb \
  --wandb_project cv20252-m2cl \
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
  --epochs 30 \
  --holdout_fraction 0.2 \
  --scheduler none \
  --hparams_profile paper \
  --num_workers 4 \
  --save_dir outputs/checkpoints \
  --skip_existing
```

## 11. Wrapper Script Alternative

The wrapper is equivalent to launching the three ResNet-18 dataset commands
above:

```bash
bash scripts/run_main.sh
```

Override options like this:

```bash
BACKBONE=resnet50 EPOCHS=30 METHODS="m2cl" DATASETS="pacs" bash scripts/run_main.sh
METHODS="erm rsc mixup coral" DATASETS="pacs vlcs" EPOCHS=30 bash scripts/run_main.sh
```

## 12. Results

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
