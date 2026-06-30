# M2-CL: Multiscale and Multilayer Contrastive Learning for Domain Generalization

Course-project implementation of M2-CL for visual domain generalization. The
project trains image classifiers on source domains and evaluates them on one
held-out target domain.

Supported methods:

```text
erm, rsc, mixup, coral, mmd, sagnet, selfreg, arm, eqrm, sagm, m2, m2cl
```

Supported benchmarks:

```text
PACS, VLCS, Office-Home
```

## Requirements

- Python 3.10 or newer
- CUDA-capable GPU recommended for full experiments
- CPU can run sanity checks, but full training will be slow

## 1. Environment Setup

Linux, macOS, WSL, or Git Bash:

```bash
git clone <repo-url>
cd Computer-Vision-20252
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
git clone <repo-url>
cd Computer-Vision-20252
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Dataset Setup

Run the downloader from the project root. The expected dataset root is
`data_root/`. The downloader tries DomainBed links first and falls back to public
mirrors where available.

```bash
python tools/download_data.py --data_root data_root --datasets pacs vlcs office_home
```

Expected download time is roughly 5-15 minutes depending on network speed.

Verify the downloaded layout before training:

```bash
python tools/check_data.py --data_root data_root --dataset all
```

Expected layout:

```text
data_root/
  pacs/
    art_painting/<class>/*.jpg
    cartoon/<class>/*.jpg
    photo/<class>/*.jpg
    sketch/<class>/*.jpg
  vlcs/
    CALTECH/full/<class_id>/*.jpg
    LABELME/full/<class_id>/*.jpg
    PASCAL/full/<class_id>/*.jpg
    SUN/full/<class_id>/*.jpg
  office_home/
    Art/<class>/*.jpg
    Clipart/<class>/*.jpg
    Product/<class>/*.jpg
    RealWorld/<class>/*.jpg
```

`Office-Home` also accepts common aliases such as `Real_World`,
`Real World`, and `Real-World`.

## 3. Quick Sanity Check

After downloading PACS, run a small smoke test before starting a full run. This
trains ERM and M2-CL for one epoch on the PACS photo held-out split.

```bash
bash scripts/smoke_test.sh
```

Expected output ends with:

```text
Smoke test finished. Outputs: outputs/smoke
```

If `bash` is not available, run the equivalent Python commands manually:

```bash
python tools/train.py --dataset pacs --test_domain photo --data_root data_root --method erm --backbone resnet18 --epochs 1 --batch_size 64 --save_dir outputs/smoke
python tools/train.py --dataset pacs --test_domain photo --data_root data_root --method m2cl --backbone resnet18 --epochs 1 --batch_size 64 --save_dir outputs/smoke
python tools/summarize_results.py --metrics_dir outputs/smoke --output_csv outputs/smoke/smoke_results.csv
```

Successful runs write:

```text
outputs/smoke/*.pth
outputs/smoke/*.json
outputs/smoke/smoke_results.csv
```

## 4. Single Training Run

Example: train M2-CL on PACS with `photo` as the unseen target domain.

```bash
python tools/train.py \
  --dataset pacs \
  --test_domain photo \
  --data_root data_root \
  --method m2cl \
  --backbone resnet18 \
  --epochs 30 \
  --hparams_profile paper \
  --save_dir outputs/checkpoints
```

Other examples:

```bash
python tools/train.py --dataset pacs --test_domain photo --data_root data_root --method erm --backbone resnet18
python tools/train.py --dataset pacs --test_domain photo --data_root data_root --method m2 --backbone resnet18 --alpha 0
python tools/train.py --dataset pacs --test_domain photo --data_root data_root --method m2cl --backbone resnet50
```

The training script saves the best checkpoint and metrics JSON under
`outputs/checkpoints/` by default.

## 5. Full Experiment Grid

The experiment launcher runs all held-out domains for the chosen dataset.
Use `--dry_run` first to inspect the generated commands without training:

```bash
python tools/run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods all \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --epochs 30 \
  --holdout_fraction 0.2 \
  --hparams_profile paper \
  --skip_existing \
  --dry_run
```

Run the actual PACS grid by removing `--dry_run`:

```bash
python tools/run_experiments.py \
  --dataset pacs \
  --data_root data_root \
  --methods all \
  --backbones resnet18 \
  --seeds 0 1 2 \
  --epochs 30 \
  --holdout_fraction 0.2 \
  --hparams_profile paper \
  --skip_existing
```

To run PACS, VLCS, and Office-Home sequentially:

```bash
bash scripts/run_main.sh
```

This runs PACS, VLCS, and Office-Home sequentially with ResNet-18, seeds 0/1/2,
and all methods, then writes a summary CSV to `outputs/main_resnet18_results.csv`.

Useful wrapper overrides:

```bash
BACKBONE=resnet50 bash scripts/run_main.sh
METHODS="erm m2 m2cl" bash scripts/run_main.sh
SEEDS="0" EPOCHS=1 bash scripts/run_main.sh
```

Supported environment variables:

| Variable | Default | Effect |
|---|---|---|
| `DATA_ROOT` | `data_root` | Path to datasets |
| `BACKBONE` | `resnet18` | `resnet18` or `resnet50` |
| `SEEDS` | `0 1 2` | Space-separated seed list |
| `METHODS` | `all` | Space-separated method names, or `all` |
| `DATASETS` | `pacs vlcs office_home` | Datasets to run |
| `EPOCHS` | config default | Override epoch count |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `SAVE_DIR` | `outputs/checkpoints` | Output directory |

Example: run only M2-CL on PACS with two seeds:

```bash
METHODS="m2cl" DATASETS="pacs" SEEDS="0 1" bash scripts/run_main.sh
```

On Windows without `bash`, run `tools/run_experiments.py` separately for each
dataset with the same options.

## 6. Evaluation and Result Summary

Evaluate saved checkpoints:

```bash
python tools/evaluate.py \
  --dataset pacs \
  --data_root data_root \
  --checkpoint_dir outputs/checkpoints \
  --method m2cl \
  --backbone resnet18
```

Aggregate metrics JSON files into CSV:

```bash
python tools/summarize_results.py \
  --metrics_dir outputs/checkpoints \
  --output_csv outputs/results.csv
```

The summary script prints per-split mean/std and dataset averages across held-out
domains.

## 7. Ablation Studies

Run the ablation tables for PACS and VLCS:

```bash
python tools/run_ablations.py \
  --tables all \
  --datasets pacs vlcs \
  --data_root data_root \
  --backbone resnet18 \
  --seeds 0 1 2 \
  --epochs 30 \
  --hparams_profile paper \
  --save_dir outputs/ablation_tables \
  --skip_existing
```

Individual table groups:

```bash
python tools/run_ablations.py --tables 5 --datasets pacs vlcs --data_root data_root
python tools/run_ablations.py --tables 6 --datasets pacs vlcs --data_root data_root
python tools/run_ablations.py --tables 7 --datasets pacs vlcs --data_root data_root
```

Wrapper script:

```bash
bash scripts/run_ablations.sh
```

Results are saved under `outputs/ablation_tables/`.

## 8. Saliency Maps

Generate gradient saliency maps from a saved checkpoint:

```bash
python tools/saliency.py \
  --dataset pacs \
  --test_domain photo \
  --data_root data_root \
  --checkpoint_dir outputs/checkpoints \
  --method m2cl \
  --backbone resnet18 \
  --max_images 8
```

Outputs are saved to:

```text
outputs/saliency/
```

## 9. Reference Numbers

Reference averages from the M2-CL paper are included only as context. Local
results may differ depending on hardware, seeds, dataset mirrors, and training
settings.

| Dataset     | ResNet-18 | ResNet-50 |
|-------------|----------:|----------:|
| PACS        |     83.54 |     85.97 |
| VLCS        |     77.78 |     78.36 |
| Office-Home |     63.27 |     71.07 |

## 10. Project Structure

```text
algorithms/
  baselines.py              ERM, DG baselines, M2, and M2-CL wrappers
configs/
  pacs.yaml
  vlcs.yaml
  office_home.yaml
data/
  pacs.py
  vlcs.py
  office_home.py
losses/
  contrastive.py            layer-wise supervised contrastive loss
models/
  m2cl.py                   M2/M2-CL architecture with ResNet-18/50 backbones
  extraction_block.py       multi-scale concentration pipeline
  architecture_baselines.py ablation architecture variants
  architecture_specs.py     ablation variant metadata
utils/
  transforms.py             ImageNet normalization and augmentation
tools/
  train.py                  single-run training
  evaluate.py               checkpoint evaluation
  run_experiments.py        experiment grid launcher
  run_ablations.py          ablation grid launcher
  summarize_results.py      metrics aggregation
  download_data.py          dataset downloader
  check_data.py             dataset layout validator
  saliency.py               gradient saliency visualization
scripts/
  run_main.sh               full experiment wrapper
  run_ablations.sh          ablation wrapper
  smoke_test.sh             quick sanity check
```

Generated files are intentionally ignored by Git:

```text
data_root/
outputs/
wandb/
*.pth
*.pt
```
