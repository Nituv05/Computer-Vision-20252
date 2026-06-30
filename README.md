# M²-CL: Multiscale and Multilayer Contrastive Learning for Domain Generalization

Course-project implementation of **M²-CL** — a domain generalization method that attaches multi-scale, multi-layer feature extraction blocks to a ResNet backbone and regularizes them with a layer-wise supervised contrastive loss.

Supported methods: `erm`, `rsc`, `mixup`, `coral`, `mmd`, `sagnet`, `selfreg`, `arm`, `eqrm`, `sagm`, `m2`, `m2cl`.

---

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU recommended (CPU works but is slow)

---

## 1. Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Download Datasets

Run the downloader script from the **project root**. It tries the official DomainBed mirrors first and falls back to HuggingFace/Mediafire automatically.

```bash
python tools/download_data.py --data_root data_root --datasets pacs vlcs office_home
```

Expected time: ~5–15 min depending on connection speed.

After downloading, verify the layout:

```bash
python tools/check_data.py --data_root data_root --dataset all
```

The expected folder structure is:

```
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

---

## 3. Quick Sanity Check (Smoke Test)

Run 1 epoch each on ERM and M²-CL to confirm the environment works before starting a full run:

```bash
bash scripts/smoke_test.sh
```

Expected output ends with:
```
Smoke test finished. Outputs: outputs/smoke
```

---

## 4. Reproducing Paper Results

### 4a. Single run (fastest way to check one number)

```bash
python tools/train.py \
  --dataset pacs \
  --test_domain photo \
  --data_root data_root \
  --method m2cl \
  --backbone resnet18
```

Checkpoint and per-run JSON metrics are saved to `outputs/checkpoints/`.

### 4b. Full reproduction — all methods, all leave-one-out splits, 3 seeds

```bash
bash scripts/run_main.sh
```

This runs PACS → VLCS → Office-Home sequentially with ResNet-18, seeds 0/1/2, and all methods, then writes a summary CSV to `outputs/main_resnet18_results.csv`.

For ResNet-50:

```bash
BACKBONE=resnet50 bash scripts/run_main.sh
```

### 4c. Controlling the run via environment variables

| Variable | Default | Effect |
|---|---|---|
| `DATA_ROOT` | `data_root` | Path to datasets |
| `BACKBONE` | `resnet18` | `resnet18` or `resnet50` |
| `SEEDS` | `0 1 2` | Space-separated list |
| `METHODS` | `all` | Space-separated method names, or `all` |
| `DATASETS` | `pacs vlcs office_home` | Which datasets to run |
| `EPOCHS` | config default | Override epoch count |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `SAVE_DIR` | `outputs/checkpoints` | Output directory |

Example — run only M²-CL on PACS with 2 seeds:

```bash
METHODS="m2cl" DATASETS="pacs" SEEDS="0 1" bash scripts/run_main.sh
```

### 4d. Summarize results to CSV

```bash
python tools/summarize_results.py \
  --metrics_dir outputs/checkpoints \
  --output_csv outputs/results.csv
```

---

## 5. Expected Results

Numbers below are mean test accuracy (%) averaged over 4 leave-one-domain-out splits and 3 seeds.

| Dataset | ResNet-18 | ResNet-50 |
|---|---:|---:|
| PACS | 83.54 | 85.97 |
| VLCS | 77.78 | 78.36 |
| Office-Home | 63.27 | 71.07 |

Per-domain breakdown for PACS ResNet-18 (Table I in the report):

| art\_painting | cartoon | photo | sketch | avg |
|---:|---:|---:|---:|---:|
| 81.66 | 78.42 | 97.00 | 77.07 | 83.54 |

---

## 6. Evaluation Only (from saved checkpoint)

```bash
python tools/evaluate.py \
  --dataset pacs \
  --data_root data_root \
  --method m2cl
```

---

## 7. Ablation Studies

Run all ablation tables (architecture variants, temperature τ, balance weight α):

```bash
bash scripts/run_ablations.sh
```

Individual tables:

```bash
# Table 5 — architecture variants
python tools/run_ablations.py --tables 5 --datasets pacs vlcs --data_root data_root

# Table 6 — temperature τ sensitivity
python tools/run_ablations.py --tables 6 --datasets pacs vlcs --data_root data_root

# Table 7 — alpha α sensitivity
python tools/run_ablations.py --tables 7 --datasets pacs vlcs --data_root data_root
```

Results are saved to `outputs/ablation_tables/results.csv`.

---

## 8. Saliency Maps

Visualize which image regions the model attends to:

```bash
python tools/saliency.py \
  --dataset pacs \
  --test_domain photo \
  --data_root data_root \
  --method m2cl
```

Side-by-side original/saliency images are saved to `outputs/saliency/`.

---

## Project Structure

```
algorithms/
  baselines.py          # ERM, RSC, Mixup, CORAL, MMD, SagNet, SelfReg, ARM, EQRM, SAGM, M2, M2CL
configs/
  pacs.yaml             # default hyperparameters for each dataset
  vlcs.yaml
  office_home.yaml
data/
  pacs.py               # dataset loaders
  vlcs.py
  office_home.py
losses/
  contrastive.py        # layer-wise supervised contrastive loss
models/
  m2cl.py               # M2/M2-CL model with ResNet-18/50 backbone
  extraction_block.py   # multi-scale concentration pipeline
  architecture_baselines.py  # ablation model variants
  architecture_specs.py      # ablation variant metadata
utils/
  transforms.py         # ImageNet normalisation, train/test augmentation
tools/
  train.py              # single-run training entry point
  evaluate.py           # checkpoint evaluation
  run_experiments.py    # full grid launcher
  run_ablations.py      # ablation study launcher
  summarize_results.py  # aggregate metrics to CSV
  download_data.py      # dataset downloader
  check_data.py         # dataset layout validator
  saliency.py           # gradient saliency visualizer
scripts/
  run_main.sh           # wrapper: full experiment grid
  run_ablations.sh      # wrapper: ablation grid
  smoke_test.sh         # quick sanity check (1 epoch)
```
