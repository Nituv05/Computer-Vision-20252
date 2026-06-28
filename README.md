# M²-CL: Multiscale and Multilayer Contrastive Learning for Domain Generalization

Course-project implementation of **M²-CL** — a domain generalization method that
attaches multi-scale, multi-layer feature extraction blocks to a ResNet backbone
and regularizes them with a layer-wise supervised contrastive loss.

Supported methods: `erm`, `rsc`, `mixup`, `coral`, `mmd`, `sagnet`, `selfreg`,
`arm`, `eqrm`, `sagm`, `m2`, `m2cl`.

---

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Layout

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

Download automatically:

```bash
python tools/download_data.py --data_root data_root --datasets pacs vlcs office_home
```

Verify layout:

```bash
python tools/check_data.py --data_root data_root --dataset all
```

---

## Training

Single run (M²-CL, PACS, leave photo out):

```bash
python tools/train.py --dataset pacs --test_domain photo --data_root data_root --method m2cl --backbone resnet18
```

Other methods:

```bash
python tools/train.py --dataset pacs --test_domain photo --method erm
python tools/train.py --dataset pacs --test_domain photo --method m2 --alpha 0
python tools/train.py --dataset pacs --test_domain photo --method m2cl --backbone resnet50
```

Full grid (all methods × all domains × 3 seeds):

```bash
python tools/run_experiments.py \
  --dataset pacs --data_root data_root \
  --methods all --backbones resnet18 --seeds 0 1 2 \
  --epochs 30 --holdout_fraction 0.2 --hparams_profile paper
```

Or with the wrapper script (runs PACS, VLCS, Office-Home sequentially):

```bash
bash scripts/run_main.sh
BACKBONE=resnet50 bash scripts/run_main.sh
```

Checkpoints and per-run metrics are saved to `outputs/checkpoints/` by default.

---

## Evaluation

```bash
python tools/evaluate.py --dataset pacs --data_root data_root --method m2cl
```

Summarize results across seeds to CSV:

```bash
python tools/summarize_results.py --metrics_dir outputs/checkpoints --output_csv outputs/results.csv
```

---

## Ablations

Architecture and hyperparameter sensitivity studies:

```bash
python tools/run_ablations.py --tables all \
  --datasets pacs vlcs --data_root data_root \
  --backbone resnet18 --seeds 0 1 2 --epochs 30 \
  --hparams_profile paper --save_dir outputs/ablations --skip_existing
```

Individual tables:

```bash
python tools/run_ablations.py --tables 5 --datasets pacs vlcs --data_root data_root  # architecture
python tools/run_ablations.py --tables 6 --datasets pacs vlcs --data_root data_root  # tau sensitivity
python tools/run_ablations.py --tables 7 --datasets pacs vlcs --data_root data_root  # alpha sensitivity
```

Or use the wrapper:

```bash
bash scripts/run_ablations.sh
```

---

## Saliency Maps

```bash
python tools/saliency.py --dataset pacs --test_domain photo --data_root data_root --method m2cl
```

Saves side-by-side original/saliency images to `outputs/saliency/`.

---

## Paper Reference Numbers

| Dataset       | ResNet-18 | ResNet-50 |
|---------------|----------:|----------:|
| PACS          |     83.54 |     85.97 |
| VLCS          |     77.78 |     78.36 |
| Office-Home   |     63.27 |     71.07 |

---

## Project Structure

```
algorithms/
  baselines.py          # ERM, RSC, Mixup, CORAL, MMD, SagNet, SelfReg, ARM, EQRM, SAGM, M2, M2CL
configs/
  pacs.yaml  vlcs.yaml  office_home.yaml
data/
  pacs.py  vlcs.py  office_home.py
losses/
  contrastive.py        # layer-wise supervised contrastive loss
models/
  m2cl.py               # M2/M2-CL model, ResNet-18/50 backbone
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
  smoke_test.sh         # quick sanity check (small epochs/batch)
other_materials/
  report/               # LaTeX source and PDF report
  docs/                 # slides, paper PDF, runbook
  demo/                 # Gradio saliency demo
```
