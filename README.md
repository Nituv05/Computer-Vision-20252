# M2-CL Reproduction for Domain Generalization

Course-project implementation of **Multiscale and Multilayer Contrastive
Learning for Domain Generalization**.

This repo supports the paper's compared baselines and the proposed methods:

- `erm`: vanilla ResNet classifier baseline.
- `rsc`: representation self-challenging.
- `mixup`: domain mixup.
- `coral`: covariance alignment.
- `mmd`: maximum mean discrepancy.
- `sagnet`: style agnostic network.
- `selfreg`: self-supervised contrastive regularization baseline.
- `arm`: adaptive risk minimization.
- `eqrm`: empirical quantile risk minimization.
- `sagm`: sharpness-aware gradient matching baseline.
- `m2`: multi-scale/multi-layer extraction blocks concatenated into the classifier.
- `m2cl`: M2 plus the paper's layer-wise supervised contrastive regularizer.

The implementation is focused on the code required for a final-project
reproduction: training, leave-one-domain/context evaluation, ablations,
saliency maps and result summaries.

The default protocol is prepared for the paper reproduction:

- source domains are split independently with a DomainBed-style
  `holdout_fraction=0.2`;
- checkpoint selection uses source validation accuracy, not target-test
  accuracy;
- `hparams_profile=paper` is the default;
- M2/M2-CL use the paper setup: ImageNet-pretrained ResNet-18/50,
  SGD, `lr=0.001`, `batch_size=128`, `epochs=30`, reduction ratio
  `r=4`, `alpha=0.01` and temperature `tau=1.0`;
- the ten comparison baselines use DomainBed-style/recommended
  baseline hyperparameters, matching the paper's baseline paragraph;
- scheduler defaults to `none`.

## Setup

```bash
pip install -r requirements.txt
```

If ImageNet weights are not cached and the machine has no internet access, add
`--no_pretrained` to training commands. The paper uses ImageNet-pretrained
ResNet backbones.

## Dataset Layout

Datasets are not included. Put them under one `data_root`:

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

`office_home/RealWorld` also accepts common aliases such as `Real_World`,
`Real World` and `Real-World`.

## Download Datasets

PACS, VLCS and Office-Home can be downloaded with the project downloader.
It tries the DomainBed Google Drive links first, then falls back to public
mirrors when Drive blocks scripted access:

```bash
python tools/download_data.py --data_root /path/to/data_root --datasets pacs vlcs office_home
```

Validate the folder layout before starting long jobs:

```bash
python tools/check_data.py --data_root /path/to/data_root --dataset pacs
python tools/check_data.py --data_root /path/to/data_root --dataset vlcs
python tools/check_data.py --data_root /path/to/data_root --dataset office_home
python tools/check_data.py --data_root /path/to/data_root --dataset all
```

## Train

PACS leave-one-domain-out:

```bash
python tools/train.py --dataset pacs --test_domain photo --data_root /path/to/data_root --method m2cl --backbone resnet18
```

VLCS and Office-Home:

```bash
python tools/train.py --dataset vlcs --test_domain CALTECH --data_root /path/to/data_root --method m2cl
python tools/train.py --dataset office_home --test_domain Art --data_root /path/to/data_root --method m2cl
```

Useful variants and baselines:

```bash
python tools/train.py --dataset pacs --test_domain photo --method erm
python tools/train.py --dataset pacs --test_domain photo --method rsc
python tools/train.py --dataset pacs --test_domain photo --method mixup
python tools/train.py --dataset pacs --test_domain photo --method coral
python tools/train.py --dataset pacs --test_domain photo --method mmd
python tools/train.py --dataset pacs --test_domain photo --method sagnet
python tools/train.py --dataset pacs --test_domain photo --method selfreg
python tools/train.py --dataset pacs --test_domain photo --method arm
python tools/train.py --dataset pacs --test_domain photo --method eqrm
python tools/train.py --dataset pacs --test_domain photo --method sagm
python tools/train.py --dataset pacs --test_domain photo --method m2 --alpha 0
python tools/train.py --dataset pacs --test_domain photo --method m2cl --backbone resnet50
```

Checkpoints and metrics JSON are saved under `outputs/checkpoints/` by default.

Run the full paper-comparison grid over all domains and seeds:

```bash
python tools/run_experiments.py --dataset pacs --data_root /path/to/data_root --methods all --backbones resnet18 --seeds 0 1 2 --epochs 30 --holdout_fraction 0.2 --scheduler none --hparams_profile paper
python tools/run_experiments.py --dataset vlcs --data_root /path/to/data_root --methods all --backbones resnet18 --seeds 0 1 2 --epochs 30 --holdout_fraction 0.2 --scheduler none --hparams_profile paper
python tools/run_experiments.py --dataset office_home --data_root /path/to/data_root --methods all --backbones resnet18 --seeds 0 1 2 --epochs 30 --holdout_fraction 0.2 --scheduler none --hparams_profile paper
```

The same ResNet-18 grid can be launched with:

```bash
bash scripts/run_main.sh
```

For the paper protocol, use the paper wrapper. It runs PACS, VLCS and
Office-Home sequentially, uses `epochs=30`, `hparams_profile=paper`,
`holdout_fraction=0.2`, `scheduler=none`, `seeds=0 1 2`, `methods=all`, and
resumes with `--skip_existing`.

```bash
bash scripts/run_paper.sh
BACKBONE=resnet50 bash scripts/run_paper.sh
```

Enable W&B logging with one project per dataset:

```bash
WANDB=1 bash scripts/run_paper.sh
BACKBONE=resnet50 WANDB=1 bash scripts/run_paper.sh
```

Useful overrides:

```bash
DATA_ROOT=/path/to/data_root WANDB=1 bash scripts/run_paper.sh
METHODS="erm mixup coral rsc sagm m2 m2cl" WANDB=1 bash scripts/run_paper.sh
DATASETS="pacs" SEEDS="0" BACKBONE=resnet50 WANDB=1 bash scripts/run_paper.sh
```

`--methods all` runs ERM, RSC, Mixup, CORAL, MMD, SagNet, SelfReg, ARM,
EQRM, SAGM, M2 and M2-CL. Use `--methods erm m2 m2cl` only as a reduced
debugging subset, not as the full paper baseline comparison.

Use `--dry_run` first to print the exact commands without training. Use
`--skip_existing` when resuming an interrupted grid.

Optional W&B logging:

```bash
wandb login
python tools/run_experiments.py --dataset pacs --data_root /path/to/data_root --methods erm mixup coral rsc sagm m2 m2cl --backbones resnet18 --seeds 0 1 2 --epochs 30 --hparams_profile paper --wandb --wandb_project cv20252-m2cl --skip_existing
```

## Evaluate

Evaluate all held-out domains for a dataset:

```bash
python tools/evaluate.py --dataset pacs --data_root /path/to/data_root --method m2cl
python tools/evaluate.py --dataset vlcs --data_root /path/to/data_root --method m2cl
python tools/evaluate.py --dataset office_home --data_root /path/to/data_root --method m2cl
```

Summarize metric JSON files across seeds/runs:

```bash
python tools/summarize_results.py --metrics_dir outputs/checkpoints --output_csv outputs/results.csv
```

## Ablations

Architecture/model baselines from the paper's ablation setup:

```bash
python tools/run_ablations.py --study architecture --dataset pacs --all_domains --data_root /path/to/data_root --seeds 0 1 2
python tools/run_ablations.py --study architecture --dataset vlcs --all_domains --data_root /path/to/data_root --seeds 0 1 2
```

The concrete model classes are implemented in
`models/architecture_baselines.py`. The ablation runner in
`tools/run_ablations.py` calls `tools/train.py` for each variant.

This runs the plain ResNet ERM baseline, M2 cascading/parallel variants with
different reduction ratios and dropout settings, and the full M2-CL model. Each
variant is saved with a `--tag`, so summaries do not mix several M2 variants
under one name.

Architecture ablation for pipeline type, reduction ratio, dropout and loss
on a single target domain:

```bash
python tools/run_ablations.py --dataset pacs --test_domain photo --data_root /path/to/data_root --study architecture
```

Sensitivity studies:

```bash
python tools/run_ablations.py --dataset pacs --test_domain photo --data_root /path/to/data_root --study tau
python tools/run_ablations.py --dataset pacs --test_domain photo --data_root /path/to/data_root --study alpha
```

To reproduce the slide/report ablation tables, run the table mode. This expands
to all domains of PACS and VLCS and supports `--seeds`, `--skip_existing`,
W&B logging and `--dry_run`:

```bash
python tools/run_ablations.py \
  --tables all \
  --datasets pacs vlcs \
  --data_root /path/to/data_root \
  --backbone resnet18 \
  --seeds 0 1 2 \
  --epochs 30 \
  --batch_size 128 \
  --lr 0.001 \
  --hparams_profile paper \
  --holdout_fraction 0.2 \
  --scheduler none \
  --num_workers 4 \
  --save_dir outputs/ablation_tables \
  --skip_existing
```

Individual table modes are available:

```bash
python tools/run_ablations.py --tables 5 --datasets pacs vlcs --data_root /path/to/data_root
python tools/run_ablations.py --tables 6 --datasets pacs vlcs --data_root /path/to/data_root
python tools/run_ablations.py --tables 7 --datasets pacs vlcs --data_root /path/to/data_root
```

Use `--dry_run` to print commands without running them. After training, summarize:

```bash
python tools/summarize_results.py \
  --metrics_dir outputs/ablation_tables \
  --output_csv outputs/ablation_tables/results.csv
```

## Saliency Maps

After training a checkpoint:

```bash
python tools/saliency.py --dataset pacs --test_domain photo --data_root /path/to/data_root --method m2cl
```

The script saves side-by-side original/saliency images to `outputs/saliency/`.

## Paper Reference Results

Top-1 accuracy from the paper:

| Dataset | ResNet-18 M2-CL | ResNet-50 M2-CL |
|---|---:|---:|
| PACS avg | 83.54 | 85.97 |
| VLCS avg | 77.78 | 78.36 |
| Office-Home avg | 63.27 | 71.07 |

The paper compares against DomainBed baselines ERM, RSC, Mixup, CORAL, MMD,
SagNet, SelfReg, ARM, EQRM and SAGM. This project repo implements all ten
baseline method entry points plus M2 and M2-CL directly.

## Project Structure

```text
models/
  extraction_block.py   # official-style M2 concentration pipeline
  m2cl.py               # ERM/M2/M2-CL model builder, ResNet-18/50
  architecture_baselines.py # explicit architecture baseline model classes
  architecture_specs.py # shared architecture baseline tags and run settings
algorithms/
  baselines.py          # ERM, RSC, Mixup, CORAL, MMD, SagNet, SelfReg, ARM, EQRM, SAGM
losses/
  contrastive.py        # layer-wise M2-CL contrastive objective
data/
  pacs.py vlcs.py office_home.py
configs/
  pacs.yaml vlcs.yaml office_home.yaml
docs/
  CODE_READING_GUIDE.md
  SERVER_RUNBOOK.md
  papers/m2cl_paper.pdf
  slides/HUST_THEME_BEAMER/
reportcv/
  m2cl_report.tex m2cl_report.pdf result_tables.tex result_tables.pdf
scripts/
  run_main.sh run_paper.sh run_ablations.sh
  sync-train.sh smoke_test.sh check_server_ready.sh
  make_colab_notebooks.py
tools/
  train.py evaluate.py run_experiments.py summarize_results.py
  download_data.py check_data.py saliency.py
  run_ablations.py
```
