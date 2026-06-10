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
  nico/
    <class>/<context>/*.jpg
```

`office_home/RealWorld` also accepts common aliases such as `Real_World`,
`Real World` and `Real-World`.

## Download Datasets

PACS, VLCS and Office-Home can be downloaded with the project downloader.
It tries the DomainBed Google Drive links first, then falls back to public
mirrors when Drive blocks scripted access:

```bash
python download_data.py --data_root /path/to/data_root --datasets pacs vlcs office_home
```

NICO is distributed by the official project site through Dropbox/Baidu. Download
the archive manually from https://nico.thumedialab.com/, then normalize it:

```bash
python download_data.py --data_root /path/to/data_root --datasets nico --nico_archive /path/to/NICO.zip
```

Validate the folder layout before starting long jobs:

```bash
python check_data.py --data_root /path/to/data_root --dataset pacs
python check_data.py --data_root /path/to/data_root --dataset vlcs
python check_data.py --data_root /path/to/data_root --dataset office_home
```

Use `--dataset all` only after NICO is also present.

## Train

PACS leave-one-domain-out:

```bash
python train.py --dataset pacs --test_domain photo --data_root /path/to/data_root --method m2cl --backbone resnet18
```

VLCS and Office-Home:

```bash
python train.py --dataset vlcs --test_domain CALTECH --data_root /path/to/data_root --method m2cl
python train.py --dataset office_home --test_domain Art --data_root /path/to/data_root --method m2cl
```

NICO leave-multiple-contexts-out:

```bash
python train.py --dataset nico --n_heldout 7 --data_root /path/to/data_root --method m2cl
```

Useful variants and baselines:

```bash
python train.py --dataset pacs --test_domain photo --method erm
python train.py --dataset pacs --test_domain photo --method rsc
python train.py --dataset pacs --test_domain photo --method mixup
python train.py --dataset pacs --test_domain photo --method coral
python train.py --dataset pacs --test_domain photo --method mmd
python train.py --dataset pacs --test_domain photo --method sagnet
python train.py --dataset pacs --test_domain photo --method selfreg
python train.py --dataset pacs --test_domain photo --method arm
python train.py --dataset pacs --test_domain photo --method eqrm
python train.py --dataset pacs --test_domain photo --method sagm
python train.py --dataset pacs --test_domain photo --method m2 --alpha 0
python train.py --dataset pacs --test_domain photo --method m2cl --backbone resnet50
```

Checkpoints and metrics JSON are saved under `outputs/checkpoints/` by default.

Run the full paper-comparison grid over all domains and seeds:

```bash
python run_experiments.py --dataset pacs --data_root /path/to/data_root --methods all --backbones resnet18 --seeds 0 1 2 --epochs 30 --holdout_fraction 0.2 --scheduler none --hparams_profile paper
python run_experiments.py --dataset vlcs --data_root /path/to/data_root --methods all --backbones resnet18 --seeds 0 1 2 --epochs 30 --holdout_fraction 0.2 --scheduler none --hparams_profile paper
python run_experiments.py --dataset office_home --data_root /path/to/data_root --methods all --backbones resnet18 --seeds 0 1 2 --epochs 30 --holdout_fraction 0.2 --scheduler none --hparams_profile paper
```

The same ResNet-18 grid can be launched with:

```bash
bash scripts/run_main.sh
```

For the paper protocol, use the explicit backbone scripts. They run PACS,
VLCS and Office-Home sequentially, use `epochs=30`, `hparams_profile=paper`,
`holdout_fraction=0.2`, `scheduler=none`, `seeds=0 1 2`, `methods=all`, and
resume with `--skip_existing`.

```bash
bash scripts/run_paper_resnet18.sh
bash scripts/run_paper_resnet50.sh
```

Enable W&B logging with one project per dataset:

```bash
WANDB=1 bash scripts/run_paper_resnet18.sh
WANDB=1 bash scripts/run_paper_resnet50.sh
```

Useful overrides:

```bash
DATA_ROOT=/path/to/data_root WANDB=1 bash scripts/run_paper_resnet18.sh
METHODS="erm mixup coral rsc sagm m2 m2cl" WANDB=1 bash scripts/run_paper_resnet18.sh
DATASETS="pacs" SEEDS="0" WANDB=1 bash scripts/run_paper_resnet50.sh
```

`--methods all` runs ERM, RSC, Mixup, CORAL, MMD, SagNet, SelfReg, ARM,
EQRM, SAGM, M2 and M2-CL. Use `--methods erm m2 m2cl` only as a reduced
debugging subset, not as the full paper baseline comparison.

Use `--dry_run` first to print the exact commands without training. Use
`--skip_existing` when resuming an interrupted grid.

Optional W&B logging:

```bash
wandb login
python run_experiments.py --dataset pacs --data_root /path/to/data_root --methods erm mixup coral rsc sagm m2 m2cl --backbones resnet18 --seeds 0 1 2 --epochs 30 --hparams_profile paper --wandb --wandb_project cv20252-m2cl --skip_existing
```

## Evaluate

Evaluate all held-out domains for a dataset:

```bash
python evaluate.py --dataset pacs --data_root /path/to/data_root --method m2cl
python evaluate.py --dataset vlcs --data_root /path/to/data_root --method m2cl
python evaluate.py --dataset office_home --data_root /path/to/data_root --method m2cl
```

Evaluate NICO N=3,5,7:

```bash
python evaluate.py --dataset nico --data_root /path/to/data_root --method m2cl
```

Summarize metric JSON files across seeds/runs:

```bash
python summarize_results.py --metrics_dir outputs/checkpoints --output_csv outputs/results.csv
```

## Ablations

Architecture/model baselines from the paper's ablation setup:

```bash
python architecture_baselines.py --dataset pacs --data_root /path/to/data_root --seeds 0 1 2
python architecture_baselines.py --dataset vlcs --data_root /path/to/data_root --seeds 0 1 2
```

The concrete model classes are implemented in
`models/architecture_baselines.py`. The root-level `architecture_baselines.py`
file is only the experiment runner that calls `train.py` for each variant.

This runs the plain ResNet ERM baseline, M2 cascading/parallel variants with
different reduction ratios and dropout settings, and the full M2-CL model. Each
variant is saved with a `--tag`, so summaries do not mix several M2 variants
under one name.

Architecture ablation for pipeline type, reduction ratio, dropout and loss
on a single target domain:

```bash
python ablation.py --dataset pacs --test_domain photo --data_root /path/to/data_root --study architecture
```

Sensitivity studies:

```bash
python ablation.py --dataset pacs --test_domain photo --data_root /path/to/data_root --study tau
python ablation.py --dataset pacs --test_domain photo --data_root /path/to/data_root --study alpha
```

To reproduce the slide/report ablation tables, run the table mode. This expands
to all domains of PACS and VLCS and supports `--seeds`, `--skip_existing`,
W&B logging and `--dry_run`:

```bash
python ablation.py \
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
python ablation.py --tables 5 --datasets pacs vlcs --data_root /path/to/data_root
python ablation.py --tables 6 --datasets pacs vlcs --data_root /path/to/data_root
python ablation.py --tables 7 --datasets pacs vlcs --data_root /path/to/data_root
```

Use `--dry_run` to print commands without running them. After training, summarize:

```bash
python summarize_results.py \
  --metrics_dir outputs/ablation_tables \
  --output_csv outputs/ablation_tables/results.csv
```

## Saliency Maps

After training a checkpoint:

```bash
python saliency.py --dataset pacs --test_domain photo --data_root /path/to/data_root --method m2cl
```

The script saves side-by-side original/saliency images to `outputs/saliency/`.

## Paper Reference Results

Top-1 accuracy from the paper:

| Dataset | ResNet-18 M2-CL | ResNet-50 M2-CL |
|---|---:|---:|
| PACS avg | 83.54 | 85.97 |
| VLCS avg | 77.78 | 78.36 |
| Office-Home avg | 63.27 | 71.07 |
| NICO N=3 | 87.93 | 89.30 |
| NICO N=5 | 84.10 | 87.68 |
| NICO N=7 | 82.14 | 86.90 |

The paper compares against DomainBed baselines ERM, RSC, Mixup, CORAL, MMD,
SagNet, SelfReg, ARM, EQRM and SAGM. This project repo implements all ten
baseline method entry points plus M2 and M2-CL directly.

## Project Structure

```text
models/
  extraction_block.py   # official-style M2 concentration pipeline
  m2cl.py               # ERM/M2/M2-CL model builder, ResNet-18/50
  architecture_baselines.py # explicit architecture baseline model classes
architecture_specs.py   # shared architecture baseline tags and run settings
algorithms/
  baselines.py          # ERM, RSC, Mixup, CORAL, MMD, SagNet, SelfReg, ARM, EQRM, SAGM
losses/
  contrastive.py        # layer-wise M2-CL contrastive objective
data/
  pacs.py vlcs.py office_home.py nico.py
configs/
  pacs.yaml vlcs.yaml office_home.yaml nico.yaml
download_data.py
check_data.py
train.py
evaluate.py
run_experiments.py
architecture_baselines.py
summarize_results.py
ablation.py
saliency.py
```
