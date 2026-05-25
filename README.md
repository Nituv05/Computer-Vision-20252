# Domain Generalization for Image Classification

## Quick Start

```bash
pip install -r requirements.txt

# Generate the presentation slides
python slides/generate_slides.py
# → m2cl_presentation.pptx

# Generate the reproduction report PDF
python paper/generate_report.py
# → m2cl_report.pdf

# Smoke-test the model (no dataset required)
python models/m2cl.py
python losses/contrastive.py
```

---

## Training

```bash
# PACS — test on Photo domain
python train.py --dataset pacs --test_domain photo --data_root /path/to/datasets

# VLCS — test on Caltech101 domain
python train.py --dataset vlcs --test_domain Caltech101 --data_root /path/to/datasets

# Office-Home — test on Art domain
python train.py --dataset office_home --test_domain Art --data_root /path/to/datasets
```

## Evaluation (leave-one-domain-out)

```bash
python evaluate.py --dataset pacs --data_root /path/to/datasets
python evaluate.py --dataset vlcs --data_root /path/to/datasets
python evaluate.py --dataset office_home --data_root /path/to/datasets
```

---

## Dataset Preparation

| Dataset     | Download |
|-------------|----------|
| PACS        | http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017 |
| VLCS        | https://github.com/belaalb/G2DM |
| Office-Home | https://hemanthdv.github.io/officehome-dataset/ |
| NICO        | https://nicochallenge.com/ |

Expected root structure:
```
data_root/
  pacs/
    art_painting/<class>/*.jpg
    cartoon/<class>/*.jpg
    photo/<class>/*.jpg
    sketch/<class>/*.jpg
  vlcs/
    PASCAL/<class>/*.jpg
    LabelMe/<class>/*.jpg
    Caltech101/<class>/*.jpg
    SUN09/<class>/*.jpg
  office_home/
    Art/<class>/*.jpg
    ...
  nico/
    <class>/<context>/*.jpg
```

---

## Paper Results (ResNet-18)

| Dataset     | M²-CL Avg | 2nd Best (SAGM) | Δ      |
|-------------|-----------|-----------------|--------|
| PACS        | 83.54%    | 82.68%          | +0.86% |
| VLCS        | 77.78%    | 75.17%          | +2.61% |
| Office-Home | 63.27%    | 59.66%          | +3.61% |
| NICO (N=7)  | 62.19%    | 59.10%          | +3.09% |

---

## Project Structure

```
root/
├── models/
│   ├── extraction_block.py   # Extraction block with concentration pipelines
│   └── m2cl.py               # Full M²-CL model (ResNet-18 + 13 extraction blocks)
├── losses/
│   └── contrastive.py        # Multi-layer contrastive loss
├── data/
│   ├── pacs.py / vlcs.py / office_home.py / nico.py
├── utils/
│   └── transforms.py
├── configs/
│   ├── pacs.yaml / vlcs.yaml / office_home.yaml / nico.yaml
├── slides/
│   └── generate_slides.py    # Generates m2cl_presentation.pptx
├── paper/
│   └── generate_report.py    # Generates m2cl_report.pdf
├── train.py
└── evaluate.py
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| α (loss weight) | 0.01 | α = 1.0 causes ~20% drop |
| τ (temperature) | 1.0  | Stable in range 0.1–2.0 |
| r (reduction)   | 4    | Channel compression ratio |
| dropout p       | 0.5  | Spatial (channel) dropout |
| embed_dim       | 128  | Per-layer embedding size |
