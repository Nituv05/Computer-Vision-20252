"""Sinh notebook Colab (mỗi dataset+backbone 1 file) chạy độc lập.

Chạy từ root repo:
    python scripts/make_colab_notebooks.py
"""
import json
from pathlib import Path

REPO_URL = "https://github.com/Nituv05/Computer-Vision-20252.git"

# (dataset, backbone). Mỗi cặp = 1 notebook = 1 GPU riêng.
JOBS = [
    ("pacs", "resnet50"),
    ("vlcs", "resnet50"),
    ("office_home", "resnet50"),
    ("office_home", "resnet18"),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def notebook(ds, bb):
    bbtag = "r18" if bb == "resnet18" else "r50"
    return {
        "cells": [
            md([
                f"# M2-CL trên Colab — `{ds}` / {bb} (GPU riêng)\n",
                "Chạy ĐỘC LẬP 1 cặp dataset+backbone trên 1 Colab runtime/GPU riêng. Mở nhiều notebook = nhiều GPU song song thật.\n",
                "**Data** để ở ổ LOCAL `/content` (nhanh, ổn định — KHÔNG giải nén lên Drive). **Output + log + checkpoint** lưu trên Drive → ngắt session vẫn không mất, Run all lại là train tiếp (`--skip_existing`). Data tải lại mỗi session vài phút.\n",
                "Đủ 12 method = 10 baseline (erm,rsc,mixup,coral,mmd,sagnet,selfreg,arm,eqrm,sagm) + m2 + m2cl.\n",
                "**W&B**: điền key vào cell 4b (key để TRỐNG trong file). Dùng cá nhân — ĐỪNG commit/up file ĐÃ điền key lên GitHub public.\n",
                "Dùng: Runtime → Change runtime type → **A100 GPU** → Run all. (Batch tự dò = 128 = batch paper; A100 40GB chạy đúng paper + nhanh.)",
            ]),
            code([
                "# 1) GPU + mount Drive\n",
                "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')",
            ]),
            code([
                "# 2) Clone code + cài deps thiếu (giữ torch sẵn của Colab)\n",
                "import os\n",
                "WORK = '/content/drive/MyDrive/cv20252'\n",
                "REPO = '/content/Computer-Vision-20252'\n",
                "os.makedirs(WORK, exist_ok=True)\n",
                f"if not os.path.isdir(REPO):\n",
                f"    !git clone {REPO_URL} {{REPO}}\n",
                "%cd {REPO}\n",
                "!git pull -q\n",
                "!pip install -q 'gdown>=5.2,<6.0' 'huggingface_hub[hf_xet]>=0.36' 'pyarrow>=14.0' 'wandb>=0.17'\n",
                "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))",
            ]),
            code([
                f"# 3) Cấu hình cho job NÀY (data + output RIÊNG theo dataset+backbone, không đụng notebook khác)\n",
                f"DS = '{ds}'\n",
                f"BB = '{bb}'\n",
                f"BBTAG = '{bbtag}'\n",
                "DATA_ROOT = f'/content/data_{DS}'              # DATA ở ổ LOCAL /content (nhanh, ổn định). KHÔNG để trên Drive!\n",
                "SAVE_DIR  = f'{WORK}/outputs/{DS}_{BBTAG}'    # output trên Drive, RIÊNG theo backbone\n",
                "import time; LOG = f'{WORK}/logs/{DS}_{BBTAG}_' + time.strftime('%Y%m%d_%H%M%S') + '.log'\n",
                "os.makedirs(f'{WORK}/logs', exist_ok=True)\n",
                "free_mb = torch.cuda.mem_get_info()[0] // 1024**2\n",
                "BATCH_SIZE = 128 if free_mb >= 24000 else 64 if free_mb >= 20000 else 48 if free_mb >= 15000 else 32\n",
                "print(f'DS={DS} backbone={BB} batch_size={BATCH_SIZE} log={LOG}')",
            ]),
            code([
                "# 4) Tải dataset NÀY vào ổ LOCAL + kiểm tra layout.\n",
                "#    Để data trên /content (không phải Drive): tránh lỗi giải nén 15k+ file qua Drive FUSE,\n",
                "#    và train đọc ảnh nhanh hơn NHIỀU. Mỗi session tải lại vài phút là đáng.\n",
                "if not os.path.isdir(f'{DATA_ROOT}/{DS}'):\n",
                "    !python tools/download_data.py --data_root {DATA_ROOT} --datasets {DS}\n",
                "!python tools/check_data.py --data_root {DATA_ROOT} --dataset {DS}",
            ]),
            code([
                "# 4b) Điền W&B API key của bạn vào đây (dùng cá nhân; ĐỪNG up file đã điền lên GitHub public).\n",
                "WANDB_API_KEY = ''   # <-- DÁN KEY VÀO GIỮA 2 DẤU NHÁY\n",
                "assert WANDB_API_KEY, 'Hãy dán WANDB_API_KEY vào ô trên rồi chạy lại cell.'\n",
                "os.environ['WANDB_API_KEY'] = WANDB_API_KEY\n",
                "WANDB_PROJECT = f'cv20252-{DS.replace(\"_\", \"-\")}'   # mỗi dataset 1 project (chung r18/r50)\n",
                "!wandb login --relogin $WANDB_API_KEY\n",
                "print('W&B project:', WANDB_PROJECT)",
            ]),
            code([
                "# 5) TRAIN đủ 12 method + log W&B, log ĐẦY ĐỦ ra cả màn hình lẫn file trên Drive (tee).\n",
                "#    --skip_existing: Run all lại là train tiếp phần còn dở.\n",
                "!python tools/run_experiments.py --dataset {DS} --data_root {DATA_ROOT} \\\n",
                "    --methods all --backbones {BB} --seeds 0 1 2 --epochs 30 \\\n",
                "    --holdout_fraction 0.2 --scheduler none --hparams_profile paper \\\n",
                "    --num_workers 2 --batch_size {BATCH_SIZE} --save_dir {SAVE_DIR} --skip_existing \\\n",
                "    --wandb --wandb_project {WANDB_PROJECT} --wandb_group {DS}-{BB}-paper30 \\\n",
                "    --wandb_tags {DS} {BB} paper30 full \\\n",
                "    2>&1 | tee -a {LOG}",
            ]),
            code([
                "# 6) Tổng hợp kết quả job này -> CSV trên Drive\n",
                "!python tools/summarize_results.py --metrics_dir {SAVE_DIR} --output_csv {WORK}/outputs/{DS}_{BBTAG}_results.csv\n",
                "import pandas as pd; pd.read_csv(f'{WORK}/outputs/{DS}_{BBTAG}_results.csv')",
            ]),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


output_dir = Path("notebooks")
output_dir.mkdir(exist_ok=True)

for ds, bb in JOBS:
    bbtag = "r18" if bb == "resnet18" else "r50"
    path = output_dir / f"colab_{ds}_{bbtag}.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(ds, bb), f, ensure_ascii=False, indent=1)
    print("wrote", path)
