"""
Generate paper-style saliency comparison grids: input, baseline and method.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from algorithms import METHODS, build_algorithm
from train import build_raw_loaders, checkpoint_path, load_config
from utils.transforms import IMAGENET_MEAN, IMAGENET_STD


def unnormalize(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def saliency_map(image_tensor: torch.Tensor, algorithm, device) -> np.ndarray:
    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    algorithm.zero_grad(set_to_none=True)
    logits = algorithm.predict(input_tensor)
    pred = int(logits.argmax(1).item())
    logits[0, pred].backward()
    saliency = input_tensor.grad.detach().abs()[0].max(dim=0).values
    saliency = saliency.cpu().numpy()
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency


def saliency_to_image(saliency: np.ndarray) -> Image.Image:
    saliency = np.power(saliency, 0.65)
    base = np.full((*saliency.shape, 3), 235, dtype=np.uint8)
    base[..., 0] = (235 - saliency * 120).clip(0, 255).astype(np.uint8)
    base[..., 1] = (240 - saliency * 185).clip(0, 255).astype(np.uint8)
    base[..., 2] = (245 - saliency * 45).clip(0, 255).astype(np.uint8)
    return Image.fromarray(base)


def load_algorithm(args, cfg, method: str, split_name: str, device):
    ckpt_path = checkpoint_path(
        args.checkpoint_dir,
        args.dataset,
        split_name,
        method,
        args.backbone,
        args.seed,
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    algorithm_args = ckpt.get("model_args", {})
    algorithm_args.setdefault("num_classes", cfg["num_classes"])
    algorithm_args.setdefault("method", method)
    algorithm_args.setdefault("backbone", args.backbone)
    algorithm_args["pretrained"] = False
    algorithm = build_algorithm(**algorithm_args).to(device)
    algorithm.load_state_dict(ckpt["state_dict"])
    algorithm.eval()
    return algorithm


def draw_centered(draw, box, text, font):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def make_grid(samples, baseline_model, method_model, args, device):
    font = ImageFont.load_default()
    cell = args.image_size
    header_h = 36
    gap = 16
    margin = 18
    rows = len(samples)
    cols = 3
    width = margin * 2 + cols * cell + (cols - 1) * gap
    height = margin * 2 + header_h + rows * cell + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    headers = ["Input Image", args.baseline_label, args.method_label]
    for col, header in enumerate(headers):
        x = margin + col * (cell + gap)
        draw_centered(draw, (x, margin, x + cell, margin + header_h), header, font)

    y0 = margin + header_h
    for row, (image_tensor, _) in enumerate(samples):
        original = unnormalize(image_tensor).resize((cell, cell))
        baseline = saliency_to_image(
            saliency_map(image_tensor, baseline_model, device)
        ).resize((cell, cell))
        method = saliency_to_image(
            saliency_map(image_tensor, method_model, device)
        ).resize((cell, cell))
        for col, image in enumerate([original, baseline, method]):
            x = margin + col * (cell + gap)
            y = y0 + row * (cell + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=(150, 150, 150))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["pacs", "vlcs", "office_home", "nico"])
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--checkpoint_dir", default="./outputs/checkpoints")
    parser.add_argument("--baseline_method", choices=METHODS, default="erm")
    parser.add_argument("--method", choices=METHODS, default="m2cl")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"],
                        default="resnet18")
    parser.add_argument("--test_domain", default=None)
    parser.add_argument("--n_heldout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=180)
    parser.add_argument("--baseline_label", default="Baseline")
    parser.add_argument("--method_label", default="Our Method")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    if args.dataset == "nico":
        if args.n_heldout is None:
            raise ValueError("--n_heldout is required for NICO")
        split_name = f"N{args.n_heldout}"
    else:
        if args.test_domain is None:
            raise ValueError("--test_domain is required")
        split_name = args.test_domain

    _, test_set, _ = build_raw_loaders(
        args.dataset,
        args.data_root,
        args.test_domain,
        args.n_heldout,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
    )
    samples = [test_set[i] for i in range(min(args.max_images, len(test_set)))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = load_algorithm(
        args, cfg, args.baseline_method, split_name, device
    )
    method_model = load_algorithm(args, cfg, args.method, split_name, device)
    grid = make_grid(samples, baseline_model, method_model, args, device)

    output = args.output
    if output is None:
        output = (
            f"outputs/saliency_compare/{args.dataset}_{split_name}_"
            f"{args.baseline_method}_vs_{args.method}_{args.backbone}_seed{args.seed}.png"
        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved saliency comparison: {output_path}")


if __name__ == "__main__":
    main()
