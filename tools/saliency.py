"""
Generate simple gradient saliency visualizations for a saved checkpoint.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _bootstrap  # noqa: F401
from algorithms import METHODS, build_algorithm
from tools.train import build_raw_loaders, checkpoint_path, load_config
from utils.transforms import IMAGENET_MEAN, IMAGENET_STD


def unnormalize(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return image


def saliency_overlay(image: np.ndarray, saliency: np.ndarray) -> Image.Image:
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    heat = np.zeros_like(image)
    heat[..., 0] = (saliency * 255).astype(np.uint8)
    overlay = (0.60 * image + 0.40 * heat).clip(0, 255).astype(np.uint8)

    canvas = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
    canvas[:, :image.shape[1]] = image
    canvas[:, image.shape[1]:] = overlay
    return Image.fromarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["pacs", "vlcs", "office_home"])
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--checkpoint_dir", default="./outputs/checkpoints")
    parser.add_argument("--method", choices=METHODS, default="m2cl")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"],
                        default="resnet18")
    parser.add_argument("--test_domain", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--output_dir", default="./outputs/saliency")
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    if args.test_domain is None:
        raise ValueError("--test_domain is required for saliency")
    split_name = args.test_domain

    _, test_set, _ = build_raw_loaders(
        args.dataset,
        args.data_root,
        args.test_domain,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
    )

    ckpt_path = checkpoint_path(
        args.checkpoint_dir,
        args.dataset,
        split_name,
        args.method,
        args.backbone,
        args.seed,
        args.tag,
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    algorithm_args = ckpt.get("model_args", {})
    algorithm_args.setdefault("num_classes", cfg["num_classes"])
    algorithm_args.setdefault("method", args.method)
    algorithm_args.setdefault("backbone", args.backbone)
    algorithm_args.setdefault("architecture_tag", args.tag)
    algorithm_args["pretrained"] = False
    algorithm = build_algorithm(
        **algorithm_args,
    ).to(device)
    algorithm.load_state_dict(ckpt["state_dict"])
    algorithm.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(min(args.max_images, len(test_set))):
        image, label = test_set[index]
        input_tensor = image.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        algorithm.zero_grad(set_to_none=True)
        logits = algorithm.predict(input_tensor)
        pred = int(logits.argmax(1).item())
        logits[0, pred].backward()

        grad = input_tensor.grad.detach().abs()[0].max(dim=0).values.cpu().numpy()
        original = unnormalize(image)
        result = saliency_overlay(original, grad)
        result.save(output_dir / f"{args.dataset}_{split_name}_{index:03d}.png")

    print(f"Saved saliency images to {output_dir}")


if __name__ == "__main__":
    main()
