"""
NICO loader for leave-multiple-contexts-out evaluation.

Expected structure:
  <root>/nico/
    <class>/<context>/*.jpg
"""
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils.transforms import get_train_transform, get_test_transform


CLASSES_19 = [
    "bear", "bird", "cat", "cow", "dog", "elephant", "horse",
    "monkey", "rat", "sheep", "airplane", "bicycle", "boat",
    "bus", "car", "helicopter", "motorcycle", "train", "truck",
]


def _image_files(directory: Path):
    for path in directory.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            yield path


def get_class_names(root: str) -> list[str]:
    nico_root = Path(root) / "nico"
    existing = {path.name for path in nico_root.iterdir() if path.is_dir()}
    ordered = [name for name in CLASSES_19 if name in existing]
    extras = sorted(existing - set(ordered))
    return ordered + extras


def list_contexts_by_class(root: str) -> dict[str, list[str]]:
    nico_root = Path(root) / "nico"
    contexts = {}
    for cls_name in get_class_names(root):
        cls_dir = nico_root / cls_name
        contexts[cls_name] = sorted(
            path.name for path in cls_dir.iterdir() if path.is_dir()
        )
    return contexts


def build_context_split(root: str, n_heldout: int,
                        seed: int = 0) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Randomly hold out n contexts per class, matching the protocol described
    in the paper for NICO N=3,5,7.
    """
    rng = random.Random(seed)
    train_contexts = {}
    test_contexts = {}
    for cls_name, contexts in list_contexts_by_class(root).items():
        if len(contexts) <= n_heldout:
            raise ValueError(
                f"NICO class '{cls_name}' has {len(contexts)} contexts, "
                f"cannot hold out {n_heldout}."
            )
        heldout = sorted(rng.sample(contexts, n_heldout))
        train_contexts[cls_name] = [ctx for ctx in contexts if ctx not in heldout]
        test_contexts[cls_name] = heldout
    return train_contexts, test_contexts


class NICODataset(Dataset):
    def __init__(self, root: str, contexts_by_class: dict[str, list[str]],
                 split: str = "train"):
        nico_root = Path(root) / "nico"
        self.transform = get_train_transform() if split == "train" else get_test_transform()
        self.classes = get_class_names(root)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []

        for cls_name in self.classes:
            for context in contexts_by_class.get(cls_name, []):
                ctx_dir = nico_root / cls_name / context
                if not ctx_dir.exists():
                    continue
                for img_path in _image_files(ctx_dir):
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_nico_loaders(root: str, n_heldout: int = 7, batch_size: int = 128,
                     num_workers: int = 4, seed: int = 0):
    train_contexts, test_contexts = build_context_split(root, n_heldout, seed)
    train_set = NICODataset(root, train_contexts, "train")
    test_set = NICODataset(root, test_contexts, "test")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
