"""
NICO dataset loader.
Download: https://nicochallenge.com/
Expected structure:
  <root>/nico/
    <class>/<context>/*.jpg
"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.transforms import get_train_transform, get_test_transform

CLASSES_19 = [
    "bear", "bird", "cat", "cow", "dog", "elephant", "horse",
    "monkey", "rat", "sheep", "airplane", "bicycle", "boat",
    "bus", "car", "helicopter", "motorcycle", "train", "truck",
]


class NICODataset(Dataset):
    def __init__(self, root: str, contexts: list, split: str = "train"):
        nico_root = Path(root) / "nico"
        self.transform = get_train_transform() if split == "train" else get_test_transform()
        self.samples = []
        cls_names = [d.name for d in nico_root.iterdir() if d.is_dir()]
        self.classes = sorted(cls_names)
        cls2idx = {c: i for i, c in enumerate(self.classes)}
        for cls_name in self.classes:
            for ctx in contexts:
                ctx_dir = nico_root / cls_name / ctx
                if ctx_dir.exists():
                    for img_path in ctx_dir.iterdir():
                        if img_path.suffix.lower() in (".jpg", ".png", ".jpeg"):
                            self.samples.append((str(img_path), cls2idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_nico_loaders(root: str, test_contexts: list, all_contexts: list,
                     batch_size: int = 128, num_workers: int = 4):
    train_contexts = [c for c in all_contexts if c not in test_contexts]
    train_set = NICODataset(root, train_contexts, "train")
    test_set = NICODataset(root, test_contexts, "test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
