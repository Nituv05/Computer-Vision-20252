"""
PACS dataset loader.
Download: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
Expected structure:
  <root>/pacs/
    art_painting/<class>/*.jpg
    cartoon/<class>/*.jpg
    photo/<class>/*.jpg
    sketch/<class>/*.jpg
"""
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.transforms import get_train_transform, get_test_transform

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


class PACSDataset(Dataset):
    def __init__(self, root: str, domain: str, split: str = "train"):
        self.root = Path(root) / "pacs" / domain
        self.transform = get_train_transform() if split == "train" else get_test_transform()
        self.samples = []
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), cls_idx))
                for img_path in cls_dir.glob("*.png"):
                    self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_pacs_loaders(root: str, test_domain: str, batch_size: int = 128,
                     num_workers: int = 4):
    train_domains = [d for d in DOMAINS if d != test_domain]
    train_datasets = [PACSDataset(root, d, "train") for d in train_domains]

    from torch.utils.data import ConcatDataset
    train_set = ConcatDataset(train_datasets)
    test_set = PACSDataset(root, test_domain, "test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
