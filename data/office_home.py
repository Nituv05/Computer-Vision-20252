"""
Office-Home dataset loader.
Download: https://hemanthdv.github.io/officehome-dataset/
Expected structure:
  <root>/office_home/
    Art/<class>/*.jpg
    Clipart/<class>/*.jpg
    Product/<class>/*.jpg
    RealWorld/<class>/*.jpg
"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.transforms import get_train_transform, get_test_transform

DOMAINS = ["Art", "Clipart", "Product", "RealWorld"]


class OfficeHomeDataset(Dataset):
    def __init__(self, root: str, domain: str, split: str = "train"):
        domain_root = Path(root) / "office_home" / domain
        self.transform = get_train_transform() if split == "train" else get_test_transform()
        self.samples = []
        self.classes = sorted([d.name for d in domain_root.iterdir() if d.is_dir()])
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = domain_root / cls_name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".png", ".jpeg"):
                    self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_office_home_loaders(root: str, test_domain: str, batch_size: int = 128,
                             num_workers: int = 4):
    train_domains = [d for d in DOMAINS if d != test_domain]
    train_set = ConcatDataset([OfficeHomeDataset(root, d, "train") for d in train_domains])
    test_set = OfficeHomeDataset(root, test_domain, "test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
