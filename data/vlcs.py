"""
VLCS dataset loader.
Download: https://github.com/belaalb/G2DM (follow VLCS instructions)
Expected structure:
  <root>/vlcs/ or <root>/VLCS/
    CALTECH/full/<class_id>/*.jpg
    LABELME/full/<class_id>/*.jpg
    PASCAL/full/<class_id>/*.jpg
    SUN/full/<class_id>/*.jpg
"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.transforms import get_train_transform, get_test_transform

DOMAINS = ["CALTECH", "LABELME", "PASCAL", "SUN"]
DOMAIN_ALIASES = {
    "caltech": "CALTECH",
    "caltech101": "CALTECH",
    "labelme": "LABELME",
    "pascal": "PASCAL",
    "voc2007": "PASCAL",
    "sun": "SUN",
    "sun09": "SUN",
}
CLASSES = ["0", "1", "2", "3", "4"]


def canonical_domain(domain: str) -> str:
    return DOMAIN_ALIASES.get(domain.replace("_", "").lower(), domain)


def resolve_dataset_root(root: str) -> Path:
    base = Path(root)
    for name in ("vlcs", "VLCS"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / "vlcs"


def resolve_domain_root(root: str, domain: str) -> Path:
    dataset_root = resolve_dataset_root(root)
    domain_root = dataset_root / canonical_domain(domain)
    full_root = domain_root / "full"
    return full_root if full_root.exists() else domain_root


class VLCSDataset(Dataset):
    def __init__(self, root: str, domain: str, split: str = "train"):
        self.root = resolve_domain_root(root, domain)
        self.transform = get_train_transform() if split == "train" else get_test_transform()
        self.samples = []
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.iterdir():
                    if img_path.suffix.lower() in (".jpg", ".png", ".jpeg"):
                        self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_vlcs_loaders(root: str, test_domain: str, batch_size: int = 128,
                     num_workers: int = 4):
    train_domains = [d for d in DOMAINS if d != test_domain]
    train_set = ConcatDataset([VLCSDataset(root, d, "train") for d in train_domains])
    test_set = VLCSDataset(root, test_domain, "test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
