"""
Validate dataset folders before launching long training runs.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTS)


def status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def check_class_domain(root: Path, dataset_name: str, domains: list[str],
                       classes: list[str] | None = None):
    print(f"\n[{dataset_name}] root={root}")
    if not root.exists():
        print("  MISSING dataset root")
        return False

    all_ok = True
    for domain in domains:
        domain_root = root / domain
        if not domain_root.exists():
            print(f"  {domain:20s} {status(False)}")
            all_ok = False
            continue

        if classes is None:
            class_dirs = sorted(path for path in domain_root.iterdir() if path.is_dir())
        else:
            class_dirs = [domain_root / cls for cls in classes]

        missing_classes = [path.name for path in class_dirs if not path.exists()]
        image_count = count_images(domain_root)
        ok = image_count > 0 and not missing_classes
        all_ok = all_ok and ok
        suffix = ""
        if missing_classes:
            suffix = f" missing_classes={len(missing_classes)}"
        print(
            f"  {domain:20s} {status(ok):8s} "
            f"classes={len(class_dirs) - len(missing_classes):3d} "
            f"images={image_count:6d}{suffix}"
        )
    return all_ok


def check_pacs(data_root: Path):
    from data.pacs import CLASSES, DOMAINS
    return check_class_domain(data_root / "pacs", "PACS", DOMAINS, CLASSES)


def check_vlcs(data_root: Path):
    from data.vlcs import CLASSES, DOMAINS, resolve_dataset_root, resolve_domain_root

    root = resolve_dataset_root(str(data_root))
    print(f"\n[VLCS] root={root}")
    if not root.exists():
        print("  MISSING dataset root")
        return False

    all_ok = True
    for domain in DOMAINS:
        domain_root = resolve_domain_root(str(data_root), domain)
        missing_classes = [
            cls_name for cls_name in CLASSES
            if not (domain_root / cls_name).exists()
        ]
        image_count = count_images(domain_root)
        ok = image_count > 0 and not missing_classes
        all_ok = all_ok and ok
        suffix = ""
        if missing_classes:
            suffix = f" missing_classes={len(missing_classes)}"
        print(
            f"  {domain:20s} {status(ok):8s} "
            f"classes={len(CLASSES) - len(missing_classes):3d} "
            f"images={image_count:6d}{suffix}"
        )
    return all_ok


def check_office_home(data_root: Path):
    from data.office_home import DOMAINS, resolve_domain_root
    print(f"\n[Office-Home] root={data_root / 'office_home'}")
    all_ok = True
    for domain in DOMAINS:
        domain_root = resolve_domain_root(str(data_root), domain)
        exists = domain_root.exists()
        class_count = (
            len([path for path in domain_root.iterdir() if path.is_dir()])
            if exists else 0
        )
        image_count = count_images(domain_root)
        ok = exists and class_count > 0 and image_count > 0
        all_ok = all_ok and ok
        print(
            f"  {domain:20s} {status(ok):8s} "
            f"classes={class_count:3d} images={image_count:6d} "
            f"path={domain_root.name}"
        )
    return all_ok


def check_nico(data_root: Path):
    nico_root = data_root / "nico"
    print(f"\n[NICO] root={nico_root}")
    if not nico_root.exists():
        print("  MISSING dataset root")
        return False

    class_dirs = sorted(path for path in nico_root.iterdir() if path.is_dir())
    all_ok = bool(class_dirs)
    for cls_dir in class_dirs:
        contexts = sorted(path for path in cls_dir.iterdir() if path.is_dir())
        image_count = count_images(cls_dir)
        ok = len(contexts) > 7 and image_count > 0
        all_ok = all_ok and ok
        print(
            f"  {cls_dir.name:20s} {status(ok):8s} "
            f"contexts={len(contexts):3d} images={image_count:6d}"
        )
    return all_ok


CHECKERS = {
    "pacs": check_pacs,
    "vlcs": check_vlcs,
    "office_home": check_office_home,
    "nico": check_nico,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", choices=["all", *CHECKERS.keys()],
                        default="all")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    datasets = CHECKERS.keys() if args.dataset == "all" else [args.dataset]
    ok = True
    for dataset in datasets:
        ok = CHECKERS[dataset](data_root) and ok

    if not ok:
        raise SystemExit("\nDataset check failed. Fix folder layout before training.")
    print("\nDataset check passed.")


if __name__ == "__main__":
    main()
