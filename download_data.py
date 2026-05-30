"""
Download and normalize dataset folders used by this project.

PACS, VLCS and Office-Home are first tried from the official DomainBed
download links. Public mirrors are used as fallback when Google Drive blocks
scripted access.
"""
import argparse
import os
import re
import shutil
import tarfile
from html import unescape
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm


DOMAINBED_URLS = {
    "vlcs": {
        "url": "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8",
        "archive": "VLCS.tar.gz",
        "extracted": "VLCS",
        "target": "vlcs",
    },
    "pacs": {
        "url": "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
        "archive": "PACS.zip",
        "extracted": "kfold",
        "target": "pacs",
    },
    "office_home": {
        "url": "https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw",
        "archive": "office_home.zip",
        "extracted": "OfficeHomeDataset_10072016",
        "target": "office_home",
    },
}


HF_FALLBACKS = {
    "pacs": {
        "repo_id": "flwrlabs/pacs",
        "repo_type": "dataset",
        "filename": "data/train-00000-of-00001.parquet",
        "local_dir": "hf_pacs",
        "target": "pacs",
        "kind": "pacs_parquet",
    },
    "office_home": {
        "repo_id": "huangyuyang11/officehome",
        "repo_type": "model",
        "filename": "OfficeHomeDataset_10072016.zip",
        "local_dir": "hf_office_home",
        "target": "office_home",
        "kind": "archive",
        "extracted": "OfficeHomeDataset_10072016",
    },
}


MEDIAFIRE_FALLBACKS = {
    "vlcs": {
        "url": "https://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file",
        "archive": "VLCS.tar.gz",
        "extracted": "VLCS",
        "target": "vlcs",
    },
}


PACS_LABELS = [
    "dog", "elephant", "giraffe", "guitar", "horse", "house", "person",
]


CHECKABLE_DATASETS = ["pacs", "vlcs", "office_home", "nico"]


NICO_HINT = """
NICO is distributed by the official project site through Dropbox/Baidu.
Download it manually from:
  https://nico.thumedialab.com/

Then run:
  python download_data.py --data_root <root> --datasets nico --nico_archive <path-to-nico-archive>
"""


def remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def extract_archive(archive_path: Path, output_dir: Path):
    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_dir)
    elif suffixes.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tar:
            tar.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_hf_file(spec: dict, data_root: Path) -> Path:
    hf_home = data_root / ".hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    from huggingface_hub import hf_hub_download

    local_dir = data_root / spec["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            filename=spec["filename"],
            local_dir=str(local_dir),
        )
    )


def convert_pacs_parquet(parquet_path: Path, target: Path, force: bool):
    if target.exists():
        if not force:
            print(f"[SKIP] PACS converted folder exists: {target}")
            return
        remove_path(target)

    import pyarrow.parquet as pq

    print(f"[CONVERT] {parquet_path} -> {target}")
    table = pq.read_table(parquet_path)
    images = table["image"].to_pylist()
    domains = table["domain"].to_pylist()
    labels = table["label"].to_pylist()

    for idx, (image, domain, label) in enumerate(
        tqdm(zip(images, domains, labels), total=table.num_rows, desc="pacs")
    ):
        class_name = PACS_LABELS[int(label)]
        original_name = Path(image.get("path") or f"{idx:05d}.jpg").name
        if not Path(original_name).suffix:
            original_name = f"{original_name}.jpg"
        output_dir = target / domain / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{idx:05d}_{original_name}"
        output_path.write_bytes(image["bytes"])


def download_hf_fallback(name: str, data_root: Path, force: bool,
                         keep_archive: bool):
    spec = HF_FALLBACKS[name]
    print(f"[HF] {name}: {spec['repo_id']}/{spec['filename']}")
    downloaded = download_hf_file(spec, data_root)

    if spec["kind"] == "pacs_parquet":
        convert_pacs_parquet(downloaded, data_root / spec["target"], force)
        return

    archive_path = data_root / Path(spec["filename"]).name
    if downloaded != archive_path:
        shutil.copyfile(downloaded, archive_path)
    print(f"[EXTRACT] {archive_path}")
    extract_archive(archive_path, data_root)
    rename_extracted(data_root, spec["extracted"], spec["target"], force=True)
    if not keep_archive:
        archive_path.unlink(missing_ok=True)


def google_drive_id(url: str) -> str:
    match = re.search(r"id=([^&]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"/d/([^/]+)", url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse Google Drive file id from {url}")


def _confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    text = response.text
    match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
    if match:
        return match.group(1)
    return None


def download_google_drive(url: str, output: Path):
    file_id = google_drive_id(url)
    session = requests.Session()
    base_url = "https://docs.google.com/uc?export=download"
    response = session.get(base_url, params={"id": file_id}, stream=True)
    token = _confirm_token(response)
    if token:
        response.close()
        response = session.get(
            base_url, params={"id": file_id, "confirm": token}, stream=True
        )
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=output.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_direct(url: str, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(output, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=output.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def mediafire_direct_url(page_url: str) -> str:
    response = requests.get(page_url, timeout=30)
    response.raise_for_status()
    match = re.search(
        r'href="(https://download[^"]+)"\s+id="downloadButton"',
        response.text,
    )
    if not match:
        raise RuntimeError("Could not find MediaFire download button URL")
    return unescape(match.group(1))


def download_mediafire_fallback(name: str, data_root: Path, force: bool,
                                keep_archive: bool):
    spec = MEDIAFIRE_FALLBACKS[name]
    archive_path = data_root / spec["archive"]
    print(f"[MEDIAFIRE] {name}: {spec['url']}")
    direct_url = mediafire_direct_url(spec["url"])
    download_direct(direct_url, archive_path)
    print(f"[EXTRACT] {archive_path}")
    extract_archive(archive_path, data_root)
    rename_extracted(data_root, spec["extracted"], spec["target"], force=True)
    if not keep_archive:
        archive_path.unlink(missing_ok=True)


def rename_extracted(data_root: Path, extracted_name: str, target_name: str,
                     force: bool):
    src = data_root / extracted_name
    dst = data_root / target_name
    if not src.exists() and dst.exists():
        return
    if not src.exists():
        candidates = [
            path for path in data_root.iterdir()
            if path.is_dir() and path.name.lower() == extracted_name.lower()
        ]
        if candidates:
            src = candidates[0]
    if not src.exists():
        raise FileNotFoundError(f"Expected extracted folder not found: {src}")
    if src.exists() and dst.exists():
        src_norm = os.path.normcase(str(src.resolve()))
        dst_norm = os.path.normcase(str(dst.resolve()))
        if src_norm == dst_norm:
            return
    if dst.exists():
        if not force:
            print(f"[SKIP] {dst} exists. Use --force to replace it.")
            remove_path(src)
            return
        remove_path(dst)
    src.rename(dst)


def download_domainbed_dataset(name: str, data_root: Path, force: bool,
                               keep_archive: bool):
    spec = DOMAINBED_URLS[name]
    target = data_root / spec["target"]
    if target.exists() and not force:
        print(f"[SKIP] {name}: {target} exists")
        return
    if target.exists() and force:
        remove_path(target)

    archive_path = data_root / spec["archive"]
    print(f"[DOWNLOAD] {name} -> {archive_path}")
    try:
        gdown_home = data_root / ".gdown_home"
        gdown_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(gdown_home)
        os.environ["USERPROFILE"] = str(gdown_home)
        import gdown
        result = gdown.download(
            spec["url"], str(archive_path), quiet=False, fuzzy=True
        )
        if result is None:
            raise RuntimeError("gdown could not retrieve the public file")
    except ImportError:
        download_google_drive(spec["url"], archive_path)
    except Exception as exc:
        if name in MEDIAFIRE_FALLBACKS:
            print(f"[WARN] DomainBed download failed for {name}: {exc}")
            print(f"[WARN] Falling back to MediaFire mirror for {name}.")
            download_mediafire_fallback(name, data_root, force, keep_archive)
            print(f"[OK] {name}: {target}")
            return
        if name not in HF_FALLBACKS:
            raise
        print(f"[WARN] DomainBed download failed for {name}: {exc}")
        print(f"[WARN] Falling back to Hugging Face mirror for {name}.")
        download_hf_fallback(name, data_root, force, keep_archive)
        print(f"[OK] {name}: {target}")
        return
    print(f"[EXTRACT] {archive_path}")
    extract_archive(archive_path, data_root)
    rename_extracted(data_root, spec["extracted"], spec["target"], force=True)
    if not keep_archive:
        archive_path.unlink(missing_ok=True)
    print(f"[OK] {name}: {target}")


def normalize_nico(data_root: Path, archive_path: Path | None, force: bool,
                   keep_archive: bool):
    target = data_root / "nico"
    if target.exists() and not force:
        print(f"[SKIP] nico: {target} exists")
        return
    if archive_path is None:
        print(NICO_HINT.strip())
        return
    if target.exists() and force:
        remove_path(target)

    archive_path = archive_path.resolve()
    print(f"[EXTRACT] {archive_path}")
    extract_archive(archive_path, data_root)

    candidates = [
        "nico", "NICO", "NICO_DG", "NICO-dataset", "NICO_Dataset",
        "NICO-master",
    ]
    for candidate in candidates:
        src = data_root / candidate
        if src.exists():
            if src != target:
                src.rename(target)
            break
    else:
        print(
            "NICO archive extracted, but I could not infer the top folder. "
            "Rename the extracted class/context folder to 'nico'."
        )
        return

    if not keep_archive and archive_path.parent == data_root:
        archive_path.unlink(missing_ok=True)
    print(f"[OK] nico: {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--datasets", nargs="+",
                        choices=["pacs", "vlcs", "office_home", "nico", "all"],
                        default=["pacs", "vlcs", "office_home"])
    parser.add_argument("--nico_archive", default=None,
                        help="Path to a manually downloaded NICO archive")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep_archive", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["pacs", "vlcs", "office_home", "nico"]

    for dataset in datasets:
        if dataset in DOMAINBED_URLS:
            download_domainbed_dataset(
                dataset, data_root, args.force, args.keep_archive
            )
        elif dataset == "nico":
            normalize_nico(
                data_root,
                Path(args.nico_archive) if args.nico_archive else None,
                args.force,
                args.keep_archive,
            )

    print("\nRun this after downloads finish:")
    for dataset in datasets:
        print(f"  python check_data.py --data_root {data_root} --dataset {dataset}")
    if set(datasets) == set(CHECKABLE_DATASETS):
        print(f"  python check_data.py --data_root {data_root} --dataset all")


if __name__ == "__main__":
    main()
