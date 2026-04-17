import hashlib
import random
import shutil
from pathlib import Path
from PIL import Image

RAW_DIR = Path("dataset_raw")
OUT_DIR = Path("dataset")

CLASSES = ["drunk", "normal"]
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)


def file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def ensure_dirs():
    for split in SPLIT_RATIOS:
        for cls in CLASSES:
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def collect_clean_files(class_name: str):
    src_dir = RAW_DIR / class_name
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing folder: {src_dir}")

    seen_hashes = set()
    clean_files = []

    for path in src_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTS:
            continue
        if not is_valid_image(path):
            continue

        h = file_hash(path)
        if h in seen_hashes:
            continue

        seen_hashes.add(h)
        clean_files.append(path)

    return clean_files


def split_files(files):
    random.shuffle(files)
    n = len(files)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }


def copy_split_files(class_name: str, split_map):
    for split, files in split_map.items():
        for i, src in enumerate(files, start=1):
            dst = OUT_DIR / split / class_name / f"{class_name}_{i}{src.suffix.lower()}"
            shutil.copy2(src, dst)


def main():
    ensure_dirs()

    for cls in CLASSES:
        files = collect_clean_files(cls)
        print(f"{cls}: {len(files)} clean unique images found")
        split_map = split_files(files)
        copy_split_files(cls, split_map)

        for split in split_map:
            print(f"  {split}: {len(split_map[split])}")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()