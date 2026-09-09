"""Create deterministic train/validation/test folders from raw images."""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "raw"
SPLIT_DIR = PROJECT_DIR / "split"
CLASSES = ("real", "fake")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Train validation and test ratios must all be greater than zero")

    source_files: dict[str, list[Path]] = {}
    for class_name in CLASSES:
        class_dir = RAW_DIR / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {class_dir}")
        files = image_files(class_dir)
        if len(files) < 3:
            raise ValueError(f"Expected at least 3 images in {class_dir} but found {len(files)}")
        source_files[class_name] = files

    rng = random.Random(args.seed)
    counts: dict[str, dict[str, int]] = {}

    with tempfile.TemporaryDirectory(prefix="split_build_", dir=PROJECT_DIR) as temp_root:
        staged_split = Path(temp_root) / "split"
        for class_name in CLASSES:
            files = source_files[class_name].copy()
            rng.shuffle(files)
            n_train = int(len(files) * args.train_ratio)
            n_val = int(len(files) * args.val_ratio)
            groups = {
                "train": files[:n_train],
                "val": files[n_train : n_train + n_val],
                "test": files[n_train + n_val :],
            }
            counts[class_name] = {name: len(items) for name, items in groups.items()}

            for split_name, items in groups.items():
                destination = staged_split / split_name / class_name
                destination.mkdir(parents=True, exist_ok=True)
                for source in items:
                    shutil.copy2(source, destination / source.name)

        if SPLIT_DIR.exists():
            shutil.rmtree(SPLIT_DIR)
        shutil.move(str(staged_split), str(SPLIT_DIR))

    print("Split counts:")
    for class_name in CLASSES:
        print(f"  {class_name}: {counts[class_name]}")
    for split_name in ("train", "val", "test"):
        total = sum(counts[class_name][split_name] for class_name in CLASSES)
        print(f"  total {split_name}: {total}")


if __name__ == "__main__":
    main()
