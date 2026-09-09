"""Validate class counts dimensions color modes and image readability."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "raw"
CLASSES = ("real", "fake")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    total = 0
    errors: list[str] = []
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()

    for class_name in CLASSES:
        folder = RAW_DIR / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")
        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        print(f"{class_name}: {len(files)} images")
        total += len(files)
        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    sizes[image.size] += 1
                    modes[image.mode] += 1
            except Exception as error:  # reports the exact problem file
                errors.append(f"{path}: {error}")

    print(f"total: {total} images")
    print(f"dimensions: {dict(sizes)}")
    print(f"color modes: {dict(modes)}")
    if errors:
        print("Unreadable images:")
        print("\n".join(errors))
        raise SystemExit(1)
    print("Dataset validation passed")


if __name__ == "__main__":
    main()
