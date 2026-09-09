"""Decode split images and save 96 x 96 NumPy arrays for the CNN."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "split"
ARRAY_DIR = PROJECT_DIR / "arrays"
CLASSES = ("real", "fake")
IMAGE_SIZE = (96, 96)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def class_images(split_name: str, class_name: str) -> list[Path]:
    folder = SPLIT_DIR / split_name / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing split folder: {folder}. Run split_data.py first")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label, class_name in enumerate(CLASSES):
        files = class_images(split_name, class_name)
        if not files:
            raise ValueError(f"No supported images found for {split_name}/{class_name}")
        for path in files:
            try:
                with Image.open(path) as image:
                    prepared = image.convert("RGB").resize(IMAGE_SIZE)
                    images.append(np.asarray(prepared, dtype=np.uint8))
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Could not read image: {path}") from error
            labels.append(label)

    x_data = np.stack(images, axis=0)
    y_data = np.asarray(labels, dtype=np.float32)
    indices = np.random.default_rng(42).permutation(len(y_data))
    return x_data[indices], y_data[indices]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="array_build_", dir=PROJECT_DIR) as temp_root:
        staged_arrays = Path(temp_root) / "arrays"
        staged_arrays.mkdir()

        for split_name in ("train", "val", "test"):
            x_data, y_data = load_split(split_name)
            np.save(staged_arrays / f"X_{split_name}.npy", x_data)
            np.save(staged_arrays / f"y_{split_name}.npy", y_data)
            print(
                split_name,
                x_data.shape,
                y_data.shape,
                "real:",
                int((y_data == 0).sum()),
                "fake:",
                int((y_data == 1).sum()),
            )

        if ARRAY_DIR.exists():
            shutil.rmtree(ARRAY_DIR)
        shutil.move(str(staged_arrays), str(ARRAY_DIR))

    print(f"Arrays saved to {ARRAY_DIR}")


if __name__ == "__main__":
    main()
