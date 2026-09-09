"""Shared model-loading and single-image prediction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from tensorflow import keras


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "outputs" / "best_model.keras"
IMAGE_SIZE = (96, 96)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> keras.Model:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return keras.models.load_model(path, compile=False)


def prepare_image(source: str | Path | BinaryIO | Image.Image) -> np.ndarray:
    if isinstance(source, Image.Image):
        prepared = source.convert("RGB").resize(IMAGE_SIZE)
        return np.expand_dims(np.asarray(prepared, dtype=np.uint8), axis=0)

    with Image.open(source) as image:
        prepared = image.convert("RGB").resize(IMAGE_SIZE)
        return np.expand_dims(np.asarray(prepared, dtype=np.uint8), axis=0)


def predict_image(
    model: keras.Model,
    source: str | Path | BinaryIO | Image.Image,
    threshold: float = 0.5,
) -> dict[str, float | str]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    probability_fake = float(model.predict(prepare_image(source), verbose=0).reshape(-1)[0])
    probability_real = 1.0 - probability_fake
    label = "fake" if probability_fake >= threshold else "real"
    confidence = probability_fake if label == "fake" else probability_real
    return {
        "label": label,
        "confidence": confidence,
        "probability_fake": probability_fake,
        "probability_real": probability_real,
    }
