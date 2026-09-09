"""Predict whether one image is real or fake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_utils import DEFAULT_MODEL_PATH, load_model, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to a JPG PNG BMP or WebP image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    result = predict_image(load_model(args.model), args.image, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
