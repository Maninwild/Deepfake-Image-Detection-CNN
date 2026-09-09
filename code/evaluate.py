"""Evaluate the saved CNN on the held-out test arrays and recreate its plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tensorflow import keras


PROJECT_DIR = Path(__file__).resolve().parent.parent
ARRAY_DIR = PROJECT_DIR / "arrays"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.keras"
CLASS_NAMES = ("real", "fake")


def required_file(path: Path, instruction: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. {instruction}")
    return path


def main() -> None:
    required_file(ARRAY_DIR / "X_test.npy", "Run preprocess_arrays.py first")
    required_file(ARRAY_DIR / "y_test.npy", "Run preprocess_arrays.py first")
    required_file(MODEL_PATH, "Train the model or restore best_model.keras")
    OUTPUT_DIR.mkdir(exist_ok=True)

    x_test = np.load(ARRAY_DIR / "X_test.npy")
    y_test = np.load(ARRAY_DIR / "y_test.npy").astype(int)
    model = keras.models.load_model(MODEL_PATH, compile=False)

    probabilities = model.predict(x_test, batch_size=32, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", pos_label=1, zero_division=0
    )
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": matrix.tolist(),
        "n_test": int(len(y_test)),
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print("=== TEST METRICS ===")
    print(json.dumps(metrics, indent=2))
    print(report)

    history_path = OUTPUT_DIR / "history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for axis, metric, title in zip(
            axes, ("loss", "accuracy", "auc"), ("Loss", "Accuracy", "AUC")
        ):
            axis.plot(history[metric], label="train")
            axis.plot(history[f"val_{metric}"], label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.legend()
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], CLASS_NAMES)
    axis.set_yticks([0, 1], CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion Matrix (Test Set)")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontsize=13,
            )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve (Test Set)")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close(figure)
    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
