"""Train the custom CNN in resumable steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
MAX_EPOCHS = 25
PROJECT_DIR = Path(__file__).resolve().parent.parent
ARRAY_DIR = PROJECT_DIR / "arrays"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.keras"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.keras"
HISTORY_PATH = OUTPUT_DIR / "history.json"
STATE_PATH = OUTPUT_DIR / "state.json"

tf.random.set_seed(SEED)
np.random.seed(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epochs", nargs="?", type=int, default=4)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard training state and train a new model from epoch 1",
    )
    return parser.parse_args()


def build_model(input_shape: tuple[int, int, int] = (96, 96, 3)) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="deepfake_custom_cnn")


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = ("X_train", "y_train", "X_val", "y_val")
    paths = [ARRAY_DIR / f"{name}.npy" for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing training arrays:\n  " + "\n  ".join(missing) + "\nRun preprocess_arrays.py first"
        )
    return tuple(np.load(path) for path in paths)  # type: ignore[return-value]


def reset_training_files() -> None:
    for path in (CHECKPOINT_PATH, BEST_MODEL_PATH, HISTORY_PATH, STATE_PATH):
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    OUTPUT_DIR.mkdir(exist_ok=True)

    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.is_file() else None
    if state and state.get("epochs_done", 0) >= MAX_EPOCHS and not args.restart:
        print("The packaged model has already reached 25 epochs")
        print("Use --restart only when you intentionally want to retrain it from scratch")
        return

    x_train, y_train, x_val, y_val = load_arrays()
    if args.restart:
        reset_training_files()
        print("Previous training state removed")

    n_real = int((y_train == 0).sum())
    n_fake = int((y_train == 1).sum())
    if n_real == 0 or n_fake == 0:
        raise ValueError("Training arrays must contain both real and fake images")
    total = n_real + n_fake
    class_weight = {0: total / (2.0 * n_real), 1: total / (2.0 * n_fake)}

    if CHECKPOINT_PATH.is_file():
        if not state:
            raise RuntimeError("checkpoint.keras exists but state.json is missing")
        print(f"Resuming from {CHECKPOINT_PATH}")
        model = keras.models.load_model(CHECKPOINT_PATH)
        start_epoch = int(state["epochs_done"])
        best_val_auc = float(state["best_val_auc"])
        epochs_since_improve = int(state["epochs_since_improve"])
    else:
        if not args.restart and (state or BEST_MODEL_PATH.is_file()):
            raise RuntimeError(
                "Saved results exist without a resumable checkpoint. "
                "Use --restart to intentionally train a new model"
            )
        print("Building new model")
        model = build_model((*IMAGE_SIZE, 3))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        start_epoch = 0
        best_val_auc = -1.0
        epochs_since_improve = 0
        model.summary()
        with (OUTPUT_DIR / "model_summary.txt").open("w", encoding="utf-8") as output:
            model.summary(print_fn=lambda line: output.write(line + "\n"))

    remaining = MAX_EPOCHS - start_epoch
    epochs_to_run = min(args.epochs, remaining)
    if epochs_to_run <= 0:
        print("Already reached the maximum number of epochs")
        return

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            initial_value_threshold=best_val_auc,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, verbose=1
        ),
    ]
    final_epoch = start_epoch + epochs_to_run
    print(f"Training epochs {start_epoch + 1} to {final_epoch} (maximum {MAX_EPOCHS})")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        initial_epoch=start_epoch,
        epochs=final_epoch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    previous_history = (
        json.loads(HISTORY_PATH.read_text(encoding="utf-8")) if HISTORY_PATH.is_file() else {}
    )
    for name, values in history.history.items():
        previous_history.setdefault(name, [])
        previous_history[name].extend(float(value) for value in values)
    HISTORY_PATH.write_text(json.dumps(previous_history, indent=2), encoding="utf-8")

    validation_aucs = history.history.get("val_auc", [])
    for value in validation_aucs:
        if value > best_val_auc + 1e-4:
            best_val_auc = float(value)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

    epochs_completed = start_epoch + len(history.epoch)
    STATE_PATH.write_text(
        json.dumps(
            {
                "epochs_done": epochs_completed,
                "best_val_auc": best_val_auc,
                "epochs_since_improve": epochs_since_improve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model.save(CHECKPOINT_PATH)
    print(
        f"Checkpoint saved: epochs_done={epochs_completed} "
        f"best_val_auc={best_val_auc:.4f}"
    )


if __name__ == "__main__":
    main()
