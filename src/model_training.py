"""Stage 4 — train a CNN (TensorFlow) or MLP (TensorFlow / scikit-learn)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.utils import ROOT, load_params, save_json, set_seed


def _artifact_dir() -> Path:
    artifact_dir = ROOT / "models" / "classifier"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _has_tensorflow() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def _tf_optimizer(name: str, learning_rate: float):
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop

    mapping = {
        "adam": Adam(learning_rate=learning_rate),
        "sgd": SGD(learning_rate=learning_rate, momentum=0.9),
        "rmsprop": RMSprop(learning_rate=learning_rate),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported optimizer '{name}'. Use adam, sgd, or rmsprop.")
    return mapping[name]


def build_cnn(image_shape: tuple[int, ...], cfg: dict, num_classes: int):
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D

    l2 = regularizers.l2(float(cfg["l2_regularization"]))
    dropout = float(cfg["dropout_rate"])
    model = Sequential([Input(shape=image_shape)])
    for units in cfg["conv_filters"]:
        model.add(Conv2D(int(units), 3, activation="relu", padding="same", kernel_regularizer=l2))
        model.add(MaxPooling2D())
        model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(int(cfg["dense_units"]), activation="relu", kernel_regularizer=l2))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def build_mlp(n_features: int, cfg: dict, num_classes: int):
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import Dense, Dropout, Input

    l2 = regularizers.l2(float(cfg["l2_regularization"]))
    dropout = float(cfg["dropout_rate"])
    units = int(cfg["dense_units"])
    return Sequential(
        [
            Input(shape=(n_features,)),
            Dense(units, activation="relu", kernel_regularizer=l2),
            Dropout(dropout),
            Dense(max(units // 2, 16), activation="relu", kernel_regularizer=l2),
            Dropout(dropout),
            Dense(num_classes, activation="softmax"),
        ]
    )


def _train_tensorflow(cfg: dict, architecture: str, train, val, num_classes: int) -> dict:
    from tensorflow.keras.callbacks import EarlyStopping

    if architecture == "mlp":
        x_train, x_val = train["features"], val["features"]
        model = build_mlp(x_train.shape[1], cfg, num_classes)
    else:
        x_train, x_val = train["images"], val["images"]
        model = build_cnn(x_train.shape[1:], cfg, num_classes)

    y_train, y_val = train["labels"], val["labels"]
    model.compile(
        optimizer=_tf_optimizer(str(cfg["optimizer"]).lower(), float(cfg["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        callbacks=[
            EarlyStopping(
                monitor="val_accuracy",
                patience=int(cfg["early_stopping_patience"]),
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    artifact_dir = _artifact_dir()
    model_path = artifact_dir / "model.keras"
    model.save(model_path)
    payload = {key: [float(v) for v in values] for key, values in history.history.items()}
    payload.update(
        {
            "architecture": architecture,
            "backend": "tensorflow",
            "epochs_ran": len(history.history["loss"]),
            "model_path": "classifier/model.keras",
        }
    )
    save_json({"backend": "tensorflow", "model_file": "model.keras"}, artifact_dir / "format.json")
    return payload


def _train_sklearn(cfg: dict, train, val) -> dict:
    x_train = train["features"]
    y_train = train["labels"]
    x_val = val["features"]
    y_val = val["labels"]

    solver = "adam" if str(cfg["optimizer"]).lower() in {"adam", "rmsprop"} else "sgd"
    model = MLPClassifier(
        hidden_layer_sizes=(int(cfg["dense_units"]), max(int(cfg["dense_units"]) // 2, 16)),
        activation="relu",
        solver=solver,
        learning_rate_init=float(cfg["learning_rate"]),
        batch_size=int(cfg["batch_size"]),
        max_iter=int(cfg["epochs"]) * 20,
        early_stopping=True,
        n_iter_no_change=int(cfg["early_stopping_patience"]),
        random_state=42,
        verbose=False,
    )
    model.fit(x_train, y_train)
    artifact_dir = _artifact_dir()
    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path)
    save_json({"backend": "sklearn", "model_file": "model.joblib"}, artifact_dir / "format.json")
    train_acc = float(model.score(x_train, y_train))
    val_acc = float(model.score(x_val, y_val))
    print("TensorFlow is not installed; trained an sklearn MLP on selected features.")
    print(f"train_accuracy={train_acc:.4f} val_accuracy={val_acc:.4f}")
    return {
        "architecture": "mlp",
        "backend": "sklearn",
        "accuracy": [train_acc],
        "val_accuracy": [val_acc],
        "epochs_ran": int(getattr(model, "n_iter_", 0)),
        "model_path": "classifier/model.joblib",
        "optimizer": solver,
        "dropout_rate": float(cfg["dropout_rate"]),
    }


def main() -> None:
    params = load_params()
    set_seed(int(params["base"]["random_seed"]))
    cfg = params["model_training"]
    num_classes = int(params["base"]["num_classes"])
    feature_dir = ROOT / "data" / "features"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(feature_dir / "train.npz")
    val = np.load(feature_dir / "val.npz")
    architecture = str(cfg["architecture"]).lower()

    if _has_tensorflow():
        history = _train_tensorflow(cfg, architecture, train, val, num_classes)
    else:
        history = _train_sklearn(cfg, train, val)

    save_json(history, models_dir / "training_history.json")
    print(f"Saved training artifacts to {models_dir}")


if __name__ == "__main__":
    main()
