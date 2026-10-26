"""Stage 5 — score the trained model and write assignment reports."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import ROOT, class_names_cifar10, ensure_dir, load_json, load_params, save_json, set_seed


def _predict(history: dict, x_images: np.ndarray, x_features: np.ndarray) -> np.ndarray:
    backend = str(history.get("backend", "tensorflow")).lower()
    architecture = str(history.get("architecture", "cnn")).lower()
    if backend == "sklearn":
        model = joblib.load(ROOT / "models" / history.get("model_path", "classifier/model.joblib"))
        return model.predict(x_features)

    from tensorflow.keras.models import load_model

    model = load_model(ROOT / "models" / history.get("model_path", "classifier/model.keras"))
    x_test = x_features if architecture == "mlp" else x_images
    return np.argmax(model.predict(x_test, verbose=0), axis=1)


def _save_confusion_plot(matrix: np.ndarray, names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    params = load_params()
    set_seed(int(params["base"]["random_seed"]))
    eval_cfg = params["model_evaluation"]
    names = class_names_cifar10()
    average = str(eval_cfg.get("average", "macro"))

    feature_dir = ROOT / "data" / "features"
    reports_dir = ensure_dir(ROOT / "reports")
    history = load_json(ROOT / "models" / "training_history.json")

    test = np.load(feature_dir / "test.npz")
    y_true = test["labels"]
    y_pred = _predict(history, test["images"], test["features"])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "architecture": history.get("architecture"),
        "backend": history.get("backend"),
        "test_samples": int(len(y_true)),
        "average": average,
    }
    save_json(metrics, reports_dir / "metrics.json")

    report_text = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    (reports_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    with (reports_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual", "predicted"])
        for actual_idx, predicted_idx in zip(y_true, y_pred):
            writer.writerow([names[int(actual_idx)], names[int(predicted_idx)]])

    if bool(eval_cfg.get("save_plots", True)):
        _save_confusion_plot(matrix, names, reports_dir / "confusion_matrix.png")

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Reports written to {reports_dir}")


if __name__ == "__main__":
    main()
