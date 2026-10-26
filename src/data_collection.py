"""Stage 1 — collect raw images and write them to disk."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image

from src.utils import (
    ROOT,
    class_names_cifar10,
    ensure_dir,
    load_params,
    save_json,
    set_seed,
)


def _save_split(images: np.ndarray, labels: np.ndarray, split_dir: Path, names: list[str]) -> int:
    count = 0
    for idx, (image, label) in enumerate(zip(images, labels)):
        class_name = names[int(label)]
        class_dir = ensure_dir(split_dir / class_name)
        out_path = class_dir / f"{class_name}_{idx:05d}.png"
        Image.fromarray(image.astype(np.uint8)).save(out_path)
        count += 1
    return count


def _write_manifest(raw_dir: Path, names: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        for class_name in names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.glob("*.png")):
                rows.append(
                    {
                        "filepath": str(image_path.relative_to(raw_dir)),
                        "split": split,
                        "label": class_name,
                    }
                )

    with (raw_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "split", "label"])
        writer.writeheader()
        writer.writerows(rows)


def collect_cifar10(max_samples: int, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from tensorflow.keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    train_n = min(max_samples, len(x_train))
    test_n = min(max(max_samples // 5, 200), len(x_test))

    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(x_train), size=train_n, replace=False)
    test_idx = rng.choice(len(x_test), size=test_n, replace=False)
    return x_train[train_idx], y_train[train_idx], x_test[test_idx], y_test[test_idx]


def collect_synthetic(max_samples: int, names: list[str], image_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback dataset: colored shapes, one class per CIFAR-10 label name."""
    rng = np.random.default_rng(42)
    colors = [
        (200, 40, 40),
        (40, 40, 200),
        (40, 180, 40),
        (220, 180, 40),
        (160, 80, 200),
        (40, 180, 180),
        (220, 100, 40),
        (120, 80, 40),
        (80, 80, 80),
        (240, 240, 240),
    ]

    def make_image(label: int) -> np.ndarray:
        canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        canvas[:] = rng.integers(10, 40, size=3)
        color = np.array(colors[label], dtype=np.int32)
        color = np.clip(color + rng.integers(-20, 21, size=3), 0, 255).astype(np.uint8)
        if label % 2 == 0:
            canvas[8:24, 8:24] = color
        else:
            yy, xx = np.ogrid[:image_size, :image_size]
            mask = (yy - 16) ** 2 + (xx - 16) ** 2 <= 10**2
            canvas[mask] = color
        noise = rng.integers(0, 18, size=canvas.shape, dtype=np.uint8)
        return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    train_n = max_samples
    test_n = max(max_samples // 5, 200)
    y_train = rng.integers(0, len(names), size=train_n)
    y_test = rng.integers(0, len(names), size=test_n)
    x_train = np.stack([make_image(int(lbl)) for lbl in y_train])
    x_test = np.stack([make_image(int(lbl)) for lbl in y_test])
    return x_train, y_train, x_test, y_test


def main() -> None:
    params = load_params()
    seed = int(params["base"]["random_seed"])
    set_seed(seed)

    cfg = params["data_collection"]
    names = class_names_cifar10()
    raw_dir = ensure_dir(ROOT / cfg["output_dir"])
    dataset_name = str(cfg["dataset_name"]).lower()
    max_samples = int(cfg["max_samples"])

    try:
        if dataset_name == "synthetic":
            raise RuntimeError("synthetic dataset requested")
        x_train, y_train, x_test, y_test = collect_cifar10(max_samples, names)
        source = "cifar10"
    except Exception as exc:
        print(f"CIFAR-10 unavailable ({exc}). Falling back to a synthetic dataset.")
        x_train, y_train, x_test, y_test = collect_synthetic(max_samples, names)
        source = "synthetic"

    train_count = _save_split(x_train, y_train, ensure_dir(raw_dir / "train"), names)
    test_count = _save_split(x_test, y_test, ensure_dir(raw_dir / "test"), names)
    _write_manifest(raw_dir, names)

    save_json(
        {
            "source": source,
            "train_images": train_count,
            "test_images": test_count,
            "classes": names,
        },
        raw_dir / "collection_meta.json",
    )
    print(f"Saved {train_count} train and {test_count} test images to {raw_dir}")


if __name__ == "__main__":
    main()
