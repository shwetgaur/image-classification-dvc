"""Flask API for image classification (EC2 / Docker deployment)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from src.feature_selection import _transform
from src.utils import class_names_cifar10, load_json, load_params

app = Flask(__name__)

_selector = None
_model = None
_history = None
_params = None


def _load_artifacts() -> None:
    global _selector, _model, _history, _params
    _params = load_params()
    _history = load_json(ROOT / "models" / "training_history.json")
    _selector = joblib.load(ROOT / "data" / "features" / "selector.joblib")

    backend = str(_history.get("backend", "sklearn")).lower()
    model_path = ROOT / "models" / _history.get("model_path", "classifier/model.joblib")
    if backend == "sklearn":
        _model = joblib.load(model_path)
    else:
        from tensorflow.keras.models import load_model

        _model = load_model(model_path)


def _preprocess_image(file_bytes: bytes, image_size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    return arr.astype(np.float32) / 255.0


def predict_image(file_bytes: bytes) -> dict:
    if _model is None:
        _load_artifacts()

    image_size = int(_params["data_processing"]["image_size"])
    image = _preprocess_image(file_bytes, image_size)
    batch = np.expand_dims(image, axis=0)
    flat = batch.reshape(1, -1)
    features = _transform(_selector, flat)

    backend = str(_history.get("backend", "sklearn")).lower()
    architecture = str(_history.get("architecture", "mlp")).lower()
    names = class_names_cifar10()

    if backend == "sklearn" or architecture == "mlp":
        pred_idx = int(_model.predict(features)[0])
    else:
        pred_idx = int(np.argmax(_model.predict(batch, verbose=0), axis=1)[0])

    return {
        "predicted_class": names[pred_idx],
        "class_index": pred_idx,
        "backend": backend,
        "architecture": architecture,
    }


HOME_HTML = """
<!DOCTYPE html>
<html>
<head><title>Image Classification DVC</title></head>
<body style="font-family: Arial; max-width: 640px; margin: 2rem auto;">
  <h1>Image Classification (DVC + Docker + AWS)</h1>
  <p>Upload a 32x32-style object image (CIFAR-10 classes).</p>
  <form method="post" action="/predict" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required />
    <button type="submit">Classify</button>
  </form>
  <p><a href="/health">Health check JSON</a></p>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HOME_HTML)


@app.get("/health")
def health():
    try:
        _load_artifacts()
        ok = True
        msg = "Model and selector loaded"
    except Exception as exc:
        ok = False
        msg = str(exc)
    return jsonify({"status": "ok" if ok else "degraded", "message": msg})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        result = predict_image(file.read())
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
