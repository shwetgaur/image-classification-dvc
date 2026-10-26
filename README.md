# Modular Image Classification with DVC

End-to-end MLOps pipeline for image classification. Each stage is a separate Python module, all tunables live in `params.yaml`, and `dvc.yaml` wires the stages into one reproducible run.

## Pipeline

```
data_collection → data_processing → feature_selection → model_training → model_evaluation
```

| Stage | Module | What it does |
| --- | --- | --- |
| Data collection | `src/data_collection.py` | Downloads a CIFAR-10 subset (or builds a synthetic fallback) and saves labeled PNGs |
| Data processing | `src/data_processing.py` | Resizes, normalizes, train/val splits, and applies augmentation |
| Feature selection | `src/feature_selection.py` | Variance filter + SelectKBest / PCA on flattened pixels |
| Model training | `src/model_training.py` | Trains a CNN (default) or MLP from `params.yaml` |
| Model evaluation | `src/model_evaluation.py` | Writes accuracy/precision/recall/F1, a report, and a confusion matrix |

## Project layout

```
image-classification-dvc/
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── src/
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── data/          # created by the pipeline
├── models/        # classifier.keras + training history
└── reports/       # metrics.json, plots, classification report
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
dvc init
```

If this repo already contains a `.dvc/` folder, skip `dvc init`.

## Run the full pipeline

```bash
dvc repro
```

DVC only re-runs stages whose code, params, or inputs changed.

Useful commands:

```bash
dvc dag                 # show stage graph
dvc metrics show        # print reports/metrics.json
dvc plots show          # open the confusion-matrix plot
dvc repro model_training
```

### Run stages without DVC

```bash
python -m src.data_collection
python -m src.data_processing
python -m src.feature_selection
python -m src.model_training
python -m src.model_evaluation
```

## Tunable parameters (`params.yaml`)

There are more than 10 knobs. The main ones:

1. `data_processing.image_size`
2. `data_processing.validation_split`
3. `data_processing.augmentation.rotation_range`
4. `data_processing.augmentation.horizontal_flip`
5. `data_processing.augmentation.zoom_range`
6. `feature_selection.variance_threshold`
7. `feature_selection.k_best`
8. `model_training.learning_rate`
9. `model_training.batch_size`
10. `model_training.epochs`
11. `model_training.optimizer`
12. `model_training.dropout_rate`
13. `model_training.dense_units`
14. `model_training.early_stopping_patience`
15. `model_training.l2_regularization`

After changing any value, run `dvc repro` again. DVC invalidates only the affected stages.

## Dataset

Default source is **CIFAR-10** (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). `data_collection.max_samples` keeps the download small for assignment demos.

If CIFAR-10 cannot be downloaded, the collector automatically writes a synthetic colored-shape dataset so the rest of the pipeline still runs.

To force the fallback:

```yaml
data_collection:
  dataset_name: synthetic
```

## Model

- `architecture: cnn` — small ConvNet on processed images (default, needs TensorFlow)
- `architecture: mlp` — dense network on the selected feature vectors

If TensorFlow is missing, training automatically falls back to a scikit-learn MLP on the selected features so `dvc repro` still completes. Install TensorFlow to use the CNN path:

```bash
pip install tensorflow
```

## Outputs

- `models/classifier/` — trained model (`model.keras` or `model.joblib`)
- `models/training_history.json` — loss / accuracy per epoch
- `reports/metrics.json` — DVC-tracked metrics
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

## Notes for graders

- Stages are isolated `.py` modules, not a single notebook.
- Pipeline automation is defined entirely in `dvc.yaml`.
- All hyperparameters are centralized in `params.yaml`.
