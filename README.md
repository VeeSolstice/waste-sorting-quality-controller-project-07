# Waste Sorting and Recycling Quality Controller

A local Python 3.13 prototype that analyzes a controlled-background image with
several **separated** waste items. It suggests one of six categories for each
localized region:

- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

The polished Streamlit page is the primary demonstration interface. The saved
ResNet-18 checkpoint, SmallCNN baseline, dataset manifests, and historical test
evidence remain unchanged.

## What the current prototype does

The scene workflow has two explicit stages:

1. **Controlled-background localization.** OpenCV and NumPy estimate the border
   background color, create a foreground mask, clean small noise, find connected
   regions, filter by area, and return bounding boxes. This is classical image
   processing—not YOLO and not a learned object detector.
2. **Crop classification.** Each localized PIL crop is passed in memory to the
   existing ResNet-18 with the checkpoint's saved deterministic preprocessing.
   The model predicts cardboard, glass, metal, paper, plastic, or trash.

The intended scene contains roughly 2–4 separated objects on a plain,
contrasting background. Touching items may merge into one region. A cluttered,
textured, low-contrast, or nearly full-frame foreground can fail localization;
the app then reports the problem instead of silently classifying the whole scene
as one item.

## Sorting suggestions and optional target-bin check

Target-bin checking is optional. Without a target bin, each sufficiently
confident crop receives its predicted class as a sorting suggestion.

The provisional rule is:

1. If the highest softmax score is below `0.60`, mark the item **Manual review**.
2. If no target bin is selected, show the predicted category as a suggestion.
3. If a target bin is selected and the prediction matches it, show
   **Matches selected bin**.
4. Otherwise show **Potential contaminant—review**.

Softmax scores are not calibrated guarantees. This is a prediction-based review
rule, not measured contamination accuracy, bin purity, or physical conveyor
control.

## Run the Streamlit app

From PowerShell in the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\src\streamlit_app.py
```

The page supports:

- JPG, JPEG, PNG, BMP, and WebP uploads;
- original and annotated scene views;
- numbered per-item boxes, predictions, scores, and review flags;
- counts by suggested category;
- optional selected-bin comparison;
- downloadable annotated PNG and per-item CSV;
- local cached checkpoint loading with no inference API or internet dependency.

For the most reliable live demonstration, use good lighting, a plain background,
clear object separation, and an image no larger than 20 MB or 20 million pixels.

## Local composite demonstration

The following command builds an explicitly labelled **COMPOSITE DEMONSTRATION**
from three validation-split TrashNet images:

```powershell
.\.venv\Scripts\python.exe .\src\create_composite_demo.py
```

It writes `data/demo/composite_validation_scene.png`. The `data/` directory is
ignored by Git. This composite is a pipeline smoke-test scene, not a factory
photograph and not multi-object ground truth. It does not use test-split images
to tune localization.

## Single-item CLI compatibility

The original command-line predictor remains available and keeps the mandatory
target-bin interface:

```powershell
.\.venv\Scripts\python.exe .\src\predict.py .\path\to\item.jpg plastic
```

The old Tkinter file remains in `src/gui.py` for historical compatibility, but
Streamlit is the recommended interface.

## Dataset provenance

This project was developed with the 2,527-image
[TrashNet dataset-resized collection](https://github.com/garythung/trashnet),
credited upstream to Gary Thung and Mindy Yang. The upstream README asks users
to cite the repository. It contains an
[MIT License](https://github.com/garythung/trashnet/blob/master/LICENSE), with
Copyright (c) 2017 Gary Thung.

TrashNet contains single-item class photos; it does not provide real multi-item
bounding boxes or contamination labels. Raw images and downloaded archives are
kept locally and are not committed.

Place the separately obtained dataset at:

```text
data/raw/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Reproducible environment

The verified local environment is:

- Python 3.13.15
- PyTorch 2.11.0 with CUDA 12.8 (`2.11.0+cu128`)
- torchvision 0.26.0 with CUDA 12.8 (`0.26.0+cu128`)
- NVIDIA GeForce RTX 3050 6 GB Laptop GPU
- NumPy 2.5.4
- Pillow 13.0.0
- OpenCV Python 5.1.0.95
- Streamlit 1.67.0
- Windows with data-loader workers set to zero

Create the matching environment from PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
  torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r .\requirements-app.txt
```

The application does not download pretrained weights at inference time; it
constructs ResNet-18 without weights and loads `artifacts/waste_resnet18.pt`.

## Verification commands

Compile the source and run the controlled-scene tests:

```powershell
.\.venv\Scripts\python.exe -m compileall -q .\src .\tests
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

The tests cover separated regions, blank scenes, touching regions, small noise,
too many substantial regions, oversized dimensions, invalid image bytes,
optional target-bin behavior, and PNG/CSV result consistency.

## Dataset preparation and training history

Audit and deterministic split generation:

```powershell
.\.venv\Scripts\python.exe .\src\audit_dataset.py
.\.venv\Scripts\python.exe .\src\make_splits.py
```

The committed manifest contains 1,764 training, 380 validation, and 377 test
images. Six cross-class byte-identical files are excluded before splitting.

Historical training commands:

```powershell
.\.venv\Scripts\python.exe .\src\train_model.py
.\.venv\Scripts\python.exe .\src\train_baseline.py
```

Both checkpoints were selected only by validation macro F1. This Streamlit
milestone does not retrain either model and does not rerun the held-out test
evaluation.

## Measured held-out single-item classification results

Both checkpoints were fixed using validation results before the 377-image test
split was evaluated.

| Model | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| ResNet-18 | 343/377 | 0.9098 | 0.8959 |
| SmallCNN | 238/377 | 0.6313 | 0.5647 |

These are six-class **single-item classification** results. They are not
multi-object localization accuracy, multi-object detection accuracy, mAP, IoU,
contamination accuracy, recycling purity, latency, FPS, or production conveyor
performance.

Detailed historical metrics, per-class results, confusion matrices, and
per-image test predictions remain under `artifacts/`.

## Remaining limitations

- There are no real multi-object bounding-box annotations for quantitative
  localization evaluation.
- Real-scene validation with an independently photographed 2–4 item scene is
  still pending; the repository currently provides only the labelled composite
  smoke test.
- Touching or overlapping items can merge, while texture and shadows can create
  false regions.
- The classifier was trained on single-item TrashNet photographs, so crop and
  real-scene distribution shift remains unmeasured.
- The provisional `0.60` review threshold has not been calibrated.
- No production latency, FPS, conveyor, camera, purity, or contamination study
  has been performed.
