# Waste Sorting and Recycling Quality Controller

A Python 3.13 image-classification prototype for sorting one main waste item
into one of six classes:

- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

The primary model is a pretrained ResNet-18 fine-tuned on this dataset. A
small CNN trained from scratch provides an experimental baseline. The project
includes dataset auditing and reproducible split generation, training,
held-out evaluation, command-line prediction, and a simple Tkinter GUI.

## Scope and quality-control rule

The prototype expects **one main item per image**. It does not perform object
detection, segmentation, or direct measurement of material contamination.

For prediction, the user selects a target bin. The initial operational rule is:

1. If the highest softmax score is below `0.60`, return **Manual review**.
2. Otherwise, if the predicted class equals the selected bin, return
   **Matches selected bin**.
3. Otherwise, return **Potential contaminant—review**.

This flag is a prediction-based rule. Softmax scores are not calibrated
guarantees, and the test results below are classification results—not measured
contamination-detection accuracy.

## Dataset

This project was developed with the 2,527-image
[TrashNet dataset-resized collection](https://github.com/garythung/trashnet)
from Gary Thung and Mindy Yang. Raw images and downloaded archives are not
stored in this repository.

Obtain the dataset separately from the upstream project and extract it so the
six class folders are located at:

```text
data/raw/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

One PowerShell download method is:

```powershell
New-Item -ItemType Directory -Force .\data\raw | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip" `
  -OutFile .\data\raw\dataset-resized.zip
Expand-Archive .\data\raw\dataset-resized.zip -DestinationPath .\data\raw -Force
```

Review the upstream source before using or redistributing the dataset. This
repository does not assert dataset license terms that have not been verified.

## Environment and setup

The recorded training environment was:

- Python 3.13.15
- Pillow 13.0.0
- PyTorch 2.11.0 with CUDA 12.8 (`2.11.0+cu128`)
- torchvision 0.26.0 with CUDA 12.8 (`0.26.0+cu128`)
- NVIDIA GeForce RTX 3050 6GB Laptop GPU
- Windows, with `num_workers=0` for reliable data loading

Create the matching environment from PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install Pillow==13.0.0
.\.venv\Scripts\python.exe -m pip install `
  torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu128
```

Tkinter is included with the standard Windows Python installation used here.
CUDA is selected automatically when available; otherwise the programs use CPU.

## Dataset preparation

Audit the local images and create deterministic train/validation/test manifests:

```powershell
.\.venv\Scripts\python.exe .\src\audit_dataset.py
.\.venv\Scripts\python.exe .\src\make_splits.py
```

The committed manifest contains 1,764 training, 380 validation, and 377 test
images. Cross-class byte-identical images are excluded before splitting.

## Training

Fine-tune ResNet-18 for five epochs:

```powershell
.\.venv\Scripts\python.exe .\src\train_model.py
```

Train the SmallCNN baseline from scratch for five epochs:

```powershell
.\.venv\Scripts\python.exe .\src\train_baseline.py
```

Both checkpoints are selected only by validation macro F1. The test split is
not loaded by either training program.

## Final evaluation

Evaluate both already selected checkpoints once on the held-out test split:

```powershell
.\.venv\Scripts\python.exe .\src\evaluate_models.py
```

The script writes metrics JSON, confusion-matrix CSV, and per-image prediction
CSV files under `artifacts/`. Confusion-matrix rows are true classes and
columns are predicted classes.

## CLI prediction

Pass an image path and selected target bin:

```powershell
.\.venv\Scripts\python.exe .\src\predict.py .\path\to\item.jpg plastic
```

The CLI prints the predicted class, all six softmax scores, and the
prediction-based quality-control result.

## Tkinter GUI

```powershell
.\.venv\Scripts\python.exe .\src\gui.py
```

Choose an image, select the intended bin, and click **Analyze**. The GUI loads
the committed ResNet-18 checkpoint once at startup.

## Measured held-out classification results

Both checkpoints were fixed using validation results before the 377-image test
split was evaluated.

| Model | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| ResNet-18 | 343/377 | 0.9098 | 0.8959 |
| SmallCNN | 238/377 | 0.6313 | 0.5647 |

These values measure six-class image classification only. The dataset does not
provide the real-bin contamination ground truth required to report measured
contamination-detection accuracy.

Detailed metrics, per-class results, confusion matrices, and individual test
predictions are available in `artifacts/`.
