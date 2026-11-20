# Waste Sorting and Recycling Quality Controller

Project 07 is a local Python 3.13 prototype with two Streamlit workflows and a
separate educational NumPy module. The primary workflow localizes separated
waste items on a controlled background and classifies each crop. An optional,
lower-performing learned detector demonstrates bounding-box detection on TACO
outdoor litter images.

This repository reports classification and bounding-box detection evidence. It
does **not** report measured recycling purity, contamination-detection accuracy,
factory-conveyor performance, or calibrated confidence.

## Streamlit workflows

Run the interface from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\src\streamlit_app.py
```

### 1. Controlled background (default)

This is the primary offline demonstration:

1. OpenCV estimates the border background colour, builds a foreground mask,
   cleans noise, finds connected regions, and conservatively merges overlapping
   fragments that appear to be one physical item.
2. Each region is classified in memory by the saved six-class TrashNet
   ResNet-18 as `cardboard`, `glass`, `metal`, `paper`, `plastic`, or `trash`.
3. The page shows numbered boxes, softmax scores, sorting suggestions, review
   flags, an optional foreground mask, and downloadable local PNG/CSV results.

The intended input is roughly 2–4 separated objects on a plain, contrasting
background. This localizer is classical image processing, not learned object
detection. Touching items can merge; shadows, texture, low contrast, or an
almost full-frame object can create bad regions.

The optional target-bin rule is explicit:

- top score below `0.60`: **Manual review**;
- otherwise prediction equals selected bin: **Matches selected bin**;
- otherwise: **Potential contaminant—review**.

Without a selected bin the app gives a sorting suggestion, not pass/fail. A
softmax score is not a calibrated guarantee, and this rule is not measured bin
purity or contamination accuracy.

Create the labelled local composite smoke-test scene with:

```powershell
.\.venv\Scripts\python.exe .\src\create_composite_demo.py
```

It writes ignored `data/demo/composite_validation_scene.png` from three
validation-split TrashNet images. It is a composite diagnostic, not a real
multi-object photograph or an accuracy dataset.

### 2. Experimental learned TACO detector (optional)

This mode uses Faster R-CNN MobileNetV3 Large 320 FPN, initialized from the
named Torchvision COCO weights, with a newly trained seven-output box head:
background `0`; cardboard `1`; glass `2`; metal `3`; paper `4`; plastic `5`;
and other `6`. `OTHER` is displayed as **annotated waste needing review** and
is not the TrashNet `trash` class.

Uploads are EXIF-oriented, converted to RGB, and aspect-preservingly resized
without cropping. The selected checkpoint's CPU long-side limit is 960; the
model transform uses minimum/maximum sides 640/960. The app maps predicted
boxes back to the displayed image.

The mode is deliberately separate from the default workflow:

- it is intended for outdoor TACO-style litter scenes, not controlled tabletop
  images or factory conveyors;
- it has no target-bin pass/fail, purity, or contamination decision;
- its fixed display-only score cutoff is `0.30`;
- Torchvision first applies the model's internal ROI-head threshold `0.05` and
  retains at most 100 detections per image.

The final saved test evaluation applied no additional application-level score
filter. The app's `0.30` display cutoff therefore does not describe the test
protocol.

The exact selected detector checkpoint and selection record are ignored local
files under `data/taco_detection/`. They are verified by SHA-256 before loading,
never downloaded or silently substituted. A GitHub clone alone can run the
default mode because the ResNet checkpoint is committed, but it **cannot** run
the experimental mode until the exact local TACO preparation, selection record,
and checkpoint have been restored.

## Single-item command line

The existing single-item classifier remains available:

```powershell
.\.venv\Scripts\python.exe .\src\predict.py .\path\to\item.jpg plastic
```

The Streamlit app is the project's interface. The command-line predictor
remains available for single-item checks and automation.

## Dataset evidence and provenance

### TrashNet classification cohort

The 2,527-image
[TrashNet dataset-resized collection](https://github.com/garythung/trashnet)
is credited upstream to Gary Thung and Mindy Yang. Its README asks users to
cite the repository, and its repository contains an
[MIT LICENSE](https://github.com/garythung/trashnet/blob/master/LICENSE) with
Copyright (c) 2017 Gary Thung.

`src/audit_dataset.py` fully decodes JPEGs with Pillow and hashes file bytes.
Six cross-class byte-identical files were excluded. The deterministic seed-42
manifest contains 1,764 training, 380 validation, and 377 held-out test images.
Exact hashes detect byte-identical files, not visually similar images.

TrashNet images are single-item class photos; they provide neither multi-item
boxes nor contamination labels. Obtain the dataset separately and place it at:

```text
data/raw/dataset-resized/
|-- cardboard/
|-- glass/
|-- metal/
|-- paper/
|-- plastic/
`-- trash/
```

Raw images and archives remain ignored local data.

### TACO detection cohort

The detection audit pins the official
[pedropro/TACO](https://github.com/pedropro/TACO) annotations at commit
`70479ad9adc3da971a8ae75ac97eb1026910eec7`; the pinned annotation SHA-256 is
`812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b`.
TACO stays completely separate from the TrashNet split and metrics.

The selected local Policy B cohort keeps five unambiguous materials plus the
explicit `other` review class. It excludes a whole image rather than silently
dropping an annotated object. All 778 selected images decoded successfully and
contain 2,594 boxes, including 461 multi-object images:

| Target class | Boxes |
|---|---:|
| cardboard | 40 |
| glass | 194 |
| metal | 229 |
| paper | 109 |
| plastic | 1,097 |
| other | 925 |

The duplicate-grouped seed-42 image split is 544 train / 117 validation / 117
test, with 1,813 / 390 / 391 boxes. The derived COCO file assigns unique local
annotation IDs while retaining original source IDs in provenance metadata.

Photograph rights require caution. The TACO software repository's license does
not by itself grant rights to every contributed photograph. The selected local
cohort preserves the source URL and observed image-license string: 316 images
are labelled `CC` and 462 are labelled `ODBL (c) OpenLitterMap & Contributors`.
The project does not interpret `CC` as a specific Creative Commons variant,
does not treat a null value as redistribution permission, and does not publish
downloaded TACO photographs, annotations, derived manifests, or audit output.

## Models, training, and selection

### TrashNet classifiers

Both classifiers used RGB inputs, CrossEntropyLoss, Adam, seed 42, five epochs,
train rows for parameter updates, and validation rows for model selection. The
held-out test split was not loaded by either training program.

- **ResNet-18:** `ResNet18_Weights.IMAGENET1K_V1`; ImageNet-normalized 224-pixel
  crops; batch 16; learning rate 0.0001. All earlier parameters were frozen;
  only `layer4` and the new six-output `fc` layer were trained. Frozen batch-norm
  layers remained in evaluation mode. Epoch 5 was selected with validation
  accuracy 0.905263 and macro F1 0.903490.
- **SmallCNN:** four convolution blocks, randomly initialized and genuinely
  trained from scratch; ImageNet-normalized 128-pixel crops; batch 32; learning
  rate 0.001. Epoch 5 was selected with validation accuracy 0.644737 and macro
  F1 0.563476.

Training used random resized crops and horizontal flips. Validation/inference
used deterministic resize and centre crop (256→224 for ResNet, 144→128 for
SmallCNN) with the same ImageNet normalization.

### TACO detector

Detector training used only the 544 train images for updates and all 117
validation images for AP50 checkpoint selection. Settings were seed 42, batch
1, Windows workers 0, CUDA mixed precision, SGD learning rate 0.0025, momentum
0.9, weight decay 0.0005, and pycocotools COCOeval box AP. The COCO-trained
backbone/ROI model was loaded from
`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`; the replacement
seven-output head began untrained.

Three isolated experiments were preserved:

| Run | Resize (CPU / model min-max) | Epochs | Best validation AP50 |
|---|---|---:|---:|
| initial | 320 / 320-320 | 3 | 0.065856 |
| high-resolution | 960 / 640-960 | 3 | 0.187336 |
| fresh high-resolution | 960 / 640-960 | 8 | **0.206085 (epoch 3)** |

The eight-epoch run was fresh, not a continuation. The final rule selected the
highest saved validation AP50; an exact numeric tie would select the
lexicographically smaller checkpoint SHA-256. Selection occurred before any
test image was loaded. The selected checkpoint hash is
`ade03c44a78c4fe653037c264c663e00a17c2d47df8ada6e40bb75291822b6ee`.

## Measured held-out results

### TrashNet: six-class single-item classification (377 images)

| Model | Correct | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 343/377 | 0.9098 | 0.8930 | 0.9009 | 0.8959 |
| SmallCNN | 238/377 | 0.6313 | 0.6305 | 0.5600 | 0.5647 |

Detailed per-class results, confusion matrices, and per-image predictions are
committed under `artifacts/`. These are classification results—not localization,
object-detection, contamination, or purity results.

### TACO: experimental bounding-box detection (117 images, 391 boxes)

The fixed selected checkpoint was evaluated once after selection. Genuine COCO
bbox metrics were AP50 **0.184587** and AP50:95 **0.104930**.

| Class | AP50 | AP50:95 | Ground-truth boxes |
|---|---:|---:|---:|
| cardboard | 0.382264 | 0.254068 | 7 |
| glass | 0.000000 | 0.000000 | 29 |
| metal | 0.310484 | 0.153120 | 35 |
| paper | 0.047390 | 0.024041 | 16 |
| plastic | 0.265350 | 0.142099 | 165 |
| other | 0.102031 | 0.056254 | 139 |

The metrics use COCO IoU thresholds 0.50:0.05:0.95, all areas, and at most 100
detections per image. The installed Torchvision ROI head applied its internal
0.05 threshold; no additional evaluator/application threshold was applied. The
glass result of zero and the small per-class supports are important limitations.
These outdoor-litter results do not validate controlled tabletop scenes or a
production conveyor.

## NumPy learning fundamentals

`src/fundamentals/` is independent of the trained waste models. It contains an
educational two-layer NumPy MLP with matrix multiplication, ReLU, stable
softmax, mean cross-entropy, explicit manual gradients, and parameter updates.
The five-example fixed-seed demo compares analytical gradients with central
finite differences and explains why inputs at a ReLU boundary are unsuitable
for a smooth numerical check.

The demos distinguish:

- full-batch gradient descent;
- true stochastic SGD using one randomly sampled example per update;
- ordinary Momentum using the current-parameter gradient; and
- Nesterov momentum using a gradient evaluated at look-ahead parameters.

Run them with:

```powershell
.\.venv\Scripts\python.exe .\src\fundamentals\demo_gradients.py
.\.venv\Scripts\python.exe .\src\fundamentals\compare_optimizers.py
```

The tiny deterministic example demonstrates equations and loss reduction; it
does not establish that one optimizer is universally better and is not part of
the ResNet, SmallCNN, or detector pipelines.

## Environment and dependencies

The verified Windows environment is:

- Python 3.13.15
- PyTorch 2.11.0+cu128 and Torchvision 0.26.0+cu128
- NVIDIA GeForce RTX 3050 6 GB Laptop GPU
- NumPy 2.5.4, Pillow 13.0.0, OpenCV Python 5.1.0.95
- Streamlit 1.67.0 and pycocotools 2.0.11

Create the matching environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
  torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r .\requirements-app.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements-detector.txt
```

The application never downloads weights at inference time. ResNet is
constructed with `weights=None` and loaded from the committed checkpoint; the
experimental mode similarly requires the hash-selected ignored local detector.

## Reproducible commands

### Read-only offline-demo audit

```powershell
.\.venv\Scripts\python.exe .\src\check_demo_readiness.py --require-experimental
```

This checks imports, versions, the default checkpoint, the optional selection
record and selected-checkpoint hash/metadata, and chosen local images. It does
not download, train, or evaluate a dataset. Override `--default-image` or
`--experimental-image` to check a different local demonstration image.

### TrashNet preparation and training

```powershell
.\.venv\Scripts\python.exe .\src\audit_dataset.py
.\.venv\Scripts\python.exe .\src\make_splits.py
.\.venv\Scripts\python.exe .\src\train_model.py --epochs 5 --batch-size 16
.\.venv\Scripts\python.exe .\src\train_baseline.py
```

`src/evaluate_models.py` is the separate historical final classification
evaluator. Do not rerun it merely to launch the demo.

### TACO local preparation and detector training

```powershell
.\.venv\Scripts\python.exe .\src\audit_taco_dataset.py
.\.venv\Scripts\python.exe .\src\prepare_taco_detection.py --policy B

.\.venv\Scripts\python.exe -u .\src\train_taco_detector.py `
  --device cuda --epochs 8 --batch-size 1 `
  --cpu-max-long-side 960 --detector-min-side 640 --detector-max-side 960 `
  --learning-rate 0.0025 --momentum 0.9 --weight-decay 0.0005 `
  --experiment-name highres_640_960_fresh_8ep `
  --output-dir .\data\taco_detection\training_highres_640_960_fresh_8ep
```

All TACO inputs and outputs stay under ignored `data/`. The final selection and
test-evaluation commands are retained in `src/evaluate_taco_detector.py`, but
the held-out test should not be rerun for an ordinary reproduction or demo.

### Engineering verification

```powershell
.\.venv\Scripts\python.exe -m compileall -q .\src .\tests
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
.\.venv\Scripts\python.exe -m pip check
```

## Remaining scope and official-compendium gaps

Implemented code should not be confused with theoretical discussion. Current
gaps include:

- no learned semantic or instance segmentation and no IoU evaluation for masks;
- no YOLO integration or YOLO training/evaluation;
- no LSTM sequence model and no GAN/VAE component;
- no implemented tensor-operation/SVD demonstration or controlled convolution
  ablation suite beyond the present models;
- no Hebbian, competitive, error-correction, or energy-based learning demos;
- no measured recycling purity, contamination ground truth, production latency,
  sustained FPS, camera/conveyor integration, or industrial field trial;
- no claim that the OpenCV or low-AP TACO path is production-ready; and
- the detector's ignored local data/checkpoint and uncertain photograph rights
  prevent a clone-only redistribution/demo of that mode.

The NumPy MLP, numerical gradient checking, GD, true stochastic SGD, Momentum,
and genuine look-ahead Nesterov demonstrations are implemented. The TACO path
is genuine learned object detection, but its low AP and zero glass AP50 make it
an optional experiment rather than a replacement for the controlled workflow.

Public repository:
[VeeSolstice/waste-sorting-quality-controller-project-07](https://github.com/VeeSolstice/waste-sorting-quality-controller-project-07)
