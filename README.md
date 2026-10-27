# ☀️ Solar Panel Stain Detection & Severity Classification Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and deep learning system that automates surface defect and soiling inspection for photovoltaic (PV) solar panels. 

The pipeline eliminates background noise (roof tiles, vegetation, sky), maps pixel-accurate stain segmentation masks (dust, bird droppings, mud), calculates exact surface coverage percentages, and identifies localized electrical hotspot risks across a $4 \times 4$ sub-panel grid.

---

## 🚀 Key Features

* **Stage A: Perspective Rectification & Background Removal:**
  Automatically identifies the 4 panel corners and uses homography warping to extract a canonical $512 \times 512$ top-down view, cutting out 100% of surrounding background interference.
* **Stage B: Stain Segmentation (U-Net with ResNet34):**
  Trained with compound **Dice + Focal Loss** to tackle severe class imbalance (resolving small bird droppings and faint particulate dust).
* **Explainable Severity Engine:**
  - **High Stain (Cleaning Required):** Total surface coverage $\ge 10.0\%$ **OR** any single localized bird-dropping blob $\ge 2.0\%$.
  - **Low Stain (Operational):** Below both operational thresholds.
* **$4 \times 4$ Sub-Panel Grid Heatmap:**
  Divides the panel into 16 localized cells to detect hotspots that threaten bypass diodes.
* **Production-Ready ONNX Runtime & Streamlit UI:**
  Exported to lightweight ONNX runtimes with verified numeric parity and packaged into an interactive web app.

---

## 📊 Benchmark Results (Quarantined Test Set)

| Evaluation Metric | Target Threshold | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Stain Mask IoU** | $\ge 0.700$ | **`0.8032`** | **Exceeded** |
| **Stain Mask Dice** | $\ge 0.800$ | **`0.8784`** | **Exceeded** |
| **High-Class Recall** | $\ge 0.950$ | **`1.0000` (100%)** | **Zero False Negatives** |
| **ROC-AUC** | $\ge 0.950$ | **`0.9529`** | **Near-Perfect Separation** |
| **Cross-Dataset F1 Drop** | $\le 0.080$ | **`0.0400`** | **Robust Generalization** |

---

## 🛠️ Repository Structure

```
solar_panel_stain_detection/
├── config.yaml               # Central configuration (thresholds, seeds, model params)
├── DATASETS.md               # Dataset provenance, licensing, and pHash protocol
├── data_prep.py              # Ingestion, pHash deduplication, 70/15/15 group-stratified split
├── train_panel_detector.py   # Stage A: Perspective homography rectifier
├── train_stain_segmenter.py  # Stage B: U-Net segmentation with Dice+Focal loss
├── train_classifier.py       # Baseline image-level classifier (ResNet18)
├── infer.py                  # End-to-end CLI inference (CSV + visual overlays)
├── evaluate.py               # Evaluation suite (IoU, Recall, Slices, Calibration, Errors)
├── export_onnx.py            # ONNX export with parity check against PyTorch
├── app.py                    # Interactive Streamlit web inspection app
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Ingested datasets (Kaggle, Mendeley, Roboflow)
│   ├── masks/                # Ground-truth polygonal masks (SAM 2 / CVAT spec)
│   └── manifest.csv          # Group-stratified dataset index
├── models/
│   ├── stain_segmenter_best.pt   # PyTorch Stage B model weights
│   ├── classifier_baseline_best.pt # Baseline classifier weights
│   ├── stain_segmenter.onnx      # Production ONNX model
│   └── classifier_baseline.onnx  # Production ONNX classifier
├── outputs/
│   ├── overlays/             # Visual diagnostic images with 4x4 grid heatmaps
│   └── predictions.csv       # Batch predictions log
└── reports/
    ├── figures/              # Confusion matrix, ROC, calibration, and slice plots
    ├── errors/               # Visual diagnostics of edge failure cases
    └── evaluation_report.md  # Detailed metrics report
```

---

## ⚡ Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/solar-panel-stain-detection.git
cd solar-panel-stain-detection
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Inference on an Image
```bash
python infer.py --image path/to/solar_panel.jpg --output-dir outputs/overlays/
```

### 3. Batch Inference on a Folder
```bash
python infer.py --input-dir data/raw/sample_folder --output-csv outputs/predictions.csv
```

### 4. Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```

---

## 📜 License
This project is open-sourced under the MIT License.
