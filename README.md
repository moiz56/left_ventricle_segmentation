# Left Ventricle Segmentation

A deep learning pipeline for automatic left ventricle segmentation from cardiac MRI (CMR) volumes using an Attention U-Net architecture.

## Overview

This project trains an Attention U-Net model on the [EMIDEC dataset](http://emidec.com/) to segment the left ventricle from 3D cardiac MRI scans. It includes a complete MLOps pipeline — data ingestion, preprocessing, training with experiment tracking via MLflow, and a FastAPI inference service deployable via Docker.

## Project Structure

```
left_ventricle_segmentation/
├── app/
│   ├── main.py              # FastAPI inference service
│   └── utils.py             # Overlay saving utilities
├── configs/
│   └── config.yaml          # Hyperparameters and model config
├── logs/
│   └── logger.py            # Logging setup
├── src/
│   ├── data/
│   │   ├── ingest.py        # Dataset organization from raw EMIDEC files
│   │   ├── preprocessing.py # Slice extraction, padding/resizing to 224×224
│   │   ├── split.py         # Train/validation split
│   │   ├── remap_masks.py   # Mask label remapping
│   │   └── cardiac_mri_dataset.py  # PyTorch Dataset with augmentations
│   ├── models/
│   │   └── attention_unet.py  # Attention U-Net model registry
│   ├── training/
│   │   └── train.py         # Training loop with DiceCE loss
│   ├── pipeline/
│   │   └── training_pipeline.py  # End-to-end training pipeline entry point
│   ├── inference/
│   │   └── inference.py     # CLI inference pipeline
│   └── utils/
│       ├── load_yaml.py     # Config loader
│       └── visualize_data.py
├── constants.py             # Path constants
├── requirements.txt
├── Dockerfile
└── pyproject.toml
```

## Dataset

This project uses the **EMIDEC** (Evaluation and Mitral valve segmentation in Echocardiographic Data for Cardiac pathologies) dataset. Place the raw dataset at:

```
dataset/emidec-dataset-1.0.1/
```

The pipeline automatically handles organization, preprocessing, and splitting.

## Setup

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (optional; CPU is supported)

### Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Usage

### 1. Run the Full Training Pipeline

The training pipeline handles all steps — ingestion, preprocessing, splitting, and training — in one command:

```bash
python -m src.pipeline.training_pipeline
```

**CLI arguments** (override config defaults):

| Argument | Default | Description |
|---|---|---|
| `--lr` | `1e-4` | Learning rate |
| `--batch_size` | `16` | Batch size |
| `--num_epochs` | `50` | Training epochs |
| `--model` | `attention_unet_extended` | Model variant |

Example:

```bash
python -m src.pipeline.training_pipeline --lr 5e-5 --num_epochs 100
```

The best model checkpoint is saved to `registry/attention_unet.pth`. Training curves (loss and Dice score) are saved to `figures/`.

### 2. Run Inference (CLI)

Visualize predictions on validation samples:

```bash
python -m src.inference.inference --model attention_unet_extended --inference_number 10
```

| Argument | Default | Description |
|---|---|---|
| `--model` | from config | Model name in the registry |
| `--inference_number` | `10` | Number of samples to visualize |
| `--save_dir` | `figures` | Output directory for overlay images |
| `--model_path` | `registry` | Directory containing `.pth` checkpoints |

### 3. Run the Inference API

#### Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### With Docker

```bash
docker build -t lv-segmentation .
docker run -p 8000:8000 lv-segmentation
```

The API will be available at `http://localhost:8000`.

#### API Endpoints

**`POST /predict/`** — Segment a single NIfTI volume

```bash
curl -X POST "http://localhost:8000/predict/" \
  -F "file=@patient_001.nii.gz" \
  -F "batch_size=8"
```

**`POST /batch_predict/`** — Segment multiple NIfTI volumes at once

```bash
curl -X POST "http://localhost:8000/batch_predict/" \
  -F "files=@patient_001.nii.gz" \
  -F "files=@patient_002.nii.gz"
```

Both endpoints accept `.nii` or `.nii.gz` files, return inference timing, and save overlay images asynchronously to `app/inference_results/`.

## Model Architecture

Two Attention U-Net variants are available via the model registry:

| Name | Channels | Dropout | Notes |
|---|---|---|---|
| `attention_unet` | (32, 64, 128, 256, 512) | None | Baseline |
| `attention_unet_extended` | (64, 128, 256, 512, 1048) | 0.5 | Default, larger capacity |

Both are 2D models trained slice-by-slice on 224×224 grayscale images with 2 output classes (background / left ventricle).

## Training Details

| Hyperparameter | Value |
|---|---|
| Loss | DiceCE Loss |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Epochs | 50 |
| Train/Val split | 90% / 10% |
| Input size | 224×224 |
| Metric | Dice Score |

Experiment metrics (loss, Dice score per epoch) are tracked with **MLflow**. To view the MLflow dashboard:

```bash
mlflow ui
```

## Data Pipeline

1. **Ingest** (`src/data/ingest.py`) — Flatten the EMIDEC patient-folder structure into `images/` and `contours/` directories.
2. **Preprocess** (`src/data/preprocessing.py`) — Load each NIfTI volume, extract 2D axial slices, pad slices smaller than 224×224 or resize larger ones down, and save as `.npy` files.
3. **Split** (`src/data/split.py`) — Randomly split paired image/mask slices into train (90%) and validation (10%) sets using a fixed seed.
4. **Remap** (`src/data/remap_masks.py`) — Remap mask labels from EMIDEC convention to binary (0 = background, 1 = LV).

## Logging

Logs are written per component to the `logs/` directory:

- `logs/training.log` — Training loop
- `logs/training_pipeline.log` — Full pipeline
- `logs/inference.log` — CLI inference
- `logs/api.log` — FastAPI service
- `logs/dataset.log` — Dataset loading
