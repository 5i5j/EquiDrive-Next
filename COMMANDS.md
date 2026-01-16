# EquiDrive-Next Training Guide

## Environment Setup
- Environment: `conda activate tf_gpu` (Ubuntu 22.04 P620)
- Hardware: NVIDIA RTX 3080 Ti

## Data Pipeline
1. **Bronze Extraction**: `python3 src/data/builder_bronze.py`
2. **Silver V2 (4D Features)**: `python3 src/data/extractor_v2.py`

## Model Training
- Run: `python3 src/training/train_v2.py`
- Hyperparams are stored in `params.yaml`.

## MLOps Sync
- Pull data: `dvc pull`
- Push data: `dvc push` (Remote: MinIO/S3 equidrive-ml)

