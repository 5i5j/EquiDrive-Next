# EquiDrive-Next: Autonomous Vehicle Trajectory Intent Prediction

## 🎯 Core Philosophy
EquiDrive-Next is built on the principle of **Data-Centric AI**. We believe that a balanced, high-fidelity data pipeline is more critical than complex model architectures. By mastering the transition from raw Argoverse 2 telemetry to synchronized 4D tensors, we provide a robust foundation for next-generation motion prediction.

EquiDrive-Next is a high-performance, refined reconstruction of a trajectory prediction system based on the Argoverse 2 (AV2) dataset. It focuses on classifying driving intentions (Left, Right, Straight) using deep learning architectures.

## 🚀 Project Vision
Built from the ground up ("Greenfield Project"), this repository implements a clean data engineering pipeline (Landing -> Bronze -> Silver) and a baseline LSTM model to achieve robust intent classification.

## 🛠 Tech Stack
- **Compute**: NVIDIA GeForce RTX 3080 Ti (12GB)
- **Framework**: TensorFlow 2.x / Keras
- **Data Engine**: Argoverse 2 API, Pandas, Polars
- **Version Control**: DVC (Data Version Control) for large Parquet files

## 📂 Data Pipeline (The Medallion Architecture)
1. **Landing**: Raw AV2 Parquet files (Multi-agent, global coordinates).
2. **Bronze**: Filtered Focal Tracks + Intent Labeling via Heading Delta (V2 Logic).
3. **Silver**: Normalized (110, 2) tensors in `.npz` format for GPU-accelerated training.

## 🧠 Model Architecture (V1 Baseline)
- **Type**: Single-layer LSTM (64 units)
- **Input**: `(110, 2)` - 110 frames of (x, y) coordinates.
- **Output**: 3-class Softmax (Left, Right, Straight).
- **Current Performance**: **91.4% Validation Accuracy** (trained on 10k samples).

## 🏃 How to Run
```bash
# Set up PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# 1. Process Data
python3 src/data/builder_bronze.py
python3 src/data/extractor_v1.py

# 2. Train Model
python3 src/training/train_v1.py

## 🚀 Recent Milestones: V2 "Symmetry"
- **Balanced Training**: Achieved a perfect 1:1:1 distribution (2000 samples per class) to eliminate "Straight-driving bias."
- **4D Feature Engineering**: Upgraded input tensors to $(110, 4)$ by incorporating velocity vectors ($v_x, v_y$) alongside positions ($x, y$).
- **Performance**: Reached **92.6% Validation Accuracy** on a balanced dataset, with a significant reduction in loss ($0.14$).

## 🧠 Model Architecture (V2 Advanced)
- **Input**: $(110, 4)$ - Time-series of Relative Positions & Instantaneous Velocities.
- **Backbone**: Dual-layer LSTM (64/32 units) with Dropout (0.2) for regularization.
- **Output**: 3-class Softmax (LEFT, RIGHT, STRAIGHT).

## 📂 Data Pipeline
1. **Bronze**: Class-balanced extraction from AV2 via `focal_track_id` and `heading_delta` logic.
2. **Silver V2**: Normalized 4D tensors in `.npz` format.

## 📊 V1 vs V2 Comparison
| Version | Features | Data Balance | Val Acc | Loss |
| :--- | :--- | :--- | :--- | :--- |
| **V1** | 2D (x, y) | Natural (70% Straight) | 91.4% | 0.20 |
| **V2** | **4D (x, y, vx, vy)** | **Balanced (1:1:1)** | **92.6%** | **0.14** |