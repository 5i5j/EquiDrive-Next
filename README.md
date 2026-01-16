# EquiDrive-Next: Autonomous Vehicle Trajectory Intent Prediction

## 🎯 Core Philosophy
EquiDrive-Next is built on the principle of **Data-Centric AI**. We believe that a balanced, high-fidelity data pipeline is more critical than complex model architectures. By mastering the transition from raw Argoverse 2 telemetry to synchronized 4D tensors, we provide a robust foundation for next-generation motion prediction.

## 🏗️ System Architecture (The Four Pillars)
1. **Data Engineering**: Automated pipeline from raw AV2 telemetry to balanced 4D tensors.
2. **MLOps**: Full-stack versioning using **Git** (Code) and **DVC** (Data/Models) with **MinIO/S3** backend.
3. **Kinematic Modeling**: Dual-layer LSTM architecture optimized for RTX 3080 Ti.
4. **Quantitative Benchmarking**: Real-time inference visualization and confidence scoring.

## 🚀 Recent Milestones: V2 "Symmetry"
- **Balanced Training**: Achieved a perfect **1:1:1 distribution** (2000 samples per class) to eliminate "Straight-driving bias."
- **4D Feature Engineering**: Upgraded input tensors to $(110, 4)$ by incorporating velocity vectors ($v_x, v_y$) alongside positions ($x, y$).
- **Performance**: Reached **92.6% Validation Accuracy** on a balanced dataset, with a significant reduction in loss ($0.14$).
- **Visual Validation**: Confirmed **99.9% confidence** on complex S-curve trajectories using V2 inference.

## 📂 Data Pipeline (The Medallion Architecture)
1. **Landing**: Raw AV2 Parquet files (Multi-agent, global coordinates).
2. **Bronze**: Filtered Focal Tracks + Intent Labeling via Heading Delta.
3. **Silver (V2)**: Normalized **4D tensors** $(110, 4)$ in `.npz` format for GPU-accelerated training.

## 🧠 Model Architecture (V2 Advanced)
- **Input**: $(110, 4)$ - Relative Positions + Instantaneous Velocities.
- **Backbone**: Dual-layer LSTM (64/32 units) with Dropout (0.2) for regularization.
- **Output**: 3-class Softmax (LEFT, RIGHT, STRAIGHT).

## 📊 V1 vs V2 Comparison
| Version | Features | Data Balance | Val Acc | Loss | Focus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | 2D (x, y) | Natural (70% Straight) | 91.4% | 0.20 | Baseline Setup |
| **V2** | **4D (x,y,vx,vy)** | **Balanced (1:1:1)** | **92.6%** | **0.14** | **Kinematic Physics** |

## 📦 Reproducibility & MLOps
EquiDrive-Next uses **DVC** to manage datasets and models, keeping the GitHub repository lightweight.
- **Code**: [GitHub - 5i5j/EquiDrive-Next](https://github.com/5i5j/EquiDrive-Next)
- **Data/Models**: Hosted on `s3://equidrive-ml` (MinIO/AWS S3).

```bash
# To reproduce the V2 state:
git pull origin main
dvc pull
python3 src/evaluation/visualizer.py

