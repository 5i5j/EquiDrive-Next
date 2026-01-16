# EquiDrive-Next 实验记录

## [2026-01-16] V3.1 大规模数据演进
### 1. 核心变更
- **特征降维**: 剔除 $x, y$ 绝对坐标，仅保留 `[vx, vy, angle_err, lat_dist]` (4D)。
- **数据扩容**: 从 6,000 提升至 30,000 样本 (`QUOTA_PER_CLASS = 10000`)。
- **环境**: P620 (16-core parallel extraction).

### 2. 关键命令行
- 数据提取: `python3 src/data/extractor_v3.py`
- 训练: `python3 src/training/train_v3.py`
- 备份: `dvc add data/silver_v3 models/*.h5 && git commit -m "..."`

### 3. 实验结果
- **Val Accuracy**: 94.92% (Epoch 45)
- **结论**: 证明了“去坐标化”能显著增强模型的几何推理能力，降低过拟合。