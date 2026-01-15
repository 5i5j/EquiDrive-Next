import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
# 导入咱们刚刚定义的模型
from src.models.baseline_v1 import build_v1_model

def load_silver_data(data_dir="data/silver_v1"):
    """
    复习点：从 NPZ 中提取特征和标签
    """
    files = list(Path(data_dir).glob("*.npz"))
    x_data, y_data = [], []
    
    print(f"📦 正在载入 {len(files)} 个白银样本...")
    for f in files:
        data = np.load(f)
        x_data.append(data['x'])
        y_data.append(data['y'])
    
    return np.array(x_data), np.array(y_data)

def start_training():
    # 1. 准备数据
    X, y = load_silver_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 构建模型
    model = build_v1_model(input_shape=(110, 2))
    
    # 复习点：为什么用 SparseCategoricalCrossentropy？
    # 因为我们的标签是数字 (0, 1, 2)，不是 One-hot 向量
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. 开始炼丹
    print("\n🔥 P620 启动，开始 V1 Baseline 训练...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )

    # 4. 保存模型 (Greenfield 的第一个模型资产)
    model_path = Path("models/v1_baseline.h5")
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    print(f"\n✅ 训练完成！模型已保存至: {model_path}")

if __name__ == "__main__":
    start_training()