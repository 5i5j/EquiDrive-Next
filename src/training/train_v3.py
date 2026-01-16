import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 配置路径
DATA_DIR = Path("data/silver_v3")
MODEL_SAVE_PATH = Path("models/equidrive_v3_refined.h5")
SCALER_SAVE_PATH = Path("models/scaler_v3_refined.pkl")

def load_v3_data():
    X_raw, y = [], []
    files = list(DATA_DIR.glob("*.npz"))
    
    print(f"📦 加载数据并执行【去坐标化】映射 (3万样本规模)...")
    for f in files:
        data = np.load(f)
        # 仅保留 [vx, vy, angle_err, lat_dist]
        X_raw.append(data['x'][:, 2:]) 
        y.append(data['y'])
    
    X_raw = np.array(X_raw)
    y = np.array(y)

    N, T, F = X_raw.shape
    X_reshaped = X_raw.reshape(-1, F)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped).reshape(N, T, F)
    
    SCALER_SAVE_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"⚖️ 特征标准化完成，Scaler 已保存。")
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def build_v3_refined_model():
    model = models.Sequential([
        layers.Input(shape=(110, 4)),
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_v3_data()
    
    model = build_v3_refined_model()
    model.summary()
    
    print("\n🚀 V3.1 [4D-Refined] 大规模数据训练启动...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks
    )
    
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ 训练完成！Refined 模型已存至: {MODEL_SAVE_PATH}")
