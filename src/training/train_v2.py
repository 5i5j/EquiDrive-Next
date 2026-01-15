import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.models.baseline_v2 import build_v2_model

def run_v2():
    # 1. Load 4D Data
    files = list(Path("data/silver_v2").glob("*.npz"))
    X, y = [], []
    for f in files:
        data = np.load(f)
        X.append(data['x'])
        y.append(data['y'])
    
    X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), test_size=0.2)

    # 2. Build & Compile
    model = build_v2_model(input_shape=(110, 4))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 3. Ignite 3080 Ti
    print("\n🔥 V2 (Balanced & 4D) Training Started...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
    
    model.save("models/v2_advanced.h5")
    print("\n✅ V2 Mission Accomplished!")

if __name__ == "__main__":
    run_v2()