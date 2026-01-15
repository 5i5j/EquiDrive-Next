import tensorflow as tf
from tensorflow.keras import layers, Model

def build_v2_model(input_shape=(110, 4), num_classes=3):
    """
    V2 模型：处理 (x, y, vx, vy)
    """
    inputs = layers.Input(shape=input_shape)
    
    # 增加一层层级，更好地处理多维特征
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x) # 增加 Dropout 防止过拟合
    
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="EquiDrive_V2_Advanced")
    return model

if __name__ == "__main__":
    model = build_v2_model()
    model.summary()