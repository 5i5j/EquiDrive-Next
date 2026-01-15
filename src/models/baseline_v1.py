import tensorflow as tf
from tensorflow.keras import layers, Model

def build_v1_model(input_shape=(110, 2), num_classes=3):
    """
    【复习重点】V1 Baseline 架构：
    1. 输入层：接收 (110, 2)
    2. LSTM 层：提取时间序列特征
    3. 全连接层：映射到分类空间
    4. Softmax：输出概率分布
    """
    inputs = layers.Input(shape=input_shape)
    
    # LSTM 层：64个单元
    # return_sequences=False 表示我们只需要最后一个时间步的输出（即整条轨迹的总结）
    x = layers.LSTM(64, return_sequences=False)(inputs)
    
    # 增加一个全连接层来“消化”特征
    x = layers.Dense(32, activation='relu')(x)
    
    # 输出层：3个类别（左、右、直）
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="EquiDrive_V1_Baseline")
    return model

if __name__ == "__main__":
    # 打印模型结构，确认参数量
    model = build_v1_model()
    model.summary()