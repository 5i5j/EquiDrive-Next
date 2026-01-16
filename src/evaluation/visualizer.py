import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import random

def run_visual_inference():
    # 1. 创建保存目录
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    # 2. 加载模型
    model_path = "models/v2_advanced.h5"
    model = tf.keras.models.load_model(model_path)
    
    # 3. 采样数据
    silver_dir = Path("data/silver_v2")
    test_files = list(silver_dir.glob("*.npz"))
    sample_file = random.choice(test_files)
    
    data = np.load(sample_file)
    x_input = data['x']  # (110, 4) -> x, y, vx, vy
    y_true = data['y']
    label_names = ["LEFT", "RIGHT", "STRAIGHT"]
    
    # 4. 推理
    prediction = model.predict(x_input[np.newaxis, ...], verbose=0)
    y_pred = np.argmax(prediction)
    confidence = np.max(prediction)

    # 5. 绘图
    plt.figure(figsize=(10, 8))
    # 画出轨迹
    plt.plot(x_input[:, 0], x_input[:, 1], 'b-', label='Vehicle Path', linewidth=2)
    # 画出起始点和终点
    plt.scatter(x_input[0, 0], x_input[0, 1], c='g', s=100, label='Start (t=0)') 
    plt.scatter(x_input[-1, 0], x_input[-1, 1], c='r', s=100, label='End (t=11s)')
    
    # 画出速度矢量 (展示 4D 特征的魔力)
    # 每隔 20 个点画一个箭头
    for i in range(0, 110, 20):
        plt.arrow(x_input[i, 0], x_input[i, 1], x_input[i, 2], x_input[i, 3], 
                  head_width=0.2, color='orange', alpha=0.5)

    res_color = 'green' if y_pred == y_true else 'red'
    plt.title(f"V2 Advanced Inference\nTrue: {label_names[y_true]} | Pred: {label_names[y_pred]} ({confidence:.1%})", 
              color=res_color, fontsize=14)
    plt.xlabel("Relative X (meters)")
    plt.ylabel("Relative Y (meters)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')

    # 6. 保存并退出
    save_path = plot_dir / f"pred_{sample_file.stem}.png"
    plt.savefig(save_path)
    print(f"✅ 预测成功！图像已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_visual_inference()