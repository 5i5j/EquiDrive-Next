import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_v1_tensor(parquet_path):
    df = pd.read_parquet(parquet_path)
    
    # 核心复习点：归一化 (减去第一帧)
    x0 = df['position_x'].iloc[0]
    y0 = df['position_y'].iloc[0]
    
    rel_x = df['position_x'].values - x0
    rel_y = df['position_y'].values - y0
    
    # 组合成 (110, 2)
    features = np.stack([rel_x, rel_y], axis=-1).astype(np.float32)
    
    return features

def main():
    bronze_dir = Path("data/bronze")
    silver_dir = Path("data/silver_v1") # 标记为 v1 方便对比
    silver_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义标签映射
    label_map = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2}
    
    print("💎 正在将 Bronze 转化为 Silver V1 (110, 2) 张量...")
    
    for label_str, label_idx in label_map.items():
        files = list((bronze_dir / label_str).glob("*.parquet"))
        for f in tqdm(files, desc=f"Processing {label_str}"):
            feat = process_v1_tensor(f)
            # 保存为压缩的 npz
            np.savez_compressed(silver_dir / f"{f.stem}.npz", x=feat, y=label_idx)

if __name__ == "__main__":
    main()