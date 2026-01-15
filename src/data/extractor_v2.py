import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_v2_features():
    bronze_dir = Path("data/bronze")
    silver_dir = Path("data/silver_v2")
    silver_dir.mkdir(parents=True, exist_ok=True)
    
    label_map = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2}
    all_files = list(bronze_dir.rglob("*.parquet"))
    
    print(f"💎 正在提炼 4D 特征 (x, y, vx, vy) -> Silver V2...")

    for f in tqdm(all_files):
        df = pd.read_parquet(f)
        label = label_map[f.parent.name]
        
        # 1. 位置归一化：以第一帧为原点 (0,0)
        x0, y0 = df['position_x'].iloc[0], df['position_y'].iloc[0]
        rel_x = df['position_x'].values - x0
        rel_y = df['position_y'].values - y0
        
        # 2. 提取速度：直接使用原始速度 (vx, vy)
        vx = df['velocity_x'].values
        vy = df['velocity_y'].values
        
        # 3. 堆叠成 (110, 4) 张量
        features = np.stack([rel_x, rel_y, vx, vy], axis=-1)
        
        # 4. 保存为 NPZ
        np.savez_compressed(silver_dir / f"{f.stem}.npz", x=features, y=label)

if __name__ == "__main__":
    extract_v2_features()