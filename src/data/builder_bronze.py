import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

def classify_intent(df):
    h_start = df['heading'].iloc[0]
    h_end = df['heading'].iloc[-1]
    # 计算航向角差值，处理 [-pi, pi] 突变
    delta_h = (h_end - h_start + np.pi) % (2 * np.pi) - np.pi
    if delta_h > 0.20: return "LEFT"
    if delta_h < -0.20: return "RIGHT"
    return "STRAIGHT"

def build_balanced_bronze():
    src_root = "data/landing/train"
    dest_root = Path("data/bronze")
    LIMIT = 2000 
    counts = {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0}

    print("🔍 正在深度扫描 AV2 格式数据...")
    
    scenario_files = []
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.startswith("scenario_") and file.endswith(".parquet"):
                scenario_files.append(os.path.join(root, file))
        if len(scenario_files) >= 60000: break # 预留足够样本

    print(f"⚖️ 开始均衡处理 {len(scenario_files)} 个备选文件...")

    for f_path in tqdm(scenario_files):
        if all(c >= LIMIT for c in counts.values()): break
            
        try:
            df = pd.read_parquet(f_path)
            
            # 修正核心逻辑：AV2 通过 focal_track_id 识别主角
            focal_id = df['focal_track_id'].iloc[0]
            focal_df = df[df['track_id'] == focal_id]
            
            if len(focal_df) < 110: continue
                
            intent = classify_intent(focal_df)
            
            if counts[intent] < LIMIT:
                save_dir = dest_root / intent
                save_dir.mkdir(parents=True, exist_ok=True)
                focal_df.to_parquet(save_dir / os.path.basename(f_path))
                counts[intent] += 1
        except Exception as e:
            continue

    print(f"\n✅ 成功！最终分布: {counts}")

if __name__ == "__main__":
    build_balanced_bronze()