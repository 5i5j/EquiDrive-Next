import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.map_geometry_v3 import extract_v3_features

# --- 配置参数 ---
INPUT_DIR = Path("data/landing/train")
OUTPUT_DIR = Path("data/silver_v3")
QUOTA_PER_CLASS = 10000  # 每类采集2000个，总计6000个
MAX_WORKERS = os.cpu_count() // 2  # 使用一半的核心，留一半陪老大弹琴

def process_single_scene(scene_path):
    """处理单个场景的逻辑，供并行调用"""
    try:
        parquet_file = list(scene_path.glob("*.parquet"))[0]
        json_file = list(scene_path.glob("*.json"))[0]
        
        # 1. 快速判定标签 (Heading Delta)
        df = pd.read_parquet(parquet_file)
        focal_id = df['focal_track_id'].iloc[0]
        agent_df = df[df['track_id'] == focal_id]
        
        if len(agent_df) < 110:
            return None
        
        agent_df = agent_df.iloc[:110]
        start_h = agent_df['heading'].iloc[0]
        end_h = agent_df['heading'].iloc[-1]
        delta = (end_h - start_h + np.pi) % (2 * np.pi) - np.pi
        
        label = 2 # STRAIGHT
        if delta > 0.15: label = 0 # LEFT
        elif delta < -0.15: label = 1 # RIGHT
        
        # 2. 提取特征
        with open(json_file, 'r') as f:
            map_data = json.load(f)
        
        start_pos = agent_df[['position_x', 'position_y']].iloc[0].values
        lanes = map_data.get('lane_segments', {})
        
        # 寻找最近车道中心线
        best_line = None
        min_d = float('inf')
        for props in lanes.values():
            line = np.array([[pt['x'], pt['y']] for pt in props['centerline']])
            d = np.min(np.linalg.norm(line - start_pos, axis=1))
            if d < min_d:
                min_d = d
                best_line = line
        
        if best_line is None: return None

        features = []
        for i in range(110):
            row = agent_df.iloc[i]
            v_state = [row['position_x'], row['position_y'], row['heading']]
            # 调用咱们刚写的几何算法
            err, lat = extract_v3_features(v_state, best_line)
            
            features.append([
                row['position_x'], row['position_y'], 
                row['velocity_x'], row['velocity_y'],
                err, lat
            ])
            
        return {
            'scene_id': scene_path.name,
            'x': np.array(features, dtype=np.float32),
            'y': label
        }
    except Exception:
        return None

def build_v3_silver_dataset():
    OUTPUT_DIR.mkdir(exist_ok=True)
    # 使用 iglob 代替 list，实现流式扫描，解决 20 万文件扫描慢的问题
    import glob
    scenarios_iter = Path(INPUT_DIR).iterdir()
    
    counts = {0: 0, 1: 0, 2: 0}
    print(f"🚀 多核流水线已就绪 (Workers: {MAX_WORKERS})，正在流式分发任务...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 逐个提交任务，而不是等 20 万个列完
        future_to_scene = {}
        
        for scene_path in scenarios_iter:
            if not scene_path.is_dir(): continue
            
            # 提交任务
            future = executor.submit(process_single_scene, scene_path)
            future_to_scene[future] = scene_path
            
            # 这里的逻辑：每积压 100 个任务就处理一波结果，防止内存溢出
            if len(future_to_scene) > MAX_WORKERS * 2:
                for f in as_completed(future_to_scene):
                    result = f.result()
                    if result:
                        label = result['y']
                        if counts[label] < QUOTA_PER_CLASS:
                            save_path = OUTPUT_DIR / f"{result['scene_id']}.npz"
                            np.savez(save_path, x=result['x'], y=result['y'])
                            counts[label] += 1
                            print(f"\r📊 实时进度: [L:{counts[0]} R:{counts[1]} S:{counts[2]}] 总计:{sum(counts.values())}", end="", flush=True)
                    
                    del future_to_scene[f]
                    break # 跳出内层，继续分发新任务
            
            # 熔断退出
            if all(c >= QUOTA_PER_CLASS for c in counts.values()):
                break

    print(f"\n✅ V3 提炼完成！")

if __name__ == "__main__":
    build_v3_silver_dataset()