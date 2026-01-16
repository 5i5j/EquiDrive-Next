import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_local_map(json_path, parquet_path=None):
    with open(json_path, 'r') as f:
        map_data = json.load(f)
    
    lanes = map_data.get('lane_segments', {})
    
    plt.figure(figsize=(12, 12))
    
    # 1. 绘制所有车道中心线
    for lane_id, lane_props in lanes.items():
        centerline = lane_props.get('centerline', [])
        if centerline:
            # 提取 x, y 坐标
            xs = [pt['x'] for pt in centerline]
            ys = [pt['y'] for pt in centerline]
            plt.plot(xs, ys, color='gray', alpha=0.3, linestyle='--', linewidth=1)
            
    # 2. 如果提供了轨迹数据，把车也画上去
    if parquet_path and Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path)
        # 筛选出 Focal Agent
        focal_id = df['focal_track_id'].iloc[0]
        agent_df = df[df['track_id'] == focal_id]
        
        plt.plot(agent_df['position_x'], agent_df['position_y'], 
                 color='blue', linewidth=3, label='Focal Agent')
        plt.scatter(agent_df['position_x'].iloc[0], agent_df['position_y'].iloc[0], 
                    c='g', s=100, label='Start')
        plt.scatter(agent_df['position_x'].iloc[-1], agent_df['position_y'].iloc[-1], 
                    c='r', s=100, label='End')

    plt.title(f"V3 World View: {json_path.parent.name}")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 保存结果
    output_path = Path("plots/v3_world_view.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"🌍 上帝视角地图已保存至: {output_path}")

if __name__ == "__main__":
    scenario_dir = Path("data/landing/train/404dde4c-a8f5-4db8-bd00-984935fbb412")
    json_file = scenario_dir / "log_map_archive_404dde4c-a8f5-4db8-bd00-984935fbb412.json"
    parquet_file = scenario_dir / "scenario_404dde4c-a8f5-4db8-bd00-984935fbb412.parquet"
    
    plot_local_map(json_file, parquet_file)