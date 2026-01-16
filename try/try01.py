import json
from pathlib import Path

def check_lane_attributes(json_path):
    with open(json_path, 'r') as f:
        map_data = json.load(f)
    
    lanes = map_data.get('lane_segments', {})
    # 打印前几个车道的详细属性，看看有没有 'turn_direction' 或 'lane_type'
    for i, (lane_id, props) in enumerate(lanes.items()):
        if i < 5:  # 采样前5条
            print(f"ID: {lane_id} | Type: {props.get('lane_type')} | Exit: {props.get('exit_departure')}")

if __name__ == "__main__":
    json_path = Path("data/landing/train/404dde4c-a8f5-4db8-bd00-984935fbb412/log_map_archive_404dde4c-a8f5-4db8-bd00-984935fbb412.json")
    check_lane_attributes(json_path)