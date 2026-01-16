import numpy as np

def extract_v3_features(vehicle_state, centerline_points):
    """
    vehicle_state: [x, y, heading]
    centerline_points: np.array([[x, y], ...]) 
    """
    v_pos = vehicle_state[:2]
    v_heading = vehicle_state[2]

    # 1. 找到车道中心线上离车最近的点
    dists = np.linalg.norm(centerline_points - v_pos, axis=1)
    closest_idx = np.argmin(dists)
    closest_pt = centerline_points[closest_idx]

    # 2. 计算【横向偏移量 (Lateral Distance)】
    # 原理：计算车辆到中心线切线的垂直距离
    # 找下一个点来确定切线方向
    next_idx = min(closest_idx + 1, len(centerline_points) - 1)
    lane_vec = centerline_points[next_idx] - closest_pt
    lane_heading = np.arctan2(lane_vec[1], lane_vec[0])
    
    # 向量叉乘原理算点到线距离（带正负号，代表左右偏移）
    # 简化版：计算车辆相对于车道线的相对坐标
    rel_pos = v_pos - closest_pt
    # 旋转矩阵将相对位置转到车道坐标系
    cos_h, sin_h = np.cos(-lane_heading), np.sin(-lane_heading)
    lateral_dist = rel_pos[0] * sin_h + rel_pos[1] * cos_h

    # 3. 计算【航向夹角误差 (Heading Error)】
    angle_error = v_heading - lane_heading
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi # 标准化到 [-pi, pi]

    return angle_error, lateral_dist

# 测试代码
# 假设车在 (0.5, 0)，车道中心线在 (0, 0) -> (10, 0)，车头正北
print(f"📐 航向误差: {extract_v3_features([0,0,np.pi/2], np.array([[0,0],[10,0]]))[0]:.2f} rad")
print(f"📏 横向偏移: {extract_v3_features([0.5,0,np.pi/2], np.array([[0,0],[10,0]]))[1]:.2f} m")