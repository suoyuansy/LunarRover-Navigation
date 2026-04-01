"""
局部终点选择、候选终点生成、评分与重规划逻辑
评分公式: J = w6*S_future - w1*d_goal - w2*d_path_dev - w3*L_path - w4*C_clearance - w5*C_heading - w7*C_turn
评分越高越好
"""

import math
import shutil
from pathlib import Path

import numpy as np

from config import (
    LOCAL_DEM_RANGE,
    LOCAL_DEM_RESOLUTION,
    LOCAL_GOAL_LOOKAHEAD_POINTS,
    CANDIDATE_SEARCH_MAX_RADIUS_CELLS,
    CLEARANCE_NEIGHBORHOOD_RADIUS_CELLS,
    LOCAL_PLANNER_METHOD_PHASE1,
)
from coordinate_transform import world_to_local, local_to_dem_grid, dem_grid_to_local, local_to_world
from utils import calculate_distance, calculate_yaw_between_points, normalize_angle, ensure_dir
from visualization import load_costmap_txt, save_ranked_path_visualization


# ========================================
# 评分权重配置（可以根据需要调整）
# ========================================
# 惩罚项权重（越大表示越不希望该项发生）
W_D_GOAL = 1.0          # 候选终点离原终点的距离
W_PATH_DEV = 2.0        # 局部路径到全局路径片段的平均偏差
W_PATH_LEN = 1.0        # 局部路径长度
W_CLEARANCE = 1.5       # 终点附近障碍密度
W_HEADING = 1.0         # 到达终点时姿态与后续全局路径方向的差异
W_TURN = 0.5            # 转向代价（平滑性）

# 奖励项权重（越大表示越希望该项发生）
W_FUTURE = 2.0          # 下一轮 DEM 能覆盖到的后续全局路径收益

# ========================================
# DEM 边界安全边距（栅格数）
# ========================================
BOUNDARY_SAFE_MARGIN_CELLS = 8  # 距离 DEM 边界至少 5 个栅格


def inside_grid(col, row, cols, rows):
    """检查栅格是否在范围内"""
    return 0 <= col < cols and 0 <= row < rows


def is_safe_from_boundary(col, row, grid_size, min_margin_cells=BOUNDARY_SAFE_MARGIN_CELLS):
    """
    检查栅格是否远离边界至少 min_margin_cells 个栅格

    参数:
        col, row: 栅格坐标
        grid_size: DEM 栅格尺寸（假设为正方形）
        min_margin_cells: 距离边界的最小栅格数（默认2）
    """
    return (
        min_margin_cells <= col < grid_size - min_margin_cells and
        min_margin_cells <= row < grid_size - min_margin_cells
    )


def waypoint_to_dem_grid(waypoint_xyz, current_pose_xy_yaw, dem_range, resolution):
    """将路径点转换为 DEM 栅格坐标"""
    current_x, current_y, current_yaw = current_pose_xy_yaw
    gx, gy = waypoint_xyz[0], waypoint_xyz[1]
    x_local, y_local = world_to_local(gx, gy, current_x, current_y, current_yaw)
    col, row = local_to_dem_grid(x_local, y_local, dem_range, resolution)
    return x_local, y_local, col, row


def choose_local_goal_index(
    waypoints_with_yaw,
    current_global_idx,
    current_pose_xy_yaw,
    dem_range=LOCAL_DEM_RANGE,
    resolution=LOCAL_DEM_RESOLUTION,
    lookahead_points=LOCAL_GOAL_LOOKAHEAD_POINTS,
):
    """
    选择局部规划终点：
    条件：
    1. 在 DEM 范围内
    2. 距离 DEM 各个边界 >= BOUNDARY_SAFE_MARGIN_CELLS (默认2个栅格)
    3. 在满足以上条件的点中，选择序列号最大的（最远的）

    如果最终终点满足条件，优先选择最终终点
    """
    total_segments = len(waypoints_with_yaw) - 1
    grid_size = int((dem_range * 2) / resolution)

    # 首先检查最终终点
    final_idx = total_segments
    final_wp = waypoints_with_yaw[final_idx]
    _, _, final_col, final_row = waypoint_to_dem_grid(final_wp, current_pose_xy_yaw, dem_range, resolution)

    # 检查最终终点是否在 DEM 范围内，且距离边界 >= 安全边距
    if (inside_grid(final_col, final_row, grid_size, grid_size) and
            is_safe_from_boundary(final_col, final_row, grid_size)):
        print(
            f"【终点选择】选择最终终点: 索引={final_idx}, 栅格=({final_col}, {final_row}), "
            f"距离边界 >= {BOUNDARY_SAFE_MARGIN_CELLS} 栅格"
        )
        return final_idx, final_col, final_row

    # 向后查找 lookahead_points 个点
    max_check_idx = min(current_global_idx + lookahead_points, total_segments)
    best_idx = None
    best_col = None
    best_row = None

    for idx in range(current_global_idx + 1, max_check_idx + 1):
        wp = waypoints_with_yaw[idx]
        _, _, col, row = waypoint_to_dem_grid(wp, current_pose_xy_yaw, dem_range, resolution)

        if not inside_grid(col, row, grid_size, grid_size):
            continue

        if not is_safe_from_boundary(col, row, grid_size):
            print(f"【终点选择】跳过索引 {idx}: 栅格=({col}, {row}), 距离边界过近")
            continue

        best_idx = idx
        best_col = col
        best_row = row

    if best_idx is not None:
        print(
            f"【终点选择】选择局部目标: 索引={best_idx}, 栅格=({best_col}, {best_row}), "
            f"距离边界 >= {BOUNDARY_SAFE_MARGIN_CELLS} 栅格"
        )
        return best_idx, best_col, best_row

    print(f"【终点选择】警告：未找到距离边界 >= {BOUNDARY_SAFE_MARGIN_CELLS} 个栅格的安全目标点")
    return None


def generate_candidate_goals(costmap, original_goal, max_radius=CANDIDATE_SEARCH_MAX_RADIUS_CELLS):
    """
    在原终点周围搜索所有非障碍候选点
    不再做 Python 侧连通性检查，直接返回候选点列表
    """
    gx, gy = original_goal
    rows, cols = costmap.shape
    candidates = []
    seen = set()

    for radius in range(1, max_radius + 1):
        x_min = max(0, gx - radius)
        x_max = min(cols - 1, gx + radius)
        y_min = max(0, gy - radius)
        y_max = min(rows - 1, gy + radius)

        shell_points = []
        for x in range(x_min, x_max + 1):
            shell_points.append((x, y_min))
            shell_points.append((x, y_max))
        for y in range(y_min + 1, y_max):
            shell_points.append((x_min, y))
            shell_points.append((x_max, y))

        shell_points = list(dict.fromkeys(shell_points))
        shell_points.sort(key=lambda p: math.hypot(p[0] - gx, p[1] - gy))

        for col, row in shell_points:
            if (col, row) in seen:
                continue
            seen.add((col, row))

            if costmap[row, col] >= 1.0:
                continue

            candidates.append((col, row))

    return candidates


def dem_path_to_world_path(path_points_dem, current_pose_xy_yaw, start_z_build, dem_grid, dem_range, resolution, lidar_height_offset):
    """将 DEM 路径点转换为世界坐标路径"""
    world_path = []
    current_x, current_y, current_yaw = current_pose_xy_yaw

    for col, row in path_points_dem:
        x_local, y_local = dem_grid_to_local(col, row, dem_range, resolution)
        x_world, y_world = local_to_world(x_local, y_local, current_x, current_y, current_yaw)
        z_relative = dem_grid[row, col]
        z_world = start_z_build + z_relative - lidar_height_offset
        world_path.append((x_world, y_world, z_world))

    return world_path


def compute_path_length_world(path_world):
    """计算世界坐标路径长度"""
    if path_world is None or len(path_world) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(path_world)):
        length += calculate_distance(
            path_world[i - 1][0],
            path_world[i - 1][1],
            path_world[i][0],
            path_world[i][1],
        )
    return length


def compute_turn_penalty(path_world):
    """
    计算路径转向惩罚（平滑性）
    返回路径的总转向角度（弧度）
    """
    if path_world is None or len(path_world) < 3:
        return 0.0

    total_turn = 0.0
    for i in range(1, len(path_world) - 1):
        x0, y0 = path_world[i - 1][0], path_world[i - 1][1]
        x1, y1 = path_world[i][0], path_world[i][1]
        x2, y2 = path_world[i + 1][0], path_world[i + 1][1]
        yaw1 = calculate_yaw_between_points(x0, y0, x1, y1)
        yaw2 = calculate_yaw_between_points(x1, y1, x2, y2)
        total_turn += abs(normalize_angle(yaw2 - yaw1))
    return total_turn


def get_global_fragment(waypoints_with_yaw, current_global_idx, original_goal_idx):
    """获取全局路径片段（从当前位置到原终点）"""
    frag = []
    for idx in range(current_global_idx, min(original_goal_idx + 1, len(waypoints_with_yaw))):
        frag.append(waypoints_with_yaw[idx])
    return frag


def compute_path_deviation(path_world, global_fragment):
    """
    计算路径偏离度
    局部路径到全局路径片段的平均偏差
    """
    if not path_world or not global_fragment:
        return 0.0

    deviations = []
    global_xy = [(p[0], p[1]) for p in global_fragment]
    for px, py, _ in path_world:
        d = min(calculate_distance(px, py, gx, gy) for gx, gy in global_xy)
        deviations.append(d)

    return float(np.mean(deviations)) if deviations else 0.0


def compute_clearance_penalty(costmap, goal, neighborhood_radius=CLEARANCE_NEIGHBORHOOD_RADIUS_CELLS):
    """
    计算终点附近障碍密度 / 安全余量代价
    返回 [0, 1] 之间的值，越大表示障碍越多/越不安全
    """
    gx, gy = goal
    rows, cols = costmap.shape
    x_min = max(0, gx - neighborhood_radius)
    x_max = min(cols - 1, gx + neighborhood_radius)
    y_min = max(0, gy - neighborhood_radius)
    y_max = min(rows - 1, gy + neighborhood_radius)

    local_patch = costmap[y_min:y_max + 1, x_min:x_max + 1]
    if local_patch.size == 0:
        return 1.0

    obstacle_ratio = float(np.mean(local_patch >= 1.0))
    return obstacle_ratio


def compute_heading_penalty(path_world, waypoints_with_yaw, original_goal_idx):
    """
    计算航向偏差惩罚
    到达终点时姿态与后续全局路径方向的差异
    """
    if path_world is None or len(path_world) < 2:
        return math.pi

    end_prev = path_world[-2]
    end_curr = path_world[-1]
    path_heading = calculate_yaw_between_points(end_prev[0], end_prev[1], end_curr[0], end_curr[1])

    if original_goal_idx < len(waypoints_with_yaw) - 1:
        g0 = waypoints_with_yaw[original_goal_idx]
        g1 = waypoints_with_yaw[min(original_goal_idx + 1, len(waypoints_with_yaw) - 1)]
        global_heading = calculate_yaw_between_points(g0[0], g0[1], g1[0], g1[1])
    else:
        g0 = waypoints_with_yaw[max(0, original_goal_idx - 1)]
        g1 = waypoints_with_yaw[original_goal_idx]
        global_heading = calculate_yaw_between_points(g0[0], g0[1], g1[0], g1[1])

    return abs(normalize_angle(path_heading - global_heading))


def compute_future_benefit(
    path_world,
    original_goal_idx,
    waypoints_with_yaw,
    dem_range=LOCAL_DEM_RANGE,
    resolution=LOCAL_DEM_RESOLUTION,
):
    """
    计算未来收益（奖励项）
    到达新目标点后，新 DEM 能覆盖到的“原终点之后”的全局路径点数量
    """
    grid_size = int((dem_range * 2) / resolution)
    count = 0

    if not path_world:
        return 0

    if len(path_world) >= 2:
        yaw = calculate_yaw_between_points(
            path_world[-2][0], path_world[-2][1],
            path_world[-1][0], path_world[-1][1],
        )
    else:
        yaw = 0.0

    pose_xy_yaw = (path_world[-1][0], path_world[-1][1], yaw)

    for idx in range(original_goal_idx + 1, len(waypoints_with_yaw)):
        wp = waypoints_with_yaw[idx]
        _, _, col, row = waypoint_to_dem_grid(wp, pose_xy_yaw, dem_range, resolution)
        if inside_grid(col, row, grid_size, grid_size) and is_safe_from_boundary(col, row, grid_size):
            count += 1

    return count


def normalize_metrics(metric_list, key):
    """
    对指定指标进行归一化处理（Min-Max归一化）
    结果映射到 [0, 1] 区间
    """
    values = [float(m[key]) for m in metric_list]
    vmin = min(values)
    vmax = max(values)

    if abs(vmax - vmin) < 1e-9:
        for m in metric_list:
            m[f"norm_{key}"] = 0.0
        return

    for m in metric_list:
        m[f"norm_{key}"] = (float(m[key]) - vmin) / (vmax - vmin)


def score_candidates(candidate_results, waypoints_with_yaw, current_global_idx, original_goal_idx):
    """
    对候选终点进行评分
    评分公式: J = w6*S_future - w1*d_goal - w2*d_path_dev - w3*L_path - w4*C_clearance - w5*C_heading - w7*C_turn
    评分越高越好
    """
    global_fragment = get_global_fragment(waypoints_with_yaw, current_global_idx, original_goal_idx)
    metrics = []

    for item in candidate_results:
        original_goal_world = item["original_goal_world"]
        candidate_goal_world = item["candidate_goal_world"]
        path_world = item["path_world"]
        costmap = item["costmap"]
        goal_dem = item["candidate_goal_dem"]

        metric = {
            "item": item,
            "d_goal": calculate_distance(
                candidate_goal_world[0], candidate_goal_world[1],
                original_goal_world[0], original_goal_world[1],
            ),
            "d_path_dev": compute_path_deviation(path_world, global_fragment),
            "L_path": compute_path_length_world(path_world),
            "C_clearance": compute_clearance_penalty(costmap, goal_dem),
            "C_heading": compute_heading_penalty(path_world, waypoints_with_yaw, original_goal_idx),
            "C_turn": compute_turn_penalty(path_world),
            "S_future": compute_future_benefit(path_world, original_goal_idx, waypoints_with_yaw),
        }
        metrics.append(metric)

    for key in ["d_goal", "d_path_dev", "L_path", "C_clearance", "C_heading", "C_turn", "S_future"]:
        normalize_metrics(metrics, key)

    for metric in metrics:
        reward = W_FUTURE * metric["norm_S_future"]
        penalty = (
            W_D_GOAL * metric["norm_d_goal"] +
            W_PATH_DEV * metric["norm_d_path_dev"] +
            W_PATH_LEN * metric["norm_L_path"] +
            W_CLEARANCE * metric["norm_C_clearance"] +
            W_HEADING * metric["norm_C_heading"] +
            W_TURN * metric["norm_C_turn"]
        )

        metric["score"] = reward - penalty
        metric["reward"] = reward
        metric["penalty"] = penalty

    metrics.sort(key=lambda x: x["score"], reverse=True)
    return metrics


def write_path_txt(path_points_dem, output_path):
    """写入路径文件"""
    output_path = Path(output_path)
    text = "->".join(f"({col},{row})" for col, row in path_points_dem)
    output_path.write_text(text, encoding="utf-8")


def export_ranked_candidates(metrics, replan_dir, costmap_path):
    """导出排序后的候选结果（按评分从高到低排序）"""
    replan_dir = Path(replan_dir)
    ensure_dir(replan_dir)

    for rank, metric in enumerate(metrics, start=1):
        item = metric["item"]

        ranked_path_file = replan_dir / f"{rank}_path.txt"
        ranked_vis_file = replan_dir / f"{rank}_path_vis.jpg"
        ranked_score_file = replan_dir / f"{rank}_score.txt"

        write_path_txt(item["path_dem"], ranked_path_file)

        save_ranked_path_visualization(
            costmap_path=costmap_path,
            start=item["start_dem"],
            goal=item["candidate_goal_dem"],
            path_points=item["path_dem"],
            output_image_path=ranked_vis_file,
        )

        with open(ranked_score_file, "w", encoding="utf-8") as f:
            f.write(f"rank = {rank}\n")
            f.write(f"score = {metric['score']:.6f}\n")
            f.write(f"reward = {metric['reward']:.6f}\n")
            f.write(f"penalty = {metric['penalty']:.6f}\n")
            f.write(f"d_goal = {metric['d_goal']:.6f} (norm={metric['norm_d_goal']:.6f})\n")
            f.write(f"d_path_dev = {metric['d_path_dev']:.6f} (norm={metric['norm_d_path_dev']:.6f})\n")
            f.write(f"L_path = {metric['L_path']:.6f} (norm={metric['norm_L_path']:.6f})\n")
            f.write(f"C_clearance = {metric['C_clearance']:.6f} (norm={metric['norm_C_clearance']:.6f})\n")
            f.write(f"C_heading = {metric['C_heading']:.6f} (norm={metric['norm_C_heading']:.6f})\n")
            f.write(f"C_turn = {metric['C_turn']:.6f} (norm={metric['norm_C_turn']:.6f})\n")
            f.write(f"S_future = {metric['S_future']:.6f} (norm={metric['norm_S_future']:.6f})\n")
            f.write(f"candidate_goal_dem = {item['candidate_goal_dem']}\n")


def plan_with_goal_adjustment(
    local_path_planner,
    dem_grid,
    start_dem,
    original_goal_dem,
    original_goal_world,
    current_pose_xy_yaw,
    start_z_build,
    waypoints_with_yaw,
    current_global_idx,
    original_goal_idx,
    lidar_height_offset,
    base_costmap_path,
    output_dir,
    resolution=LOCAL_DEM_RESOLUTION,
    dem_range=LOCAL_DEM_RANGE,
):
    """
    终点挪动策略
    返回: (status, best_item 或 None)
    """
    costmap_path = Path(base_costmap_path)
    output_dir = Path(output_dir)

    if not costmap_path.exists():
        return "NO_COSTMAP", None

    costmap = load_costmap_txt(costmap_path)
    candidates = generate_candidate_goals(costmap, original_goal_dem)

    if len(candidates) == 0:
        return "NO_CANDIDATE_GOAL", None

    ensure_dir(output_dir)

    work_dir = output_dir / "_working"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    ensure_dir(work_dir)

    candidate_results = []

    # 遍历所有候选点，使用 BidirectionalAStar 检查通路
    for candidate_goal in candidates:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        ensure_dir(work_dir)

        status, path_points_dem = local_path_planner.plan_path_from_costmap(
            costmap_path=costmap_path,
            start_col=start_dem[0],
            start_row=start_dem[1],
            goal_col=candidate_goal[0],
            goal_row=candidate_goal[1],
            output_dir=work_dir,
            method=LOCAL_PLANNER_METHOD_PHASE1,  # BidirectionalAStar
        )

        if status != "OK" or not path_points_dem:
            continue

        path_world = dem_path_to_world_path(
            path_points_dem=path_points_dem,
            current_pose_xy_yaw=current_pose_xy_yaw,
            start_z_build=start_z_build,
            dem_grid=dem_grid,
            dem_range=dem_range,
            resolution=resolution,
            lidar_height_offset=lidar_height_offset,
        )
        goal_world = path_world[-1]

        candidate_results.append({
            "status": status,
            "candidate_goal_dem": candidate_goal,
            "candidate_goal_world": goal_world,
            "original_goal_world": original_goal_world,
            "path_dem": path_points_dem,
            "path_world": path_world,
            "costmap": costmap,
            "start_dem": start_dem,
        })

    if work_dir.exists():
        shutil.rmtree(work_dir)

    if not candidate_results:
        return "NO_VALID_REPLAN_PATH", None

    metrics = score_candidates(candidate_results, waypoints_with_yaw, current_global_idx, original_goal_idx)
    export_ranked_candidates(metrics, output_dir, costmap_path)

    best_item = metrics[0]["item"]
    best_item["ranked_metrics"] = metrics[0]
    return "OK", best_item