"""
全局 Costmap 管理模块
负责全局 costmap 的融合、更新和重规划
"""

from pathlib import Path
import numpy as np
import cv2

from config import (
    GLOBAL_COSTMAP_RESOLUTION,
    GLOBAL_OBSTACLE_INFLATION_RADIUS,
    GLOBAL_REPLAN_SKIP_POINTS,
)
from coordinate_transform import dem_grid_to_local, local_to_world
from utils import generate_waypoints_with_elevation, calculate_yaw_between_points, ensure_dir
from visualization import (
    load_costmap_txt,
    save_costmap_txt,
    make_costmap_gray_image,
)


def world_to_global_costmap_cell(x_world, y_world, resolution=GLOBAL_COSTMAP_RESOLUTION):
    col = int(round(x_world / resolution))
    row = int(round(y_world / resolution))
    return col, row


def global_costmap_cell_to_world(col, row, resolution=GLOBAL_COSTMAP_RESOLUTION):
    x_world = col * resolution
    y_world = row * resolution
    return x_world, y_world


def inside_global_costmap(costmap, col, row):
    rows, cols = costmap.shape
    return 0 <= col < cols and 0 <= row < rows


def inflate_obstacle_to_global_costmap(global_costmap, col, row, radius=GLOBAL_OBSTACLE_INFLATION_RADIUS):
    rows, cols = global_costmap.shape
    x_min = max(0, col - radius)
    x_max = min(cols - 1, col + radius)
    y_min = max(0, row - radius)
    y_max = min(rows - 1, row + radius)
    global_costmap[y_min:y_max + 1, x_min:x_max + 1] = 1.0


def fuse_single_local_observation(global_costmap, observation):
    """
    将单个局部 costmap 观测融合到全局 costmap
    注意：排除局部 costmap 最外层（四条边）的障碍
    返回整个局部DEM在全局costmap中的范围框
    """
    raw_costmap_path = observation["raw_costmap_path"]
    current_x, current_y, current_yaw = observation["pose_xy_yaw"]
    dem_range = observation["dem_range"]
    resolution = observation["resolution"]

    local_costmap = load_costmap_txt(raw_costmap_path)
    rows, cols = local_costmap.shape
    
    # 创建掩码，排除最外层
    inner_mask = np.zeros_like(local_costmap, dtype=bool)
    inner_mask[1:rows-1, 1:cols-1] = True
    
    # 只考虑内部的障碍点
    obstacle_indices = np.argwhere((local_costmap >= 1.0) & inner_mask)

    # 计算整个局部DEM在全局costmap中的四个角点
    # 局部DEM的四个角（在局部坐标系中）
    corners_local = [
        (dem_range, -dem_range),   # 前左
        (dem_range, dem_range),    # 前右
        (-dem_range, dem_range),   # 后右
        (-dem_range, -dem_range),  # 后左
    ]
    
    # 转换到世界坐标，再转换到全局costmap栅格
    corners_global = []
    for x_local, y_local in corners_local:
        x_world, y_world = local_to_world(x_local, y_local, current_x, current_y, current_yaw)
        gcol, grow = world_to_global_costmap_cell(x_world, y_world)
        corners_global.append((gcol, grow))
    
    # 计算包围盒
    all_cols = [c for c, r in corners_global]
    all_rows = [r for c, r in corners_global]
    min_c, max_c = min(all_cols), max(all_cols)
    min_r, max_r = min(all_rows), max(all_rows)

    # 融合障碍
    for row, col in obstacle_indices:
        x_local, y_local = dem_grid_to_local(int(col), int(row), dem_range, resolution)
        x_world, y_world = local_to_world(x_local, y_local, current_x, current_y, current_yaw)
        gcol, grow = world_to_global_costmap_cell(x_world, y_world)

        if inside_global_costmap(global_costmap, gcol, grow):
            inflate_obstacle_to_global_costmap(global_costmap, gcol, grow, GLOBAL_OBSTACLE_INFLATION_RADIUS)

    return (min_c, min_r, max_c, max_r)


def build_fused_global_costmap(base_global_costmap, local_obstacle_observations):
    """构建融合后的全局 costmap"""
    fused = base_global_costmap.copy()
    all_boxes = []

    for obs in local_obstacle_observations:
        box = fuse_single_local_observation(fused, obs)
        if box is not None:
            all_boxes.append(box)

    return fused, all_boxes


def save_global_costmap_merge_visualization(costmap, current_box, output_image_path):
    """
    保存全局 costmap 融合可视化
    只显示当前（最后一次）融合的局部DEM范围的红框（正方形）
    """
    gray = make_costmap_gray_image(costmap)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 只画当前融合的框（整个局部DEM范围）
    if current_box:
        x1, y1, x2, y2 = current_box
        cv2.rectangle(color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(str(output_image_path), color)


def save_local_costmap_artifacts(costmap_path, dem_build_dir):
    """保存局部 costmap 相关文件到 dem_build 目录"""
    dem_build_dir = Path(dem_build_dir)
    costmap = load_costmap_txt(costmap_path)

    dem_costmap_vis = dem_build_dir / "dem_costmap_vis.jpg"

    gray = make_costmap_gray_image(costmap)
    cv2.imwrite(str(dem_costmap_vis), gray)
    print(f"已保存: {dem_costmap_vis}")

    return costmap_path, dem_costmap_vis


def save_global_merge_artifacts(base_global_costmap, local_obstacle_observations, dem_build_dir):
    """保存全局融合相关文件"""
    dem_build_dir = Path(dem_build_dir)

    fused_costmap, boxes = build_fused_global_costmap(base_global_costmap, local_obstacle_observations)

    merge_txt = dem_build_dir / "global_dem_costmap_merge.txt"
    merge_vis = dem_build_dir / "global_dem_costmap_merge_vis.jpg"

    save_costmap_txt(fused_costmap, merge_txt)
    
    # 只传入最后一个框（当前融合的局部DEM范围）
    current_box = boxes[-1] if boxes else None
    save_global_costmap_merge_visualization(fused_costmap, current_box, merge_vis)
    print(f"已保存: {merge_txt}")
    print(f"已保存: {merge_vis}")

    return fused_costmap, merge_txt, merge_vis


def build_waypoints_with_yaw_from_global_path_cells(path_points_cells):
    """从全局路径栅格点构建带航向的路径点"""
    xy_points = []
    for col, row in path_points_cells:
        x_world, y_world = global_costmap_cell_to_world(col, row)
        xy_points.append((x_world, y_world))

    waypoints = generate_waypoints_with_elevation(xy_points)

    waypoints_with_yaw = []
    for i, (x, y, z) in enumerate(waypoints):
        if i < len(waypoints) - 1:
            next_x, next_y, _ = waypoints[i + 1]
            yaw = calculate_yaw_between_points(x, y, next_x, next_y)
        else:
            yaw = waypoints_with_yaw[-1][3] if waypoints_with_yaw else 0.0
        waypoints_with_yaw.append((x, y, z, yaw))

    return waypoints_with_yaw


def merge_replanned_global_path(new_segment_waypoints, old_waypoints_with_yaw, reconnect_global_idx):
    """合并重规划后的全局路径"""
    merged = list(new_segment_waypoints)
    if reconnect_global_idx + 1 < len(old_waypoints_with_yaw):
        merged.extend(old_waypoints_with_yaw[reconnect_global_idx + 1:])

    refined = []
    for i, (x, y, z, _) in enumerate(merged):
        if i < len(merged) - 1:
            nx, ny, _, _ = merged[i + 1]
            yaw = calculate_yaw_between_points(x, y, nx, ny)
        else:
            yaw = refined[-1][3] if refined else 0.0
        refined.append((x, y, z, yaw))
    return refined


def save_global_replan_path_visualization(costmap_path, start, goal, path_points, output_path, old_waypoints=None):
    """
    保存全局重规划路径可视化
    原全局路径用蓝线显示，局部修改的全局路径用红线显示
    """
    costmap = load_costmap_txt(costmap_path)
    gray = make_costmap_gray_image(costmap)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 画起点和终点
    sx, sy = start
    gx, gy = goal
    cv2.circle(color, (sx, sy), 5, (255, 0, 0), -1)
    cv2.circle(color, (gx, gy), 5, (0, 255, 0), -1)
    
    # 画新路径（红线）
    if path_points and len(path_points) >= 2:
        pts = np.array(path_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(color, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    
    # 画旧全局路径（蓝线）- 如果有
    if old_waypoints and len(old_waypoints) >= 2:
        old_points = []
        for wp in old_waypoints:
            col, row = world_to_global_costmap_cell(wp[0], wp[1])
            old_points.append([col, row])
        old_pts = np.array(old_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(color, [old_pts], isClosed=False, color=(255, 0, 0), thickness=1)
    
    cv2.imwrite(str(output_path), color)


def run_global_replan(
    local_path_planner,
    base_global_costmap,
    local_obstacle_observations,
    current_world_xyz,
    waypoints_with_yaw,
    original_local_goal_idx,
    output_dir,
):
    """
    执行全局路径重规划
    返回: (status, new_waypoints_with_yaw, reconnect_idx)
    """
    from pathlib import Path
    from visualization import save_costmap_txt

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    fused_global_costmap, _ = build_fused_global_costmap(base_global_costmap, local_obstacle_observations)

    # 不再正式保存 global_costmap.txt，只生成一个临时文件供 C++ 重规划使用
    temp_fused_costmap_path = output_dir / "_temp_global_costmap_for_replan.txt"
    save_costmap_txt(fused_global_costmap, temp_fused_costmap_path)

    total_segments = len(waypoints_with_yaw) - 1
    reconnect_idx = min(original_local_goal_idx + GLOBAL_REPLAN_SKIP_POINTS, total_segments)
    reconnect_world = waypoints_with_yaw[reconnect_idx]

    start_col, start_row = world_to_global_costmap_cell(current_world_xyz[0], current_world_xyz[1])
    goal_col, goal_row = world_to_global_costmap_cell(reconnect_world[0], reconnect_world[1])

    status, path_points = local_path_planner.plan_path_from_costmap(
        costmap_path=temp_fused_costmap_path,
        start_col=start_col,
        start_row=start_row,
        goal_col=goal_col,
        goal_row=goal_row,
        output_dir=output_dir,
    )

    used_costmap_path = temp_fused_costmap_path

    if status in ("START_IS_OBSTACLE", "START_AND_GOAL_ARE_OBSTACLES", "NO_PATH_FOUND"):
        status, path_points, soften_costmap_path, used_radius = local_path_planner.plan_path_from_costmap_with_start_relaxation(
            costmap_path=temp_fused_costmap_path,
            start_col=start_col,
            start_row=start_row,
            goal_col=goal_col,
            goal_row=goal_row,
            output_dir=output_dir,
            revision_filename="soften_global_costmap.txt",
            stop_when_start_cleared_only=False,
        )
        used_costmap_path = soften_costmap_path
        print(f"全局重规划起点软化结束，状态: {status}，半径: {used_radius}")

    path_vis_path = output_dir / "path.jpg"
    save_global_replan_path_visualization(
        costmap_path=used_costmap_path,
        start=(start_col, start_row),
        goal=(goal_col, goal_row),
        path_points=path_points,
        output_path=path_vis_path,
        old_waypoints=waypoints_with_yaw if status == "OK" else None
    )

    if status != "OK" or not path_points:
        if temp_fused_costmap_path.exists():
            try:
                temp_fused_costmap_path.unlink()
            except Exception:
                pass
        return status, None, reconnect_idx

    new_segment_waypoints = build_waypoints_with_yaw_from_global_path_cells(path_points)
    merged_waypoints = merge_replanned_global_path(new_segment_waypoints, waypoints_with_yaw, reconnect_idx)

    final_path_txt = output_dir / "path.txt"
    text = "->".join(f"({c},{r})" for c, r in path_points)
    final_path_txt.write_text(text, encoding="utf-8")

    # 删除临时全局融合 costmap，不再保留
    if temp_fused_costmap_path.exists():
        try:
            temp_fused_costmap_path.unlink()
        except Exception:
            pass

    return "OK", merged_waypoints, reconnect_idx