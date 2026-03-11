"""
可视化模块
包含AirSim调试绘图和Matplotlib可视化功能
"""

import airsim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# 导入配置
from config import DRAW_DEBUG, PLANNED_PATH_Z_OFFSET


# ========================================
# AirSim 调试绘图
# ========================================
def draw_line(client, p1, p2, color=(0, 0, 1), thickness=3.0, persistent=True, vehicle_name="Car1"):
    """绘制线段"""
    if not DRAW_DEBUG:
        return

    line_points = [
        airsim.Vector3r(p1[0], p1[1], p1[2] - 0.5),
        airsim.Vector3r(p2[0], p2[1], p2[2] - 0.5)
    ]
    client.simPlotLineList(
        line_points,
        color_rgba=[*color, 1],
        thickness=thickness,
        is_persistent=persistent
    )


def draw_waypoint_marker(client, x, y, z=0, color=(0, 0, 1), size=8.0, vehicle_name="Car1"):
    """绘制目标点标记"""
    if not DRAW_DEBUG:
        return
    client.simPlotPoints(
        [airsim.Vector3r(x, y, z - 0.5)],
        color_rgba=[*color, 1],
        size=size,
        is_persistent=True
    )


def draw_planned_path(client, waypoints_with_yaw):
    """预先绘制规划路径（蓝色持久线）"""
    print("绘制全局规划路径（蓝色，高度降低显示）...")

    for i in range(len(waypoints_with_yaw) - 1):
        p1 = waypoints_with_yaw[i]
        p2 = waypoints_with_yaw[i + 1]

        draw_line(
            client,
            (p1[0], p1[1], p1[2] + PLANNED_PATH_Z_OFFSET),
            (p2[0], p2[1], p2[2] + PLANNED_PATH_Z_OFFSET),
            color=(0, 0, 1), thickness=3.0, persistent=True
        )

        draw_waypoint_marker(
            client, p1[0], p1[1], p1[2] + PLANNED_PATH_Z_OFFSET, color=(0, 0, 1)
        )

    last = waypoints_with_yaw[-1]
    draw_waypoint_marker(
        client, last[0], last[1], last[2] + PLANNED_PATH_Z_OFFSET, color=(0, 0, 1)
    )

    print(f"已绘制 {len(waypoints_with_yaw)} 个路径点（高度偏移: {PLANNED_PATH_Z_OFFSET}m）")


def draw_local_path(client, path_points_world, color=(1, 0, 1), thickness=2.0):
    """绘制局部规划路径（紫色）"""
    if not DRAW_DEBUG or len(path_points_world) < 2:
        return
    
    for i in range(len(path_points_world) - 1):
        p1 = path_points_world[i]
        p2 = path_points_world[i + 1]
        draw_line(
            client,
            (p1[0], p1[1], p1[2]),
            (p2[0], p2[1], p2[2]),
            color=color, thickness=thickness, persistent=True
        )


# ========================================
# Matplotlib 可视化（DEM和路径规划结果）
# ========================================
def load_dem_txt(dem_path):
    """读取矩形 DEM txt"""
    dem_path = Path(dem_path)
    rows = []
    with dem_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split()]
            rows.append(row)

    if not rows:
        raise RuntimeError(f"DEM 文件为空: {dem_path}")

    col_num = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != col_num:
            raise RuntimeError(f"DEM 不是规则矩形: {dem_path}")

    return np.array(rows, dtype=np.float64)


def load_costmap_txt(costmap_path):
    """读取 costmap txt"""
    costmap_path = Path(costmap_path)
    rows = []
    with costmap_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split()]
            rows.append(row)

    if not rows:
        raise RuntimeError(f"costmap 文件为空: {costmap_path}")

    col_num = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != col_num:
            raise RuntimeError(f"costmap 不是规则矩形: {costmap_path}")

    return np.array(rows, dtype=np.float64)


def parse_path_txt(path_path):
    """解析 path.txt"""
    path_path = Path(path_path)
    
    if not path_path.exists():
        return "NO_PATH_FOUND", None
        
    content = path_path.read_text(encoding="utf-8").strip()

    if content == "START_IS_OBSTACLE":
        return "START_IS_OBSTACLE", None
    if content == "GOAL_IS_OBSTACLE":
        return "GOAL_IS_OBSTACLE", None
    if content == "START_AND_GOAL_ARE_OBSTACLES":
        return "START_AND_GOAL_ARE_OBSTACLES", None
    if content == "NO_PATH_FOUND":
        return "NO_PATH_FOUND", None
    if not content:
        return "EMPTY", None

    matches = re.findall(r"\((\-?\d+),(\-?\d+)\)", content)
    if not matches:
        return "INVALID", None

    path = [(int(x), int(y)) for x, y in matches]
    return "OK", path


def make_costmap_gray_image(costmap):
    """把 costmap 转成 8 位灰度图"""
    img = np.zeros_like(costmap, dtype=np.uint8)

    ge_one = costmap >= 1.0
    lt_one = ~ge_one

    img[ge_one] = 255
    scaled = np.clip(costmap[lt_one], 0.0, 1.0) * 255.0
    img[lt_one] = scaled.astype(np.uint8)

    return img


def draw_start_goal_on_gray(gray_img, start, goal):
    """在灰度图上标注起点终点"""
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    sx, sy = start
    gx, gy = goal

    cv2.circle(color_img, (sx, sy), 5, (255, 0, 0), -1)   # 蓝色起点
    cv2.circle(color_img, (gx, gy), 5, (0, 255, 0), -1)   # 绿色终点

    cv2.putText(color_img, "Start", (sx + 8, sy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(color_img, "Goal", (gx + 8, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return color_img


def draw_path_on_image(color_img, path_points):
    """在图上用红线画路径"""
    out = color_img.copy()
    if not path_points or len(path_points) < 2:
        return out

    pts = np.array(path_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    return out


def save_3d_scene(dem, start, goal, path_points, output_path):
    """保存三维场景图"""
    output_path = Path(output_path)
    rows, cols = dem.shape

    xs = np.arange(cols)
    ys = np.arange(rows)
    X, Y = np.meshgrid(xs, ys)
    Z = dem.astype(np.float32)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        cmap="gray",
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.9
    )

    sx, sy = start
    gx, gy = goal

    sz = Z[sy, sx]
    gz = Z[gy, gx]

    z_range = np.max(Z) - np.min(Z)
    z_offset = max(z_range * 0.01, 0.5)

    ax.scatter([sx], [sy], [sz + z_offset], c="blue", s=20, depthshade=False, label="Start")
    ax.scatter([gx], [gy], [gz + z_offset], c="green", s=20, depthshade=False, label="Goal")

    # 只有当 path_points 不为 None 且长度大于1时才画路径
    if path_points is not None and len(path_points) > 1:
        px = np.array([p[0] for p in path_points], dtype=np.int32)
        py = np.array([p[1] for p in path_points], dtype=np.int32)
        pz = Z[py, px] + z_offset

        ax.plot(px, py, pz, color="red", linewidth=2, label="Path")

    ax.set_title("3D DEM with Start / Goal / Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.legend()
    ax.view_init(elev=45, azim=-60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def visualize_planning_results(dem_path, planning_output_dir, start_col, start_row, goal_col, goal_row):
    """
    可视化局部路径规划结果，包括：
    - costmap_with_start_goal.jpg
    - costmap_with_start_goal_and_path.jpg（仅当路径存在时）
    - dem_3d_scene.jpg
    """
    # 确保路径是 Path 对象
    dem_path = Path(dem_path)
    planning_output_dir = Path(planning_output_dir)
    
    # 读取 DEM
    try:
        dem = load_dem_txt(dem_path)
    except Exception as e:
        print(f"读取 DEM 失败: {e}")
        return False

    # 读取 costmap
    costmap_path = planning_output_dir / "costmap.txt"
    try:
        costmap = load_costmap_txt(costmap_path)
    except Exception as e:
        print(f"读取 costmap 失败: {e}")
        return False

    # 读取路径（可能不存在）
    path_file = planning_output_dir / "path.txt"
    path_status, path_points = parse_path_txt(path_file)
    
    # 如果 path.txt 不存在或为空，path_points 为 None
    if path_status != "OK":
        print(f"路径规划状态: {path_status}，将仅可视化 costmap 和 DEM")
        path_points = None  # 确保为 None

    # 生成 costmap 灰度图
    gray = make_costmap_gray_image(costmap)

    # 标注起终点
    start = (start_col, start_row)
    goal = (goal_col, goal_row)
    gray_with_points = draw_start_goal_on_gray(gray, start, goal)

    # 保存带起终点的代价地图图像（无论路径是否成功都保存）
    costmap_points_img_path = planning_output_dir / "costmap_with_start_goal.jpg"
    cv2.imwrite(str(costmap_points_img_path), gray_with_points)
    print(f"已保存: {costmap_points_img_path}")

    # 如果有路径，保存带路径的图像
    if path_points is not None and len(path_points) >= 2:
        path_vis = draw_path_on_image(gray_with_points, path_points)
        path_vis_path = planning_output_dir / "costmap_with_start_goal_and_path.jpg"
        cv2.imwrite(str(path_vis_path), path_vis)
        print(f"已保存: {path_vis_path}")

    # 三维场景可视化（无论路径是否成功都保存，只是不画路径）
    try:
        save_3d_scene(
            dem=dem,
            start=start,
            goal=goal,
            path_points=path_points if path_points is not None and len(path_points) >= 2 else None,
            output_path=planning_output_dir / "dem_3d_scene.jpg"
        )
        print(f"已保存: {planning_output_dir / 'dem_3d_scene.jpg'}")
    except Exception as e:
        print(f"保存三维场景失败: {e}")
        return False

    print(f"可视化结果已保存到: {planning_output_dir}")
    return True