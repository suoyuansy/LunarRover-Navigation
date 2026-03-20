"""
可视化模块
包含AirSim调试绘图和Matplotlib可视化功能
"""

from pathlib import Path
import re
import airsim
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import DRAW_DEBUG, PLANNED_PATH_Z_OFFSET


def draw_line(client, p1, p2, color=(0, 0, 1), thickness=3.0, persistent=True, vehicle_name="Car1"):
    if not DRAW_DEBUG:
        return
    line_points = [
        airsim.Vector3r(p1[0], p1[1], p1[2] - 0.5),
        airsim.Vector3r(p2[0], p2[1], p2[2] - 0.5),
    ]
    client.simPlotLineList(line_points, color_rgba=[*color, 1], thickness=thickness, is_persistent=persistent)


def draw_waypoint_marker(client, x, y, z=0, color=(0, 0, 1), size=8.0, vehicle_name="Car1"):
    if not DRAW_DEBUG:
        return
    client.simPlotPoints([airsim.Vector3r(x, y, z - 0.5)], color_rgba=[*color, 1], size=size, is_persistent=True)


def draw_planned_path(client, waypoints_with_yaw, color=(0, 0, 1)):
    print("绘制全局规划路径...")
    for i in range(len(waypoints_with_yaw) - 1):
        p1 = waypoints_with_yaw[i]
        p2 = waypoints_with_yaw[i + 1]
        draw_line(
            client,
            (p1[0], p1[1], p1[2] + PLANNED_PATH_Z_OFFSET),
            (p2[0], p2[1], p2[2] + PLANNED_PATH_Z_OFFSET),
            color=color,
            thickness=3.0,
            persistent=True
        )
        draw_waypoint_marker(client, p1[0], p1[1], p1[2] + PLANNED_PATH_Z_OFFSET, color=color)
    last = waypoints_with_yaw[-1]
    draw_waypoint_marker(client, last[0], last[1], last[2] + PLANNED_PATH_Z_OFFSET, color=color)
    print(f"已绘制 {len(waypoints_with_yaw)} 个路径点（高度偏移: {PLANNED_PATH_Z_OFFSET}m）")


def draw_local_path(client, path_points_world, color=(1, 0, 0), thickness=2.0):
    if not DRAW_DEBUG or len(path_points_world) < 2:
        return
    for i in range(len(path_points_world) - 1):
        p1 = path_points_world[i]
        p2 = path_points_world[i + 1]
        draw_line(client, (p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]), color=color, thickness=thickness, persistent=True)


def load_dem_txt(dem_path):
    dem_path = Path(dem_path)
    rows = []
    with dem_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"DEM 文件为空: {dem_path}")
    col_num = len(rows[0])
    for row in rows:
        if len(row) != col_num:
            raise RuntimeError(f"DEM 不是规则矩形: {dem_path}")
    return np.array(rows, dtype=np.float64)


def load_costmap_txt(costmap_path):
    costmap_path = Path(costmap_path)
    rows = []
    with costmap_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"costmap 文件为空: {costmap_path}")
    col_num = len(rows[0])
    for row in rows:
        if len(row) != col_num:
            raise RuntimeError(f"costmap 不是规则矩形: {costmap_path}")
    return np.array(rows, dtype=np.float64)


def save_costmap_txt(costmap, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in costmap:
            f.write(" ".join(f"{float(v):.6f}" for v in row))
            f.write("\n")


def parse_path_txt(path_path):
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
    return "OK", [(int(x), int(y)) for x, y in matches]


def make_costmap_gray_image(costmap):
    img = np.zeros_like(costmap, dtype=np.uint8)
    ge_one = costmap >= 1.0
    lt_one = ~ge_one
    img[ge_one] = 255
    scaled = np.clip(costmap[lt_one], 0.0, 1.0) * 255.0
    img[lt_one] = scaled.astype(np.uint8)
    return img


def draw_start_goal_on_gray(gray_img, start, goal):
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    sx, sy = start
    gx, gy = goal
    cv2.circle(color_img, (sx, sy), 5, (255, 0, 0), -1)
    cv2.circle(color_img, (gx, gy), 5, (0, 255, 0), -1)
    cv2.putText(color_img, "Start", (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(color_img, "Goal", (gx + 8, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return color_img


def draw_path_on_image(color_img, path_points):
    out = color_img.copy()
    if not path_points or len(path_points) < 2:
        return out
    pts = np.array(path_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    return out


def save_ranked_path_visualization(costmap_path, start, goal, path_points, output_image_path):
    costmap = load_costmap_txt(costmap_path)
    gray = make_costmap_gray_image(costmap)
    gray_with_points = draw_start_goal_on_gray(gray, start, goal)
    path_vis = draw_path_on_image(gray_with_points, path_points)
    cv2.imwrite(str(output_image_path), path_vis)


def save_3d_scene(dem, start, goal, path_points, output_path, debug=False):
    """
    保存 DEM 三维可视化图，路径点贴合 DEM 表面
    
    参数:
        dem: DEM 数据 (H, W)
        start: (col, row) 起点栅格坐标
        goal: (col, row) 终点栅格坐标  
        path_points: [(col, row), ...] 路径点列表
        output_path: 输出图片路径
        debug: 是否打印调试信息
    """
    output_path = Path(output_path)
    rows, cols = dem.shape
    
    if debug:
        print(f"DEM shape: {dem.shape}")
        print(f"Start: {start}, Goal: {goal}")
        print(f"Path points count: {len(path_points) if path_points else 0}")
    
    # 创建网格坐标
    xs = np.arange(cols)
    ys = np.arange(rows)
    X, Y = np.meshgrid(xs, ys)
    
    # DEM 高程数据
    Z = dem.astype(np.float32)
    
    # 计算合适的 z_offset（基于 DEM 高程范围）
    z_range = np.max(Z) - np.min(Z)
    z_offset = max(z_range * 0.02, 0.05)  # 至少 0.05 米
    
    if debug:
        print(f"Z range: {z_range:.3f}, z_offset: {z_offset:.3f}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # 绘制 DEM 表面（使用灰度色图更好地显示地形）
    surf = ax.plot_surface(X, Y, Z, cmap="gray", linewidth=0, antialiased=True, 
                          shade=True, alpha=0.9, rstride=1, cstride=1)
    
    # 处理起点坐标
    sx, sy = int(start[0]), int(start[1])
    sx = max(0, min(sx, cols - 1))
    sy = max(0, min(sy, rows - 1))
    sz = Z[sy, sx]
    
    # 处理终点坐标
    gx, gy = int(goal[0]), int(goal[1])
    gx = max(0, min(gx, cols - 1))
    gy = max(0, min(gy, rows - 1))
    gz = Z[gy, gx]
    
    if debug:
        print(f"Start Z: {sz:.3f}, Goal Z: {gz:.3f}")
    
    # 绘制起点（蓝色，大一点，带白边）
    ax.scatter([sx], [sy], [sz + z_offset], c="blue", s=80, depthshade=False, 
              label="Start", marker='o', edgecolors='white', linewidths=2, alpha=1.0)
    
    # 绘制终点（绿色，大一点，带白边）
    ax.scatter([gx], [gy], [gz + z_offset], c="green", s=80, depthshade=False, 
              label="Goal", marker='o', edgecolors='white', linewidths=2, alpha=1.0)
    
    # 绘制路径
    if path_points is not None and len(path_points) >= 2:
        # 提取路径坐标
        px_list = []
        py_list = []
        pz_list = []
        
        for i, p in enumerate(path_points):
            if len(p) >= 2:
                px, py = int(p[0]), int(p[1])
                # 确保在有效范围内
                px = max(0, min(px, cols - 1))
                py = max(0, min(py, rows - 1))
                pz = Z[py, px]
                
                px_list.append(px)
                py_list.append(py)
                pz_list.append(pz)
                
                if debug and i < 3:
                    print(f"Path point {i}: ({px}, {py}) -> Z={pz:.3f}")
        
        if len(px_list) >= 2:
            px_arr = np.array(px_list)
            py_arr = np.array(py_list)
            pz_arr = np.array(pz_list) + z_offset  # 添加偏移使路径可见
            
            # 绘制路径线（红色，较粗）
            ax.plot(px_arr, py_arr, pz_arr, color="red", linewidth=3, label="Path", alpha=0.9)
            
            # 每隔几个点画一个点标记
            step = max(1, len(px_list) // 20)
            ax.scatter(px_arr[::step], py_arr[::step], pz_arr[::step], 
                      c="darkred", s=15, depthshade=False, alpha=0.7)
    
    # 设置标题和标签
    ax.set_title("3D DEM with Start / Goal / Path", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (grid)", fontsize=11)
    ax.set_ylabel("Y (grid)", fontsize=11)
    ax.set_zlabel("Elevation", fontsize=11)
    
    # 设置坐标轴范围（留一些边距）
    margin = 5
    ax.set_xlim([-margin, cols + margin])
    ax.set_ylim([-margin, rows + margin])
    z_min, z_max = np.min(Z), np.max(Z)
    ax.set_zlim([z_min - z_range*0.1, z_max + z_range*0.2])
    
    # 设置等比例（可选）
    # ax.set_box_aspect([cols, rows, z_range])
    
    # 调整视角
    ax.view_init(elev=35, azim=-60)
    
    # 添加颜色条显示高程
    m = plt.cm.ScalarMappable(cmap="gray")
    m.set_array(Z)
    cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Elevation', rotation=270, labelpad=15)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"已保存 3D 可视化: {output_path}")


def save_costmap_path_visualization(costmap_path, start, goal, path_points, output_dir, prefix="costmap"):
    """保存costmap可视化，包括起点终点和路径"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    costmap = load_costmap_txt(costmap_path)
    gray = make_costmap_gray_image(costmap)
    gray_with_points = draw_start_goal_on_gray(gray, start, goal)
    cv2.imwrite(str(output_dir / f"{prefix}_with_start_goal.jpg"), gray_with_points)
    
    if path_points is not None and len(path_points) >= 2:
        path_vis = draw_path_on_image(gray_with_points, path_points)
        cv2.imwrite(str(output_dir / f"{prefix}_with_start_goal_and_path.jpg"), path_vis)


def visualize_planning_results(dem_path, costmap_path, planning_output_dir, start_col, start_row, goal_col, goal_row, has_path):
    """
    可视化规划结果
    生成:
    - costmap_with_start_goal.jpg
    - costmap_with_start_goal_and_path.jpg (如果有路径)
    - dem_3d_with_start_goal_and_path.jpg (如果有路径)
    """
    dem_path = Path(dem_path)
    costmap_path = Path(costmap_path)
    planning_output_dir = Path(planning_output_dir)
    
    try:
        dem = load_dem_txt(dem_path)
    except Exception as e:
        print(f"读取 DEM 失败: {e}")
        return False
    
    try:
        costmap = load_costmap_txt(costmap_path)
    except Exception as e:
        print(f"读取 costmap 失败: {e}")
        return False
    
    # 解析路径
    path_file = planning_output_dir / "path.txt"
    path_status, path_points = parse_path_txt(path_file)
    
    start = (start_col, start_row)
    goal = (goal_col, goal_row)
    
    # 1. 生成 costmap_with_start_goal.jpg
    gray = make_costmap_gray_image(costmap)
    gray_with_points = draw_start_goal_on_gray(gray, start, goal)
    costmap_start_goal_path = planning_output_dir / "costmap_with_start_goal.jpg"
    cv2.imwrite(str(costmap_start_goal_path), gray_with_points)
    print(f"已保存: {costmap_start_goal_path}")
    
    # 2. 如果有路径，生成其他可视化文件
    if has_path and path_points is not None and len(path_points) >= 2:
        # costmap_with_start_goal_and_path.jpg
        path_vis = draw_path_on_image(gray_with_points, path_points)
        path_vis_path = planning_output_dir / "costmap_with_start_goal_and_path.jpg"
        cv2.imwrite(str(path_vis_path), path_vis)
        print(f"已保存: {path_vis_path}")
        
        # dem_3d_with_start_goal_and_path.jpg
        try:
            save_3d_scene(
                dem=dem,
                start=start,
                goal=goal,
                path_points=path_points,
                output_path=planning_output_dir / "dem_3d_with_start_goal_and_path.jpg"
            )
            print(f"已保存: {planning_output_dir / 'dem_3d_with_start_goal_and_path.jpg'}")
        except Exception as e:
            print(f"保存三维场景失败: {e}")
            return False
    
    print(f"可视化结果已保存到: {planning_output_dir}")
    return True