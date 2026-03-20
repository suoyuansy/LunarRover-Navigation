"""
工具函数模块
包含数学计算、文件操作、坐标转换等通用工具函数
"""

import math
import numpy as np
import os
from datetime import datetime

try:
    from config import (
        DEM_DATA, DEM_ROWS, DEM_COLS, ORIGIN_HEIGHT,
        OUTPUT_ROOT_DIR
    )
except ImportError:
    DEM_DATA = None
    DEM_ROWS = None
    DEM_COLS = None
    ORIGIN_HEIGHT = 0.0
    OUTPUT_ROOT_DIR = "local_planningpath"


# ========================================
# 数学工具函数
# ========================================
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_yaw_between_points(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)


def quaternion_to_eulerian_angles(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_rotation_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return rz @ ry @ rx


def interpolate_yaw(current_yaw, target_yaw, alpha):
    angle_diff = normalize_angle(target_yaw - current_yaw)
    return normalize_angle(current_yaw + angle_diff * alpha)


# ========================================
# 文件与目录操作
# ========================================
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def create_run_output_folder():
    """创建运行输出文件夹"""
    from pathlib import Path
    
    root_dir = Path(os.getcwd()) / OUTPUT_ROOT_DIR
    ensure_dir(root_dir)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir_name = f"local_planningpath_{run_timestamp}"
    run_dir = root_dir / run_dir_name
    ensure_dir(run_dir)

    return root_dir, run_dir, run_timestamp  # 返回 Path 对象，不是 str


def create_single_build_folder(run_dir, prefix="dem_build"):
    """创建单次 DEM 构建文件夹"""
    from pathlib import Path
    
    run_dir = Path(run_dir)
    build_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    build_dir_name = f"{prefix}_{build_timestamp}"
    build_dir = run_dir / build_dir_name
    ensure_dir(build_dir)

    point_data_dir = build_dir / "point_data"
    ensure_dir(point_data_dir)

    return build_dir, point_data_dir, build_timestamp  # 返回 Path 对象，不是 str

# ========================================
# 全局 DEM 文件操作
# ========================================
def read_dem_file(filename="dem.txt"):
    global DEM_DATA, ORIGIN_HEIGHT, DEM_ROWS, DEM_COLS
    print("开始读取全局 DEM 数据...")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"DEM 文件 {filename} 不存在")

    filename = os.path.abspath(filename)

    print("从 txt 读取 DEM...")
    DEM_DATA = np.loadtxt(filename, dtype=np.float32)

    if DEM_DATA.ndim == 1:
        DEM_DATA = DEM_DATA.reshape(1, -1)

    DEM_ROWS, DEM_COLS = DEM_DATA.shape
    print(f"读取完成，数据总共 {DEM_ROWS} 行 {DEM_COLS} 列")

    ORIGIN_HEIGHT = float(DEM_DATA[0, 0])
    print(f"原点(0,0)高度: {ORIGIN_HEIGHT}m")


def calculate_airsim_elevation():
    global DEM_DATA, ORIGIN_HEIGHT
    print("开始计算 AirSim 坐标系下的高程坐标...")
    DEM_DATA = -(DEM_DATA - ORIGIN_HEIGHT)
    print("AirSim 坐标系高程坐标计算完毕")


def get_elevation(x, y):
    global DEM_DATA, DEM_ROWS, DEM_COLS

    col = int(round(x))
    row = int(round(y))

    if row < 0 or row >= DEM_ROWS or col < 0 or col >= DEM_COLS:
        print(f"警告：坐标({x}, {y})超出 DEM 范围，使用默认高度 0")
        return 0.0

    return float(DEM_DATA[row, col])


def read_path_file(filename):
    print(f"开始读取路径文件 {filename}...")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"路径文件 {filename} 不存在")

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    waypoints = []
    content = content.replace(' ', '').replace('\n', '')
    points_str = content.split('->')

    for point_str in points_str:
        point_str = point_str.strip()
        if not point_str:
            continue

        if point_str.startswith('(') and point_str.endswith(')'):
            point_str = point_str[1:-1]
            xy = point_str.split(',')
            if len(xy) == 2:
                x = float(xy[0])
                y = float(xy[1])
                waypoints.append((x, y))

    print(f"路径文件读取完成，共 {len(waypoints)} 个路径点")
    return waypoints


def generate_waypoints_with_elevation(path_points):
    print("开始生成带高程的路径点...")

    waypoints = []
    for i, (x, y) in enumerate(path_points):
        z = get_elevation(x, y)
        waypoints.append((x, y, z))

        if i < 3 or i >= len(path_points) - 3:
            print(f" 路径点 {i}: ({x}, {y}, {z:.3f})")
        elif i == 3:
            print(" ...")

    print(f"路径点生成完成，共 {len(waypoints)} 个点")
    return waypoints