import airsim
import time
import math
import numpy as np
import os
import traceback
import subprocess
import re
from datetime import datetime
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

# 尝试使用 GDAL 填充空洞；如果环境未安装 GDAL，则自动退回 scipy 插值
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except Exception:
    GDAL_AVAILABLE = False


# ========================================
# 基于 AirSim CV 模式 + LiDAR 的局部路径规划与 DEM 构建
# ========================================

# ========================================
# 全局配置参数
# ========================================
VEHICLE_NAME = "Car1"

# 路径跟踪相关
STEP_DISTANCE = 0.25
STEP_DISTANCE_DELAY = 0.1
STEP_ANGLE = 20
STEP_ANGLE_DELAY = 0.1
DRAW_DEBUG = True
PLANNED_PATH_Z_OFFSET = 0.5

# 全局 DEM 数据（从文件读取，用于给全局路径点补高程）
DEM_DATA = None
DEM_ROWS = None
DEM_COLS = None
ORIGIN_HEIGHT = 0.0
WAYPOINTS = []

# LiDAR 配置参数
LIDAR_SENSOR_NAME = "LidarSensor"
LIDAR_SCAN_RANGE = 30.0
LIDAR_FRAMES_PER_SAMPLE = 1
LIDAR_SAMPLE_INTERVAL = 0.1

# 新增：每次设定完位姿后，等待多少帧"新的雷达点云"
LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET = 3

# 新增：等待新帧时的轮询间隔和超时
LIDAR_POLL_INTERVAL = 0.005
LIDAR_WAIT_TIMEOUT_PER_FRAME = 10

# 点云预处理
LIDAR_MIN_VALID_DISTANCE = 1.0
LIDAR_MAX_VALID_DISTANCE = 50.0
LIDAR_MIN_Z = -100.0
LIDAR_MAX_Z = 100.0

# 局部 DEM 参数
LOCAL_DEM_RANGE = 10.0
LOCAL_DEM_RESOLUTION = 0.1
LOCAL_DEM_GRID_SIZE = int((LOCAL_DEM_RANGE * 2) / LOCAL_DEM_RESOLUTION)

# DEM 插值参数
DEM_GAUSSIAN_SIGMA = 1.0
DEM_USE_GDAL_FILL = True

# 每栅格融合方式
GRID_FUSION_METHOD = "median"

# 构建 DEM 的策略
INITIAL_BUILD_AFTER_SEGMENT_INDEX = 0
BUILD_EVERY_N_SEGMENTS = 4

# 输出目录
OUTPUT_ROOT_DIR = "local_planningpath"

# C++ 路径规划器配置
PATH_PLANNING_EXE = r"D:\Graduation_design\code\path_planning_based_on_lunar_DEM\out\build\x64-Debug\path_planning_based_on_lunar_DEM.exe"
PATH_PLANNING_TIMEOUT = 15.0

# 雷达传感器离地面的高度（Z轴向下为正，所以实际地面比传感器低 LIDAR_HEIGHT_OFFSET）
LIDAR_HEIGHT_OFFSET = 0.5  # 80cm = 0.8m


# ========================================
# 工具函数
# ========================================
def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_distance(x1, y1, x2, y2):
    """计算平面两点距离"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_yaw_between_points(x1, y1, x2, y2):
    """计算从点1指向点2的偏航角 yaw（弧度）"""
    return math.atan2(y2 - y1, x2 - x1)


def ensure_dir(dir_path):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def create_run_output_folder():
    """创建本次运行根目录"""
    root_dir = os.path.join(os.getcwd(), OUTPUT_ROOT_DIR)
    ensure_dir(root_dir)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir_name = f"local_planningpath_{run_timestamp}"
    run_dir = os.path.join(root_dir, run_dir_name)
    ensure_dir(run_dir)

    return root_dir, run_dir, run_timestamp


def create_single_build_folder(run_dir, prefix="dem_build"):
    """创建 DEM 构建目录"""
    build_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    build_dir_name = f"{prefix}_{build_timestamp}"
    build_dir = os.path.join(run_dir, build_dir_name)
    ensure_dir(build_dir)

    point_data_dir = os.path.join(build_dir, "point_data")
    ensure_dir(point_data_dir)

    return build_dir, point_data_dir, build_timestamp


def read_dem_file(filename="dem.txt"):
    """读取全局 DEM 文件"""
    global DEM_DATA, ORIGIN_HEIGHT, DEM_ROWS, DEM_COLS
    print("开始读取全局 DEM 数据...")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"DEM 文件 {filename} 不存在")

    dem_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split()]
            dem_list.append(values)

    DEM_DATA = np.array(dem_list, dtype=np.float32)
    DEM_ROWS, DEM_COLS = DEM_DATA.shape

    print(f"读取完成，数据总共 {DEM_ROWS} 行 {DEM_COLS} 列")

    ORIGIN_HEIGHT = DEM_DATA[0, 0]
    print(f"原点(0,0)高度: {ORIGIN_HEIGHT}m")


def calculate_airsim_elevation():
    """将全局 DEM 转为 AirSim 坐标系下的 Z 值（Z向下）"""
    global DEM_DATA
    print("开始计算 AirSim 坐标系下的高程坐标...")
    DEM_DATA = -(DEM_DATA - ORIGIN_HEIGHT)
    print("AirSim 坐标系高程坐标计算完毕")


def get_elevation(x, y):
    """根据坐标获取全局 DEM 中的 AirSim Z"""
    global DEM_DATA, DEM_ROWS, DEM_COLS

    col = int(round(x))
    row = int(round(y))

    if row < 0 or row >= DEM_ROWS or col < 0 or col >= DEM_COLS:
        print(f"警告：坐标({x}, {y})超出 DEM 范围，使用默认高度 0")
        return 0.0

    return float(DEM_DATA[row, col])


def read_path_file(filename):
    """读取路径文件"""
    global WAYPOINTS
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
    """生成带 AirSim Z 的路径点列表"""
    global WAYPOINTS
    print("开始生成带高程的路径点...")

    WAYPOINTS = []
    for i, (x, y) in enumerate(path_points):
        z = get_elevation(x, y)
        WAYPOINTS.append((x, y, z))

        if i < 3 or i >= len(path_points) - 3:
            print(f" 路径点 {i}: ({x}, {y}, {z:.3f})")
        elif i == 3:
            print(" ...")

    print(f"路径点生成完成，共 {len(WAYPOINTS)} 个点")


def quaternion_to_eulerian_angles(qx, qy, qz, qw):
    """四元数转欧拉角"""
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
    """欧拉角转旋转矩阵"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return rz @ ry @ rx


def get_vehicle_pose_full(client):
    """获取车辆位姿"""
    pose = client.simGetVehiclePose(VEHICLE_NAME)
    pos = pose.position
    ori = pose.orientation

    roll, pitch, yaw = quaternion_to_eulerian_angles(
        ori.x_val, ori.y_val, ori.z_val, ori.w_val
    )

    return pos.x_val, pos.y_val, pos.z_val, roll, pitch, yaw


def set_vehicle_pose(client, x, y, z, yaw, pitch=0.0, roll=0.0):
    """直接设置车辆位姿"""
    pose = airsim.Pose()
    pose.position = airsim.Vector3r(x, y, z)
    pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
    client.simSetVehiclePose(pose, True, VEHICLE_NAME)


def interpolate_yaw(current_yaw, target_yaw, alpha):
    """角度插值"""
    angle_diff = normalize_angle(target_yaw - current_yaw)
    return normalize_angle(current_yaw + angle_diff * alpha)


# ========================================
# 坐标转换函数（局部路径规划专用）
# ========================================
def world_to_local(x_world, y_world, x_origin, y_origin, yaw_origin):
    """
    将世界坐标转换为以 (x_origin, y_origin, yaw_origin) 为原点的局部坐标系
    局部坐标系定义：x向前，y向右
    """
    dx = x_world - x_origin
    dy = y_world - y_origin
    
    cos_yaw = math.cos(yaw_origin)
    sin_yaw = math.sin(yaw_origin)
    
    x_local = dx * cos_yaw + dy * sin_yaw
    y_local = -dx * sin_yaw + dy * cos_yaw
    
    return x_local, y_local


def local_to_dem_grid(x_local, y_local, dem_range, resolution):
    """
    将局部坐标转换为 DEM 栅格坐标 (col, row)
    row: 前 -> 后，对应 x_local 从 +range -> -range
    col: 左 -> 右，对应 y_local 从 -range -> +range
    """
    row_float = (dem_range - x_local) / resolution
    col_float = (y_local + dem_range) / resolution
    
    col = int(round(col_float))
    row = int(round(row_float))
    
    return col, row


def dem_grid_to_local(col, row, dem_range, resolution):
    """将 DEM 栅格坐标转换回局部坐标"""
    y_local = col * resolution - dem_range
    x_local = dem_range - row * resolution
    
    return x_local, y_local


def local_to_world(x_local, y_local, x_origin, y_origin, yaw_origin):
    """将局部坐标转换回世界坐标"""
    cos_yaw = math.cos(yaw_origin)
    sin_yaw = math.sin(yaw_origin)
    
    x_world = x_origin + x_local * cos_yaw - y_local * sin_yaw
    y_world = y_origin + x_local * sin_yaw + y_local * cos_yaw
    
    return x_world, y_world


# ========================================
# 绘图函数
# ========================================
def draw_line(client, p1, p2, color=(0, 0, 1), thickness=3.0, persistent=True):
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


def draw_waypoint_marker(client, x, y, z=0, color=(0, 0, 1), size=8.0):
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
# LiDAR 数据采集与 DEM 构建
# ========================================
class PointCloudAccumulator:
    """在一段运动过程中持续采集 LiDAR 点云"""

    def __init__(self, vehicle_name=VEHICLE_NAME, sensor_name=LIDAR_SENSOR_NAME):
        self.vehicle_name = vehicle_name
        self.sensor_name = sensor_name
        self.client = None
        self.frames = []
        self.last_lidar_timestamp = None

    def initialize(self, client):
        self.client = client
        print("LiDAR 点云采集模块初始化完成")
        print(f" 传感器名: {self.sensor_name}")

    def reset(self, clear_timestamp=False):
        """重置累计缓存"""
        self.frames = []
        if clear_timestamp:
            self.last_lidar_timestamp = None

    def _wait_for_next_new_lidar_frame(self, timeout=LIDAR_WAIT_TIMEOUT_PER_FRAME, poll_interval=LIDAR_POLL_INTERVAL):
        """等待一帧新的 timestamp LiDAR 数据"""
        if self.client is None:
            raise RuntimeError("PointCloudAccumulator 尚未 initialize(client)")

        deadline = time.time() + timeout

        while time.time() < deadline:
            lidar_data = self.client.getLidarData(self.sensor_name, self.vehicle_name)

            if lidar_data is None or lidar_data.point_cloud is None or len(lidar_data.point_cloud) < 3:
                time.sleep(poll_interval)
                continue

            ts = lidar_data.time_stamp

            if self.last_lidar_timestamp is not None and ts == self.last_lidar_timestamp:
                time.sleep(poll_interval)
                continue

            self.last_lidar_timestamp = ts
            return lidar_data

        return None

    def collect_once_wait_n_frames(self, expected_pose, wait_new_frames=LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET,
                                   wait_timeout_per_frame=LIDAR_WAIT_TIMEOUT_PER_FRAME,
                                   poll_interval=LIDAR_POLL_INTERVAL,
                                   debug_prefix="采集点云"):
        """设置完位姿后，等待 N 帧新点云，保存最后一帧"""
        if self.client is None:
            raise RuntimeError("PointCloudAccumulator 尚未 initialize(client)")

        last_new_lidar = None

        for i in range(wait_new_frames):
            lidar_data = self._wait_for_next_new_lidar_frame(
                timeout=wait_timeout_per_frame,
                poll_interval=poll_interval
            )

            if lidar_data is None:
                print(f"  {debug_prefix}: 等待第 {i + 1}/{wait_new_frames} 帧新 LiDAR 数据超时")
                return 0

            last_new_lidar = lidar_data
            print(f"  {debug_prefix}: 获取到第 {i + 1}/{wait_new_frames} 帧新 LiDAR, timestamp={lidar_data.time_stamp}")

        if last_new_lidar is None:
            return 0

        pts = np.array(last_new_lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        pts = self._filter_lidar_points(pts)

        if len(pts) == 0:
            print(f"  {debug_prefix}: 最终保存帧过滤后无有效点")
            return 0

        self.frames.append({
            "points_local": pts,
            "pose": expected_pose,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "lidar_timestamp": last_new_lidar.time_stamp
        })

        return len(pts)

    def _filter_lidar_points(self, points_local):
        """LiDAR 点云预处理"""
        if points_local is None or len(points_local) == 0:
            return np.empty((0, 3), dtype=np.float32)

        finite_mask = np.isfinite(points_local).all(axis=1)
        pts = points_local[finite_mask]

        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        dists = np.linalg.norm(pts, axis=1)

        valid_mask = (
            (dists >= LIDAR_MIN_VALID_DISTANCE) &
            (dists <= LIDAR_MAX_VALID_DISTANCE) &
            (pts[:, 2] >= LIDAR_MIN_Z) &
            (pts[:, 2] <= LIDAR_MAX_Z)
        )

        pts = pts[valid_mask]
        return pts

    def get_all_frames(self):
        """获取当前累计的所有点云帧"""
        return self.frames

    def save_frames_to_point_data(self, point_data_dir):
        """保存每帧点云到 point_data 目录"""
        for idx, frame in enumerate(self.frames, start=1):
            _, _, _, roll, pitch, yaw = frame["pose"]

            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            lidar_ts = frame.get("lidar_timestamp", 0)

            filename = (
                f"{idx:04d}_ts({lidar_ts})_roll({roll_deg:.2f})_pitch({pitch_deg:.2f})_yaw({yaw_deg:.2f}).txt"
            )
            filepath = os.path.join(point_data_dir, filename)
            np.savetxt(filepath, frame["points_local"], fmt="%.6f")


class LocalDEMBuilder:
    """根据累计的多帧点云，在"当前传感器坐标系"下构建局部 DEM"""

    def __init__(self, dem_range=LOCAL_DEM_RANGE, resolution=LOCAL_DEM_RESOLUTION):
        self.dem_range = dem_range
        self.resolution = resolution
        self.grid_size = int((dem_range * 2) / resolution)

    def build_dem_from_frames(self, frames, current_pose):
        """
        输入：
            frames: 累积的点云帧
            current_pose: 当前构建时刻的传感器位姿 (cx,cy,cz,roll,pitch,yaw)
        输出：
            dem_grid: HxW，存储的是"当前传感器坐标系下的 z 值（Z向下）"
            mask_valid: HxW
            all_points_in_current_sensor: 融合到当前传感器坐标系后的所有点
        """
        if frames is None or len(frames) == 0:
            print("警告：无历史点云帧，无法构建 DEM")
            return None, None, None

        cx, cy, cz, croll, cpitch, cyaw = current_pose

        # 当前构建时刻：当前传感器坐标系 -> 世界坐标系 的旋转
        R_current = euler_to_rotation_matrix(croll, cpitch, cyaw)
        t_current = np.array([cx, cy, cz], dtype=np.float64)

        # 需要的是：世界 -> 当前传感器坐标系
        R_world_to_current = R_current.T

        transformed_points_list = []

        for frame in frames:
            pts_local_old = frame["points_local"]
            fx, fy, fz, froll, fpitch, fyaw = frame["pose"]

            if pts_local_old is None or len(pts_local_old) == 0:
                continue

            # 旧采样时刻：旧传感器坐标系 -> 世界坐标系
            R_old = euler_to_rotation_matrix(froll, fpitch, fyaw)
            t_old = np.array([fx, fy, fz], dtype=np.float64)

            # 1) old_sensor -> world
            pts_world = (R_old @ pts_local_old.T).T + t_old

            # 2) world -> current_sensor
            pts_current = (R_world_to_current @ (pts_world - t_current).T).T

            transformed_points_list.append(pts_current)

        if not transformed_points_list:
            print("警告：没有有效点云帧可用于 DEM 构建")
            return None, None, None

        all_points_in_current_sensor = np.vstack(transformed_points_list)

        # 在当前传感器坐标系下裁剪 DEM 范围
        x_local = all_points_in_current_sensor[:, 0]  # forward
        y_local = all_points_in_current_sensor[:, 1]  # right
        z_local = all_points_in_current_sensor[:, 2]  # down

        left_local = -y_local

        range_mask = (
            (np.abs(x_local) <= self.dem_range) &
            (np.abs(left_local) <= self.dem_range)
        )

        x_local = x_local[range_mask]
        y_local = y_local[range_mask]
        z_local = z_local[range_mask]

        if len(x_local) < 20:
            print("警告：局部范围内点云不足，无法构建 DEM")
            return None, None, None

        # 左前方为原点：
        # row: 前 -> 后，对应 x_local 从 +range -> -range
        # col: 左 -> 右，对应 y_local 从 -range -> +range
        row_float = (self.dem_range - x_local) / self.resolution
        col_float = (y_local + self.dem_range) / self.resolution

        rows = np.floor(row_float).astype(np.int32)
        cols = np.floor(col_float).astype(np.int32)

        valid_idx = (
            (rows >= 0) & (rows < self.grid_size) &
            (cols >= 0) & (cols < self.grid_size)
        )

        rows = rows[valid_idx]
        cols = cols[valid_idx]
        zs = z_local[valid_idx]

        if len(zs) < 20:
            print("警告：落入栅格的有效点不足，无法构建 DEM")
            return None, None, None

        dem_grid = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)

        # 将每格对应的 z 聚合
        grid_buckets = {}
        for r, c, z in zip(rows, cols, zs):
            key = (r, c)
            if key not in grid_buckets:
                grid_buckets[key] = []
            grid_buckets[key].append(float(z))

        for (r, c), z_list in grid_buckets.items():
            z_arr = np.array(z_list, dtype=np.float32)

            if GRID_FUSION_METHOD == "mean":
                dem_grid[r, c] = float(np.mean(z_arr))
            elif GRID_FUSION_METHOD == "min":
                dem_grid[r, c] = float(np.min(z_arr))
            elif GRID_FUSION_METHOD == "max":
                dem_grid[r, c] = float(np.max(z_arr))
            else:
                dem_grid[r, c] = float(np.median(z_arr))

        mask_valid = ~np.isnan(dem_grid)

        if np.sum(mask_valid) < 20:
            print("警告：栅格有效点过少，无法构建 DEM")
            return None, None, None

        dem_grid = self.fill_dem_holes(dem_grid)
        dem_grid = gaussian_filter(dem_grid, sigma=DEM_GAUSSIAN_SIGMA)

        return dem_grid, mask_valid, all_points_in_current_sensor

    def fill_dem_holes(self, dem_grid):
        """优先使用 GDAL FillNodata；若不可用则退回 scipy.griddata"""
        if DEM_USE_GDAL_FILL and GDAL_AVAILABLE:
            try:
                return self._fill_with_gdal(dem_grid)
            except Exception as e:
                print(f"GDAL FillNodata 失败，退回 scipy 插值: {e}")

        return self._fill_with_scipy(dem_grid)

    def _fill_with_gdal(self, dem_grid):
        """使用 GDAL FillNodata 填充空洞"""
        rows, cols = dem_grid.shape
        driver = gdal.GetDriverByName('MEM')

        ds = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
        band = ds.GetRasterBand(1)

        data = dem_grid.copy().astype(np.float32)
        nodata_value = -9999.0
        nan_mask = np.isnan(data)
        data[nan_mask] = nodata_value

        band.WriteArray(data)
        band.SetNoDataValue(nodata_value)

        mask_ds = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
        mask_band = mask_ds.GetRasterBand(1)
        mask_array = np.where(np.isnan(dem_grid), 0, 1).astype(np.uint8)
        mask_band.WriteArray(mask_array)

        gdal.FillNodata(
            targetBand=band,
            maskBand=mask_band,
            maxSearchDist=50,
            smoothingIterations=1
        )

        filled = band.ReadAsArray().astype(np.float32)

        remain_mask = (filled == nodata_value) | (~np.isfinite(filled))
        if np.any(remain_mask):
            valid = filled[~remain_mask]
            fill_val = float(np.median(valid)) if len(valid) > 0 else 0.0
            filled[remain_mask] = fill_val

        return filled

    def _fill_with_scipy(self, dem_grid):
        """使用 scipy.griddata 填充空洞"""
        rows, cols = dem_grid.shape
        valid_mask = ~np.isnan(dem_grid)

        if np.sum(valid_mask) == 0:
            return np.zeros_like(dem_grid, dtype=np.float32)

        rr, cc = np.mgrid[0:rows, 0:cols]
        valid_points = np.column_stack((rr[valid_mask], cc[valid_mask]))
        valid_values = dem_grid[valid_mask]

        filled = griddata(
            valid_points,
            valid_values,
            (rr, cc),
            method='linear'
        )

        remain_mask = np.isnan(filled)
        if np.any(remain_mask):
            filled_nearest = griddata(
                valid_points,
                valid_values,
                (rr, cc),
                method='nearest'
            )
            filled[remain_mask] = filled_nearest[remain_mask]

        return filled.astype(np.float32)

    def save_dem_results(self, dem_grid, current_pose, build_dir):
        """保存 DEM 结果"""
        cx, cy, cz, roll, pitch, yaw = current_pose

        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        dem_txt_path = os.path.join(build_dir, "dem.txt")
        meta_txt_path = os.path.join(build_dir, "dem_meta.txt")
        jpg_path = os.path.join(build_dir, "dem_3d.jpg")

        # 保存时取反，使得 Z 向上为正（符合常规 DEM 格式）
        np.savetxt(dem_txt_path, -dem_grid, fmt="%.4f")

        with open(meta_txt_path, "w", encoding="utf-8") as f:
            f.write("===== Local DEM Metadata =====\n")
            f.write(f"center_x = {cx:.6f}\n")
            f.write(f"center_y = {cy:.6f}\n")
            f.write(f"center_z = {cz:.6f}\n")
            f.write(f"roll_rad = {roll:.8f}\n")
            f.write(f"pitch_rad = {pitch:.8f}\n")
            f.write(f"yaw_rad = {yaw:.8f}\n")
            f.write(f"roll_deg = {roll_deg:.6f}\n")
            f.write(f"pitch_deg = {pitch_deg:.6f}\n")
            f.write(f"yaw_deg = {yaw_deg:.6f}\n")
            f.write(f"dem_range = {self.dem_range}\n")
            f.write(f"resolution = {self.resolution}\n")
            f.write(f"grid_size = {self.grid_size}\n")
            f.write(f"fusion_method = {GRID_FUSION_METHOD}\n")
            f.write("coordinate_frame = current_sensor_frame\n")
            f.write("origin_definition = top-left is front-left of current sensor\n")
            f.write("row_direction = front -> back\n")
            f.write("col_direction = left -> right\n")
            f.write("x_axis = forward_positive\n")
            f.write("y_axis = right_positive\n")
            f.write("z_axis = down_positive\n")
            f.write("dem_z_direction = down_positive (stored value), up_positive in txt file\n")

        self._save_3d_visualization(dem_grid, jpg_path)

        print(f"DEM 已保存: {dem_txt_path}")
        print(f"元数据已保存: {meta_txt_path}")
        print(f"三维图已保存: {jpg_path}")

        return dem_txt_path

    def _save_3d_visualization(self, dem_grid, jpg_path):
        """保存 DEM 三维可视化图"""
        rows, cols = dem_grid.shape

        row_coords = np.arange(rows) * self.resolution
        col_coords = np.arange(cols) * self.resolution
        xx, yy = np.meshgrid(col_coords, row_coords)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 可视化时取反，使得向上为正
        zz_vis = -dem_grid

        ax.plot_surface(xx, yy, zz_vis, cmap='terrain', linewidth=0, antialiased=True)

        ax.invert_xaxis()
        ax.set_title("Local DEM 3D Visualization")
        ax.set_xlabel("Column (left -> right, m)")
        ax.set_ylabel("Row (front -> back, m)")
        ax.set_zlabel("Height (upward positive, m)")

        plt.tight_layout()
        plt.savefig(jpg_path, dpi=200)
        plt.close(fig)


# ========================================
# 局部路径规划器接口（C++ 程序调用）
# ========================================
class LocalPathPlanner:
    """调用 C++ 路径规划器进行局部路径规划"""

    def __init__(self, exe_path=PATH_PLANNING_EXE):
        self.exe_path = Path(exe_path)
        self.timeout = PATH_PLANNING_TIMEOUT

    def plan_path(self, dem_path, start_col, start_row, goal_col, goal_row, 
                  grid_size, output_dir):
        """
        调用 C++ 路径规划器
        返回：
            ("OK", path_points_dem) - 成功，path_points_dem 是 [(col, row), ...]
            其他状态码 - 失败
        """
        dem_path = Path(dem_path)
        output_dir = Path(output_dir)

        if not self.exe_path.exists():
            print(f"错误：路径规划器可执行文件不存在: {self.exe_path}")
            return "EXE_NOT_FOUND", None

        if not dem_path.exists():
            print(f"错误：DEM 文件不存在: {dem_path}")
            return "DEM_NOT_FOUND", None

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.exe_path),
            str(dem_path),
            str(start_col),
            str(start_row),
            str(goal_col),
            str(goal_row),
            str(grid_size),
            str(output_dir),
        ]

        print(f"调用路径规划器:")
        print(f"  DEM: {dem_path}")
        print(f"  起点: ({start_col}, {start_row})")
        print(f"  终点: ({goal_col}, {goal_row})")
        print(f"  分辨率: {grid_size}m")
        print(f"  输出: {output_dir}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            print(f"启动路径规划器失败: {e}")
            return "PROCESS_ERROR", None

        status, result = self._wait_for_outputs(process, output_dir)

        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                print(f"\nC++ stdout:\n{stdout}")
            if stderr:
                print(f"\nC++ stderr:\n{stderr}")
        except:
            pass

        if status == "success":
            path_file = output_dir / "path.txt"
            path_status, path_points = self._parse_path_txt(path_file)
            return path_status, path_points
        elif status == "timeout":
            print("路径规划超时")
            process.kill()
            return "TIMEOUT", None
        elif status == "process_error":
            print("路径规划进程错误")
            return "PROCESS_ERROR", None

        return status, result

    def _wait_for_outputs(self, process, output_dir):
        """等待 path.txt 和 costmap.txt 生成"""
        path_file = output_dir / "path.txt"
        costmap_file = output_dir / "costmap.txt"

        start_time = time.time()

        while True:
            if path_file.exists() and costmap_file.exists():
                return "success", None

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    return "process_error", None
                else:
                    return "process_error", None

            if time.time() - start_time > self.timeout:
                return "timeout", None

            time.sleep(0.2)

    def _parse_path_txt(self, path_file):
        """解析 path.txt"""
        if not path_file.exists():
            return "NO_PATH_FOUND", None

        content = path_file.read_text(encoding="utf-8").strip()

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


# ========================================
# 局部路径规划结果可视化
# ========================================
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

    # 修改这里：只有当 path_points 不为 None 且长度大于1时才画路径
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


# ========================================
# 核心运动控制
# ========================================
def move_to_target_constant_yaw(client, accumulator,
                                current_x, current_y, current_z, current_yaw,
                                target_x, target_y, target_z, target_yaw,
                                trajectory_points, segment_idx):
    """控制车辆从当前位置移动到目标位置，一步一动，采集点云"""
    total_distance = calculate_distance(current_x, current_y, target_x, target_y)
    num_steps = max(int(total_distance / STEP_DISTANCE), 1)

    current_yaw_deg = math.degrees(current_yaw)
    target_yaw_deg = math.degrees(target_yaw)

    print(f"路段 {segment_idx}: ({current_x:.1f}, {current_y:.1f}) -> ({target_x:.1f}, {target_y:.1f})")
    print(f"距离: {total_distance:.2f}m, 步数: {num_steps}")

    prev_x, prev_y, prev_z = current_x, current_y, current_z

    # 第一阶段：直线移动
    for step in range(1, num_steps + 1):
        alpha = step / num_steps

        x = current_x + (target_x - current_x) * alpha
        y = current_y + (target_y - current_y) * alpha
        z = current_z + (target_z - current_z) * alpha
        yaw = current_yaw

        set_vehicle_pose(client, x, y, z, yaw)
        trajectory_points.append((x, y, z))

        # 绘制红色轨迹
        draw_line(
            client,
            (prev_x, prev_y, prev_z),
            (x, y, z),
            color=(1, 0, 0),
            thickness=2.0,
            persistent=True
        )
        prev_x, prev_y, prev_z = x, y, z

        # 采集点云
        expected_pose = (x, y, z, 0.0, 0.0, yaw)
        num_pts = accumulator.collect_once_wait_n_frames(
            expected_pose=expected_pose,
            wait_new_frames=LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET,
            wait_timeout_per_frame=LIDAR_WAIT_TIMEOUT_PER_FRAME,
            poll_interval=LIDAR_POLL_INTERVAL,
            debug_prefix="直线采集点云"
        )
        if num_pts > 0:
            print(f"  采集点云: {num_pts} 点")

    # 保证精确到达
    set_vehicle_pose(client, target_x, target_y, target_z, current_yaw)
    trajectory_points.append((target_x, target_y, target_z))

    print(f"到达目标位置 ({target_x:.2f}, {target_y:.2f})")

    # 第二阶段：原地转向
    if abs(normalize_angle(target_yaw - current_yaw)) > 0.01:
        yaw_diff = abs(normalize_angle(target_yaw - current_yaw))
        yaw_deg_diff = math.degrees(yaw_diff)
        turn_steps = max(int(yaw_deg_diff / STEP_ANGLE), 1)

        print(f"开始转向: {current_yaw_deg:.1f}° -> {target_yaw_deg:.1f}°")

        for step in range(1, turn_steps + 1):
            alpha = step / turn_steps
            yaw = interpolate_yaw(current_yaw, target_yaw, alpha)

            set_vehicle_pose(client, target_x, target_y, target_z, yaw)

            expected_pose = (target_x, target_y, target_z, 0.0, 0.0, yaw)
            num_pts = accumulator.collect_once_wait_n_frames(
                expected_pose=expected_pose,
                wait_new_frames=LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET,
                wait_timeout_per_frame=LIDAR_WAIT_TIMEOUT_PER_FRAME,
                poll_interval=LIDAR_POLL_INTERVAL,
                debug_prefix="转向采集点云"
            )
            if num_pts > 0:
                print(f"  转向采集点云: {num_pts} 点")

        print(f"转向完成，当前 yaw: {target_yaw_deg:.1f}°")

    return target_x, target_y, target_z, target_yaw


def move_along_local_path(client, accumulator, local_path_points_world, 
                          start_z_world, dem_grid, dem_range, resolution,
                          trajectory_points, current_yaw):
    """
    沿局部规划路径行走
    Z坐标已经在外部计算好（考虑了雷达高度偏移）
    """
    if len(local_path_points_world) < 2:
        print("警告：局部路径点数量不足")
        return local_path_points_world[0][0], local_path_points_world[0][1], start_z_world, current_yaw

    current_x, current_y = local_path_points_world[0][0], local_path_points_world[0][1]
    current_z = local_path_points_world[0][2]  # 使用路径点中的Z（已修正）

    print(f"\n开始沿局部路径行走，共 {len(local_path_points_world)} 个点")
    print(f"起点: ({current_x:.2f}, {current_y:.2f}, {current_z:.2f})")

    # 遍历路径点（跳过第一个点）
    for i in range(1, len(local_path_points_world)):
        target_x = local_path_points_world[i][0]
        target_y = local_path_points_world[i][1]
        target_z = local_path_points_world[i][2]  # 使用已修正的Z
        
        # 计算目标 yaw
        if i < len(local_path_points_world) - 1:
            next_x = local_path_points_world[i + 1][0]
            next_y = local_path_points_world[i + 1][1]
            target_yaw = calculate_yaw_between_points(target_x, target_y, next_x, next_y)
        else:
            prev_x = local_path_points_world[i - 1][0]
            prev_y = local_path_points_world[i - 1][1]
            target_yaw = calculate_yaw_between_points(prev_x, prev_y, target_x, target_y)

        # 移动到目标点
        current_x, current_y, current_z, current_yaw = move_to_target_constant_yaw(
            client=client,
            accumulator=pointcloud_accumulator,
            current_x=current_x,
            current_y=current_y,
            current_z=current_z,
            current_yaw=current_yaw,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            target_yaw=target_yaw,
            trajectory_points=trajectory_points,
            segment_idx=i
        )

        print(f"  已到达局部路径点 {i}/{len(local_path_points_world)-1}")

    return current_x, current_y, current_z, current_yaw


# ========================================
# 主函数
# ========================================
pointcloud_accumulator = PointCloudAccumulator(VEHICLE_NAME, LIDAR_SENSOR_NAME)
local_dem_builder = LocalDEMBuilder(LOCAL_DEM_RANGE, LOCAL_DEM_RESOLUTION)
local_path_planner = LocalPathPlanner(PATH_PLANNING_EXE)


def main():
    print("=" * 70)
    print("AirSim CV模式 + LiDAR 局部路径规划与 DEM 构建")
    print("流程: 初始全局路径行走 → 构建DEM → 局部路径规划 → 沿局部路径行走 → 构建DEM → ...")
    print(f"雷达离地面高度偏移: {LIDAR_HEIGHT_OFFSET}m (已用于修正局部路径高程)")
    print("=" * 70)

    # 1. 读取全局 DEM 与路径
    read_dem_file("dem.txt")
    calculate_airsim_elevation()

    path_file = None
    for file in os.listdir('.'):
        if file.startswith("path") and file.endswith(".txt"):
            path_file = file
            break

    if path_file is None:
        raise FileNotFoundError("未找到路径文件")

    path_points = read_path_file(path_file)
    generate_waypoints_with_elevation(path_points)

    if len(WAYPOINTS) < 2:
        raise ValueError("路径点数量不足，至少需要2个点")

    print(f"\n准备执行路径跟踪，共 {len(WAYPOINTS)} 个路径点")
    print("=" * 70)

    # 2. 创建本次运行目录
    output_root_dir, run_dir, run_timestamp = create_run_output_folder()
    print(f"总输出目录: {output_root_dir}")
    print(f"本次运行目录: {run_dir}")

    # 3. AirSim 连接
    client = airsim.VehicleClient()
    client.confirmConnection()
    client.enableApiControl(True, VEHICLE_NAME)
    client.simFlushPersistentMarkers()

    print(f"\n已启用 API 控制: {VEHICLE_NAME}")

    # 4. 初始化模块
    pointcloud_accumulator.initialize(client)

    # 5. 计算全局路径点朝向
    waypoints_with_yaw = []
    for i, (x, y, z) in enumerate(WAYPOINTS):
        if i < len(WAYPOINTS) - 1:
            next_x, next_y, next_z = WAYPOINTS[i + 1]
            yaw = calculate_yaw_between_points(x, y, next_x, next_y)
        else:
            yaw = waypoints_with_yaw[-1][3] if waypoints_with_yaw else 0.0
        waypoints_with_yaw.append((x, y, z, yaw))

    # 6. 绘制全局路径
    draw_planned_path(client, waypoints_with_yaw)
    time.sleep(0.5)

    trajectory_points = []

    # 7. 瞬移到起始点
    first_wp = waypoints_with_yaw[0]
    print(f"\n【瞬移到起始点】({first_wp[0]}, {first_wp[1]}, {first_wp[2]})")
    set_vehicle_pose(client, first_wp[0], first_wp[1], first_wp[2], first_wp[3])
    time.sleep(1.0)

    current_x, current_y, current_z, roll, pitch, current_yaw = get_vehicle_pose_full(client)
    print(f"当前位置: ({current_x:.2f}, {current_y:.2f}, {current_z:.2f})")

    # 清空缓存，准备采集
    pointcloud_accumulator.reset(clear_timestamp=True)
    print("已在瞬移到起点后清空点云缓存")

    total_segments = len(waypoints_with_yaw) - 1
    current_global_idx = 0  # 当前在全局路径中的索引

    try:
        # ===== 阶段1：初始全局路径行走（走到 INITIAL_BUILD_AFTER_SEGMENT_INDEX）=====
        target_initial_idx = min(INITIAL_BUILD_AFTER_SEGMENT_INDEX + 1, total_segments)
        
        if target_initial_idx > 0:
            print(f"\n{'=' * 70}")
            print(f"【初始全局路径行走】从起点到全局路径点 {target_initial_idx}")
            print(f"{'=' * 70}")

            while current_global_idx < target_initial_idx:
                wp_start = waypoints_with_yaw[current_global_idx]
                wp_target = waypoints_with_yaw[current_global_idx + 1]

                print(f"\n全局路段 {current_global_idx + 1}/{total_segments}")
                print(f"({wp_start[0]:.2f}, {wp_start[1]:.2f}) -> ({wp_target[0]:.2f}, {wp_target[1]:.2f})")

                current_x, current_y, current_z, current_yaw = move_to_target_constant_yaw(
                    client=client,
                    accumulator=pointcloud_accumulator,
                    current_x=current_x,
                    current_y=current_y,
                    current_z=current_z,
                    current_yaw=current_yaw,
                    target_x=wp_target[0],
                    target_y=wp_target[1],
                    target_z=wp_target[2],
                    target_yaw=wp_target[3],
                    trajectory_points=trajectory_points,
                    segment_idx=current_global_idx + 1
                )

                current_global_idx += 1
                print(f"已到达全局路径点 {current_global_idx}，位置: ({current_x:.2f}, {current_y:.2f}, {current_z:.2f})")

        # ===== 主循环：构建 DEM → 局部路径规划 → 沿局部路径行走 → 构建 DEM → ... =====
        while current_global_idx < total_segments:
            print(f"\n{'=' * 70}")
            print(f"【DEM 构建阶段】当前位于全局路径点 {current_global_idx}")
            print(f"{'=' * 70}")

            # 获取当前位姿
            latest_pose_full = get_vehicle_pose_full(client)
            current_x, current_y, current_z, roll, pitch, current_yaw = latest_pose_full
            
            # 获取累计的点云帧
            frames = pointcloud_accumulator.get_all_frames()

            if frames is None or len(frames) == 0:
                print("警告：当前累计点云帧为空，无法构建 DEM")
                print("程序退出")
                return

            # 创建 DEM 构建目录
            dem_build_dir, point_data_dir, dem_timestamp = create_single_build_folder(run_dir, "dem_build")
            print(f"本次 DEM 构建目录: {dem_build_dir}")

            # 保存原始点云帧
            pointcloud_accumulator.save_frames_to_point_data(point_data_dir)

            # 构建 DEM
            dem_grid, valid_mask, fused_points_current = local_dem_builder.build_dem_from_frames(
                frames=frames,
                current_pose=latest_pose_full
            )

            if dem_grid is None:
                print("错误：DEM 构建失败")
                print("程序退出")
                return

            # 保存 DEM 结果
            dem_txt_path = local_dem_builder.save_dem_results(
                dem_grid=dem_grid,
                current_pose=latest_pose_full,
                build_dir=dem_build_dir
            )

            # 清空缓存，为下一轮采集做准备（但保留 timestamp）
            pointcloud_accumulator.reset(clear_timestamp=False)

            # ===== 局部路径规划阶段 =====
            print(f"\n{'=' * 70}")
            print(f"【局部路径规划阶段】")
            print(f"{'=' * 70}")

            # 确定局部路径规划的终点（BUILD_EVERY_N_SEGMENTS 个全局路径点之后）
            goal_global_idx = min(current_global_idx + BUILD_EVERY_N_SEGMENTS, total_segments)
            
            if goal_global_idx <= current_global_idx:
                print("已到达终点附近，结束循环")
                break

            # 起点和终点的世界坐标
            start_world = (current_x, current_y, current_z)
            goal_world = (waypoints_with_yaw[goal_global_idx][0], 
                         waypoints_with_yaw[goal_global_idx][1],
                         waypoints_with_yaw[goal_global_idx][2])

            print(f"起点（世界坐标）: ({start_world[0]:.2f}, {start_world[1]:.2f}, {start_world[2]:.2f})")
            print(f"终点（世界坐标）: ({goal_world[0]:.2f}, {goal_world[1]:.2f}, {goal_world[2]:.2f})")

            # 世界 -> 局部坐标转换（以当前位置为原点，当前 yaw 为方向）
            goal_local_x, goal_local_y = world_to_local(
                goal_world[0], goal_world[1],
                start_world[0], start_world[1], current_yaw
            )

            print(f"终点（局部坐标）: ({goal_local_x:.2f}, {goal_local_y:.2f})")

            # 局部 -> DEM 栅格坐标转换
            dem_range = LOCAL_DEM_RANGE
            resolution = LOCAL_DEM_RESOLUTION
            
            # 起点在局部坐标系原点 -> DEM 中心
            start_col, start_row = local_to_dem_grid(0, 0, dem_range, resolution)
            
            # 终点转换
            goal_col, goal_row = local_to_dem_grid(goal_local_x, goal_local_y, dem_range, resolution)

            print(f"起点（DEM栅格）: ({start_col}, {start_row})")
            print(f"终点（DEM栅格）: ({goal_col}, {goal_row})")

            # 检查是否在有效范围内
            grid_size = local_dem_builder.grid_size
            
            # 如果终点超出范围，调整到边界
            if not (0 <= goal_col < grid_size and 0 <= goal_row < grid_size):
                print(f"警告：终点 ({goal_col}, {goal_row}) 超出 DEM 范围 [0, {grid_size})")
                goal_col = max(0, min(goal_col, grid_size - 1))
                goal_row = max(0, min(goal_row, grid_size - 1))
                print(f"调整后终点（DEM栅格）: ({goal_col}, {goal_row})")

            # 创建局部路径规划输出目录
            planning_output_dir = os.path.join(dem_build_dir, "local_path_planning_result")
            ensure_dir(planning_output_dir)

            # 调用 C++ 路径规划器
            status, path_points_dem = local_path_planner.plan_path(
                dem_path=dem_txt_path,
                start_col=start_col,
                start_row=start_row,
                goal_col=goal_col,
                goal_row=goal_row,
                grid_size=resolution,
                output_dir=planning_output_dir
            )

            # 处理路径规划结果
            if status == "OK" and path_points_dem is not None:
                print(f"局部路径规划成功，得到 {len(path_points_dem)} 个路径点")
                
                # 可视化规划结果（修复：传入字符串路径）
                visualize_planning_results(dem_txt_path, planning_output_dir, 
                                          start_col, start_row, goal_col, goal_row)

                # 将 DEM 栅格坐标转换为世界坐标
                local_path_points_world = []
                
                # 记录构建DEM时的传感器Z（作为基准）
                start_z_build = current_z
                
                for col, row in path_points_dem:
                    # DEM -> 局部坐标
                    x_local, y_local = dem_grid_to_local(col, row, dem_range, resolution)
                    
                    # 局部 -> 世界坐标（XY）
                    x_world, y_world = local_to_world(
                        x_local, y_local,
                        start_world[0], start_world[1], current_yaw
                    )
                    
                    # 关键修正：Z坐标计算
                    # dem_grid[row, col] 是相对于构建时刻传感器的Z偏移（Z向下为正）
                    # 地面实际Z = 传感器Z + 偏移 + 雷达高度偏移（因为DEM是相对于雷达的，地面在雷达下方）
                    z_relative = dem_grid[row, col]
                    z_world = start_z_build + z_relative - LIDAR_HEIGHT_OFFSET
                    
                    local_path_points_world.append((x_world, y_world, z_world))

                # 绘制局部路径（紫色）
                draw_local_path(client, local_path_points_world, color=(1, 0, 1))
                print(f"局部路径已转换到世界坐标，共 {len(local_path_points_world)} 个点")

                # ===== 沿局部路径行走阶段 =====
                print(f"\n{'=' * 70}")
                print(f"【沿局部路径行走阶段】")
                print(f"{'=' * 70}")

                # 沿局部路径行走（采集点云）
                current_x, current_y, current_z, current_yaw = move_along_local_path(
                    client=client,
                    accumulator=pointcloud_accumulator,
                    local_path_points_world=local_path_points_world,
                    start_z_world=start_z_build,
                    dem_grid=dem_grid,
                    dem_range=dem_range,
                    resolution=resolution,
                    trajectory_points=trajectory_points,
                    current_yaw=current_yaw
                )

                # 更新全局索引
                current_global_idx = goal_global_idx
                print(f"已完成局部路径行走，当前全局索引: {current_global_idx}")

            else:
                # 路径规划失败，退出程序
                print(f"错误：局部路径规划失败，状态: {status}")
                
                # 尝试可视化失败结果（如果有costmap）
                try:
                    visualize_planning_results(dem_txt_path, planning_output_dir, 
                                              start_col, start_row, goal_col, goal_row)
                except Exception as e:
                    print(f"可视化失败结果失败: {e}")
                
                print("程序退出")
                return

            time.sleep(0.2)

        print("\n" + "=" * 70)
        print("路径跟踪与局部 DEM 构建完成！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        traceback.print_exc()
    finally:
        client.enableApiControl(False, VEHICLE_NAME)
        print("已禁用 API 控制")


if __name__ == "__main__":
    main()