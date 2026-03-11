import airsim
import time
import math
import numpy as np
import os
import traceback
from datetime import datetime
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# 尝试使用 GDAL 填充空洞；如果环境未安装 GDAL，则自动退回 scipy 插值
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except Exception:
    GDAL_AVAILABLE = False


# ========================================
# 基于 AirSim CV 模式 + LiDAR 的路径行走与 DEM 构建
# 当前版本：
# 1) 不做局部路径规划
# 2) 只在指定路径段上累积点云并构建局部 DEM
# 3) 每次构建 DEM 时，将历史各帧点云统一变换到“当前构建时刻的传感器坐标系”下
# 4) 核心修正：
#    set_vehicle_pose() 后，不立刻保存 LiDAR
#    而是等待 N 帧“新的 timestamp”点云后，再保存最后一帧
# 5) 直线移动与转向过程中都按上述逻辑采样
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
PLANNED_PATH_Z_OFFSET = 0.5  # 蓝色规划路径降低显示，便于和红色实际轨迹区分

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

# 新增：每次设定完位姿后，等待多少帧“新的雷达点云”
LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET = 3

# 新增：等待新帧时的轮询间隔和超时
LIDAR_POLL_INTERVAL = 0.005
LIDAR_WAIT_TIMEOUT_PER_FRAME = 10   # 每等一帧新数据的最长时间

# 点云预处理
LIDAR_MIN_VALID_DISTANCE = 1.0   # 去掉离传感器过近的点，避免打到车体/传感器自身
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
# 可选：'median' / 'mean' / 'min' / 'max'
# 对当前问题，推荐 median，更抗离群点
GRID_FUSION_METHOD = "median"

# 构建 DEM 的策略
INITIAL_BUILD_AFTER_SEGMENT_INDEX = 0
BUILD_EVERY_N_SEGMENTS = 4

# 输出目录
OUTPUT_ROOT_DIR = "local_planningpath"


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
    """
    创建本次运行根目录：
    ./local_planningpath/local_planningpath_{run_timestamp}/
    """
    root_dir = os.path.join(os.getcwd(), OUTPUT_ROOT_DIR)
    ensure_dir(root_dir)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir_name = f"local_planningpath_{run_timestamp}"
    run_dir = os.path.join(root_dir, run_dir_name)
    ensure_dir(run_dir)

    return root_dir, run_dir, run_timestamp


def create_single_build_folder(run_dir):
    """
    在本次运行目录下创建一次 DEM 构建目录：
    ./local_planningpath/local_planningpath_{run_timestamp}/local_planningpath_{build_timestamp}/
    """
    build_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    build_dir_name = f"local_planningpath_{build_timestamp}"
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
    """
    将全局 DEM 转为 AirSim 坐标系下的 Z 值：
    原始 DEM 高度：向上为正
    当前使用坐标系：Z 向下，所以高度越高，Z 越小
    """
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
    """
    四元数转欧拉角
    返回: roll, pitch, yaw（弧度）
    """
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
    """
    欧拉角转旋转矩阵
    旋转顺序：Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ], dtype=np.float64)

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ], dtype=np.float64)

    rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ], dtype=np.float64)

    return rz @ ry @ rx


def get_vehicle_pose_full(client):
    """
    获取车辆位姿
    返回:
        x, y, z, roll, pitch, yaw
    注意：
        坐标系按 AirSim/UE 当前仿真坐标，Z 向下
    """
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
    """
    角度插值
    alpha: 0.0 -> 当前角度, 1.0 -> 目标角度
    """
    angle_diff = normalize_angle(target_yaw - current_yaw)
    return normalize_angle(current_yaw + angle_diff * alpha)


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
    """预先绘制规划路径（蓝色持久线），高度降低以区分于实际轨迹"""
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


# ========================================
# LiDAR 数据采集与 DEM 构建
# ========================================
class PointCloudAccumulator:
    """
    在一段运动过程中持续采集 LiDAR 点云
    注意：
    - 每帧点云保存在其“采样时刻的传感器局部坐标系”下
    - 同时记录该帧采样时刻的传感器位姿
    - 构建 DEM 时，再统一变换到“当前构建时刻的传感器坐标系”
    - 核心修正：
      set_vehicle_pose 后，不立刻保存点云，而是等待若干帧“新的 timestamp”后再保存最后一帧
    """

    def __init__(self, vehicle_name=VEHICLE_NAME, sensor_name=LIDAR_SENSOR_NAME):
        self.vehicle_name = vehicle_name
        self.sensor_name = sensor_name
        self.client = None
        self.frames = []  # 每个元素: {'points_local': Nx3, 'pose': (x,y,z,roll,pitch,yaw), 'timestamp': ...}
        self.last_lidar_timestamp = None

    def initialize(self, client):
        self.client = client
        print("LiDAR 点云采集模块初始化完成")
        print(f" 传感器名: {self.sensor_name}")
        print(f" 最大有效距离: {LIDAR_MAX_VALID_DISTANCE}m")
        print(f" 最小有效距离: {LIDAR_MIN_VALID_DISTANCE}m")

    def reset(self, clear_timestamp=False):
        """
        重置累计缓存
        默认只清空 frames，不清空 last_lidar_timestamp
        """
        self.frames = []
        if clear_timestamp:
            self.last_lidar_timestamp = None

    def _wait_for_next_new_lidar_frame(self, timeout=LIDAR_WAIT_TIMEOUT_PER_FRAME, poll_interval=LIDAR_POLL_INTERVAL):
        """
        等待一帧“新的 timestamp” LiDAR 数据
        返回：
            lidar_data 或 None
        """
        if self.client is None:
            raise RuntimeError("PointCloudAccumulator 尚未 initialize(client)")

        deadline = time.time() + timeout

        while time.time() < deadline:
            lidar_data = self.client.getLidarData(self.sensor_name, self.vehicle_name)

            if lidar_data is None or lidar_data.point_cloud is None or len(lidar_data.point_cloud) < 3:
                time.sleep(poll_interval)
                continue

            ts = lidar_data.time_stamp

            # 必须是新帧
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
        """
        在设置完位姿后调用：
        先等待 wait_new_frames 帧“新的 timestamp”雷达数据，
        最终保存最后一帧点云，并把它绑定到 expected_pose

        参数：
            expected_pose: (x,y,z,roll,pitch,yaw)
            wait_new_frames: 等待多少帧新的雷达点云
        返回：
            保存的有效点数
        """
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
        """
        LiDAR 点云预处理：
        1. 去掉 NaN / inf
        2. 去掉离雷达过近点（可能打到车体、传感器自身）
        3. 去掉超量程点
        4. 可选的 Z 范围过滤
        """
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
        """
        保存每帧点云到 point_data 目录
        命名格式：
        序号_ts(... )_roll(... )_pitch(... )_yaw(... ).txt
        """
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
    """
    根据累计的多帧点云，在“当前传感器坐标系”下构建局部 DEM

    重要：
    - 历史帧点云先从各自采样时刻的传感器坐标系
      -> 世界坐标系
      -> 当前构建时刻的传感器坐标系
    - DEM 以“当前传感器”为参考
    - 左前方为 DEM 原点
    - 行：前 -> 后
    - 列：左 -> 右
    """

    def __init__(self, dem_range=LOCAL_DEM_RANGE, resolution=LOCAL_DEM_RESOLUTION):
        self.dem_range = dem_range
        self.resolution = resolution
        self.grid_size = int((dem_range * 2) / resolution)

    def build_dem_from_frames(self, frames, current_pose):
        """
        输入：
            frames: 累积的点云帧，每帧点都在各自采样时刻的传感器坐标系下
            current_pose: 当前构建时刻的传感器/车体位姿 (cx,cy,cz,roll,pitch,yaw)
        输出：
            dem_grid: HxW，存储的是“当前传感器坐标系下的 z 值（Z向下）”
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
        # 当前坐标系定义：
        # x: 前
        # y: 右
        # z: 下
        x_local = all_points_in_current_sensor[:, 0]  # forward
        y_local = all_points_in_current_sensor[:, 1]  # right
        z_local = all_points_in_current_sensor[:, 2]  # down

        # 仅用于裁剪时表达左右对称范围，左为 -y
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
                # 默认 median
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
        """
        保存：
        1. dem.txt
        2. dem_meta.txt
        3. dem_3d.jpg
        """
        cx, cy, cz, roll, pitch, yaw = current_pose

        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        dem_txt_path = os.path.join(build_dir, "dem.txt")
        meta_txt_path = os.path.join(build_dir, "dem_meta.txt")
        jpg_path = os.path.join(build_dir, "dem_3d.jpg")

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

        self._save_3d_visualization(dem_grid, jpg_path)

        print(f"DEM 已保存: {dem_txt_path}")
        print(f"元数据已保存: {meta_txt_path}")
        print(f"三维图已保存: {jpg_path}")

    def _save_3d_visualization(self, dem_grid, jpg_path):
        """保存 DEM 三维可视化图"""
        rows, cols = dem_grid.shape

        # DEM 存储定义：
        # row = 0 -> 最前
        # row = end -> 最后
        # col = 0 -> 最左
        # col = end -> 最右
        #
        # 左前方原点即：
        # (row=0, col=0)

        row_coords = np.arange(rows) * self.resolution
        col_coords = np.arange(cols) * self.resolution
        xx, yy = np.meshgrid(col_coords, row_coords)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 当前 DEM 存的 z 是“向下为正”
        # 为了可视化上更直观，这里取负值显示
        zz_vis = -dem_grid

        ax.plot_surface(xx, yy, zz_vis, cmap='terrain', linewidth=0, antialiased=True)

        # 反转 x 轴显示方向，使得 col=0 与 row=0 出现在同一个左前角
        ax.invert_xaxis()
        ax.set_title("Local DEM 3D Visualization")
        ax.set_xlabel("Column (left -> right, m)")
        ax.set_ylabel("Row (front -> back, m)")
        ax.set_zlabel("Height for visualization (-Z)")

        plt.tight_layout()
        plt.savefig(jpg_path, dpi=200)
        plt.close(fig)


# ========================================
# 核心运动控制
# ========================================
def move_to_target_constant_yaw(client, accumulator,
                                current_x, current_y, current_z, current_yaw,
                                target_x, target_y, target_z, target_yaw,
                                trajectory_points, segment_idx):
    """
    控制车辆从当前位置移动到目标位置
    运动时保持 yaw 不变，到达后才转向（yaw 渐变）
    实时绘制红色运动轨迹
    并在移动与转向过程中持续采集点云

    核心修改：
    - 每次 set_vehicle_pose() 后，不立刻保存 LiDAR
    - 先等待三帧新的雷达点云(timestamp区分)
    - 再保存最后一帧作为该停顿点对应点云
    - 转向过程中也照常采样，同样每次转动后等待三帧新点云再保存
    """
    total_distance = calculate_distance(current_x, current_y, target_x, target_y)
    num_steps = max(int(total_distance / STEP_DISTANCE), 1)

    current_yaw_deg = math.degrees(current_yaw)
    target_yaw_deg = math.degrees(target_yaw)

    print(f"路段 {segment_idx}: ({current_x:.1f}, {current_y:.1f}) -> ({target_x:.1f}, {target_y:.1f})")
    print(f"距离: {total_distance:.2f}m, 步数: {num_steps}")

    prev_x, prev_y, prev_z = current_x, current_y, current_z

    # 第一阶段：直线移动（yaw 保持不变）
    for step in range(1, num_steps + 1):
        alpha = step / num_steps

        x = current_x + (target_x - current_x) * alpha
        y = current_y + (target_y - current_y) * alpha
        z = current_z + (target_z - current_z) * alpha
        yaw = current_yaw

        set_vehicle_pose(client, x, y, z, yaw)

        trajectory_points.append((x, y, z))

        # 实时绘制实际轨迹（红色持久线）
        draw_line(
            client,
            (prev_x, prev_y, prev_z),
            (x, y, z),
            color=(1, 0, 0),
            thickness=2.0,
            persistent=True
        )
        prev_x, prev_y, prev_z = x, y, z

        # 关键修正：
        # 设置完位姿之后，先等待三帧新的雷达点云数据，再保存最后一帧
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

        # 获取三帧新的雷达数据之后，再额外等待 STEP_DISTANCE_DELAY
        #time.sleep(STEP_DISTANCE_DELAY)

    # 保证精确到达
    set_vehicle_pose(client, target_x, target_y, target_z, current_yaw)
    trajectory_points.append((target_x, target_y, target_z))

    print(f"到达目标位置 ({target_x:.2f}, {target_y:.2f})")

    # 第二阶段：原地转向
    if abs(normalize_angle(target_yaw - current_yaw)) > 0.01:
        yaw_diff = abs(normalize_angle(target_yaw - current_yaw))
        yaw_deg_diff = math.degrees(yaw_diff)
        turn_steps = max(int(yaw_deg_diff / STEP_ANGLE), 1)

        print(f"开始转向: {current_yaw_deg:.1f}° -> {target_yaw_deg:.1f}° (差值: {yaw_deg_diff:.1f}°)")

        for step in range(1, turn_steps + 1):
            alpha = step / turn_steps
            yaw = interpolate_yaw(current_yaw, target_yaw, alpha)

            set_vehicle_pose(client, target_x, target_y, target_z, yaw)

            # 转向过程照常采样：
            # 每次转动一次位姿，等待三帧新的雷达点云数据，再保存最后一帧
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

            # 获取三帧新的雷达数据之后，再额外等待 STEP_ANGLE_DELAY
            #time.sleep(STEP_ANGLE_DELAY)

        print(f"转向完成，当前 yaw: {target_yaw_deg:.1f}°")

    return target_x, target_y, target_z, target_yaw


def should_build_dem_after_segment(segment_index, total_segments):
    """
    判定某一段结束后，是否需要构建 DEM
    规则：
    1. 第0段结束后构建
    2. 之后每4段构建一次
    3. 最后一段若不整齐落在规则点上，也补构一次
    """
    if segment_index == INITIAL_BUILD_AFTER_SEGMENT_INDEX:
        return True

    if segment_index > INITIAL_BUILD_AFTER_SEGMENT_INDEX:
        if (segment_index - INITIAL_BUILD_AFTER_SEGMENT_INDEX) % BUILD_EVERY_N_SEGMENTS == 0:
            return True

    if segment_index == total_segments - 1:
        return True

    return False


# ========================================
# 主函数
# ========================================
pointcloud_accumulator = PointCloudAccumulator(VEHICLE_NAME, LIDAR_SENSOR_NAME)
local_dem_builder = LocalDEMBuilder(LOCAL_DEM_RANGE, LOCAL_DEM_RESOLUTION)


def main():
    print("=" * 70)
    print("AirSim CV模式 + LiDAR 多帧融合局部 DEM 构建")
    print("当前流程: 沿全局路径运动 → 分段累积点云 → 变换到当前传感器坐标系 → 构建局部 DEM")
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

    # 2. 本次运行开始前，先创建本次运行目录
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

    # 瞬移到起点后清空缓存，同时重置 timestamp，同步起点后的第一轮采样
    pointcloud_accumulator.reset(clear_timestamp=True)
    print("已在瞬移到起点后清空点云缓存，后续仅累计出发点之后的点云数据")

    total_segments = len(waypoints_with_yaw) - 1

    try:
        for seg_idx in range(total_segments):
            wp_start = waypoints_with_yaw[seg_idx]
            wp_target = waypoints_with_yaw[seg_idx + 1]

            print(f"\n{'=' * 70}")
            print(f"路段 {seg_idx + 1}/{total_segments}")
            print(f"起点: ({wp_start[0]:.2f}, {wp_start[1]:.2f})")
            print(f"终点: ({wp_target[0]:.2f}, {wp_target[1]:.2f})")
            print(f"{'=' * 70}")

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
                segment_idx=seg_idx + 1
            )

            print(f"已到达路段终点，当前位置: ({current_x:.2f}, {current_y:.2f}, {current_z:.2f})")

            # 判断当前段结束后是否构建 DEM
            do_build = should_build_dem_after_segment(seg_idx, total_segments)

            if do_build:
                print("\n开始构建局部 DEM ...")

                latest_pose_full = get_vehicle_pose_full(client)
                frames = pointcloud_accumulator.get_all_frames()

                if frames is None or len(frames) == 0:
                    print("警告：当前累计点云帧为空，跳过本次 DEM 构建")
                else:
                    build_dir, point_data_dir, build_timestamp = create_single_build_folder(run_dir)
                    print(f"本次构建目录: {build_dir}")

                    # 先保存每帧点云
                    pointcloud_accumulator.save_frames_to_point_data(point_data_dir)

                    dem_grid, valid_mask, fused_points_current = local_dem_builder.build_dem_from_frames(
                        frames=frames,
                        current_pose=latest_pose_full
                    )

                    if dem_grid is not None:
                        local_dem_builder.save_dem_results(
                            dem_grid=dem_grid,
                            current_pose=latest_pose_full,
                            build_dir=build_dir
                        )
                    else:
                        print("警告：DEM 构建失败，本次不保存 DEM")

                # 本次构建完成后，清空缓存，准备下一轮累计
                # 默认不清 timestamp，避免把上一轮最后旧帧再次当成新帧
                pointcloud_accumulator.reset(clear_timestamp=False)

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