"""
全局配置参数模块
包含所有可调参数和常量配置
"""

# ========================================
# 车辆与仿真配置
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

# ========================================
# LiDAR 配置参数
# ========================================
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

# ========================================
# 局部 DEM 参数
# ========================================
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
# 数据文件夹路径（相对于当前工作目录）
DATA_DIR = "global_path_file"

# ========================================
# C++ 路径规划器配置
# ========================================
PATH_PLANNING_EXE = r"D:\Graduation_design\code\path_planning_based_on_lunar_DEM\out\build\x64-Debug\path_planning_based_on_lunar_DEM.exe"
PATH_PLANNING_TIMEOUT = 15.0

# 雷达传感器离地面的高度（Z轴向下为正，所以实际地面比传感器低 LIDAR_HEIGHT_OFFSET）
LIDAR_HEIGHT_OFFSET = 0.5  # 50cm = 0.5m