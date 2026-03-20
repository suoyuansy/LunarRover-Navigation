"""
全局配置参数模块
包含所有可调参数和常量配置
"""

from pathlib import Path

# ========================================
# 路径基础配置
# ========================================
PROJECT_ROOT = Path(__file__).resolve().parent

# 全局路径文件目录（兼容原逻辑）
DATA_DIR = str(PROJECT_ROOT / "global_path_file")
OUTPUT_ROOT_DIR = "local_planningpath"

# 全局原始输入数据目录（用于调用 C++ 全局交互规划）
GLOBAL_SOURCE_DATA_DIR = PROJECT_ROOT / "data"
GLOBAL_DEM_TIF_PATH = GLOBAL_SOURCE_DATA_DIR / "CE7DEM_1km.tif"
GLOBAL_COLOR_PNG_PATH = GLOBAL_SOURCE_DATA_DIR / "CE7DEM_1km_color.png"

# 全局交互规划输出目录
GLOBAL_OUTPUT_DIR = PROJECT_ROOT / "global_path_file"

# 是否启用 C++ 全局交互路径规划
# False: 不调用 C++，直接读取 global_path_file 下已有的 dem/path/costmap
# True : 先调用 C++ 全局交互规划器，再读取 global_path_file 下结果
ENABLE_CPP_GLOBAL_PATH_PLANNING = True

# 全局 DEM 分辨率（米）
GLOBAL_DEM_RESOLUTION = 1.0

# ========================================
# 车辆与仿真配置
# ========================================
VEHICLE_NAME = "Car1"

# 路径跟踪相关
STEP_DISTANCE = 0.25
STEP_DISTANCE_DELAY = 0.1
STEP_ANGLE = 30
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

LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET = 3
LIDAR_POLL_INTERVAL = 0.005
LIDAR_WAIT_TIMEOUT_PER_FRAME = 10

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

LOCAL_GOAL_LOOKAHEAD_POINTS = 20

DEM_GAUSSIAN_SIGMA = 1.0
DEM_USE_GDAL_FILL = True
GRID_FUSION_METHOD = "median"

INITIAL_BUILD_AFTER_SEGMENT_INDEX = 0

# ========================================
# C++ 路径规划器配置
# ========================================
PATH_PLANNING_EXE = r"D:\Graduation_design\code\path_planning_based_on_lunar_DEM\out\build\x64-Debug\path_planning_based_on_lunar_DEM.exe"
PATH_PLANNING_TIMEOUT = 300.0

# 雷达传感器离地面的高度（Z轴向下为正）
LIDAR_HEIGHT_OFFSET = 1.5

# ========================================
# 终点挪动与评分参数
# ========================================
PATH_REPLAN_ROOT_DIRNAME = "path_replan"
LOCAL_MOVE_ENDPOINT_DIRNAME = "local_replan(move_the_endpoint)"
LOCAL_SOFTEN_DIRNAME = "local_replan(soften_obstacles)"
GLOBAL_REPLAN_DIRNAME = "global_path_replan"

# 候选终点搜索最大半径（栅格）
CANDIDATE_SEARCH_MAX_RADIUS_CELLS = 8

# 终点评分时的邻域统计半径（栅格）
CLEARANCE_NEIGHBORHOOD_RADIUS_CELLS = 10

# ========================================
# 全局 costmap / 修正参数
# ========================================
# 注意：启用 C++ 全局交互规划后，最终仍然会从 global_path_file/costmap.txt 读取
GLOBAL_COSTMAP_FILENAME = "costmap.txt"
GLOBAL_COSTMAP_RESOLUTION = 1.0
GLOBAL_OBSTACLE_INFLATION_RADIUS = 1

# 全局重规划时：以“原局部终点之后 3 个全局路径点”作为连接点
GLOBAL_REPLAN_SKIP_POINTS = 3

# 软化障碍时把 1 改成 0.99
REVISION_SOFT_COST = 0.99

# 局部 / 全局起点软化的最大尝试半径
START_RELAX_MAX_RADIUS_CELLS = 20
