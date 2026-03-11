import airsim
import time
import math
import numpy as np
import os

# ========================================
# 计算机视觉模式路径跟踪控制
# 直接控制位姿，无物理模拟
# ========================================

# 全局配置参数
VEHICLE_NAME = "Car1"
STEP_DISTANCE = 0.5          # 每步移动距离（米）
STEP_DISTANCE_DELAY = 0.1    # 每步延迟（秒），控制移动速度
STEP_ANGLE = 10               # 每步移动角度（°）
STEP_ANGLE_DELAY = 0.25/2      # 每步延迟（秒），控制角度变化速度

DRAW_DEBUG = True
# 高度偏移（蓝色规划线比实际路径低的高度）
PLANNED_PATH_Z_OFFSET = 0.5  # 蓝色线绘制在实际高度下方0.5米处

# DEM数据存储
DEM_DATA = None  # 将存储1000x1000的高程数据
DEM_ROWS = 1000
DEM_COLS = 1000
ORIGIN_HEIGHT = 0.0  # 原点(0,0)的高度

# 路径点将由文件读取生成
WAYPOINTS = []

# 测试路径点定义 (x, y, z)
WAYPOINTS_TEST = [
    (0, 1, 0),       # 起点
    (1, 1, 0),      # 点1
    (1, 2, 0),     # 点2
    (2, 2, 0),      # 点3
    (3, 2, 0),       # 点4（终点）
]

def read_dem_file(filename="dem.txt"):
    """读取DEM高程数据文件"""
    global DEM_DATA, ORIGIN_HEIGHT
    
    print("开始读取DEM数据...")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"DEM文件 {filename} 不存在")
    
    dem_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割每行的数值
            values = [float(v) for v in line.split()]
            dem_list.append(values)
    
    # 转换为numpy数组
    DEM_DATA = np.array(dem_list)
    rows, cols = DEM_DATA.shape
    
    print(f"读取完成，数据总共{rows}行{cols}列")
    
    # 获取原点高度（第1行第1列，索引为[0][0]）
    ORIGIN_HEIGHT = DEM_DATA[0, 0]
    print(f"原点(0,0)高度: {ORIGIN_HEIGHT}m")

def calculate_airsim_elevation():
    """计算AirSim坐标系下的高程坐标（相对高差取反）"""
    global DEM_DATA
    
    print("开始计算AirSim坐标系下的高程坐标...")
    
    # AirSim坐标系Z朝下，相对高差 = -(实际高度 - 原点高度)
    # 即：比原点高的地方，Z为负值；比原点低的地方，Z为正值
    DEM_DATA = -(DEM_DATA - ORIGIN_HEIGHT)
    
    print("AirSim坐标系高程坐标计算完毕")

def get_elevation(x, y):
    """根据坐标获取AirSim坐标系下的高程"""
    global DEM_DATA
    
    # 坐标转换为整数索引
    col = int(round(x))
    row = int(round(y))
    
    # 边界检查
    if row < 0 or row >= DEM_ROWS or col < 0 or col >= DEM_COLS:
        print(f"警告：坐标({x}, {y})超出DEM范围，使用默认高度0")
        return 0.0
    
    return float(DEM_DATA[row, col])

def read_path_file(filename):
    """读取路径文件"""
    global WAYPOINTS
    
    print(f"开始读取路径文件 {filename}...")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"路径文件 {filename} 不存在")
    
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    # 解析路径点，格式：(520,465)->(519,464)->...
    waypoints = []
    # 移除所有空白字符
    content = content.replace(' ', '').replace('\n', '')
    
    # 分割箭头
    points_str = content.split('->')
    
    for point_str in points_str:
        point_str = point_str.strip()
        if not point_str:
            continue
        
        # 解析 (x,y) 格式
        if point_str.startswith('(') and point_str.endswith(')'):
            point_str = point_str[1:-1]  # 移除括号
            xy = point_str.split(',')
            if len(xy) == 2:
                x = float(xy[0])
                y = float(xy[1])
                waypoints.append((x, y))
    
    print(f"路径文件读取完成，共{len(waypoints)}个路径点")
    return waypoints

def generate_waypoints_with_elevation(path_points):
    """生成包含高程的路径点列表"""
    global WAYPOINTS
    
    print("开始生成带高程的路径点...")
    
    WAYPOINTS = []
    for i, (x, y) in enumerate(path_points):
        # 获取高程（AirSim坐标系）
        z = get_elevation(x, y)
        WAYPOINTS.append((x, y, z))
        if i < 3 or i >= len(path_points) - 3:  # 打印前3个和后3个
            print(f"  路径点 {i}: ({x}, {y}, {z:.3f})")
        elif i == 3:
            print("  ...")
    
    print(f"路径点生成完成，共{len(WAYPOINTS)}个点")

def get_vehicle_pose(client):
    """获取车辆位姿 (x, y, z, yaw)"""
    pose = client.simGetVehiclePose(VEHICLE_NAME)
    pos = pose.position
    ori = pose.orientation
    
    # 四元数转欧拉角 (yaw)
    x, y, z, w = ori.x_val, ori.y_val, ori.z_val, ori.w_val
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    return pos.x_val, pos.y_val, pos.z_val, yaw


def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_distance(x1, y1, x2, y2):
    """计算两点距离"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_yaw_between_points(x1, y1, x2, y2):
    """计算从点1指向点2的朝向角（弧度）"""
    return math.atan2(y2 - y1, x2 - x1)


def set_vehicle_pose(client, x, y, z, yaw, pitch=0, roll=0):
    """直接设置车辆位姿"""
    pose = airsim.Pose()
    pose.position = airsim.Vector3r(x, y, z)
    # 使用 airsim.to_quaternion(pitch, roll, yaw) 转换
    pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
    client.simSetVehiclePose(pose, True, VEHICLE_NAME)


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
    client.simPlotLineList(line_points, color_rgba=[*color, 1], thickness=thickness, is_persistent=persistent)


def draw_waypoint_marker(client, x, y, z=0, color=(0, 0, 1), size=8.0):
    """绘制目标点标记"""
    if not DRAW_DEBUG:
        return
    client.simPlotPoints([airsim.Vector3r(x, y, z - 0.5)], color_rgba=[*color, 1], size=size, is_persistent=True)


# ========================================
# 核心运动控制
# ========================================

def interpolate_yaw(current_yaw, target_yaw, alpha):
    """
    角度插值
    alpha: 0.0 -> 当前角度, 1.0 -> 目标角度
    """
    angle_diff = normalize_angle(target_yaw - current_yaw)
    return normalize_angle(current_yaw + angle_diff * alpha)


def draw_planned_path(client, waypoints_with_yaw):
    """预先绘制规划路径（蓝色持久线），高度降低以区分于实际轨迹"""
    print("绘制全局规划路径（蓝色，高度降低显示）...")
    
    for i in range(len(waypoints_with_yaw) - 1):
        p1 = waypoints_with_yaw[i]
        p2 = waypoints_with_yaw[i + 1]
        
        # 蓝色规划线：高度降低 PLANNED_PATH_Z_OFFSET
        draw_line(client, 
                 (p1[0], p1[1], p1[2] + PLANNED_PATH_Z_OFFSET), 
                 (p2[0], p2[1], p2[2] + PLANNED_PATH_Z_OFFSET), 
                 color=(0, 0, 1), thickness=3.0, persistent=True)
        
        # 蓝色路径点标记：同样降低高度
        draw_waypoint_marker(client, p1[0], p1[1], p1[2] + PLANNED_PATH_Z_OFFSET, color=(0, 0, 1))
    
    # 绘制最后一个点
    last = waypoints_with_yaw[-1]
    draw_waypoint_marker(client, last[0], last[1], last[2] + PLANNED_PATH_Z_OFFSET, color=(0, 0, 1))
    
    print(f"已绘制 {len(waypoints_with_yaw)} 个路径点（高度偏移: {PLANNED_PATH_Z_OFFSET}m）")

def move_to_target_constant_yaw(client, current_x, current_y, current_z, current_yaw,
                               target_x, target_y, target_z, target_yaw, 
                               trajectory_points, segment_idx):
    """
    控制车辆从当前位置移动到目标位置
    运动时保持yaw不变，到达后才转向（yaw渐变）
    实时绘制红色运动轨迹
    """
    global TRAJECTORY_POINTS

    # 第一阶段：直线移动（保持当前yaw不变）
    total_distance = calculate_distance(current_x, current_y, target_x, target_y)
    num_steps = max(int(total_distance / STEP_DISTANCE), 1)
    
    current_yaw_deg = math.degrees(current_yaw)
    target_yaw_deg = math.degrees(target_yaw)
    
    print(f"路段 {segment_idx}: ({current_x:.1f}, {current_y:.1f}) -> ({target_x:.1f}, {target_y:.1f})")
    print(f"距离: {total_distance:.2f}m, 步数: {num_steps}")
    #print(f"    移动时yaw保持: {current_yaw_deg:.1f}°")
    
    prev_x, prev_y, prev_z = current_x, current_y, current_z
    
    # 直线移动阶段（yaw恒定）
    for step in range(1, num_steps + 1):
        alpha = step / num_steps
        
        # 位置线性插值，yaw保持不变
        x = current_x + (target_x - current_x) * alpha
        y = current_y + (target_y - current_y) * alpha
        z = current_z + (target_z - current_z) * alpha
        yaw = current_yaw  # 保持yaw不变
        
        # 设置位姿
        set_vehicle_pose(client, x, y, z, yaw)
        
        # 记录轨迹点
        trajectory_points.append((x, y, z))
        
        # 实时绘制实际轨迹（红色持久线）
        draw_line(client, (prev_x, prev_y, prev_z), (x, y, z),color=(1, 0, 0), thickness=2.0, persistent=True)
        
        prev_x, prev_y, prev_z = x, y, z
        
        # 控制速度
        time.sleep(STEP_DISTANCE_DELAY)
    
    # 确保精确到达目标点（位置）
    set_vehicle_pose(client, target_x, target_y, target_z, current_yaw)
    trajectory_points.append((target_x, target_y, target_z))
    
    print(f"到达目标位置 ({target_x:.2f}, {target_y:.2f})")
    
    # 第二阶段：原地转向（yaw渐变）
    if abs(normalize_angle(target_yaw - current_yaw)) > 0.01:  # 需要转向
        yaw_diff = abs(normalize_angle(target_yaw - current_yaw))
        yaw_deg_diff = math.degrees(yaw_diff)
        turn_steps = max(int(yaw_deg_diff / STEP_ANGLE), 1)  # 每5度一步，至少1步
        
        print(f"开始转向: {current_yaw_deg:.1f}° -> {target_yaw_deg:.1f}° (差值: {yaw_deg_diff:.1f}°)")
        
        for step in range(1, turn_steps + 1):
            alpha = step / turn_steps
            yaw = interpolate_yaw(current_yaw, target_yaw, alpha)
            
            # 在目标位置原地旋转
            set_vehicle_pose(client, target_x, target_y, target_z, yaw)
            
            time.sleep(STEP_ANGLE_DELAY)  # 转向可以快一点
        
        print(f"转向完成，当前yaw: {target_yaw_deg:.1f}°")
    
    return target_x, target_y, target_z, target_yaw


# ========================================
# 主函数
# ========================================

def main():
    """主程序入口"""
    print("=" * 60)
    print("airsim计算机视觉模式路径跟踪控制程序")
    print("=" * 60)
    
    # 读取DEM数据
    read_dem_file("dem.txt")
    # 计算AirSim坐标系高程
    calculate_airsim_elevation()
    # 读取路径文件（自动查找匹配的文件）
    # 查找当前目录下的路径文件
    path_file = None
    for file in os.listdir('.'):
        if file.startswith("path") and file.endswith(".txt"):
            path_file = file
            break

    if path_file is None:
        raise FileNotFoundError("未找到路径文件，请确保文件名为 path(...).txt 格式")

    path_points = read_path_file(path_file)
    # 生成带高程的路径点
    generate_waypoints_with_elevation(path_points)

    if len(WAYPOINTS) < 2:
        raise ValueError("路径点数量不足，至少需要2个点")
    
    print(f"\n准备执行路径跟踪，共{len(WAYPOINTS)}个路径点")
    print("=" * 60)
    
    # ========== AirSim连接阶段 ==========
    # 连接模拟器（CV模式使用 VehicleClient 或 MultirotorClient 都可以）
    client = airsim.VehicleClient()
    client.confirmConnection()
    
    # 初始化
    client.enableApiControl(True, VEHICLE_NAME)
    client.simFlushPersistentMarkers()
    
    print(f"\n已启用API控制: {VEHICLE_NAME}")
    print("已清除之前的调试标记")
    
    # 计算每个路径点的朝向
    # 每个点的朝向 = 该点指向下一个点的方向，最后一个点保持前一个朝向
    waypoints_with_yaw = []
    
    for i, (x, y, z) in enumerate(WAYPOINTS):
        if i < len(WAYPOINTS) - 1:
            # 不是最后一个点：朝向下一个点
            next_x, next_y, next_z = WAYPOINTS[i + 1]
            yaw = calculate_yaw_between_points(x, y, next_x, next_y)
        else:
            # 最后一个点：保持上一个点的朝向
            yaw = waypoints_with_yaw[-1][3] if waypoints_with_yaw else 0
        
        waypoints_with_yaw.append((x, y, z, yaw))
    
    # 预先绘制规划路径（蓝色）
    draw_planned_path(client, waypoints_with_yaw)
    time.sleep(2)
    # 执行路径跟踪
    trajectory_points = []
    
    # 瞬移到起始点，yaw设置为指向第一个目标点的方向
    first_wp = waypoints_with_yaw[0]
    start_yaw = first_wp[3]  # 起点朝向 = 指向第一个目标点的方向
    
    print(f"\n【瞬移到起始点】({first_wp[0]}, {first_wp[1]}, {first_wp[2]}), 起点yaw设置为: {math.degrees(start_yaw):.1f}°")
    time.sleep(3)

    # 瞬移设置位姿
    set_vehicle_pose(client, first_wp[0], first_wp[1], first_wp[2], start_yaw)
    time.sleep(0.5)
    
    # 获取当前实际位姿
    current_x, current_y, current_z, current_yaw = get_vehicle_pose(client)
    print(f"  当前位置: ({current_x:.2f}, {current_y:.2f}, {current_z:.2f}), 朝向: {math.degrees(current_yaw):.1f}°")
    
    try:
        for i in range(len(waypoints_with_yaw) - 1):
            # 当前路径点
            wp_current = waypoints_with_yaw[i]
            # 下一个路径点（目标）
            wp_target = waypoints_with_yaw[i + 1]
            
            print(f"\n{'='*60}")
            print(f"路段 {i+1}/{len(waypoints_with_yaw)-1}")
            print(f"{'='*60}")
            
            # 从当前位置移动到目标位置
            # 运动时保持当前yaw，到达后转向到目标yaw
            current_x, current_y, current_z, current_yaw = move_to_target_constant_yaw(
                client,
                current_x, current_y, current_z, current_yaw,  # 起始位姿
                wp_target[0], wp_target[1], wp_target[2], wp_target[3],  # 目标位姿
                trajectory_points,
                i + 1
            )
            # 路径点间短暂停顿
            time.sleep(0.2)
        
        print("路径跟踪完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.enableApiControl(False, VEHICLE_NAME)
        print("已禁用API控制")


if __name__ == "__main__":
    main()