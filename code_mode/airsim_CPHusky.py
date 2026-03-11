import airsim
import time
import math
import numpy as np

# ========================================
# CPHusky小车路径跟踪控制程序
# 基于Cosys-AirSim SkidVehicle模式
# Yaw正方向：顺时针（右转）
# steering > 0：右转，steering < 0：左转
# ========================================

# 全局配置参数
VEHICLE_NAME = "Car1"
TARGET_SPEED = 0.5
POSITION_TOLERANCE = 0.1
ANGLE_TOLERANCE = 0.1
MAX_THROTTLE = 0.6
MIN_THROTTLE = 0.15
DRAW_DEBUG = True

# 路径点定义
WAYPOINTS = [
    (0, 1),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 3)    
]


# ========================================
# 基础工具函数
# ========================================

def get_vehicle_pose(client):
    """获取车辆位姿 (x, y, z, yaw)"""
    state = client.getCarState(VEHICLE_NAME)
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    
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


def calculate_target_angle(current_x, current_y, target_x, target_y):
    """计算目标朝向角"""
    return math.atan2(target_y - current_y, target_x - current_x)


# ========================================
# 控制函数
# ========================================

def set_car_controls(client, throttle, steering, brake=False):
    """发送车辆控制指令"""
    controls = airsim.CarControls()
    controls.throttle = max(min(throttle, 1.0), -1.0)
    controls.steering = max(min(steering, 1.0), -1.0)
    controls.handbrake = brake
    controls.is_manual_gear = False
    client.setCarControls(controls, VEHICLE_NAME)


def stop_vehicle(client):
    """停止车辆"""
    controls = airsim.CarControls()
    controls.throttle = 0
    controls.steering = 0
    controls.handbrake = True
    controls.is_manual_gear = False
    client.setCarControls(controls, VEHICLE_NAME)
    time.sleep(0.3)


def rotate_in_place(client, target_yaw):
    """
    原地转向到目标角度
    """
    while True:
        _, _, _, current_yaw = get_vehicle_pose(client)
        angle_diff = normalize_angle(target_yaw - current_yaw)

        # 到达目标角度
        if abs(angle_diff) < ANGLE_TOLERANCE:
            stop_vehicle(client)
            break

        # 判断方向
        if angle_diff > 0:
            steering = 1.0   # 右转
        else:
            steering = -1.0  # 左转

        if steering < 0:
            set_car_controls(client, throttle=0.02, steering=steering, brake=False)
        else:
            set_car_controls(client, throttle=0.0, steering=steering, brake=False)

        time.sleep(0.05)

    stop_vehicle(client)


def calculate_throttle(current_speed, target_speed, distance_to_target):
    """计算油门值，包含坡道补偿"""
    # 接近目标时减速
    if distance_to_target < 2.0:
        target_speed = target_speed * (distance_to_target / 2.0)
        target_speed = max(target_speed, 0.1)
    
    speed_error = target_speed - current_speed
    
    if speed_error > 0:
        # 需要加速（上坡或起步）
        throttle = MIN_THROTTLE + (speed_error / TARGET_SPEED) * 0.3
        throttle = min(throttle, MAX_THROTTLE)
        return throttle, False
    else:
        # 需要减速（下坡）
        if abs(speed_error) / TARGET_SPEED > 0.3:
            return 0.0, True  # 刹车
        else:
            return MIN_THROTTLE * 0.5, False  # 滑行


# ========================================
# 绘图函数
# ========================================

def draw_waypoint_connection(client, x1, y1, x2, y2):
    """绘制路径规划蓝线"""
    if not DRAW_DEBUG:
        return
    line_points = [
        airsim.Vector3r(x1, y1, -0.5),
        airsim.Vector3r(x2, y2, -0.5)
    ]
    client.simPlotLineList(line_points, color_rgba=[0, 0, 1, 1], thickness=3.0, is_persistent=True)


def draw_waypoint_marker(client, x, y):
    """绘制目标点蓝点"""
    if not DRAW_DEBUG:
        return
    client.simPlotPoints([airsim.Vector3r(x, y, -0.5)], color_rgba=[0, 0, 1, 1], size=8.0, is_persistent=True)


def draw_trajectory_segment(client, p1, p2):
    """绘制轨迹红线段（实时刷新）"""
    if not DRAW_DEBUG:
        return
    line_points = [
        airsim.Vector3r(p1[0], p1[1], p1[2] - 0.3),
        airsim.Vector3r(p2[0], p2[1], p2[2] - 0.3)
    ]
    client.simPlotLineList(line_points, color_rgba=[1, 0, 0, 1], thickness=2.0, is_persistent=False)


def draw_full_trajectory(client, trajectory):
    """绘制完整轨迹（持久化）"""
    if not DRAW_DEBUG or len(trajectory) < 2:
        return
    
    batch_size = 50
    for batch_start in range(0, len(trajectory) - 1, batch_size):
        batch_end = min(batch_start + batch_size, len(trajectory) - 1)
        for j in range(batch_start, batch_end):
            p1, p2 = trajectory[j], trajectory[j + 1]
            line_points = [
                airsim.Vector3r(p1[0], p1[1], p1[2] - 0.2),
                airsim.Vector3r(p2[0], p2[1], p2[2] - 0.2)
            ]
            client.simPlotLineList(line_points, color_rgba=[1, 0, 0, 0.8], thickness=3.0, is_persistent=True)


# ========================================
# 核心运动控制
# ========================================

def move_to_target(client, target_x, target_y, prev_x, prev_y, trajectory):
    """
    控制车辆移动到目标点
    1. 原地转向对准（支持左转和右转）
    2. 直线行驶（带坡道补偿）
    """
    # 绘制规划路径
    draw_waypoint_connection(client, prev_x, prev_y, target_x, target_y)
    draw_waypoint_marker(client, target_x, target_y)
    
    # 获取当前状态
    current_x, current_y, current_z, current_yaw = get_vehicle_pose(client)
    target_angle = calculate_target_angle(current_x, current_y, target_x, target_y)
    
    # 原地转向对准
    raw_angle_diff = normalize_angle(target_angle - current_yaw)
    angle_diff = abs(raw_angle_diff)
    
    if angle_diff > ANGLE_TOLERANCE:
        # 关键修正：交换左右判断
        # raw_angle_diff > 0：目标在右边，右转
        # raw_angle_diff < 0：目标在左边，左转
        turn_direction = "右转" if raw_angle_diff > 0 else "左转"
        print(f"目标点 ({target_x}, {target_y}) - {turn_direction}对准 {math.degrees(target_angle):.1f}度")
        rotate_in_place(client, target_angle)
    else:
        print(f"目标点 ({target_x}, {target_y}) - 已对准，前进")
    
    # 直线行驶到目标
    print(f"  前进到目标...")
    last_print_time = time.time()
    last_position = (current_x, current_y)
    stuck_counter = 0
    
    while True:
        current_x, current_y, current_z, current_yaw = get_vehicle_pose(client)
        state = client.getCarState(VEHICLE_NAME)
        
        # 计算速度和距离
        linear_vel = state.kinematics_estimated.linear_velocity
        current_speed = math.sqrt(linear_vel.x_val**2 + linear_vel.y_val**2)
        distance_to_target = calculate_distance(current_x, current_y, target_x, target_y)
        
        # 卡住检测
        if time.time() - last_print_time > 2.0:
            moved = calculate_distance(current_x, current_y, last_position[0], last_position[1])
            if moved < 0.1:
                stuck_counter += 1
                if stuck_counter > 3:
                    print("  检测到卡住，加大动力...")
                    set_car_controls(client, MAX_THROTTLE, 0, False)
                    time.sleep(0.5)
            else:
                stuck_counter = 0
            last_position = (current_x, current_y)
            last_print_time = time.time()
        
        # 到达检查
        if distance_to_target < POSITION_TOLERANCE:
            stop_vehicle(client)
            print(f"  到达目标点 ({target_x}, {target_y})")
            break
        
        # 计算控制量
        target_angle = calculate_target_angle(current_x, current_y, target_x, target_y)
        angle_error = normalize_angle(target_angle - current_yaw)
        throttle, need_brake = calculate_throttle(current_speed, TARGET_SPEED, distance_to_target)
        
        # 转向控制（P控制器）- 修正方向
        # 注意：这里需要反转符号，因为steering和angle_error的正方向相反
        # angle_error > 0（目标在右边）-> 需要右转 -> steering > 0
        steering = max(min(angle_error * 1.5, 0.6), -0.6)
        
        # 角度偏差过大，重新对准
        if abs(angle_error) > 0.3:
            # 关键修正：交换左右判断
            turn_direction = "右转" if angle_error > 0 else "左转"
            print(f"  角度偏差过大({math.degrees(angle_error):.1f}度)，{turn_direction}重新对准...")
            rotate_in_place(client, target_angle)
            continue
        
        set_car_controls(client, throttle, steering, need_brake)
        
        # 记录并绘制轨迹
        trajectory.append((current_x, current_y, current_z))
        if len(trajectory) > 1 and len(trajectory) % 5 == 0:
            draw_trajectory_segment(client, trajectory[-2], trajectory[-1])
        
        time.sleep(0.05)


# ========================================
# 主函数
# ========================================

def main():
    """主程序入口"""
    print("=" * 50)
    print("CPHusky路径跟踪控制程序启动")
    print("=" * 50)
    
    # 连接模拟器
    client = airsim.CarClient()
    client.confirmConnection()
    
    # 初始化
    client.enableApiControl(True, VEHICLE_NAME)
    client.armDisarm(True, VEHICLE_NAME)
    client.simFlushPersistentMarkers()
    
    print(f"已启用API控制: {VEHICLE_NAME}")
    print("车辆已解锁")
    print("已清除之前的调试标记")
    
    # 获取初始位置
    start_x, start_y, start_z, start_yaw = get_vehicle_pose(client)
    print(f"初始位置: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f}), 朝向: {math.degrees(start_yaw):.1f}度")
    
    # 执行路径跟踪
    trajectory = []
    prev_x, prev_y = start_x, start_y
    
    try:
        for i, (target_x, target_y) in enumerate(WAYPOINTS):
            print(f"\n--- 路径点 {i+1}/{len(WAYPOINTS)}: ({target_x}, {target_y}) ---")
            move_to_target(client, target_x, target_y, prev_x, prev_y, trajectory)
            prev_x, prev_y = target_x, target_y
            time.sleep(0.5)
        
        print("\n" + "=" * 50)
        print("路径跟踪完成！")
        print("=" * 50)
        
        # 绘制完整轨迹
        draw_full_trajectory(client, trajectory)
        print("完整轨迹已保存")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_vehicle(client)
        client.enableApiControl(False, VEHICLE_NAME)
        print("已禁用API控制")


if __name__ == "__main__":
    main()