"""
核心运动控制模块
包含车辆移动控制和局部路径行走功能
"""

import airsim
import math
import time

# 导入配置
from config import (
    VEHICLE_NAME, STEP_DISTANCE, STEP_ANGLE,
    LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET,
    LIDAR_WAIT_TIMEOUT_PER_FRAME, LIDAR_POLL_INTERVAL
)

# 导入工具函数
from utils import (
    normalize_angle, calculate_distance, calculate_yaw_between_points,
    interpolate_yaw, quaternion_to_eulerian_angles
)

# 导入可视化
from visualization import draw_line


# ========================================
# 车辆位姿操作
# ========================================
def get_vehicle_pose_full(client, vehicle_name=VEHICLE_NAME):
    """获取车辆位姿"""
    pose = client.simGetVehiclePose(vehicle_name)
    pos = pose.position
    ori = pose.orientation

    roll, pitch, yaw = quaternion_to_eulerian_angles(
        ori.x_val, ori.y_val, ori.z_val, ori.w_val
    )

    return pos.x_val, pos.y_val, pos.z_val, roll, pitch, yaw


def set_vehicle_pose(client, x, y, z, yaw, pitch=0.0, roll=0.0, vehicle_name=VEHICLE_NAME):
    """直接设置车辆位姿"""
    pose = airsim.Pose()
    pose.position = airsim.Vector3r(x, y, z)
    pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
    client.simSetVehiclePose(pose, True, vehicle_name)


# ========================================
# 运动控制
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
            thickness=6.0,
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
            accumulator=accumulator,
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