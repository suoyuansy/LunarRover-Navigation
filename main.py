"""
主程序入口
基于 AirSim CV 模式 + LiDAR 的局部路径规划与 DEM 构建
流程: 初始全局路径行走 → 构建DEM → 局部路径规划 → 沿局部路径行走 → 构建DEM → ...
"""

import airsim
import time
import traceback
import os

# 导入配置
from config import (
    VEHICLE_NAME, LIDAR_SENSOR_NAME,
    LOCAL_DEM_RANGE, LOCAL_DEM_RESOLUTION,
    INITIAL_BUILD_AFTER_SEGMENT_INDEX, BUILD_EVERY_N_SEGMENTS,
    LIDAR_HEIGHT_OFFSET,DATA_DIR
)

# 导入工具函数
from utils import (
    read_dem_file, calculate_airsim_elevation, read_path_file,
    generate_waypoints_with_elevation, calculate_yaw_between_points,
    create_run_output_folder, create_single_build_folder, ensure_dir
)

# 导入坐标转换
from coordinate_transform import (
    world_to_local, local_to_dem_grid, dem_grid_to_local, local_to_world
)

# 导入可视化
from visualization import (
    draw_planned_path, draw_local_path, visualize_planning_results
)

# 导入LiDAR和DEM
from lidar_dem import PointCloudAccumulator, LocalDEMBuilder

# 导入路径规划
from path_planner import LocalPathPlanner

# 导入运动控制
from motion_control import (
    get_vehicle_pose_full, set_vehicle_pose,
    move_to_target_constant_yaw, move_along_local_path
)


# ========================================
# 全局模块实例（在主函数中初始化）
# ========================================
pointcloud_accumulator = None
local_dem_builder = None
local_path_planner = None


# ========================================
# 主函数
# ========================================
def main():
    global pointcloud_accumulator, local_dem_builder, local_path_planner
    
    print("=" * 70)
    print("AirSim CV模式 + LiDAR 局部路径规划与 DEM 构建")
    print("流程: 初始全局路径行走 → 构建DEM → 局部路径规划 → 沿局部路径行走 → 构建DEM → ...")
    print(f"雷达离地面高度偏移: {LIDAR_HEIGHT_OFFSET}m (已用于修正局部路径高程)")
    print("=" * 70)

    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"数据目录不存在: {os.path.abspath(DATA_DIR)}")
    
    # 1. 读取全局 DEM 与路径
    dem_file_path = os.path.join(DATA_DIR, "dem.txt")
    if not os.path.exists(dem_file_path):
        raise FileNotFoundError(f"DEM文件不存在: {dem_file_path}")
    
    read_dem_file(dem_file_path)
    calculate_airsim_elevation()

    # 在数据目录下查找 path 文件
    path_file = None
    for file in os.listdir(DATA_DIR):
        if file.startswith("path") and file.endswith(".txt"):
            path_file = os.path.join(DATA_DIR, file)
            break

    if path_file is None:
        raise FileNotFoundError(f"未在 {DATA_DIR} 目录下找到 path*.txt 文件")

    path_points = read_path_file(path_file)
    waypoints = generate_waypoints_with_elevation(path_points)

    if len(waypoints) < 2:
        raise ValueError("路径点数量不足，至少需要2个点")

    print(f"\n准备执行路径跟踪，共 {len(waypoints)} 个路径点")
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

    # 4. 初始化模块实例
    pointcloud_accumulator = PointCloudAccumulator(VEHICLE_NAME, LIDAR_SENSOR_NAME)
    local_dem_builder = LocalDEMBuilder(LOCAL_DEM_RANGE, LOCAL_DEM_RESOLUTION)
    local_path_planner = LocalPathPlanner()
    
    # 初始化点云采集器
    pointcloud_accumulator.initialize(client)

    # 5. 计算全局路径点朝向
    waypoints_with_yaw = []
    for i, (x, y, z) in enumerate(waypoints):
        if i < len(waypoints) - 1:
            next_x, next_y, next_z = waypoints[i + 1]
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
                
                # 可视化规划结果
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
                # 路径规划失败，但仍然可视化失败结果
                print(f"错误：局部路径规划失败，状态: {status}")
                
                # 尝试可视化失败结果（costmap和DEM）
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