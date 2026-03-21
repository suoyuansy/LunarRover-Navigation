"""
主程序入口
基于 AirSim CV 模式 + LiDAR 的局部路径规划与 DEM 构建
"""

import os
import time
import traceback
import shutil
from pathlib import Path

import airsim

from config import (
    VEHICLE_NAME,
    LIDAR_SENSOR_NAME,
    LOCAL_DEM_RANGE,
    LOCAL_DEM_RESOLUTION,
    INITIAL_BUILD_AFTER_SEGMENT_INDEX,
    LIDAR_HEIGHT_OFFSET,
    GLOBAL_COSTMAP_FILENAME,
    ENABLE_CPP_GLOBAL_PATH_PLANNING,
    GLOBAL_DEM_TIF_PATH,
    GLOBAL_COLOR_PNG_PATH,
    GLOBAL_OUTPUT_DIR,
    GLOBAL_DEM_RESOLUTION,
)
from utils import (
    read_dem_file,
    calculate_airsim_elevation,
    read_path_file,
    generate_waypoints_with_elevation,
    calculate_yaw_between_points,
    create_run_output_folder,
    create_single_build_folder,
    ensure_dir,
)
from coordinate_transform import dem_grid_to_local, local_to_world
from visualization import (
    draw_planned_path,
    draw_local_path,
    visualize_planning_results,
    load_costmap_txt,
)
from lidar_dem import PointCloudAccumulator, LocalDEMBuilder
from local_path_planner import LocalPathPlanner
from motion_control import (
    get_vehicle_pose_full,
    set_vehicle_pose,
    move_to_target_constant_yaw,
    move_along_local_path,
)
from endpoint_scoring import choose_local_goal_index
from global_costmap_manager import (
    save_local_costmap_artifacts,
    save_global_merge_artifacts,
)
from recovery_manager import recover_local_plan


pointcloud_accumulator = None
local_dem_builder = None
local_path_planner = None


def convert_dem_path_to_world(
    path_points_dem,
    start_world,
    current_yaw,
    dem_grid,
    dem_range,
    resolution,
    start_z_build,
):
    """将 DEM 路径点转换为世界坐标路径"""
    local_path_points_world = []
    for col, row in path_points_dem:
        x_local, y_local = dem_grid_to_local(col, row, dem_range, resolution)
        x_world, y_world = local_to_world(x_local, y_local, start_world[0], start_world[1], current_yaw)
        z_relative = dem_grid[row, col]
        z_world = start_z_build + z_relative - LIDAR_HEIGHT_OFFSET
        local_path_points_world.append((x_world, y_world, z_world))
    return local_path_points_world


def build_waypoints_with_yaw(waypoints):
    """构建带航向的路径点"""
    waypoints_with_yaw = []
    for i, (x, y, z) in enumerate(waypoints):
        if i < len(waypoints) - 1:
            next_x, next_y, _ = waypoints[i + 1]
            yaw = calculate_yaw_between_points(x, y, next_x, next_y)
        else:
            yaw = waypoints_with_yaw[-1][3] if waypoints_with_yaw else 0.0
        waypoints_with_yaw.append((x, y, z, yaw))
    return waypoints_with_yaw


def load_global_data(use_cpp_global_planner=False, planner=None):
    """
    加载全局数据。

    两种模式：
    1) use_cpp_global_planner = False
       直接从 global_path_file 读取 dem.txt / costmap.txt / path.txt
    2) use_cpp_global_planner = True
       先调用 C++ 全局交互规划器生成文件，再读取
    """
    data_dir = Path(GLOBAL_OUTPUT_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    if use_cpp_global_planner:
        if planner is None:
            planner = LocalPathPlanner()

        status = planner.plan_global_path_interactive(
            dem_tif_path=GLOBAL_DEM_TIF_PATH,
            color_png_path=GLOBAL_COLOR_PNG_PATH,
            resolution=GLOBAL_DEM_RESOLUTION,
            output_dir=data_dir,
        )

        if status != "OK":
            raise RuntimeError(f"C++ 全局交互路径规划失败，状态: {status}")

    if not data_dir.exists():
        raise FileNotFoundError(f"全局数据目录不存在: {data_dir}")

    dem_file_path = data_dir / "dem.txt"
    if not dem_file_path.exists():
        raise FileNotFoundError(f"DEM 文件不存在: {dem_file_path}")

    global_costmap_file = data_dir / GLOBAL_COSTMAP_FILENAME
    if not global_costmap_file.exists():
        raise FileNotFoundError(f"全局 costmap 文件不存在: {global_costmap_file}")

    path_file = data_dir / "path.txt"
    if not path_file.exists():
        candidates = sorted(data_dir.glob("path*.txt"))
        if not candidates:
            raise FileNotFoundError(f"未在 {data_dir} 下找到 path.txt 或 path*.txt 文件")
        path_file = candidates[0]

    print("\n" + "=" * 70)
    print("加载全局数据")
    print("=" * 70)
    print(f"DEM: {dem_file_path}")
    print(f"Costmap: {global_costmap_file}")
    print(f"Path: {path_file}")
    print("=" * 70)

    read_dem_file(str(dem_file_path))
    calculate_airsim_elevation()

    base_global_costmap = load_costmap_txt(str(global_costmap_file))

    path_points = read_path_file(str(path_file))
    waypoints = generate_waypoints_with_elevation(path_points)
    if len(waypoints) < 2:
        raise ValueError("路径点数量不足，至少需要 2 个点")

    waypoints_with_yaw = build_waypoints_with_yaw(waypoints)
    return base_global_costmap, waypoints_with_yaw


def reorganize_planning_output(planning_output_dir, dem_build_dir, has_path):
    """
    重新组织规划输出文件
    - 将 costmap.txt 移动到 dem_build_dir 并重命名为 dem_costmap.txt
    - 根据是否有路径决定是否保留某些可视化文件
    """
    planning_output_dir = Path(planning_output_dir)
    dem_build_dir = Path(dem_build_dir)

    costmap_src = planning_output_dir / "costmap.txt"
    if costmap_src.exists():
        costmap_dst = dem_build_dir / "dem_costmap.txt"
        shutil.move(str(costmap_src), str(costmap_dst))
        print(f"已移动 costmap 到: {costmap_dst}")

    if not has_path:
        for file in ["costmap_with_start_goal_and_path.jpg", "dem_3d_with_start_goal_and_path.jpg"]:
            f = planning_output_dir / file
            if f.exists():
                f.unlink()


def main():
    global pointcloud_accumulator, local_dem_builder, local_path_planner

    print("=" * 70)
    print("AirSim CV模式 + LiDAR 局部路径规划与 DEM 构建")
    print("=" * 70)

    # 先创建规划器，这样一开始就能决定是否调用 C++ 全局交互规划
    local_path_planner = LocalPathPlanner()

    base_global_costmap, waypoints_with_yaw = load_global_data(
        use_cpp_global_planner=ENABLE_CPP_GLOBAL_PATH_PLANNING,
        planner=local_path_planner,
    )

    output_root_dir, run_dir, _ = create_run_output_folder()
    print(f"总输出目录: {output_root_dir}")
    print(f"本次运行目录: {run_dir}")

    client = airsim.VehicleClient()
    client.confirmConnection()
    client.enableApiControl(True, VEHICLE_NAME)
    client.simFlushPersistentMarkers()
    print(f"已启用 API 控制: {VEHICLE_NAME}")

    pointcloud_accumulator = PointCloudAccumulator(VEHICLE_NAME, LIDAR_SENSOR_NAME)
    local_dem_builder = LocalDEMBuilder(LOCAL_DEM_RANGE, LOCAL_DEM_RESOLUTION)
    pointcloud_accumulator.initialize(client)

    draw_planned_path(client, waypoints_with_yaw, color=(0, 0, 1))
    time.sleep(0.5)

    trajectory_points = []
    local_obstacle_observations = []

    first_wp = waypoints_with_yaw[0]
    print(f"\n【瞬移到起始点】({first_wp[0]}, {first_wp[1]}, {first_wp[2]})")
    set_vehicle_pose(client, first_wp[0], first_wp[1], first_wp[2], first_wp[3])
    time.sleep(1.0)

    current_x, current_y, current_z, roll, pitch, current_yaw = get_vehicle_pose_full(client)
    pointcloud_accumulator.reset(clear_timestamp=True)

    total_segments = len(waypoints_with_yaw) - 1
    current_global_idx = 0

    try:
        target_initial_idx = min(INITIAL_BUILD_AFTER_SEGMENT_INDEX + 1, total_segments)

        if target_initial_idx > 0:
            print(f"\n{'=' * 70}")
            print(f"【初始全局路径行走】从起点到全局路径点 {target_initial_idx}")
            print(f"{'=' * 70}")

            while current_global_idx < target_initial_idx:
                wp_target = waypoints_with_yaw[current_global_idx + 1]

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
                    segment_idx=current_global_idx + 1,
                )
                current_global_idx += 1

        while current_global_idx < total_segments:
            print(f"\n{'=' * 70}")
            print(f"【DEM 构建阶段】当前位于全局路径点 {current_global_idx}")
            print(f"{'=' * 70}")

            latest_pose_full = get_vehicle_pose_full(client)
            current_x, current_y, current_z, roll, pitch, current_yaw = latest_pose_full

            frames = pointcloud_accumulator.get_all_frames()
            if frames is None or len(frames) == 0:
                print("警告：当前累计点云帧为空，无法构建 DEM")
                return

            dem_build_dir, point_data_dir, _ = create_single_build_folder(run_dir, "dem_build")
            dem_build_dir = Path(dem_build_dir)
            point_data_dir = Path(point_data_dir)
            print(f"本次 DEM 构建目录: {dem_build_dir}")
            pointcloud_accumulator.save_frames_to_point_data(str(point_data_dir))

            dem_grid, valid_mask, fused_points_current = local_dem_builder.build_dem_from_frames(
                frames=frames,
                current_pose=latest_pose_full,
            )
            if dem_grid is None:
                print("错误：DEM 构建失败")
                return

            dem_txt_path = local_dem_builder.save_dem_results(
                dem_grid=dem_grid,
                current_pose=latest_pose_full,
                build_dir=str(dem_build_dir),
            )

            pointcloud_accumulator.reset(clear_timestamp=False)

            print(f"\n{'=' * 70}")
            print("【局部路径规划阶段】")
            print(f"{'=' * 70}")

            goal_selection = choose_local_goal_index(
                waypoints_with_yaw=waypoints_with_yaw,
                current_global_idx=current_global_idx,
                current_pose_xy_yaw=(current_x, current_y, current_yaw),
                dem_range=LOCAL_DEM_RANGE,
                resolution=LOCAL_DEM_RESOLUTION,
            )

            if goal_selection is None:
                print("当前 DEM 范围内没有新的全局路径点，无法继续局部规划")
                return

            goal_global_idx, goal_col, goal_row = goal_selection
            is_final_goal_stage = (goal_global_idx == total_segments)

            start_world = (current_x, current_y, current_z)
            goal_world = (
                waypoints_with_yaw[goal_global_idx][0],
                waypoints_with_yaw[goal_global_idx][1],
                waypoints_with_yaw[goal_global_idx][2],
            )
            start_col = local_dem_builder.grid_size // 2
            start_row = local_dem_builder.grid_size // 2

            print(f"起点（世界坐标）: ({start_world[0]:.2f}, {start_world[1]:.2f}, {start_world[2]:.2f})")
            print(f"终点（世界坐标）: ({goal_world[0]:.2f}, {goal_world[1]:.2f}, {goal_world[2]:.2f})")
            print(f"起点（DEM栅格）: ({start_col}, {start_row})")
            print(f"终点（DEM栅格）: ({goal_col}, {goal_row})")
            print(f"是否最终终点阶段: {is_final_goal_stage}")

            planning_output_dir = dem_build_dir / "local_path_planning_result"
            ensure_dir(planning_output_dir)

            status, path_points_dem = local_path_planner.plan_path(
                dem_path=dem_txt_path,
                start_col=start_col,
                start_row=start_row,
                goal_col=goal_col,
                goal_row=goal_row,
                resolution=LOCAL_DEM_RESOLUTION,
                output_dir=str(planning_output_dir),
            )

            has_path = (status == "OK" and path_points_dem is not None)

            reorganize_planning_output(planning_output_dir, dem_build_dir, has_path)

            dem_costmap_path = dem_build_dir / "dem_costmap.txt"

            local_obstacle_observations.append({
                "raw_costmap_path": str(dem_costmap_path),
                "pose_xy_yaw": (current_x, current_y, current_yaw),
                "dem_range": LOCAL_DEM_RANGE,
                "resolution": LOCAL_DEM_RESOLUTION,
            })

            save_local_costmap_artifacts(str(dem_costmap_path), str(dem_build_dir))
            save_global_merge_artifacts(base_global_costmap, local_obstacle_observations, str(dem_build_dir))

            visualize_planning_results(
                dem_path=dem_txt_path,
                costmap_path=str(dem_costmap_path),
                planning_output_dir=str(planning_output_dir),
                start_col=start_col,
                start_row=start_row,
                goal_col=goal_col,
                goal_row=goal_row,
                has_path=has_path,
            )

            recovery = None
            if not has_path:
                recovery = recover_local_plan(
                    local_path_planner=local_path_planner,
                    base_global_costmap=base_global_costmap,
                    local_obstacle_observations=local_obstacle_observations,
                    dem_grid=dem_grid,
                    dem_build_dir=str(dem_build_dir),
                    current_pose_xy_yaw=(current_x, current_y, current_yaw),
                    current_world_xyz=(current_x, current_y, current_z),
                    start_dem=(start_col, start_row),
                    original_goal_dem=(goal_col, goal_row),
                    original_goal_world=goal_world,
                    initial_status=status,
                    initial_path_points_dem=path_points_dem,
                    initial_costmap_path=str(dem_costmap_path),
                    waypoints_with_yaw=waypoints_with_yaw,
                    current_global_idx=current_global_idx,
                    original_goal_idx=goal_global_idx,
                    lidar_height_offset=LIDAR_HEIGHT_OFFSET,
                    is_final_goal_stage=is_final_goal_stage,
                )
            else:
                recovery = {
                    "mode": "LOCAL_OK",
                    "path_points_dem": path_points_dem,
                    "selected_goal_world": goal_world,
                    "selected_goal_global_idx": goal_global_idx,
                    "active_costmap_path": str(dem_costmap_path),
                    "new_waypoints_with_yaw": None,
                }

            mode = recovery.get("mode")

            if mode == "GLOBAL_REPLAN_OK":
                waypoints_with_yaw = recovery["new_waypoints_with_yaw"]
                inserted_start_idx = recovery["inserted_start_idx"]
                inserted_end_idx = recovery["inserted_end_idx"]

                total_segments = len(waypoints_with_yaw) - 1

                # 不重置 current_global_idx，保持当前全局进度
                # 旧蓝线不重画；只把新插入的重规划段画成绿色
                replanned_segment = waypoints_with_yaw[inserted_start_idx: inserted_end_idx + 1]
                if len(replanned_segment) >= 2:
                    print("绘制新的全局重规划段（绿色）...")
                    draw_planned_path(client, replanned_segment, color=(0, 1, 0))

                print("已切换为新的全局路径")

                # 关键：先沿新全局路径前进一个点，收集点云，再进入下一轮 DEM 构建
                next_global_idx = min(current_global_idx + 1, total_segments)
                if next_global_idx > current_global_idx:
                    print(f"\n{'=' * 70}")
                    print(f"【全局重规划后预行走】从全局路径点 {current_global_idx} 到 {next_global_idx}")
                    print(f"{'=' * 70}")

                    wp_target = waypoints_with_yaw[next_global_idx]

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
                        segment_idx=next_global_idx,
                    )
                    current_global_idx = next_global_idx
                    print(f"已沿新全局路径前进到点 {current_global_idx}，继续下一轮规划")
                else:
                    print("新全局路径已到末端，无需预行走")

                continue

            if mode == "FINAL_GOAL_UNREACHABLE":
                print(f"最终终点无法达到，已停留在 ({current_x:.2f}, {current_y:.2f}, {current_z:.2f})")
                return

            if mode != "LOCAL_OK":
                print(f"局部恢复失败，状态: {mode}")
                return

            path_points_dem = recovery["path_points_dem"]
            selected_goal_world = recovery["selected_goal_world"]

            local_path_points_world = convert_dem_path_to_world(
                path_points_dem=path_points_dem,
                start_world=start_world,
                current_yaw=current_yaw,
                dem_grid=dem_grid,
                dem_range=LOCAL_DEM_RANGE,
                resolution=LOCAL_DEM_RESOLUTION,
                start_z_build=current_z,
            )

            draw_local_path(client, local_path_points_world, color=(1, 0, 1))
            print(f"局部路径已转换到世界坐标，共 {len(local_path_points_world)} 个点")

            print(f"\n{'=' * 70}")
            print("【沿局部路径行走阶段】")
            print(f"{'=' * 70}")
            current_x, current_y, current_z, current_yaw = move_along_local_path(
                client=client,
                accumulator=pointcloud_accumulator,
                local_path_points_world=local_path_points_world,
                start_z_world=current_z,
                dem_grid=dem_grid,
                dem_range=LOCAL_DEM_RANGE,
                resolution=LOCAL_DEM_RESOLUTION,
                trajectory_points=trajectory_points,
                current_yaw=current_yaw,
            )

            if is_final_goal_stage:
                if selected_goal_world == goal_world:
                    print("已到达最终终点")
                else:
                    print("终点为障碍，已就近找到安全区停靠")
                break

            current_global_idx = goal_global_idx
            print(f"已完成局部路径行走，当前全局索引更新为: {current_global_idx}")
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