"""
局部规划失败后的恢复逻辑管理模块
根据不同错误状态执行不同的恢复策略
"""

from pathlib import Path
import cv2
import numpy as np

from config import (
    PATH_REPLAN_ROOT_DIRNAME,
    LOCAL_MOVE_ENDPOINT_DIRNAME,
    LOCAL_SOFTEN_DIRNAME,
    GLOBAL_REPLAN_DIRNAME,
    START_RELAX_MAX_RADIUS_CELLS,
)
from utils import ensure_dir
from visualization import (
    load_costmap_txt,
    save_costmap_txt,
    make_costmap_gray_image,
)
from endpoint_scoring import plan_with_goal_adjustment, write_path_txt
from global_costmap_manager import run_global_replan


def save_softened_costmap_artifacts(costmap_path, start, radius, output_dir, prefix="soften_local_costmap"):
    """保存软化后的 costmap 可视化"""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    costmap = load_costmap_txt(costmap_path)
    save_costmap_txt(costmap, output_dir / f"{prefix}.txt")

    gray = make_costmap_gray_image(costmap)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    sx, sy = start
    cv2.rectangle(color, (sx - radius, sy - radius), (sx + radius, sy + radius), (0, 0, 255), 1)
    cv2.circle(color, (sx, sy), 3, (255, 0, 0), -1)

    cv2.imwrite(str(output_dir / f"{prefix}_vis.jpg"), color)

def recover_local_plan(
    local_path_planner,
    base_global_costmap,
    local_obstacle_observations,
    dem_grid,
    dem_build_dir,
    current_pose_xy_yaw,
    current_world_xyz,
    start_dem,
    original_goal_dem,
    original_goal_world,
    initial_status,
    initial_path_points_dem,
    initial_costmap_path,
    waypoints_with_yaw,
    current_global_idx,
    original_goal_idx,
    lidar_height_offset,
    is_final_goal_stage,
):
    """
    局部规划恢复主函数
    根据不同状态执行不同的恢复策略
    
    返回字典包含:
    - mode: "LOCAL_OK", "GLOBAL_REPLAN_OK", "FINAL_GOAL_UNREACHABLE", "FAILED" 等
    - 其他相关字段
    """
    dem_build_dir = Path(dem_build_dir)
    
    # 辅助函数：尝试挪动终点
    def try_move_endpoint(costmap_path, output_dir):
        return plan_with_goal_adjustment(
            local_path_planner=local_path_planner,
            dem_grid=dem_grid,
            start_dem=start_dem,
            original_goal_dem=original_goal_dem,
            original_goal_world=original_goal_world,
            current_pose_xy_yaw=current_pose_xy_yaw,
            start_z_build=current_world_xyz[2],
            waypoints_with_yaw=waypoints_with_yaw,
            current_global_idx=current_global_idx,
            original_goal_idx=original_goal_idx,
            lidar_height_offset=lidar_height_offset,
            base_costmap_path=costmap_path,
            output_dir=output_dir,
        )

    # 辅助函数：尝试全局重规划
    def try_global_replan(output_dir):
        if is_final_goal_stage:
            return "FINAL_GOAL_UNREACHABLE", None

        global_status, new_waypoints_with_yaw, reconnect_idx, inserted_start_idx, inserted_end_idx = run_global_replan(
            local_path_planner=local_path_planner,
            base_global_costmap=base_global_costmap,
            local_obstacle_observations=local_obstacle_observations,
            current_world_xyz=current_world_xyz,
            waypoints_with_yaw=waypoints_with_yaw,
            current_global_idx=current_global_idx,
            original_local_goal_idx=original_goal_idx,
            output_dir=output_dir,
        )
        if global_status != "OK":
            return global_status, None

        return "OK", {
            "new_waypoints_with_yaw": new_waypoints_with_yaw,
            "reconnect_idx": reconnect_idx,
            "inserted_start_idx": inserted_start_idx,
            "inserted_end_idx": inserted_end_idx,
        }

    # 辅助函数：尝试软化起点（局部DEM，一次性）
    def try_soften_start_local(costmap_path, output_dir, max_radius):
        """局部DEM软化：一次性软化到最大半径"""
        return local_path_planner.plan_path_from_costmap_with_start_relaxation(
            costmap_path=costmap_path,
            start_col=start_dem[0],
            start_row=start_dem[1],
            goal_col=original_goal_dem[0],
            goal_row=original_goal_dem[1],
            output_dir=output_dir,
            revision_filename="soften_local_costmap.txt",
            max_radius=max_radius,
            gradual=False,  # 局部DEM：一次性
        )

    status = initial_status
    active_costmap_path = Path(initial_costmap_path)

    # ================================
    # 1. 已经成功
    # ================================
    if status == "OK" and initial_path_points_dem is not None:
        return {
            "mode": "LOCAL_OK",
            "path_points_dem": initial_path_points_dem,
            "selected_goal_world": original_goal_world,
            "selected_goal_global_idx": original_goal_idx,
            "active_costmap_path": str(active_costmap_path),
            "new_waypoints_with_yaw": None,
        }

    # 创建 path_replan 根目录（只在需要恢复时创建）
    path_replan_root = dem_build_dir / PATH_REPLAN_ROOT_DIRNAME
    ensure_dir(path_replan_root)

    # ================================
    # 2. GOAL_IS_OBSTACLE: 先挪终点 -> 软化起点（局部） -> 全局重规划
    # ================================
    if status == "GOAL_IS_OBSTACLE":
        # 第一步：尝试挪动终点
        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME
        replan_status, best_item = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
        if replan_status == "OK" and best_item is not None:
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": best_item["path_dem"],
                "selected_goal_world": best_item["candidate_goal_world"],
                "selected_goal_global_idx": original_goal_idx,
                "active_costmap_path": str(active_costmap_path),
                "new_waypoints_with_yaw": None,
            }

        # 第二步：软化起点（局部，一次性）后再挪终点
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        soften_status, _, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path), str(soften_dir), max_radius=START_RELAX_MAX_RADIUS_CELLS
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)
        active_costmap_path = softened_costmap_path

        if soften_status not in ("START_IS_OBSTACLE", "START_AND_GOAL_ARE_OBSTACLES"):
            replan_status2, best_item2 = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
            if replan_status2 == "OK" and best_item2 is not None:
                return {
                    "mode": "LOCAL_OK",
                    "path_points_dem": best_item2["path_dem"],
                    "selected_goal_world": best_item2["candidate_goal_world"],
                    "selected_goal_global_idx": original_goal_idx,
                    "active_costmap_path": str(active_costmap_path),
                    "new_waypoints_with_yaw": None,
                }

        # 第三步：全局重规划
        global_replan_dir = path_replan_root / GLOBAL_REPLAN_DIRNAME
        global_status, global_replan_result = try_global_replan(str(global_replan_dir))
        if global_status == "OK":
            return {
                "mode": "GLOBAL_REPLAN_OK",
                "path_points_dem": None,
                "selected_goal_world": None,
                "selected_goal_global_idx": None,
                "active_costmap_path": None,
                "new_waypoints_with_yaw": global_replan_result["new_waypoints_with_yaw"],
                "reconnect_idx": global_replan_result["reconnect_idx"],
                "inserted_start_idx": global_replan_result["inserted_start_idx"],
                "inserted_end_idx": global_replan_result["inserted_end_idx"],
            }

        return {"mode": global_status}

    # ================================
    # 3. START_IS_OBSTACLE: 先软化起点（局部，一次性） -> 挪终点 -> 全局重规划
    # ================================
    if status == "START_IS_OBSTACLE":
        # 第一步：软化起点（局部DEM，一次性软化）
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        soften_status, path2, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path), str(soften_dir), max_radius=START_RELAX_MAX_RADIUS_CELLS
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)
        active_costmap_path = softened_costmap_path

        if soften_status == "OK" and path2 is not None:
            # 保存软化后的路径
            write_path_txt(path2, soften_dir / "path.txt")
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": path2,
                "selected_goal_world": original_goal_world,
                "selected_goal_global_idx": original_goal_idx,
                "active_costmap_path": str(active_costmap_path),
                "new_waypoints_with_yaw": None,
            }

        # 第二步：挪动终点
        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME
        replan_status, best_item = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
        if replan_status == "OK" and best_item is not None:
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": best_item["path_dem"],
                "selected_goal_world": best_item["candidate_goal_world"],
                "selected_goal_global_idx": original_goal_idx,
                "active_costmap_path": str(active_costmap_path),
                "new_waypoints_with_yaw": None,
            }

        # 第三步：全局重规划
        global_replan_dir = path_replan_root / GLOBAL_REPLAN_DIRNAME
        global_status, global_replan_result = try_global_replan(str(global_replan_dir))
        if global_status == "OK":
            return {
                "mode": "GLOBAL_REPLAN_OK",
                "path_points_dem": None,
                "selected_goal_world": None,
                "selected_goal_global_idx": None,
                "active_costmap_path": None,
                "new_waypoints_with_yaw": global_replan_result["new_waypoints_with_yaw"],
                "reconnect_idx": global_replan_result["reconnect_idx"],
                "inserted_start_idx": global_replan_result["inserted_start_idx"],
                "inserted_end_idx": global_replan_result["inserted_end_idx"],
            }

        return {"mode": global_status}

    # ================================
    # 4. START_AND_GOAL_ARE_OBSTACLES: 先软化起点（局部，一次性） -> 挪终点 -> 全局重规划
    # ================================
    if status == "START_AND_GOAL_ARE_OBSTACLES":
        # 第一步：软化起点（局部DEM，一次性软化，只清除起点障碍）
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        soften_status, _, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path), str(soften_dir), max_radius=START_RELAX_MAX_RADIUS_CELLS
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)
        active_costmap_path = softened_costmap_path

        # 第二步：挪动终点
        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME
        replan_status, best_item = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
        if replan_status == "OK" and best_item is not None:
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": best_item["path_dem"],
                "selected_goal_world": best_item["candidate_goal_world"],
                "selected_goal_global_idx": original_goal_idx,
                "active_costmap_path": str(active_costmap_path),
                "new_waypoints_with_yaw": None,
            }

        # 第三步：全局重规划
        global_replan_dir = path_replan_root / GLOBAL_REPLAN_DIRNAME
        global_status, global_replan_result = try_global_replan(str(global_replan_dir))
        if global_status == "OK":
            return {
                "mode": "GLOBAL_REPLAN_OK",
                "path_points_dem": None,
                "selected_goal_world": None,
                "selected_goal_global_idx": None,
                "active_costmap_path": None,
                "new_waypoints_with_yaw": global_replan_result["new_waypoints_with_yaw"],
                "reconnect_idx": global_replan_result["reconnect_idx"],
                "inserted_start_idx": global_replan_result["inserted_start_idx"],
                "inserted_end_idx": global_replan_result["inserted_end_idx"],
            }

        return {"mode": global_status}

    # ================================
    # 5. NO_PATH_FOUND: 先软化起点（局部，一次性） -> 原costmap挪终点 -> 软化后costmap挪终点 -> 全局重规划
    # ================================
    if status == "NO_PATH_FOUND":
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME
        
        # 第一步：软化起点（局部DEM，一次性）
        soften_status, path2, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path), str(soften_dir), max_radius=START_RELAX_MAX_RADIUS_CELLS
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)

        if soften_status == "OK" and path2 is not None:
            # 保存路径
            write_path_txt(path2, soften_dir / "path.txt")
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": path2,
                "selected_goal_world": original_goal_world,
                "selected_goal_global_idx": original_goal_idx,
                "active_costmap_path": str(softened_costmap_path),
                "new_waypoints_with_yaw": None,
            }

        # 第二步：用原 costmap 挪动终点
        replan_status_raw, best_item_raw = try_move_endpoint(str(initial_costmap_path), str(move_endpoint_dir))
        if replan_status_raw == "OK" and best_item_raw is not None:
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": best_item_raw["path_dem"],
                "selected_goal_world": best_item_raw["candidate_goal_world"],
                "selected_goal_global_idx": original_goal_idx,
                "active_costmap_path": str(initial_costmap_path),
                "new_waypoints_with_yaw": None,
            }

        # 第三步：用软化后的 costmap 挪动终点
        replan_status_soft, best_item_soft = try_move_endpoint(str(softened_costmap_path), str(move_endpoint_dir))
        if replan_status_soft == "OK" and best_item_soft is not None:
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": best_item_soft["path_dem"],
                "selected_goal_world": best_item_soft["candidate_goal_world"],
                "selected_goal_global_idx": original_goal_idx,
                "active_costmap_path": str(softened_costmap_path),
                "new_waypoints_with_yaw": None,
            }

        # 第四步：全局重规划
        global_replan_dir = path_replan_root / GLOBAL_REPLAN_DIRNAME
        global_status, global_replan_result = try_global_replan(str(global_replan_dir))
        if global_status == "OK":
            return {
                "mode": "GLOBAL_REPLAN_OK",
                "path_points_dem": None,
                "selected_goal_world": None,
                "selected_goal_global_idx": None,
                "active_costmap_path": None,
                "new_waypoints_with_yaw": global_replan_result["new_waypoints_with_yaw"],
                "reconnect_idx": global_replan_result["reconnect_idx"],
                "inserted_start_idx": global_replan_result["inserted_start_idx"],
                "inserted_end_idx": global_replan_result["inserted_end_idx"],
            }

        return {"mode": global_status}

    return {"mode": "FAILED"}