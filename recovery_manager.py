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
    LOCAL_HYBRID_DIRNAME,
    GLOBAL_REPLAN_DIRNAME,
    START_RELAX_MAX_RADIUS_CELLS,
    LOCAL_PLANNER_METHOD_PHASE1,
    LOCAL_PLANNER_METHOD_PHASE2,
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


def save_hybrid_astar_visualization(costmap_path, start, goal, path_points, output_dir, prefix="HybridAStar_path"):
    """保存 HybridAStar 路径可视化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    costmap = load_costmap_txt(costmap_path)
    gray = make_costmap_gray_image(costmap)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    sx, sy = start
    gx, gy = goal

    # 起点蓝色，终点绿色
    cv2.circle(color, (sx, sy), 5, (255, 0, 0), -1)
    cv2.circle(color, (gx, gy), 5, (0, 255, 0), -1)
    cv2.putText(color, "Start", (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(color, "Goal", (gx + 8, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 路径红色
    if path_points and len(path_points) >= 2:
        pts = np.array(path_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(color, [pts], isClosed=False, color=(0, 0, 255), thickness=2)


    output_path = output_dir / f"{prefix}.jpg"
    cv2.imwrite(str(output_path), color)
    print(f"已保存 HybridAStar 可视化: {output_path}")
    return output_path


def run_hybrid_astar_optimization(
    local_path_planner,
    costmap_path,
    start_dem,
    selected_goal_dem,
    output_dir,
):
    """
    使用 HybridAStar 进行运动学优化规划

    返回:
        (status, path_points_dem, hybrid_astar_dir)
    """
    output_dir = Path(output_dir)
    hybrid_astar_dir = output_dir / LOCAL_HYBRID_DIRNAME
    hybrid_astar_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("【HybridAStar 运动学优化阶段】")
    print(f"{'=' * 70}")
    print(f"起点: {start_dem}")
    print(f"终点: {selected_goal_dem}")
    print(f"输出目录: {hybrid_astar_dir}")

    status, path_points = local_path_planner.plan_hybrid_astar_from_costmap(
        costmap_path=costmap_path,
        start_col=start_dem[0],
        start_row=start_dem[1],
        goal_col=selected_goal_dem[0],
        goal_row=selected_goal_dem[1],
        output_dir=str(hybrid_astar_dir),
    )

    path_file = hybrid_astar_dir / "path.txt"

    if status == "OK" and path_points:
        write_path_txt(path_points, path_file)
        save_hybrid_astar_visualization(
            costmap_path=costmap_path,
            start=start_dem,
            goal=selected_goal_dem,
            path_points=path_points,
            output_dir=hybrid_astar_dir,
        )
        print(f"HybridAStar 优化成功，路径点数: {len(path_points)}")
        return status, path_points, str(hybrid_astar_dir)

    # 失败时只保留 path.txt，不生成 png
    path_file.write_text("NO_PATH_FOUND", encoding="utf-8")
    print(f"HybridAStar 优化失败，状态: {status}")
    return status, None, str(hybrid_astar_dir)


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

    流程：
        BidirectionalAStar 快速搜索
        -> 若成功，则用最终选中的终点再跑 HybridAStar
        -> UE 优先显示 HybridAStar 结果
        -> 若 HybridAStar 失败，UE 回退显示 BidirectionalAStar 结果
    """
    dem_build_dir = Path(dem_build_dir)
    active_costmap_path = Path(initial_costmap_path)

    # 必须一开始就创建，避免 status=="OK" 分支提前使用时报错
    path_replan_root = dem_build_dir / PATH_REPLAN_ROOT_DIRNAME
    ensure_dir(path_replan_root)

    def finalize_local_success(
        fast_path_dem,
        selected_goal_dem,
        selected_goal_world,
        selected_goal_global_idx,
        active_costmap_path_for_this_success,
    ):
        """
        局部快速规划成功后的统一收口逻辑：
        1) 保留快速规划结果（BidirectionalAStar）
        2) 再用相同起终点跑 HybridAStar
        3) UE 显示 HybridAStar；若失败则显示快速规划结果
        """
        hybrid_status, hybrid_path, hybrid_dir = run_hybrid_astar_optimization(
            local_path_planner=local_path_planner,
            costmap_path=str(active_costmap_path_for_this_success),
            start_dem=start_dem,
            selected_goal_dem=selected_goal_dem,
            output_dir=path_replan_root,
        )

        if hybrid_status == "OK" and hybrid_path is not None:
            return {
                "mode": "LOCAL_OK",
                "path_points_dem": hybrid_path,                 # UE 实际显示 / 跟踪这条
                "fast_path_points_dem": fast_path_dem,         # 保留快速规划结果
                "hybrid_path_points_dem": hybrid_path,
                "selected_goal_world": selected_goal_world,
                "selected_goal_dem": selected_goal_dem,
                "selected_goal_global_idx": selected_goal_global_idx,
                "active_costmap_path": str(active_costmap_path_for_this_success),
                "new_waypoints_with_yaw": None,
                "hybrid_astar_dir": hybrid_dir,
                "used_display_planner": "HybridAStar",
            }

        return {
            "mode": "LOCAL_OK",
            "path_points_dem": fast_path_dem,                  # UE 回退显示快速规划结果
            "fast_path_points_dem": fast_path_dem,
            "hybrid_path_points_dem": None,
            "selected_goal_world": selected_goal_world,
            "selected_goal_dem": selected_goal_dem,
            "selected_goal_global_idx": selected_goal_global_idx,
            "active_costmap_path": str(active_costmap_path_for_this_success),
            "new_waypoints_with_yaw": None,
            "hybrid_astar_dir": hybrid_dir,
            "used_display_planner": "BidirectionalAStar",
        }

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

    def try_soften_start_local(costmap_path, output_dir, max_radius):
        return local_path_planner.plan_path_from_costmap_with_start_relaxation(
            costmap_path=costmap_path,
            start_col=start_dem[0],
            start_row=start_dem[1],
            goal_col=original_goal_dem[0],
            goal_row=original_goal_dem[1],
            output_dir=output_dir,
            revision_filename="soften_local_costmap.txt",
            max_radius=max_radius,
            gradual=False,
            method=LOCAL_PLANNER_METHOD_PHASE1,
        )

    status = initial_status

    # ==========================================================
    # 1. 初始局部规划已经成功：直接进入 HybridAStar 二阶段优化
    # ==========================================================
    if status == "OK" and initial_path_points_dem is not None:
        return finalize_local_success(
            fast_path_dem=initial_path_points_dem,
            selected_goal_dem=original_goal_dem,
            selected_goal_world=original_goal_world,
            selected_goal_global_idx=original_goal_idx,
            active_costmap_path_for_this_success=active_costmap_path,
        )

    # ==========================================================
    # 2. GOAL_IS_OBSTACLE
    #    挪终点 -> 软化起点后再挪终点 -> 全局重规划
    # ==========================================================
    if status == "GOAL_IS_OBSTACLE":
        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME

        # 第一步：原始 costmap 上挪终点（BidirectionalAStar）
        replan_status, best_item = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
        if replan_status == "OK" and best_item is not None:
            return finalize_local_success(
                fast_path_dem=best_item["path_dem"],
                selected_goal_dem=best_item["candidate_goal_dem"],
                selected_goal_world=best_item["candidate_goal_world"],
                selected_goal_global_idx=original_goal_idx,
                active_costmap_path_for_this_success=active_costmap_path,
            )

        # 第二步：软化起点后再挪终点
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        soften_status, _, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path),
            str(soften_dir),
            max_radius=START_RELAX_MAX_RADIUS_CELLS,
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)
        active_costmap_path = Path(softened_costmap_path)

        if soften_status not in ("START_IS_OBSTACLE", "START_AND_GOAL_ARE_OBSTACLES"):
            replan_status2, best_item2 = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
            if replan_status2 == "OK" and best_item2 is not None:
                return finalize_local_success(
                    fast_path_dem=best_item2["path_dem"],
                    selected_goal_dem=best_item2["candidate_goal_dem"],
                    selected_goal_world=best_item2["candidate_goal_world"],
                    selected_goal_global_idx=original_goal_idx,
                    active_costmap_path_for_this_success=active_costmap_path,
                )

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

    # ==========================================================
    # 3. START_IS_OBSTACLE
    #    软化起点 -> 挪终点 -> 全局重规划
    # ==========================================================
    if status == "START_IS_OBSTACLE":
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        soften_status, path2, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path),
            str(soften_dir),
            max_radius=START_RELAX_MAX_RADIUS_CELLS,
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)
        active_costmap_path = Path(softened_costmap_path)

        if soften_status == "OK" and path2 is not None:
            write_path_txt(path2, Path(soften_dir) / "path.txt")
            return finalize_local_success(
                fast_path_dem=path2,
                selected_goal_dem=original_goal_dem,
                selected_goal_world=original_goal_world,
                selected_goal_global_idx=original_goal_idx,
                active_costmap_path_for_this_success=active_costmap_path,
            )

        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME
        replan_status, best_item = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
        if replan_status == "OK" and best_item is not None:
            return finalize_local_success(
                fast_path_dem=best_item["path_dem"],
                selected_goal_dem=best_item["candidate_goal_dem"],
                selected_goal_world=best_item["candidate_goal_world"],
                selected_goal_global_idx=original_goal_idx,
                active_costmap_path_for_this_success=active_costmap_path,
            )

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

    # ==========================================================
    # 4. START_AND_GOAL_ARE_OBSTACLES
    #    软化起点 -> 挪终点 -> 全局重规划
    # ==========================================================
    if status == "START_AND_GOAL_ARE_OBSTACLES":
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        soften_status, _, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path),
            str(soften_dir),
            max_radius=START_RELAX_MAX_RADIUS_CELLS,
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)
        active_costmap_path = Path(softened_costmap_path)

        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME
        replan_status, best_item = try_move_endpoint(str(active_costmap_path), str(move_endpoint_dir))
        if replan_status == "OK" and best_item is not None:
            return finalize_local_success(
                fast_path_dem=best_item["path_dem"],
                selected_goal_dem=best_item["candidate_goal_dem"],
                selected_goal_world=best_item["candidate_goal_world"],
                selected_goal_global_idx=original_goal_idx,
                active_costmap_path_for_this_success=active_costmap_path,
            )

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

    # ==========================================================
    # 5. NO_PATH_FOUND
    #    软化起点 -> 原图挪终点 -> 软化后挪终点 -> 全局重规划
    # ==========================================================
    if status == "NO_PATH_FOUND":
        soften_dir = path_replan_root / LOCAL_SOFTEN_DIRNAME
        move_endpoint_dir = path_replan_root / LOCAL_MOVE_ENDPOINT_DIRNAME

        # 第一步：软化起点
        soften_status, path2, softened_costmap_path, used_radius = try_soften_start_local(
            str(active_costmap_path),
            str(soften_dir),
            max_radius=START_RELAX_MAX_RADIUS_CELLS,
        )
        save_softened_costmap_artifacts(softened_costmap_path, start_dem, used_radius, soften_dir)

        if soften_status == "OK" and path2 is not None:
            write_path_txt(path2, Path(soften_dir) / "path.txt")
            return finalize_local_success(
                fast_path_dem=path2,
                selected_goal_dem=original_goal_dem,
                selected_goal_world=original_goal_world,
                selected_goal_global_idx=original_goal_idx,
                active_costmap_path_for_this_success=Path(softened_costmap_path),
            )

        # 第二步：原始 costmap 挪终点
        replan_status_raw, best_item_raw = try_move_endpoint(str(initial_costmap_path), str(move_endpoint_dir))
        if replan_status_raw == "OK" and best_item_raw is not None:
            return finalize_local_success(
                fast_path_dem=best_item_raw["path_dem"],
                selected_goal_dem=best_item_raw["candidate_goal_dem"],
                selected_goal_world=best_item_raw["candidate_goal_world"],
                selected_goal_global_idx=original_goal_idx,
                active_costmap_path_for_this_success=Path(initial_costmap_path),
            )

        # 第三步：软化后的 costmap 挪终点
        replan_status_soft, best_item_soft = try_move_endpoint(str(softened_costmap_path), str(move_endpoint_dir))
        if replan_status_soft == "OK" and best_item_soft is not None:
            return finalize_local_success(
                fast_path_dem=best_item_soft["path_dem"],
                selected_goal_dem=best_item_soft["candidate_goal_dem"],
                selected_goal_world=best_item_soft["candidate_goal_world"],
                selected_goal_global_idx=original_goal_idx,
                active_costmap_path_for_this_success=Path(softened_costmap_path),
            )

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