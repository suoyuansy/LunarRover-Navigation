"""
局部路径规划器接口模块
负责调用 C++ 路径规划器
"""

import subprocess
import time
from pathlib import Path

from config import (
    PATH_PLANNING_EXE,
    PATH_PLANNING_TIMEOUT,
    REVISION_SOFT_COST,
    START_RELAX_MAX_RADIUS_CELLS,
)
from visualization import load_costmap_txt, save_costmap_txt


class LocalPathPlanner:
    def __init__(self, exe_path=PATH_PLANNING_EXE):
        self.exe_path = Path(exe_path)
        self.timeout = PATH_PLANNING_TIMEOUT

    def plan_path(self, dem_path, start_col, start_row, goal_col, goal_row, resolution, output_dir):
        """基于 DEM 进行路径规划"""
        dem_path = Path(dem_path)
        output_dir = Path(output_dir)

        if not self.exe_path.exists():
            return "EXE_NOT_FOUND", None
        if not dem_path.exists():
            return "DEM_NOT_FOUND", None

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.exe_path),
            str(dem_path),
            str(start_col),
            str(start_row),
            str(goal_col),
            str(goal_row),
            str(resolution),
            str(output_dir),
        ]

        print("调用路径规划器（DEM模式）:")
        print(f"  DEM: {dem_path}")
        print(f"  起点: ({start_col}, {start_row})")
        print(f"  终点: ({goal_col}, {goal_row})")
        print(f"  分辨率: {resolution}m")
        print(f"  输出: {output_dir}")

        return self._run_and_parse(cmd, output_dir, expect_costmap=True)

    def plan_path_from_costmap(self, costmap_path, start_col, start_row, goal_col, goal_row, output_dir):
        """基于 costmap 进行路径规划"""
        costmap_path = Path(costmap_path)
        output_dir = Path(output_dir)

        if not self.exe_path.exists():
            return "EXE_NOT_FOUND", None
        if not costmap_path.exists():
            return "COSTMAP_NOT_FOUND", None

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.exe_path),
            str(costmap_path),
            str(start_col),
            str(start_row),
            str(goal_col),
            str(goal_row),
            str(output_dir),
        ]

        print("调用路径规划器（costmap模式）:")
        print(f"  costmap: {costmap_path}")
        print(f"  起点: ({start_col}, {start_row})")
        print(f"  终点: ({goal_col}, {goal_row})")
        print(f"  输出: {output_dir}")

        return self._run_and_parse(cmd, output_dir, expect_costmap=False)

    def plan_path_from_costmap_with_start_relaxation(
        self,
        costmap_path,
        start_col,
        start_row,
        goal_col,
        goal_row,
        output_dir,
        revision_filename,
        max_radius=START_RELAX_MAX_RADIUS_CELLS,
        stop_when_start_cleared_only=False,
        gradual=False,  # 新增参数：True=逐渐软化（全局），False=一次性软化（局部）
    ):
        """
        软化起点附近障碍
        
        参数:
            gradual: True 表示逐渐扩大半径（用于全局costmap），
                    False 表示一次性软化到最大半径（用于局部costmap）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_costmap = load_costmap_txt(costmap_path)
        revision_costmap_path = output_dir / revision_filename

        if not gradual:
            # 局部DEM：一次性软化到最大半径
            revised = raw_costmap.copy()
            self._soften_obstacles_in_radius(
                costmap=revised,
                center=(start_col, start_row),
                radius=max_radius,  # 直接使用最大半径
                soft_cost=REVISION_SOFT_COST,
            )
            save_costmap_txt(revised, revision_costmap_path)
            
            # 只调用一次规划器
            status, path_points = self.plan_path_from_costmap(
                costmap_path=revision_costmap_path,
                start_col=start_col,
                start_row=start_row,
                goal_col=goal_col,
                goal_row=goal_row,
                output_dir=output_dir,
            )
            
            return status, path_points, revision_costmap_path, max_radius
        else:
            # 全局DEM：逐渐扩大半径
            last_status = "NO_PATH_FOUND"
            last_path = None
            last_radius = 0

            for radius in range(1, max_radius + 1):
                revised = raw_costmap.copy()
                self._soften_obstacles_in_radius(
                    costmap=revised,
                    center=(start_col, start_row),
                    radius=radius,
                    soft_cost=REVISION_SOFT_COST,
                )
                save_costmap_txt(revised, revision_costmap_path)

                status, path_points = self.plan_path_from_costmap(
                    costmap_path=revision_costmap_path,
                    start_col=start_col,
                    start_row=start_row,
                    goal_col=goal_col,
                    goal_row=goal_row,
                    output_dir=output_dir,
                )

                last_status = status
                last_path = path_points
                last_radius = radius

                if status == "OK":
                    return status, path_points, revision_costmap_path, radius

                if stop_when_start_cleared_only and status not in ("START_IS_OBSTACLE", "START_AND_GOAL_ARE_OBSTACLES"):
                    return status, path_points, revision_costmap_path, radius

            return last_status, last_path, revision_costmap_path, last_radius
        
    @staticmethod
    def _soften_obstacles_in_radius(costmap, center, radius, soft_cost=0.99):
        """软化指定半径内的障碍"""
        cx, cy = center
        rows, cols = costmap.shape

        x_min = max(0, cx - radius)
        x_max = min(cols - 1, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(rows - 1, cy + radius)

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if costmap[y, x] >= 1.0:
                    costmap[y, x] = soft_cost

    def _run_and_parse(self, cmd, output_dir, expect_costmap):
        """运行规划器并解析结果"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            print(f"启动路径规划器失败: {e}")
            return "PROCESS_ERROR", None

        status = self._wait_for_outputs(process, output_dir, expect_costmap)

        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                print(f"\nC++ stdout:\n{stdout}")
            if stderr:
                print(f"\nC++ stderr:\n{stderr}")
        except Exception:
            pass

        if status == "success":
            return self._parse_path_txt(Path(output_dir) / "path.txt")

        if status == "timeout":
            process.kill()
            return "TIMEOUT", None

        if status == "process_error":
            path_file = Path(output_dir) / "path.txt"
            if path_file.exists():
                return self._parse_path_txt(path_file)
            return "PROCESS_ERROR", None

        return status, None

    def _wait_for_outputs(self, process, output_dir, expect_costmap):
        """等待输出文件生成"""
        output_dir = Path(output_dir)
        path_file = output_dir / "path.txt"
        costmap_file = output_dir / "costmap.txt"
        start_time = time.time()

        while True:
            if path_file.exists():
                if (not expect_costmap) or costmap_file.exists():
                    return "success"

            return_code = process.poll()
            if return_code is not None:
                if path_file.exists():
                    return "success"
                return "process_error"

            if time.time() - start_time > self.timeout:
                return "timeout"

            time.sleep(0.2)

    def _parse_path_txt(self, path_file):
        """解析路径文件"""
        import re

        if not path_file.exists():
            return "NO_PATH_FOUND", None

        content = path_file.read_text(encoding="utf-8").strip()

        if content == "START_IS_OBSTACLE":
            return "START_IS_OBSTACLE", None
        if content == "GOAL_IS_OBSTACLE":
            return "GOAL_IS_OBSTACLE", None
        if content == "START_AND_GOAL_ARE_OBSTACLES":
            return "START_AND_GOAL_ARE_OBSTACLES", None
        if content == "NO_PATH_FOUND":
            return "NO_PATH_FOUND", None
        if not content:
            return "EMPTY", None

        matches = re.findall(r"\((\-?\d+),(\-?\d+)\)", content)
        if not matches:
            return "INVALID", None

        path = [(int(x), int(y)) for x, y in matches]
        return "OK", path