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

    # ============================================================
    # 新增：全局交互路径规划
    # ============================================================
    def plan_global_path_interactive(self, dem_tif_path, color_png_path, resolution, output_dir):
        """
        调用 C++ 全局交互路径规划器。

        这里的正确逻辑是：
        - 实时打印 C++ stdout，让 Python 终端能看到交互说明
        - 必须等待 C++ 进程真正退出（例如用户按 ESC 关闭窗口）后，才继续后续 Python 流程
        - 退出后再检查 dem.txt / costmap.txt / path.txt 是否生成
        """
        import threading
        import queue

        dem_tif_path = Path(dem_tif_path)
        color_png_path = Path(color_png_path)
        output_dir = Path(output_dir)

        if not self.exe_path.exists():
            return "EXE_NOT_FOUND"
        if not dem_tif_path.exists():
            return "DEM_TIF_NOT_FOUND"
        if not color_png_path.exists():
            return "COLOR_PNG_NOT_FOUND"

        output_dir.mkdir(parents=True, exist_ok=True)

        # 删除旧结果，避免误判
        for name in ("dem.txt", "costmap.txt", "path.txt"):
            f = output_dir / name
            if f.exists():
                f.unlink()

        cmd = self._build_global_interactive_cmd(
            dem_tif_path=dem_tif_path,
            color_png_path=color_png_path,
            resolution=resolution,
            output_dir=output_dir,
        )

        print("\n" + "=" * 70)
        print("调用 C++ 全局交互路径规划器")
        print("=" * 70)
        print(f"EXE: {self.exe_path}")
        print(f"DEM tif: {dem_tif_path}")
        print(f"Color png: {color_png_path}")
        print(f"Resolution: {resolution}m")
        print(f"Output dir: {output_dir}")
        print("请在 C++ 弹出的底图窗口中完成全局路径交互规划。")
        print("=" * 70)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            print(f"启动全局交互路径规划器失败: {e}")
            return "PROCESS_ERROR"

        # 用后台线程持续读取 stdout，避免主线程阻塞
        q = queue.Queue()
        def _reader_thread(pipe, q_):
            try:
                for line in iter(pipe.readline, ''):
                    q_.put(line)
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t = threading.Thread(target=_reader_thread, args=(process.stdout, q), daemon=True)
        t.start()

        start_time = time.time()

        # 关键：这里必须等进程退出，而不是文件一生成就返回
        while True:
            # 实时打印 C++ 输出，让交互说明能看到
            while not q.empty():
                line = q.get_nowait()
                print(line, end="")

            return_code = process.poll()
            if return_code is not None:
                break

            if self.timeout is not None and self.timeout > 0:
                if time.time() - start_time > self.timeout:
                    process.kill()
                    print("\n全局交互路径规划器运行超时，已终止。")
                    return "TIMEOUT"

            time.sleep(0.05)

        # 把退出前残留的输出再打印完
        time.sleep(0.1)
        while not q.empty():
            line = q.get_nowait()
            print(line, end="")

        dem_file = output_dir / "dem.txt"
        costmap_file = output_dir / "costmap.txt"
        path_file = output_dir / "path.txt"

        if not dem_file.exists():
            return "DEM_NOT_FOUND"
        if not costmap_file.exists():
            return "COSTMAP_NOT_FOUND"
        if not path_file.exists():
            return "PATH_NOT_FOUND"

        path_status, _ = self._parse_path_txt(path_file)
        return path_status

    def _build_global_interactive_cmd(self, dem_tif_path, color_png_path, resolution, output_dir):
        """
        与 C++ main.cpp 的 Mode 3 保持一致：
            exe <tiff_path> <tiff_color_path> <result_dir> <grid_size>
        """
        return [
            str(self.exe_path),
            str(dem_tif_path),
            str(color_png_path),
            str(output_dir),
            str(resolution),
        ]

    # ============================================================
    # 原有：局部 DEM 路径规划
    # ============================================================
    def plan_path(self, dem_path, start_col, start_row, goal_col, goal_row, resolution, output_dir):
        dem_path = Path(dem_path)
        output_dir = Path(output_dir)

        if not self.exe_path.exists():
            return "EXE_NOT_FOUND", None
        if not dem_path.exists():
            return "DEM_NOT_FOUND", None

        output_dir.mkdir(parents=True, exist_ok=True)

        # 关键：删除旧输出，避免误读残留文件
        for name in ("path.txt", "costmap.txt"):
            f = output_dir / name
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

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
        costmap_path = Path(costmap_path)
        output_dir = Path(output_dir)

        if not self.exe_path.exists():
            return "EXE_NOT_FOUND", None
        if not costmap_path.exists():
            return "COSTMAP_NOT_FOUND", None

        output_dir.mkdir(parents=True, exist_ok=True)

        # 关键：删除旧 path.txt，避免读取上一次结果
        path_file = output_dir / "path.txt"
        if path_file.exists():
            try:
                path_file.unlink()
            except Exception:
                pass

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
        gradual=False,
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
            revised = raw_costmap.copy()
            self._soften_obstacles_in_radius(
                costmap=revised,
                center=(start_col, start_row),
                radius=max_radius,
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

            return status, path_points, revision_costmap_path, max_radius
        else:
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

                print(f"\n[全局起点软化] radius = {radius}")
                print(f"  costmap: {revision_costmap_path}")

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

        path_file = Path(path_file)

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