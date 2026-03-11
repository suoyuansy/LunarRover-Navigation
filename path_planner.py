"""
局部路径规划器接口模块
负责调用C++路径规划器进行局部路径规划
"""

import subprocess
import time
from pathlib import Path

# 导入配置
from config import PATH_PLANNING_EXE, PATH_PLANNING_TIMEOUT


# ========================================
# 局部路径规划器接口（C++ 程序调用）
# ========================================
class LocalPathPlanner:
    """调用 C++ 路径规划器进行局部路径规划"""

    def __init__(self, exe_path=PATH_PLANNING_EXE):
        self.exe_path = Path(exe_path)
        self.timeout = PATH_PLANNING_TIMEOUT

    def plan_path(self, dem_path, start_col, start_row, goal_col, goal_row, 
                  grid_size, output_dir):
        """
        调用 C++ 路径规划器
        返回：
            ("OK", path_points_dem) - 成功，path_points_dem 是 [(col, row), ...]
            其他状态码 - 失败
        """
        dem_path = Path(dem_path)
        output_dir = Path(output_dir)

        if not self.exe_path.exists():
            print(f"错误：路径规划器可执行文件不存在: {self.exe_path}")
            return "EXE_NOT_FOUND", None

        if not dem_path.exists():
            print(f"错误：DEM 文件不存在: {dem_path}")
            return "DEM_NOT_FOUND", None

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.exe_path),
            str(dem_path),
            str(start_col),
            str(start_row),
            str(goal_col),
            str(goal_row),
            str(grid_size),
            str(output_dir),
        ]

        print(f"调用路径规划器:")
        print(f"  DEM: {dem_path}")
        print(f"  起点: ({start_col}, {start_row})")
        print(f"  终点: ({goal_col}, {goal_row})")
        print(f"  分辨率: {grid_size}m")
        print(f"  输出: {output_dir}")

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

        status, result = self._wait_for_outputs(process, output_dir)

        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                print(f"\nC++ stdout:\n{stdout}")
            if stderr:
                print(f"\nC++ stderr:\n{stderr}")
        except:
            pass

        if status == "success":
            path_file = output_dir / "path.txt"
            path_status, path_points = self._parse_path_txt(path_file)
            return path_status, path_points
        elif status == "timeout":
            print("路径规划超时")
            process.kill()
            return "TIMEOUT", None
        elif status == "process_error":
            print("路径规划进程错误")
            return "PROCESS_ERROR", None

        return status, result

    def _wait_for_outputs(self, process, output_dir):
        """等待 path.txt 和 costmap.txt 生成"""
        path_file = output_dir / "path.txt"
        costmap_file = output_dir / "costmap.txt"

        start_time = time.time()

        while True:
            if path_file.exists() and costmap_file.exists():
                return "success", None

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    return "process_error", None
                else:
                    return "process_error", None

            if time.time() - start_time > self.timeout:
                return "timeout", None

            time.sleep(0.2)

    def _parse_path_txt(self, path_file):
        """解析 path.txt"""
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