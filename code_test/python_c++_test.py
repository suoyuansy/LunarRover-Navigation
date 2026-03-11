import subprocess
import time
from pathlib import Path
from datetime import datetime
import re
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dem_txt(dem_path: Path) -> np.ndarray:
    """读取矩形 DEM txt，返回 float64 二维数组。"""
    rows = []
    with dem_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split()]
            rows.append(row)

    if not rows:
        raise RuntimeError(f"DEM 文件为空或无有效数据: {dem_path}")

    col_num = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != col_num:
            raise RuntimeError(f"DEM 不是规则矩形，第 {i + 1} 行列数异常: {dem_path}")

    return np.array(rows, dtype=np.float64)


def load_costmap_txt(costmap_path: Path) -> np.ndarray:
    """读取 costmap txt，返回 float64 二维数组。"""
    rows = []
    with costmap_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split()]
            rows.append(row)

    if not rows:
        raise RuntimeError(f"costmap 文件为空或无有效数据: {costmap_path}")

    col_num = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != col_num:
            raise RuntimeError(f"costmap 不是规则矩形，第 {i + 1} 行列数异常: {costmap_path}")

    return np.array(rows, dtype=np.float64)


def parse_path_txt(path_path: Path):
    """
    解析 path.txt
    返回:
      ("START_IS_OBSTACLE", None)
      ("GOAL_IS_OBSTACLE", None)
      ("START_AND_GOAL_ARE_OBSTACLES", None)
      ("NO_PATH_FOUND", None)
      ("OK", [(x1,y1), (x2,y2), ...])
    """
    content = path_path.read_text(encoding="utf-8").strip()

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


def make_costmap_gray_image(costmap: np.ndarray) -> np.ndarray:
    """
    把 costmap 转成 8 位灰度图：
    - >= 1 的点赋值 255
    - 0~1 之间线性映射到 0~255
    """
    img = np.zeros_like(costmap, dtype=np.uint8)

    ge_one = costmap >= 1.0
    lt_one = ~ge_one

    img[ge_one] = 255
    scaled = np.clip(costmap[lt_one], 0.0, 1.0) * 255.0
    img[lt_one] = scaled.astype(np.uint8)

    return img


def draw_start_goal_on_gray(gray_img: np.ndarray, start, goal) -> np.ndarray:
    """
    在灰度图上标注起点终点
    起点蓝色，终点绿色
    """
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    sx, sy = start
    gx, gy = goal

    cv2.circle(color_img, (sx, sy), 5, (255, 0, 0), -1)   # 蓝色
    cv2.circle(color_img, (gx, gy), 5, (0, 255, 0), -1)   # 绿色

    cv2.putText(color_img, "Start", (sx + 8, sy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(color_img, "Goal", (gx + 8, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return color_img


def draw_path_on_image(color_img: np.ndarray, path_points):
    """
    在图上用红线画路径
    """
    out = color_img.copy()
    if not path_points or len(path_points) < 2:
        return out

    pts = np.array(path_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=False, color=(0, 0, 255), thickness=2)  # 红色
    return out


def save_3d_scene(
    dem: np.ndarray,
    start,
    goal,
    path_points,
    output_path: Path
):
    """
    保存三维场景图（贴合 DEM 上方）
    - DEM 曲面
    - 起点
    - 终点
    - 路径（如果有）
    """

    rows, cols = dem.shape

    # 使用完整分辨率网格
    xs = np.arange(cols)
    ys = np.arange(rows)
    X, Y = np.meshgrid(xs, ys)
    Z = dem.astype(np.float32)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制 DEM 表面
    ax.plot_surface(
        X,
        Y,
        Z,
        cmap="gray",
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.9
    )

    sx, sy = start
    gx, gy = goal

    # 起点终点高程
    sz = Z[sy, sx]
    gz = Z[gy, gx]

    # 计算 Z 范围并设定一个合理的抬高偏移
    z_range = np.max(Z) - np.min(Z)
    z_offset = max(z_range * 0.01, 0.5)

    # 起点
    ax.scatter(
        [sx], [sy], [sz + z_offset],  # 抬高一点
        c="blue",
        s=20,
        depthshade=False,
        label="Start"
    )

    # 终点
    ax.scatter(
        [gx], [gy], [gz + z_offset],  # 抬高一点
        c="green",
        s=20,
        depthshade=False,
        label="Goal"
    )

    # 路径
    if path_points and len(path_points) > 1:
        px = np.array([p[0] for p in path_points], dtype=np.int32)
        py = np.array([p[1] for p in path_points], dtype=np.int32)
        pz = Z[py, px] + z_offset  # 路径点抬高

        ax.plot(
            px,
            py,
            pz,
            color="red",
            linewidth=2,
            label="Path"
        )

    ax.set_title("3D DEM with Start / Goal / Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.legend()

    # 更适合观察地形
    ax.view_init(elev=45, azim=-60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def wait_for_outputs(process, output_dir: Path, timeout_sec: float = 15.0, poll_interval: float = 0.2):
    """
    等待 path.txt 和 costmap.txt 同时生成。
    若 C++ 进程提前异常退出，则立即捕获错误并返回。
    返回：
        ("success", path_file, costmap_file, None, None)
        ("process_error", None, None, stdout, stderr)
        ("timeout", None, None, None, None)
    """
    path_file = output_dir / "path.txt"
    costmap_file = output_dir / "costmap.txt"

    start_time = time.time()

    while True:
        # 1. 文件已生成
        if path_file.exists() and costmap_file.exists():
            return "success", path_file, costmap_file, None, None

        # 2. 进程是否已经结束
        return_code = process.poll()
        if return_code is not None:
            stdout, stderr = process.communicate()
            if return_code != 0:
                return "process_error", None, None, stdout, stderr
            else:
                # 正常退出但文件没生成，也视为异常
                return "process_error", None, None, stdout, stderr

        # 3. 是否超时
        if time.time() - start_time > timeout_sec:
            return "timeout", None, None, None, None

        time.sleep(poll_interval)

def main():
    
    exe_path = Path(r"D:\Graduation_design\code\path_planning_based_on_lunar_DEM\out\build\x64-Debug\path_planning_based_on_lunar_DEM.exe")
    dem_path = Path(r"D:\Graduation_design\code\LunarRover_UE\dem.txt")
    
    start_x, start_y = 544, 700
    goal_x, goal_y = 966, 966
    grid_size = 1.0

    # 在 dem 同目录下创建输出文件夹
    parent_dir = dem_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = parent_dir / f"local_planningpath_{timestamp}"

    if not exe_path.exists():
        print(f"exe 不存在: {exe_path}")
        return

    if not dem_path.exists():
        print(f"DEM 文件不存在: {dem_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始局部路径规划")
    print(f"传入 dem 文件: {dem_path}")
    print(f"起点坐标: ({start_x}, {start_y})")
    print(f"终点坐标: ({goal_x}, {goal_y})")
    print(f"栅格分辨率: {grid_size}")
    print(f"输出文件夹名: {output_dir.name}")
    print("正在获取路径与代价地图...")

    cmd = [
        str(exe_path),
        str(dem_path),
        str(start_x),
        str(start_y),
        str(goal_x),
        str(goal_y),
        str(grid_size),
        str(output_dir),
    ]

    '''
    costmap_path = Path(r"D:\Graduation_design\code\LunarRover_UE\costmap.txt")

    start_x, start_y = 544, 700
    goal_x, goal_y = 966, 966

    # 在 costmap 同目录下创建输出文件夹
    parent_dir = costmap_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = parent_dir / f"local_planningpath_{timestamp}"

    if not exe_path.exists():
        print(f"exe 不存在: {exe_path}")
        return

    if not costmap_path.exists():
        print(f"costmap 文件不存在: {costmap_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始局部路径规划（costmap 直接输入模式）")
    print(f"传入 costmap 文件: {costmap_path}")
    print(f"起点坐标: ({start_x}, {start_y})")
    print(f"终点坐标: ({goal_x}, {goal_y})")
    print(f"输出文件夹名: {output_dir.name}")
    print("正在获取路径与代价地图...")

    cmd = [
        str(exe_path),
        str(costmap_path),
        str(start_x),
        str(start_y),
        str(goal_x),
        str(goal_y),
        str(output_dir),
    ]
    '''

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    detect_begin = time.time()
    status, path_file, costmap_file, stdout, stderr = wait_for_outputs(
        process,
        output_dir,
        timeout_sec=15.0,
        poll_interval=0.2
    )
    elapsed = time.time() - detect_begin

    if status == "timeout":
        print("获取超时，路径生成出错")
        process.kill()
        try:
            stdout, stderr = process.communicate(timeout=1)
        except Exception:
            stdout, stderr = "", ""
        if stdout:
            print("\nC++ stdout:")
            print(stdout)
        if stderr:
            print("\nC++ stderr:")
            print(stderr)
        return

    if status == "process_error":
        print("C++ 路径规划程序运行出错，已提前退出")
        if stdout:
            print("\nC++ stdout:")
            print(stdout)
        if stderr:
            print("\nC++ stderr:")
            print(stderr)
        return

    print(f"已获取路径与代价地图，用时 {elapsed:.2f} 秒")
    print(f"path 文件: {path_file}")
    print(f"costmap 文件: {costmap_file}")

    # 文件已经生成后，再等待进程彻底结束
    stdout, stderr = process.communicate()

    if stdout:
        print("\nC++ stdout:")
        print(stdout)

    if stderr:
        print("\nC++ stderr:")
        print(stderr)

    if process.returncode not in (0, None):
        print(f"\n警告：C++ 程序返回码为 {process.returncode}")
        return


    # 读取 DEM 和 costmap
    try:
        dem = load_dem_txt(dem_path)
        costmap = load_costmap_txt(costmap_file)
    except Exception as e:
        print(f"读取 DEM 或 costmap 失败: {e}")
        return

    # 生成 costmap 灰度图
    gray = make_costmap_gray_image(costmap)

    # 标注起终点
    start = (start_x, start_y)
    goal = (goal_x, goal_y)
    gray_with_points = draw_start_goal_on_gray(gray, start, goal)

    # 保存带起终点的代价地图图像
    costmap_points_img_path = output_dir / "costmap_with_start_goal.jpg"
    cv2.imwrite(str(costmap_points_img_path), gray_with_points)

    # 读取路径文件
    try:
        path_status, path_points = parse_path_txt(path_file)
    except Exception as e:
        print(f"读取 path.txt 失败: {e}")
        return

    if path_status == "START_AND_GOAL_ARE_OBSTACLES":
        print("起点和终点均为障碍，需要重新选择起点和终点")
        try:
            save_3d_scene(
                dem=dem,
                start=start,
                goal=goal,
                path_points=None,
                output_path=output_dir / "dem_3d_scene.jpg"
            )
        except Exception as e:
            print(f"保存三维场景失败: {e}")
        return

    if path_status == "START_IS_OBSTACLE":
        print("起点为障碍，需要重新选择起点")
        try:
            save_3d_scene(
                dem=dem,
                start=start,
                goal=goal,
                path_points=None,
                output_path=output_dir / "dem_3d_scene.jpg"
            )
        except Exception as e:
            print(f"保存三维场景失败: {e}")
        return

    if path_status == "GOAL_IS_OBSTACLE":
        print("目标点为障碍，需要重新选择目标点")
        try:
            save_3d_scene(
                dem=dem,
                start=start,
                goal=goal,
                path_points=None,
                output_path=output_dir / "dem_3d_scene.jpg"
            )
        except Exception as e:
            print(f"保存三维场景失败: {e}")
        return

    if path_status == "NO_PATH_FOUND":
        print("两点间不存在路径，需要重新选择目标点或者规划策略")
        try:
            save_3d_scene(
                dem=dem,
                start=start,
                goal=goal,
                path_points=None,
                output_path=output_dir / "dem_3d_scene.jpg"
            )
        except Exception as e:
            print(f"保存三维场景失败: {e}")
        return

    if path_status != "OK" or not path_points:
        print("path.txt 内容异常，无法解析路径")
        return

    print("成功得到路径")
    print("正在生成可视化结果")
    # 在灰度图上画红色路径
    path_vis = draw_path_on_image(gray_with_points, path_points)
    path_vis_path = output_dir / "costmap_with_start_goal_and_path.jpg"
    cv2.imwrite(str(path_vis_path), path_vis)

    # 三维场景可视化
    try:
        save_3d_scene(
            dem=dem,
            start=start,
            goal=goal,
            path_points=path_points,
            output_path=output_dir / "dem_3d_scene.jpg"
        )
    except Exception as e:
        print(f"保存三维场景失败: {e}")
        return

    print("可视化结果已保存到输出文件夹")
    print(f"输出目录: {output_dir}")



if __name__ == "__main__":
    main()