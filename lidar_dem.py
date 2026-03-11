"""
LiDAR数据采集与DEM构建模块
包含点云采集器和局部DEM构建器
"""

import airsim
import time
import math
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# 导入配置
from config import (
    VEHICLE_NAME, LIDAR_SENSOR_NAME,
    LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET,
    LIDAR_POLL_INTERVAL, LIDAR_WAIT_TIMEOUT_PER_FRAME,
    LIDAR_MIN_VALID_DISTANCE, LIDAR_MAX_VALID_DISTANCE,
    LIDAR_MIN_Z, LIDAR_MAX_Z,
    LOCAL_DEM_RANGE, LOCAL_DEM_RESOLUTION,
    DEM_GAUSSIAN_SIGMA, DEM_USE_GDAL_FILL,
    GRID_FUSION_METHOD
)

# 导入工具函数
from utils import euler_to_rotation_matrix

# 尝试使用 GDAL 填充空洞；如果环境未安装 GDAL，则自动退回 scipy 插值
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except Exception:
    GDAL_AVAILABLE = False


# ========================================
# LiDAR 点云采集器
# ========================================
class PointCloudAccumulator:
    """在一段运动过程中持续采集 LiDAR 点云"""

    def __init__(self, vehicle_name=VEHICLE_NAME, sensor_name=LIDAR_SENSOR_NAME):
        self.vehicle_name = vehicle_name
        self.sensor_name = sensor_name
        self.client = None
        self.frames = []
        self.last_lidar_timestamp = None

    def initialize(self, client):
        self.client = client
        print("LiDAR 点云采集模块初始化完成")
        print(f" 传感器名: {self.sensor_name}")

    def reset(self, clear_timestamp=False):
        """重置累计缓存"""
        self.frames = []
        if clear_timestamp:
            self.last_lidar_timestamp = None

    def _wait_for_next_new_lidar_frame(self, timeout=LIDAR_WAIT_TIMEOUT_PER_FRAME, poll_interval=LIDAR_POLL_INTERVAL):
        """等待一帧新的 timestamp LiDAR 数据"""
        if self.client is None:
            raise RuntimeError("PointCloudAccumulator 尚未 initialize(client)")

        deadline = time.time() + timeout

        while time.time() < deadline:
            lidar_data = self.client.getLidarData(self.sensor_name, self.vehicle_name)

            if lidar_data is None or lidar_data.point_cloud is None or len(lidar_data.point_cloud) < 3:
                time.sleep(poll_interval)
                continue

            ts = lidar_data.time_stamp

            if self.last_lidar_timestamp is not None and ts == self.last_lidar_timestamp:
                time.sleep(poll_interval)
                continue

            self.last_lidar_timestamp = ts
            return lidar_data

        return None

    def collect_once_wait_n_frames(self, expected_pose, wait_new_frames=LIDAR_WAIT_NEW_FRAMES_AFTER_POSE_SET,
                                   wait_timeout_per_frame=LIDAR_WAIT_TIMEOUT_PER_FRAME,
                                   poll_interval=LIDAR_POLL_INTERVAL,
                                   debug_prefix="采集点云"):
        """设置完位姿后，等待 N 帧新点云，保存最后一帧"""
        if self.client is None:
            raise RuntimeError("PointCloudAccumulator 尚未 initialize(client)")

        last_new_lidar = None

        for i in range(wait_new_frames):
            lidar_data = self._wait_for_next_new_lidar_frame(
                timeout=wait_timeout_per_frame,
                poll_interval=poll_interval
            )

            if lidar_data is None:
                print(f"  {debug_prefix}: 等待第 {i + 1}/{wait_new_frames} 帧新 LiDAR 数据超时")
                return 0

            last_new_lidar = lidar_data
            print(f"  {debug_prefix}: 获取到第 {i + 1}/{wait_new_frames} 帧新 LiDAR, timestamp={lidar_data.time_stamp}")

        if last_new_lidar is None:
            return 0

        pts = np.array(last_new_lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        pts = self._filter_lidar_points(pts)

        if len(pts) == 0:
            print(f"  {debug_prefix}: 最终保存帧过滤后无有效点")
            return 0

        self.frames.append({
            "points_local": pts,
            "pose": expected_pose,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "lidar_timestamp": last_new_lidar.time_stamp
        })

        return len(pts)

    def _filter_lidar_points(self, points_local):
        """LiDAR 点云预处理"""
        if points_local is None or len(points_local) == 0:
            return np.empty((0, 3), dtype=np.float32)

        finite_mask = np.isfinite(points_local).all(axis=1)
        pts = points_local[finite_mask]

        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        dists = np.linalg.norm(pts, axis=1)

        valid_mask = (
            (dists >= LIDAR_MIN_VALID_DISTANCE) &
            (dists <= LIDAR_MAX_VALID_DISTANCE) &
            (pts[:, 2] >= LIDAR_MIN_Z) &
            (pts[:, 2] <= LIDAR_MAX_Z)
        )

        pts = pts[valid_mask]
        return pts

    def get_all_frames(self):
        """获取当前累计的所有点云帧"""
        return self.frames

    def save_frames_to_point_data(self, point_data_dir):
        """保存每帧点云到 point_data 目录"""
        for idx, frame in enumerate(self.frames, start=1):
            _, _, _, roll, pitch, yaw = frame["pose"]

            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            lidar_ts = frame.get("lidar_timestamp", 0)

            filename = (
                f"{idx:04d}_ts({lidar_ts})_roll({roll_deg:.2f})_pitch({pitch_deg:.2f})_yaw({yaw_deg:.2f}).txt"
            )
            filepath = os.path.join(point_data_dir, filename)
            np.savetxt(filepath, frame["points_local"], fmt="%.6f")


# ========================================
# 局部 DEM 构建器
# ========================================
class LocalDEMBuilder:
    """根据累计的多帧点云，在"当前传感器坐标系"下构建局部 DEM"""

    def __init__(self, dem_range=LOCAL_DEM_RANGE, resolution=LOCAL_DEM_RESOLUTION):
        self.dem_range = dem_range
        self.resolution = resolution
        self.grid_size = int((dem_range * 2) / resolution)

    def build_dem_from_frames(self, frames, current_pose):
        """
        输入：
            frames: 累积的点云帧
            current_pose: 当前构建时刻的传感器位姿 (cx,cy,cz,roll,pitch,yaw)
        输出：
            dem_grid: HxW，存储的是"当前传感器坐标系下的 z 值（Z向下）"
            mask_valid: HxW
            all_points_in_current_sensor: 融合到当前传感器坐标系后的所有点
        """
        if frames is None or len(frames) == 0:
            print("警告：无历史点云帧，无法构建 DEM")
            return None, None, None

        cx, cy, cz, croll, cpitch, cyaw = current_pose

        # 当前构建时刻：当前传感器坐标系 -> 世界坐标系 的旋转
        R_current = euler_to_rotation_matrix(croll, cpitch, cyaw)
        t_current = np.array([cx, cy, cz], dtype=np.float64)

        # 需要的是：世界 -> 当前传感器坐标系
        R_world_to_current = R_current.T

        transformed_points_list = []

        for frame in frames:
            pts_local_old = frame["points_local"]
            fx, fy, fz, froll, fpitch, fyaw = frame["pose"]

            if pts_local_old is None or len(pts_local_old) == 0:
                continue

            # 旧采样时刻：旧传感器坐标系 -> 世界坐标系
            R_old = euler_to_rotation_matrix(froll, fpitch, fyaw)
            t_old = np.array([fx, fy, fz], dtype=np.float64)

            # 1) old_sensor -> world
            pts_world = (R_old @ pts_local_old.T).T + t_old

            # 2) world -> current_sensor
            pts_current = (R_world_to_current @ (pts_world - t_current).T).T

            transformed_points_list.append(pts_current)

        if not transformed_points_list:
            print("警告：没有有效点云帧可用于 DEM 构建")
            return None, None, None

        all_points_in_current_sensor = np.vstack(transformed_points_list)

        # 在当前传感器坐标系下裁剪 DEM 范围
        x_local = all_points_in_current_sensor[:, 0]  # forward
        y_local = all_points_in_current_sensor[:, 1]  # right
        z_local = all_points_in_current_sensor[:, 2]  # down

        left_local = -y_local

        range_mask = (
            (np.abs(x_local) <= self.dem_range) &
            (np.abs(left_local) <= self.dem_range)
        )

        x_local = x_local[range_mask]
        y_local = y_local[range_mask]
        z_local = z_local[range_mask]

        if len(x_local) < 20:
            print("警告：局部范围内点云不足，无法构建 DEM")
            return None, None, None

        # 左前方为原点：
        # row: 前 -> 后，对应 x_local 从 +range -> -range
        # col: 左 -> 右，对应 y_local 从 -range -> +range
        row_float = (self.dem_range - x_local) / self.resolution
        col_float = (y_local + self.dem_range) / self.resolution

        rows = np.floor(row_float).astype(np.int32)
        cols = np.floor(col_float).astype(np.int32)

        valid_idx = (
            (rows >= 0) & (rows < self.grid_size) &
            (cols >= 0) & (cols < self.grid_size)
        )

        rows = rows[valid_idx]
        cols = cols[valid_idx]
        zs = z_local[valid_idx]

        if len(zs) < 20:
            print("警告：落入栅格的有效点不足，无法构建 DEM")
            return None, None, None

        dem_grid = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)

        # 将每格对应的 z 聚合
        grid_buckets = {}
        for r, c, z in zip(rows, cols, zs):
            key = (r, c)
            if key not in grid_buckets:
                grid_buckets[key] = []
            grid_buckets[key].append(float(z))

        for (r, c), z_list in grid_buckets.items():
            z_arr = np.array(z_list, dtype=np.float32)

            if GRID_FUSION_METHOD == "mean":
                dem_grid[r, c] = float(np.mean(z_arr))
            elif GRID_FUSION_METHOD == "min":
                dem_grid[r, c] = float(np.min(z_arr))
            elif GRID_FUSION_METHOD == "max":
                dem_grid[r, c] = float(np.max(z_arr))
            else:
                dem_grid[r, c] = float(np.median(z_arr))

        mask_valid = ~np.isnan(dem_grid)

        if np.sum(mask_valid) < 20:
            print("警告：栅格有效点过少，无法构建 DEM")
            return None, None, None

        dem_grid = self.fill_dem_holes(dem_grid)
        dem_grid = gaussian_filter(dem_grid, sigma=DEM_GAUSSIAN_SIGMA)

        return dem_grid, mask_valid, all_points_in_current_sensor

    def fill_dem_holes(self, dem_grid):
        """优先使用 GDAL FillNodata；若不可用则退回 scipy.griddata"""
        if DEM_USE_GDAL_FILL and GDAL_AVAILABLE:
            try:
                return self._fill_with_gdal(dem_grid)
            except Exception as e:
                print(f"GDAL FillNodata 失败，退回 scipy 插值: {e}")

        return self._fill_with_scipy(dem_grid)

    def _fill_with_gdal(self, dem_grid):
        """使用 GDAL FillNodata 填充空洞"""
        rows, cols = dem_grid.shape
        driver = gdal.GetDriverByName('MEM')

        ds = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
        band = ds.GetRasterBand(1)

        data = dem_grid.copy().astype(np.float32)
        nodata_value = -9999.0
        nan_mask = np.isnan(data)
        data[nan_mask] = nodata_value

        band.WriteArray(data)
        band.SetNoDataValue(nodata_value)

        mask_ds = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
        mask_band = mask_ds.GetRasterBand(1)
        mask_array = np.where(np.isnan(dem_grid), 0, 1).astype(np.uint8)
        mask_band.WriteArray(mask_array)

        gdal.FillNodata(
            targetBand=band,
            maskBand=mask_band,
            maxSearchDist=50,
            smoothingIterations=1
        )

        filled = band.ReadAsArray().astype(np.float32)

        remain_mask = (filled == nodata_value) | (~np.isfinite(filled))
        if np.any(remain_mask):
            valid = filled[~remain_mask]
            fill_val = float(np.median(valid)) if len(valid) > 0 else 0.0
            filled[remain_mask] = fill_val

        return filled

    def _fill_with_scipy(self, dem_grid):
        """使用 scipy.griddata 填充空洞"""
        rows, cols = dem_grid.shape
        valid_mask = ~np.isnan(dem_grid)

        if np.sum(valid_mask) == 0:
            return np.zeros_like(dem_grid, dtype=np.float32)

        rr, cc = np.mgrid[0:rows, 0:cols]
        valid_points = np.column_stack((rr[valid_mask], cc[valid_mask]))
        valid_values = dem_grid[valid_mask]

        filled = griddata(
            valid_points,
            valid_values,
            (rr, cc),
            method='linear'
        )

        remain_mask = np.isnan(filled)
        if np.any(remain_mask):
            filled_nearest = griddata(
                valid_points,
                valid_values,
                (rr, cc),
                method='nearest'
            )
            filled[remain_mask] = filled_nearest[remain_mask]

        return filled.astype(np.float32)

    def save_dem_results(self, dem_grid, current_pose, build_dir):
        """保存 DEM 结果"""
        cx, cy, cz, roll, pitch, yaw = current_pose

        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        dem_txt_path = os.path.join(build_dir, "dem.txt")
        meta_txt_path = os.path.join(build_dir, "dem_meta.txt")
        jpg_path = os.path.join(build_dir, "dem_3d.jpg")

        # 保存时取反，使得 Z 向上为正（符合常规 DEM 格式）
        np.savetxt(dem_txt_path, -dem_grid, fmt="%.4f")

        with open(meta_txt_path, "w", encoding="utf-8") as f:
            f.write("===== Local DEM Metadata =====\n")
            f.write(f"center_x = {cx:.6f}\n")
            f.write(f"center_y = {cy:.6f}\n")
            f.write(f"center_z = {cz:.6f}\n")
            f.write(f"roll_rad = {roll:.8f}\n")
            f.write(f"pitch_rad = {pitch:.8f}\n")
            f.write(f"yaw_rad = {yaw:.8f}\n")
            f.write(f"roll_deg = {roll_deg:.6f}\n")
            f.write(f"pitch_deg = {pitch_deg:.6f}\n")
            f.write(f"yaw_deg = {yaw_deg:.6f}\n")
            f.write(f"dem_range = {self.dem_range}\n")
            f.write(f"resolution = {self.resolution}\n")
            f.write(f"grid_size = {self.grid_size}\n")
            f.write(f"fusion_method = {GRID_FUSION_METHOD}\n")
            f.write("coordinate_frame = current_sensor_frame\n")
            f.write("origin_definition = top-left is front-left of current sensor\n")
            f.write("row_direction = front -> back\n")
            f.write("col_direction = left -> right\n")
            f.write("x_axis = forward_positive\n")
            f.write("y_axis = right_positive\n")
            f.write("z_axis = down_positive\n")
            f.write("dem_z_direction = down_positive (stored value), up_positive in txt file\n")

        self._save_3d_visualization(dem_grid, jpg_path)

        print(f"DEM 已保存: {dem_txt_path}")
        print(f"元数据已保存: {meta_txt_path}")
        print(f"三维图已保存: {jpg_path}")

        return dem_txt_path

    def _save_3d_visualization(self, dem_grid, jpg_path):
        """保存 DEM 三维可视化图"""
        rows, cols = dem_grid.shape

        row_coords = np.arange(rows) * self.resolution
        col_coords = np.arange(cols) * self.resolution
        xx, yy = np.meshgrid(col_coords, row_coords)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 可视化时取反，使得向上为正
        zz_vis = -dem_grid

        ax.plot_surface(xx, yy, zz_vis, cmap='terrain', linewidth=0, antialiased=True)

        ax.invert_xaxis()
        ax.set_title("Local DEM 3D Visualization")
        ax.set_xlabel("Column (left -> right, m)")
        ax.set_ylabel("Row (front -> back, m)")
        ax.set_zlabel("Height (upward positive, m)")

        plt.tight_layout()
        plt.savefig(jpg_path, dpi=200)
        plt.close(fig)