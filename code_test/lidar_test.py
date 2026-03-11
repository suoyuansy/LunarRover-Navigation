import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
# ========================================
# 配置参数
# ========================================
resolution = 0.2  # 0.2米/格，较高分辨率
size = 30         # 10米x10米范围


class LidarHandler:
    """LiDAR数据处理类"""
    
    def __init__(self, vehicle_name="Car1", sensor_name="LidarSensor"):
        self.vehicle_name = vehicle_name
        self.sensor_name = sensor_name
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        print("LiDAR处理器初始化完成")
    
    def get_point_cloud(self):
        """获取点云数据"""
        lidar_data = self.client.getLidarData(self.sensor_name, self.vehicle_name)
        
        if lidar_data.point_cloud is None or len(lidar_data.point_cloud) < 3:
            return None
        
        # 解析为 (N, 3) 数组
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3)
        
        # 获取位置
        pos = lidar_data.pose.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val
        
        # 获取四元数并转换为欧拉角（roll, pitch, yaw）
        orientation = lidar_data.pose.orientation
        qx, qy, qz, qw = orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val
        
        # 四元数转欧拉角
        # Roll (x轴旋转)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y轴旋转)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # 使用90度
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z轴旋转)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 转换为角度并打印
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        
        print(f"LiDAR pose   : x={x:.6f}, y={y:.6f}, z={z:.6f}, "
            f"roll={roll_deg:.3f}°, pitch={pitch_deg:.3f}°, yaw={yaw_deg:.3f}°")
        
        return {
            'points': points,
            'timestamp': lidar_data.time_stamp,
            'pose': lidar_data.pose,
            'point_count': len(points)
        }
    
    def analyze_point_cloud(self, point_data):
        """分析点云数据特征"""
        points = point_data['points']
        
        # 基本统计
        print(f"\n=== LiDAR数据分析 ===")
        print(f"总点数: {point_data['point_count']}")
        print(f"X范围: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
        print(f"Y范围: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
        print(f"Z范围（airsim坐标系下，向下为正）: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
        
        # 转换为实际高度（Z向下为正，所以实际高度 = -Z）
        real_heights = -points[:, 2]
        print(f"实际高程范围: [{real_heights.min():.3f}, {real_heights.max():.3f}]")
        # 距离统计（欧氏距离）
        distances = np.sqrt(np.sum(points**2, axis=1))
        print(f"距离范围: [{distances.min():.2f}, {distances.max():.2f}] 米")
        print(f"平均距离: {distances.mean():.2f} 米")
        
        # 简单统计Z值分布（仅分析）
        # Z>0: 在传感器下方，Z<0: 在传感器上方
        ground_points = points[points[:, 2] > 0] 
        obstacle_points = points[points[:, 2] <= 0]
        
        print(f"高度低于雷达所在位置的点: {len(ground_points)} 个")
        print(f"高度高于雷达所在位置的点: {len(obstacle_points)} 个")
        
        return {
            'all': points,
            'real_heights': real_heights
        }
    
    def visualize_2d(self, point_data, save_path=None):
        """2D俯视可视化（俯视图+两个侧视图）"""
        points = point_data['points']

        # 计算实际高度（反转Z轴，向上为正）
        real_height = -points[:, 2]

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        # 俯视图（X-Y平面）
        #ax1 = axes[0]
        scatter1 = ax1.scatter(points[:, 1], points[:, 0],  c=real_height, cmap='RdYlBu_r', s=2, alpha=0.8)
        ax1.set_xlabel('Y (m)')
        ax1.set_ylabel('X (m)')
        ax1.set_title('Top View (X-Y Plane)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        # 标记传感器位置
        ax1.plot(0, 0, 'r+', markersize=15, mew=2, label='Sensor')
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='Height(m)')
        '''        
        # 侧视图（X-Z平面）
        ax2 = axes[1]
        scatter2 = ax2.scatter(points[:, 0], real_height, c=points[:, 1], cmap='coolwarm', s=2, alpha=0.8)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Side View (X-Z Plane)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Sensor Level')
        ax2.legend()
        plt.colorbar(scatter2, ax=ax2, label='Y (m)')

        # 侧视图（Y-Z平面）
        ax3 = axes[2]
        # Y-Z侧视图：从另一个角度看高度
        scatter3 = ax3.scatter(points[:, 1], real_height,c=points[:, 0], cmap='plasma', s=2, alpha=0.8)
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Height (m)')
        ax3.set_title('Side View (Y-Z Plane)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Sensor Level')
        ax3.legend()
        plt.colorbar(scatter3, ax=ax3, label='X (m)')
        '''
        plt.tight_layout()       
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图像已保存至:{save_path}")
        else:
            plt.show()
    
    def get_height_map(self, point_data, resolution=1.0, size=50):
        """
        生成高度图（用于地形分析）
        参数:
            resolution: 栅格分辨率（米/格），每个栅格代表的物理尺寸
            size: 地图总范围（米），以传感器为中心，向四周延伸size/2
        
        返回: 
            height_map: 2D数组，存储每个栅格的实际高度（向上为正）
            valid_mask: 有效数据掩码
        返回: 2D数组，每个单元格存储该位置的最大高度（Z值）
        """
        points = point_data['points']
        
        # 使用所有点（不区分地面/障碍）
        # 转换为实际高度（向上为正）
        real_heights = -points[:, 2]
      
        # 创建栅格
        grid_size = int(size / resolution)
        height_map = np.full((grid_size, grid_size), np.nan)
        
        center = grid_size // 2
        
        # 将点分配到栅格
        valid_points = 0
        for i, (x, y) in enumerate(points[:, :2]):
            h = real_heights[i]  # 实际高度（向上为正）
            
            # 坐标转换：物理坐标 -> 栅格索引
            gx = int(x / resolution) + center
            gy = int(y / resolution) + center
            
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                valid_points += 1
                # 保留最大高度（因为h是向上为正，所以用max）
                if np.isnan(height_map[gy, gx]) or h > height_map[gy, gx]:
                    height_map[gy, gx] = h
        
        # 统计
        valid_cells = np.sum(~np.isnan(height_map))
        total_cells = grid_size * grid_size
        print(f"栅格尺寸: {grid_size} × {grid_size}")
        print(f"有效点云数: {valid_points} / {len(points)}")
        print(f"有效栅格数: {valid_cells} / {total_cells} ({valid_cells/total_cells*100:.1f}%)")

        return height_map, grid_size, center
    
    def visualize_height_map(self,height_map, grid_size, center, resolution, size):
        """可视化高度图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # ========== 高度图（imshow）==========
        ax1 = axes[0]

        # 转置高度图，让X轴变为垂直方向，Y轴变为水平方向
        # 原: row->Y, col->X，转置后: row->X, col->Y
        transposed_height = height_map.T
        # 使用masked array处理nan值
        masked_height = np.ma.masked_where(np.isnan(transposed_height), transposed_height)
        # 显示高度图
        # extent参数设置坐标轴范围：[left, right, bottom, top]
        half_size = size / 2
        im = ax1.imshow(masked_height, cmap='RdYlBu_r', interpolation='nearest',extent=[-half_size, half_size, -half_size, half_size],  # Y轴反转
                        origin='lower') 
        
        ax1.set_xlabel('Y (m) - Right')
        ax1.set_ylabel('X (m) - Forward')

            # 将X轴刻度移到左侧
        ax1.yaxis.set_ticks_position('left')
        ax1.yaxis.set_label_position('left')
    
            # 将Y轴刻度移到下方（默认）
        ax1.xaxis.set_ticks_position('bottom')
        ax1.xaxis.set_label_position('bottom')

        ax1.set_title('Terrain Height Map\n(Height relative to sensor)')
        ax1.set_aspect('equal')

        # 标记传感器位置
        ax1.plot(0, 0, 'r+', markersize=15, mew=2, label='Sensor')
        ax1.legend()
        
        plt.colorbar(im, ax=ax1, label='Height (m)')
        
        # ========== 3D表面图 ==========
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # 创建网格
        x = np.linspace(-half_size, half_size, grid_size)
        y = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # 绘制表面（只绘制有效数据）
        valid_mask = ~np.isnan(height_map)
        if np.sum(valid_mask) > 0:
            # 使用scatter避免nan问题
            valid_x = X[valid_mask]
            valid_y = Y[valid_mask]
            valid_z = height_map[valid_mask]
            
            surf = ax2.scatter(valid_x, valid_y, valid_z, c=valid_z, cmap='RdYlBu_r', s=10, alpha=0.8)
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_zlabel('Height (m)')
            ax2.set_title('3D Terrain View')
            
            # 设置Z轴方向（向上为正）
            ax2.set_zlim(valid_z.min() - 1, valid_z.max() + 1)
            # 反转Y轴正方向
            ax2.invert_yaxis()
            plt.colorbar(surf, ax=ax2, label='Height (m)', shrink=0.5)
        
        plt.tight_layout()
        plt.show()
    



def main():
    """主函数：LiDAR数据获取与处理"""
    
    # 初始化处理器
    handler = LidarHandler()
    
    print("等待LiDAR数据...")
    time.sleep(2)  # 等待传感器初始化
    
    # 获取数据
    point_data = handler.get_point_cloud()
    
    if point_data is None:
        print("获取数据失败")
        return
    
    # 分析数据
    classified = handler.analyze_point_cloud(point_data)
    
    # 可视化
    print("\n生成可视化...")
    handler.visualize_2d(point_data)
    
    # 生成高度图
    print("\n生成高度图...")
    height_map, grid_size, center = handler.get_height_map(point_data, resolution= resolution, size=size)

    if height_map is not None:
      handler.visualize_height_map(height_map, grid_size, center, resolution, size)
    
    print("\n完成")


if __name__ == "__main__":
    main()