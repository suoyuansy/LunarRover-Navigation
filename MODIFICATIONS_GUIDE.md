# 路径可视化功能修改指南

## 功能概述
添加了三个路径可视化功能，在每次运行的输出文件夹中生成：
1. **global_path.png** - 基于全局costmap的全局规划路径（蓝线）
2. **actual_path.png** - 基于融合costmap的实际走过路径（红线）
3. **compare.png** - 融合costmap背景上的全局路径（蓝线）和实际路径（红线）对比

## 文件修改内容

### 1. visualization.py 新增函数

#### a. `world_to_global_costmap_cell(x_world, y_world, resolution=1.0)`
- **功能**：将世界坐标转换为全局costmap栅格坐标
- **参数**：
  - `x_world, y_world`：世界坐标
  - `resolution`：costmap分辨率（默认1.0米/栅格）
- **返回**：`(col, row)` 栅格坐标

#### b. `draw_global_path_on_costmap(costmap, waypoints_with_yaw, output_path, resolution=1.0)`
- **功能**：在全局costmap上用蓝线绘制全局规划路径
- **参数**：
  - `costmap`：全局costmap数组 (H, W)
  - `waypoints_with_yaw`：[(x, y, z, yaw), ...] 全局路径点（世界坐标）
  - `output_path`：输出图片路径
  - `resolution`：costmap分辨率
- **输出**：PNG图片，包含蓝线路径和蓝点路径点

#### c. `draw_actual_path_on_costmap(costmap, trajectory_points, output_path, resolution=1.0)`
- **功能**：在融合的global costmap上用红线绘制实际走过的路径
- **参数**：
  - `costmap`：融合后的全局costmap数组
  - `trajectory_points`：[(x, y, z), ...] 实际轨迹点（世界坐标）
  - `output_path`：输出图片路径
  - `resolution`：costmap分辨率
- **输出**：PNG图片，包含红线轨迹和红点

#### d. `draw_both_paths_on_costmap(costmap, waypoints_with_yaw, trajectory_points, output_path, resolution=1.0)`
- **功能**：在融合costmap上同时显示全局路径（蓝线）和实际路径（红线）
- **参数**：
  - `costmap`：融合后的全局costmap数组
  - `waypoints_with_yaw`：全局规划路径点
  - `trajectory_points`：实际走过的轨迹点
  - `output_path`：输出图片路径
  - `resolution`：costmap分辨率
- **输出**：PNG图片，包含蓝线规划路径、红线实际路径及图例

### 2. main.py 修改内容

#### a. 导入新函数
```python
from visualization import (
    draw_planned_path,
    draw_local_path,
    visualize_planning_results,
    load_costmap_txt,
    draw_global_path_on_costmap,        # 新增
    draw_actual_path_on_costmap,        # 新增
    draw_both_paths_on_costmap,         # 新增
)
```

#### b. 全局变量
```python
fused_global_costmap = None  # 新增，用于保存最后一次融合的全局costmap
```

#### c. 在main()函数中声明全局变量
```python
def main():
    global pointcloud_accumulator, local_dem_builder, local_path_planner, fused_global_costmap
```

#### d. 生成初始全局路径可视化
在 `load_global_data()` 后、创建AirSim客户端前添加：
```python
# 【新增】保存全局路径可视化
global_path_png = run_dir / "global_path.png"
draw_global_path_on_costmap(
    costmap=base_global_costmap,
    waypoints_with_yaw=waypoints_with_yaw,
    output_path=str(global_path_png),
    resolution=1.0,
)
```

#### e. 保存每次融合后的costmap
在 `save_global_merge_artifacts()` 调用处修改为：
```python
fused_global_costmap, _, _ = save_global_merge_artifacts(
    base_global_costmap, 
    local_obstacle_observations, 
    str(dem_build_dir)
)  # 【新增】保存融合后的costmap
```

#### f. 在程序完成时生成实际路径和对比图
在主循环 `break` 之后、完成信息前添加：
```python
# 【新增】在程序完成时保存实际路径和对比图
if fused_global_costmap is not None and len(trajectory_points) >= 2:
    actual_path_png = run_dir / "actual_path.png"
    compare_png = run_dir / "compare.png"
    
    draw_actual_path_on_costmap(
        costmap=fused_global_costmap,
        trajectory_points=trajectory_points,
        output_path=str(actual_path_png),
        resolution=1.0,
    )
    
    draw_both_paths_on_costmap(
        costmap=fused_global_costmap,
        waypoints_with_yaw=waypoints_with_yaw,
        trajectory_points=trajectory_points,
        output_path=str(compare_png),
        resolution=1.0,
    )
elif fused_global_costmap is None:
    print("警告：融合的全局costmap为空，无法保存实际路径可视化")
```

## 工作流程

### 运行时的执行流程

1. **程序启动**
   - 加载全局数据（全局costmap、路径点等）

2. **生成global_path.png** ⏱️ 立即生成
   - 在run_dir文件夹中保存全局规划路径
   - 基于初始的全局costmap
   - 显示全部全局路径点连线

3. **迭代规划和运动** 🔄 循环执行
   - 每次构建局部DEM
   - 进行局部路径规划
   - 沿局部路径运动
   - 收集trajectory_points
   - 融合局部costmap到全局costmap
   - 更新fused_global_costmap变量

4. **程序完成** ✅ 程序结束时
   - 生成actual_path.png
     - 显示实际走过的轨迹（红线）
     - 基于融合后的全局costmap
   - 生成compare.png
     - 同时显示规划路径（蓝线）和实际路径（红线）
     - 便于对比分析

## 坐标系统说明

### 世界坐标系 → Costmap栅格坐标转换
```
col = int(round(x_world / resolution))
row = int(round(y_world / resolution))
```

其中：
- `x_world, y_world`：AirSim世界坐标（米）
- `resolution`：costmap分辨率，默认1.0米/栅格
- `col, row`：costmap中的栅格坐标

### Costmap数组索引
- 行索引对应y坐标：`costmap[row, :]`
- 列索引对应x坐标：`costmap[:, col]`

## 颜色编码

| 元素 | 颜色 | RGB值 | 说明 |
|------|------|------|------|
| 规划路径线 | 蓝色 | (255, 0, 0) | 全局规划的路径 |
| 规划路径点 | 蓝色 | (255, 0, 0) | 全局规划的路径点 |
| 实际轨迹线 | 红色 | (0, 0, 255) | 实际走过的轨迹 |
| 实际轨迹点 | 红色 | (0, 0, 255) | 实际轨迹采样点 |
| 障碍区域 | 白色 | (255, 255, 255) | costmap中的障碍 |
| 自由空间 | 灰度 | (0-254) | costmap中的自由空间 |

## 输出文件位置

在 `local_planningpath/local_planningpath_YYYYMMDD_HHMMSS_ffffff/` 目录中：

```
local_planningpath_20260322_014945_840144/
├── global_path.png              # ✅ 全局规划路径
├── actual_path.png              # ✅ 实际走过路径
├── compare.png                  # ✅ 对比图
├── dem_build_20260322_001416_206992/
│   ├── dem_costmap.txt
│   ├── dem_costmap_vis.jpg
│   ├── global_dem_costmap_merge.txt
│   ├── global_dem_costmap_merge_vis.png
│   ├── dem.txt
│   ├── dem_meta.txt
│   ...
└── ...
```

## 使用示例

### 查看结果
程序完成后，在运行目录中找到这三张PNG图片：

```bash
# Windows
start local_planningpath/local_planningpath_*/global_path.png
start local_planningpath/local_planningpath_*/actual_path.png
start local_planningpath/local_planningpath_*/compare.png

# Linux
eog local_planningpath/local_planningpath_*/global_path.png &
eog local_planningpath/local_planningpath_*/actual_path.png &
eog local_planningpath/local_planningpath_*/compare.png &
```

## 常见问题

### 1. 为什么actual_path.png和compare.png没有生成？
- 检查是否有运动轨迹数据：`len(trajectory_points) >= 2`
- 检查fused_global_costmap是否为None
- 查看控制台是否有警告信息

### 2. 路径显示不完整或超出图片范围？
- 检查costmap的分辨率是否正确
- 验证坐标转换是否符合预期
- 考虑costmap的大小是否足够

### 3. 如何自定义颜色？
修改visualization.py中对应函数中的cv2.line()和cv2.circle()的颜色参数（BGR格式）：
```python
cv2.line(color_img, pt1, pt2, (B, G, R), thickness)  # BGR格式
```

## 注意事项

1. **分辨率一致性**：确保所有调用中的resolution参数一致（默认1.0）
2. **坐标范围**：生成的图片仅显示costmap范围内的路径
3. **数据量**：大量trajectory_points可能导致绘制较慢
4. **内存**：融合多次costmap后内存占用会增加

## 修改记录

| 日期 | 文件 | 修改内容 |
|------|------|---------|
| 2026-03-22 | visualization.py | 新增4个函数用于路径可视化 |
| 2026-03-22 | main.py | 导入新函数，添加全局变量，调用可视化函数 |
