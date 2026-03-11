"""
坐标转换模块（局部路径规划专用）
包含世界坐标、局部坐标、DEM栅格坐标之间的转换
"""

import math


# ========================================
# 世界坐标与局部坐标转换
# ========================================
def world_to_local(x_world, y_world, x_origin, y_origin, yaw_origin):
    """
    将世界坐标转换为以 (x_origin, y_origin, yaw_origin) 为原点的局部坐标系
    局部坐标系定义：x向前，y向右
    """
    dx = x_world - x_origin
    dy = y_world - y_origin
    
    cos_yaw = math.cos(yaw_origin)
    sin_yaw = math.sin(yaw_origin)
    
    x_local = dx * cos_yaw + dy * sin_yaw
    y_local = -dx * sin_yaw + dy * cos_yaw
    
    return x_local, y_local


def local_to_world(x_local, y_local, x_origin, y_origin, yaw_origin):
    """将局部坐标转换回世界坐标"""
    cos_yaw = math.cos(yaw_origin)
    sin_yaw = math.sin(yaw_origin)
    
    x_world = x_origin + x_local * cos_yaw - y_local * sin_yaw
    y_world = y_origin + x_local * sin_yaw + y_local * cos_yaw
    
    return x_world, y_world


# ========================================
# 局部坐标与DEM栅格坐标转换
# ========================================
def local_to_dem_grid(x_local, y_local, dem_range, resolution):
    """
    将局部坐标转换为 DEM 栅格坐标 (col, row)
    row: 前 -> 后，对应 x_local 从 +range -> -range
    col: 左 -> 右，对应 y_local 从 -range -> +range
    """
    row_float = (dem_range - x_local) / resolution
    col_float = (y_local + dem_range) / resolution
    
    col = int(round(col_float))
    row = int(round(row_float))
    
    return col, row


def dem_grid_to_local(col, row, dem_range, resolution):
    """将 DEM 栅格坐标转换回局部坐标"""
    y_local = col * resolution - dem_range
    x_local = dem_range - row * resolution
    
    return x_local, y_local