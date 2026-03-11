#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIFF影像降采样程序
功能：将TIFF影像分辨率降低2倍（如10000x10000 -> 5000x5000）
支持：多波段影像、保持地理坐标信息、多种重采样算法
"""

import os
import sys
from osgeo import gdal
from osgeo import gdalconst
import numpy as np
from tqdm import tqdm


def downsample_tiff(input_path, output_path, scale_factor=2, resample_method='bilinear'):
    """
    对TIFF影像进行降采样
    
    参数:
        input_path: 输入TIFF文件路径
        output_path: 输出TIFF文件路径
        scale_factor: 降采样倍数，默认为2（即分辨率减半）
        resample_method: 重采样方法
            - 'nearest': 最近邻（适合分类图）
            - 'bilinear': 双线性（适合连续数据，默认）
            - 'cubic': 三次卷积（适合连续数据，更平滑）
            - 'average': 平均值（适合降低噪声）
            - 'mode': 众数（适合分类图）
    """
    
    # 打开输入影像
    src_ds = gdal.Open(input_path, gdalconst.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(f"无法打开输入文件: {input_path}")
    
    # 获取影像基本信息
    src_width = src_ds.RasterXSize
    src_height = src_ds.RasterYSize
    num_bands = src_ds.RasterCount
    data_type = src_ds.GetRasterBand(1).DataType
    projection = src_ds.GetProjection()
    geotransform = src_ds.GetGeoTransform()
    
    print(f"输入影像信息:")
    print(f"  尺寸: {src_width} x {src_height}")
    print(f"  波段数: {num_bands}")
    print(f"  数据类型: {gdal.GetDataTypeName(data_type)}")
    
    # 计算输出尺寸
    dst_width = src_width // scale_factor
    dst_height = src_height // scale_factor
    
    print(f"\n输出影像信息:")
    print(f"  尺寸: {dst_width} x {dst_height}")
    print(f"  缩放比例: 1/{scale_factor}")
    
    # 更新地理变换参数
    # geotransform: (左上角X, 像素宽, 旋转, 左上角Y, 旋转, 像素高)
    dst_geotransform = list(geotransform)
    dst_geotransform[1] = geotransform[1] * scale_factor  # 像素宽度增大
    dst_geotransform[5] = geotransform[5] * scale_factor  # 像素高度增大（通常为负值）
    
    # 选择重采样算法
    method_dict = {
        'nearest': gdalconst.GRA_NearestNeighbour,
        'bilinear': gdalconst.GRA_Bilinear,
        'cubic': gdalconst.GRA_Cubic,
        'cubicspline': gdalconst.GRA_CubicSpline,
        'lanczos': gdalconst.GRA_Lanczos,
        'average': gdalconst.GRA_Average,
        'mode': gdalconst.GRA_Mode,
        'max': gdalconst.GRA_Max,
        'min': gdalconst.GRA_Min,
        'med': gdalconst.GRA_Med,
        'q1': gdalconst.GRA_Q1,
        'q3': gdalconst.GRA_Q3
    }
    
    gdal_method = method_dict.get(resample_method.lower(), gdalconst.GRA_Bilinear)
    print(f"  重采样方法: {resample_method}")
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    
    # 设置创建选项（优化大文件处理）
    create_options = [
        'BIGTIFF=YES',           # 支持大于4GB的文件
        'COMPRESS=LZW',          # 使用LZW压缩（无损，较好压缩比）
        'TILED=YES',             # 分块存储，提高读取效率
        'BLOCKXSIZE=512',        # 块宽度
        'BLOCKYSIZE=512',        # 块高度
        'INTERLEAVE=BAND'        # 波段交错存储
    ]
    
    dst_ds = driver.Create(output_path, dst_width, dst_height, num_bands, 
                          data_type, options=create_options)
    
    if dst_ds is None:
        raise RuntimeError(f"无法创建输出文件: {output_path}")
    
    # 设置地理参考信息
    dst_ds.SetProjection(projection)
    dst_ds.SetGeoTransform(dst_geotransform)
    
    # 复制波段元数据（NoData值等）
    for band_idx in range(1, num_bands + 1):
        src_band = src_ds.GetRasterBand(band_idx)
        dst_band = dst_ds.GetRasterBand(band_idx)
        
        # 复制NoData值
        nodata_val = src_band.GetNoDataValue()
        if nodata_val is not None:
            dst_band.SetNoDataValue(nodata_val)
        
        # 复制波段描述
        description = src_band.GetDescription()
        if description:
            dst_band.SetDescription(description)
    
    # 执行重采样
    print(f"\n开始降采样处理...")
    
    # 使用VRT进行高效重采样（内存优化）
    # 或者使用GDAL的ReprojectImage
    gdal.ReprojectImage(
        src_ds, dst_ds,
        projection, projection,  # 相同投影
        gdal_method,
        0, 0,  # 使用默认内存限制
        0,  # 不指定源窗口
        ['SKIP_NOSOURCE=YES']  # 跳过无源数据的区域
    )
    
    # 复制影像元数据
    metadata = src_ds.GetMetadata()
    if metadata:
        dst_ds.SetMetadata(metadata)
    
    # 清理
    src_ds = None
    dst_ds = None
    
    print(f"\n降采样完成！")
    print(f"输出文件: {output_path}")
    
    # 验证输出文件
    verify_ds = gdal.Open(output_path)
    if verify_ds:
        print(f"验证输出:")
        print(f"  实际尺寸: {verify_ds.RasterXSize} x {verify_ds.RasterYSize}")
        print(f"  文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        verify_ds = None


def downsample_tiff_manual(input_path, output_path, scale_factor=2, block_size=4096):
    """
    手动分块降采样（适用于超大影像，内存优化版本）
    使用简单的2x2平均池化
    """
    
    src_ds = gdal.Open(input_path, gdalconst.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(f"无法打开输入文件: {input_path}")
    
    src_width = src_ds.RasterXSize
    src_height = src_ds.RasterYSize
    num_bands = src_ds.RasterCount
    data_type = src_ds.GetRasterBand(1).DataType
    projection = src_ds.GetProjection()
    geotransform = src_ds.GetGeoTransform()
    
    # 确保尺寸是scale_factor的整数倍
    new_width = (src_width // scale_factor) * scale_factor
    new_height = (src_height // scale_factor) * scale_factor
    dst_width = new_width // scale_factor
    dst_height = new_height // scale_factor
    
    print(f"处理区域: {new_width} x {new_height} -> {dst_width} x {dst_height}")
    
    # 更新地理变换
    dst_geotransform = list(geotransform)
    dst_geotransform[1] = geotransform[1] * scale_factor
    dst_geotransform[5] = geotransform[5] * scale_factor
    
    # 创建输出
    driver = gdal.GetDriverByName('GTiff')
    create_options = [
        'BIGTIFF=YES',
        'COMPRESS=LZW',
        'TILED=YES',
        'BLOCKXSIZE=512',
        'BLOCKYSIZE=512'
    ]
    
    dst_ds = driver.Create(output_path, dst_width, dst_height, num_bands, 
                          data_type, options=create_options)
    dst_ds.SetProjection(projection)
    dst_ds.SetGeoTransform(dst_geotransform)
    
    # 分块处理
    print("开始分块处理...")
    for band_idx in range(1, num_bands + 1):
        src_band = src_ds.GetRasterBand(band_idx)
        dst_band = dst_ds.GetRasterBand(band_idx)
        nodata = src_band.GetNoDataValue()
        
        # 计算块数
        y_blocks = (new_height + block_size - 1) // block_size
        
        for y_block in tqdm(range(y_blocks), desc=f"波段 {band_idx}/{num_bands}"):
            y_start = y_block * block_size
            y_end = min(y_start + block_size, new_height)
            
            # 读取源数据（确保读取scale_factor的整数倍行）
            rows_to_read = ((y_end - y_start + scale_factor - 1) // scale_factor) * scale_factor
            y_end_aligned = y_start + rows_to_read
            
            data = src_band.ReadAsArray(0, y_start, new_width, rows_to_read)
            
            if data is None:
                continue
            
            # 执行平均池化降采样
            # 重塑数组以便进行平均
            h, w = data.shape
            h_new = h // scale_factor
            w_new = w // scale_factor
            
            # 使用reshape和mean进行高效降采样
            data_reshaped = data.reshape(h_new, scale_factor, w_new, scale_factor)
            
            # 处理NoData值
            if nodata is not None:
                mask = (data_reshaped == nodata)
                data_reshaped = np.where(mask, np.nan, data_reshaped)
                data_down = np.nanmean(data_reshaped, axis=(1, 3))
                data_down = np.where(np.isnan(data_down), nodata, data_down)
            else:
                data_down = data_reshaped.mean(axis=(1, 3))
            
            # 写入输出
            dst_band.WriteArray(data_down.astype(data.dtype), 0, y_start // scale_factor)
    
    src_ds = None
    dst_ds = None
    print(f"\n手动降采样完成: {output_path}")


def batch_downsample(input_folder, output_folder, scale_factor=2, suffix="_down2x"):
    """
    批量处理文件夹中的所有TIFF影像
    """
    os.makedirs(output_folder, exist_ok=True)
    
    tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    files = [f for f in os.listdir(input_folder) 
             if any(f.endswith(ext) for ext in tiff_extensions)]
    
    print(f"发现 {len(files)} 个TIFF文件")
    
    for i, filename in enumerate(files, 1):
        print(f"\n处理 [{i}/{len(files)}]: {filename}")
        input_path = os.path.join(input_folder, filename)
        
        # 生成输出文件名
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}{suffix}{ext}"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            downsample_tiff(input_path, output_path, scale_factor)
        except Exception as e:
            print(f"处理失败: {e}")
            continue


if __name__ == "__main__":
    # 使用示例
    
    # 单文件处理示例
    input_file = "data\CE7DEM_with_craters.tif"  # 修改为你的输入文件路径
    output_file = "data\CE7DEM_with_craters_downsample.tif"
    
    # 方法1: 使用GDAL内置重采样（推荐，支持多种算法）
    downsample_tiff(input_file, output_file, scale_factor=2, resample_method='bilinear')
    
    # 方法2: 手动分块处理（适合超大文件，内存受限时）
    # downsample_tiff_manual(input_file, output_file, scale_factor=2)
    
    # 批量处理示例
    # batch_downsample("input_folder", "output_folder", scale_factor=2)
