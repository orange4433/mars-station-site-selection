import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import xarray as xr
import os
import math

def adjust_data_for_local_time(u_data, shift_hours=9):
    """
    对每个 Ls 时间点，将 ut 轴（0~24小时）向前移动 shift_hours 小时（地方时校正）
    shift_hours=9 表示 UTC+9（东经135度）
    u_data: shape (n_Ls, n_ut) = (721, 48)
    """
    shift_points = shift_hours * 2  # 每半小时一个点
    # 沿着 ut 维度（axis=1）向左滚动 shift_points 个点
    return np.roll(u_data, shift_points, axis=1)

def plot_wind_speed_seasonal_variation(nc_file_path, lat=None, lon=None, min_value=0, max_value=16):
    """
    绘制风速大小随时间和季节变化的热力图
    
    参数:
    nc_file_path: NetCDF文件路径
    lat: 特定纬度值 (None表示全球平均)
    lon: 特定经度值 (None表示全球平均)
    min_value: 颜色条最小值
    max_value: 颜色条最大值
    """
    # 1. 读取NetCDF文件
    ds = xr.open_dataset(nc_file_path)
    
    # 2. 计算风速大小 √(U² + V²)
    wind_speed = np.sqrt(ds.U**2 + ds.V**2)
    
    '''
    # 3. 选择特定位置或计算全球平均
    if lat is not None and lon is not None:
        # 选择特定位置
        data = wind_speed.interp(latitude=lat, longitude=lon, method='linear')
    else:
        # 计算全球平均（纬度加权）
        weights = np.cos(np.deg2rad(ds.latitude))
        data = wind_speed.weighted(weights).mean(dim=('longitude', 'latitude'))
    '''
    u_data = ds.U.sel(latitude=3.75, longitude=135, method='nearest')
    v_data = ds.V.sel(latitude=3.75, longitude=135, method='nearest')
    data = wind_speed.sel(latitude=3.75, longitude=135, method='nearest')
    # 4. 准备绘图数据
    # 维度: (Ls, ut)
    plot_data = data.transpose('Ls', 'ut').values
    u_data = u_data.transpose('Ls', 'ut').values
    v_data = v_data.transpose('Ls', 'ut').values
    #plot_data = adjust_data_for_local_time(plot_data)
    #u_data =adjust_data_for_local_time(u_data)
    #v_data = adjust_data_for_local_time(v_data)
    print(plot_data[180][22])
    print(u_data[180][22])
    print(v_data[180][22])
    # 5. 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置颜色映射
    norm = colors.Normalize(vmin=min_value, vmax=max_value, clip=True)
    cmap = plt.get_cmap('plasma')
    
    # 创建热力图
    im = ax.imshow(plot_data, norm=norm, cmap=cmap, aspect='auto', 
                   extent=[0, 24, ds.Ls.min(), ds.Ls.max()], origin='lower')
    
    # 6. 添加颜色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Wind Speed (m/s)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    
    # 7. 设置坐标轴
    # X轴 - 本地时间 (LMST)
    ax.set_xlabel('Local Mean Solar Time (LMST)', 
                  fontdict={'family': 'Times New Roman', 'fontsize': 14})
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f'{int(h)}:00' for h in np.arange(0, 25, 3)])
    
    # Y轴 - 太阳经度 (Ls)
    ax.set_ylabel('Solar Longitude (Ls)', 
                  fontdict={'family': 'Times New Roman', 'fontsize': 14})
    
    # 添加季节标记
    seasons = {
        0: 'Vernal Equinox',
        90: 'Summer Solstice',
        180: 'Autumnal Equinox',
        270: 'Winter Solstice'
    }
    
    for ls, season in seasons.items():
        if ls <= ds.Ls.max():
            ax.axhline(y=ls, color='gray', linestyle='--', alpha=0.5)
            ax.text(24.5, ls, season, ha='left', va='center', fontsize=10)
    
    # 8. 添加标题
    if lat is not None and lon is not None:
        title = f'Wind Speed at Latitude: {lat}°, Longitude: {lon}°'
    else:
        title = 'Global Average Wind Speed'
    
    ax.set_title(title, fontdict={'family': 'Times New Roman', 'fontsize': 16})
    
    # 9. 优化布局
    plt.tight_layout()
    plt.show()
    
    return plot_data

# 使用示例
if __name__ == "__main__":
    # 指定NetCDF文件路径
    nc_file = "./surface_winds_ds_1_LS_0.00-360.00_rev1.1.nc"
    
    
    # 绘制特定位置的风速（例如：赤道0°, 0°）
    print("\nPlotting wind speed at specific location (0°, 0°)...")
    plot_wind_speed_seasonal_variation(nc_file, lat=4.5, lon=135.6)
    