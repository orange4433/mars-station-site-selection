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

def mars_sol2ls(sol):
    """
    将火星日(Sol)转换为太阳经度(Ls)
    使用精确的火星轨道参数计算
    """
    # 火星轨道参数
    year_day = 668.6  # 火星年长度（火星日）
    peri_day = 485.35  # 近日点对应的火星日
    e_elips = 0.09340  # 轨道偏心率
    timeperi = 1.90258341759902  # 近日点经度（弧度）
    
    # 计算平近点角
    M = 2 * np.pi * (sol - peri_day) / year_day
    M = M % (2 * np.pi)
    
    # 开普勒方程求解（牛顿迭代法）
    E = M
    for _ in range(10):
        delta_E = (E - e_elips * np.sin(E) - M) / (1 - e_elips * np.cos(E))
        E -= delta_E
        if abs(delta_E) < 1e-7:
            break
    
    # 计算真近点角
    true_anomaly = 2 * np.arctan(np.sqrt((1 + e_elips) / (1 - e_elips)) * np.tan(E / 2))
    
    # 计算太阳经度Ls
    Ls = (true_anomaly - timeperi) % (2 * np.pi)
    return np.degrees(Ls)

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
    
    # 3. 选择特定位置或计算全球平均
    if lat is not None and lon is not None:
        # 选择特定位置
        data = wind_speed.interp(latitude=lat, longitude=lon, method='linear')
    else:
        # 计算全球平均（纬度加权）
        weights = np.cos(np.deg2rad(ds.latitude))
        data = wind_speed.weighted(weights).mean(dim=('longitude', 'latitude'))
    
    # 4. 准备绘图数据
    # 维度: (Ls, ut)
    plot_data = data.transpose('Ls', 'ut').values
    plot_data = adjust_data_for_local_time(plot_data)
    
    # 5. 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置颜色映射
    norm = colors.Normalize(vmin=min_value, vmax=max_value, clip=True)
    cmap = plt.get_cmap('plasma')
    
    # 创建热力图 - 反转y轴方向
    im = ax.imshow(plot_data, norm=norm, cmap=cmap, aspect='auto', 
                   extent=[0, 24, 360, 0], origin='upper')
    
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
    
    # 设置y轴范围（反转方向）
    ax.set_ylim(360, 0)
    
    # 设置Ls刻度为0,45,90,...,360
    ax.set_yticks(np.arange(0, 361, 45))
    
    # 8. 添加Sol坐标轴（右侧）
    # 创建右侧坐标轴
    ax2 = ax.twinx()
    
    # 定义Sol范围（114到794）
    sol_min = 114
    sol_max = 794
    
    # 设置Sol坐标轴范围
    ax2.set_ylim(sol_min, sol_max)
    
    # 计算Ls和Sol的对应关系
    ls_values = np.arange(0, 361, 45)
    sol_values = []
    
    # 通过迭代找到每个Ls对应的Sol
    for ls in ls_values:
        # 使用二分法查找对应的Sol
        low, high = sol_min, sol_max
        for _ in range(20):  # 最多迭代20次
            mid = (low + high) / 2
            mid_ls = mars_sol2ls(mid)
            if abs(mid_ls - ls) < 0.1:
                sol_values.append(mid)
                break
            elif mid_ls < ls:
                low = mid
            else:
                high = mid
        else:
            sol_values.append((low + high) / 2)  # 如果没找到精确值，取中间值
    
    # 设置右侧坐标轴刻度
    ax2.set_yticks(sol_values)
    ax2.set_yticklabels([f"{int(round(s))}" for s in sol_values])
    ax2.set_ylabel('Sol', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    
    # 9. 添加标题
    if lat is not None and lon is not None:
        title = f'Wind Speed at Latitude: {lat}°, Longitude: {lon}°'
    else:
        title = 'Global Average Wind Speed'
    
    ax.set_title(title, fontdict={'family': 'Times New Roman', 'fontsize': 16})
    
    # 10. 优化布局
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