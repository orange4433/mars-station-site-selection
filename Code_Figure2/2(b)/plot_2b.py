import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
scale = 18
scale_value = 300
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'  # 设置全局字体


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ... 其他代码保持不变 ...

# 在绘图部分替换以下内容：

# 自定义经度标签格式化函数
def lon_formatter(x, pos):
    if x < 0:
        return f"{-x:.0f}°W"
    elif x > 0:
        return f"{x:.0f}°E"
    else:
        return "0"

# 自定义纬度标签格式化函数
def lat_formatter(y, pos):
    if y > 0:
        return f"{y:.0f}°N"
    elif y < 0:
        return f"{-y:.0f}°S"
    else:
        return "0"

def calculate_insolation_point(Ls, local_hour, longitude):
    """
    计算火星上给定点的地方时对应的直射点经纬度。
    
    参数:
        Ls (float): 火星的太阳黄经 (Solar Longitude)，单位为度。
        local_hour (int): 地方时，24小时制。
        longitude (float): 经度，东经为正，单位为度。
        
    返回:
        tuple: 直射点的纬度和经度，单位为度。
    """
    # 转换地方时到太阳时角 (Hour Angle)
    hour_angle = (local_hour - 12) * 15  # 每小时对应15度
    
    # 火星的自转轴倾角 (Obliquity of the ecliptic for Mars)
    epsilon = math.radians(25.19)  # 约25.19度
    
    # 太阳赤纬 (Solar Declination)
    delta = math.asin(math.sin(math.radians(Ls)) * math.sin(epsilon))
    
    # 直射点纬度 (Latitude of the subsolar point)
    latitude = math.degrees(delta)
    
    # 计算太阳时角对直射点经度的影响
    # 注意：这里的计算忽略了火星自转速度与地球的不同，以及轨道离心率等复杂因素
    sub_solar_longitude = longitude - hour_angle
    
    return latitude, sub_solar_longitude

def plot_wind_field_mcd(file_path, ls_value, local_hour, surface_file='surface.nc'):
    # 打开NetCDF文件
    with nc.Dataset(file_path, 'r') as dataset:
        lons = dataset.variables['longitude'][:]  # (65,)
        lats = dataset.variables['latitude'][:]   # (49,)
        ut_array = dataset.variables['ut'][:]     # (48,) 世界时
        ls_array = dataset.variables['Ls'][:]      # (721,) 太阳经度
        # 找到最接近的Ls索引
        ls_idx = np.argmin(np.abs(ls_array - ls_value))
        u_data = dataset.variables['U'][:, :, :, ls_idx]
        v_data = dataset.variables['V'][:, :, :, ls_idx]
        print(ls_idx)
        # 计算每个经度对应的世界时
        ut_target = (local_hour +9 - lons/15) % 24
        
        # 初始化插值后的风场数据
        u_interp = np.zeros((len(lons), len(lats)))
        v_interp = np.zeros((len(lons), len(lats)))
        
        # 对每个经度点进行插值
        for i, lon in enumerate(lons):
            # 当前经度对应的目标世界时
            current_ut = ut_target[i]
            
            # 扩展UT数组以处理周期边界
            ut_extended = np.append(ut_array, [24])
            
            # 修正维度问题：确保数组形状一致
            u_slice = u_data[i, :, :]  # (49, 48)
            v_slice = v_data[i, :, :]  # (49, 48)
            
            # 添加最后一个时间步的数据（即第一个时间步的数据）
            u_extended_data = np.concatenate([u_slice, u_slice[:, 0:1]], axis=1)
            v_extended_data = np.concatenate([v_slice, v_slice[:, 0:1]], axis=1)
            
            # 创建插值函数
            f_u = interp1d(ut_extended, u_extended_data, axis=1, kind='linear', 
                          bounds_error=False, fill_value="extrapolate")
            f_v = interp1d(ut_extended, v_extended_data, axis=1, kind='linear', 
                          bounds_error=False, fill_value="extrapolate")
            
            # 执行插值
            u_interp[i, :] = f_u(current_ut)
            v_interp[i, :] = f_v(current_ut)
        
        # 创建网格
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
        
        # 计算风速大小
        wind_speed = np.sqrt(u_interp**2 + v_interp**2)
        
        # ==============================================
        # 新增：输出北纬3.75°、东经135°位置的风速信息
        # ==============================================
        target_lat = 3.75  # 北纬3.75°
        target_lon = 135.0  # 东经135°
        
        # 找到最接近的经度和纬度索引
        lon_idx = np.argmin(np.abs(lons - target_lon))
        lat_idx = np.argmin(np.abs(lats - target_lat))
        
        # 获取该位置的风速信息
        u_value = u_interp[lon_idx, lat_idx]
        v_value = v_interp[lon_idx, lat_idx]
        wind_speed_value = wind_speed[lon_idx, lat_idx]
        
        # 计算风向（角度，0°为正北，顺时针增加）
        wind_direction = np.degrees(np.arctan2(-u_value, -v_value)) % 360
        
        # 输出结果
        print("\n" + "="*60)
        print(f"Wind information at {target_lat}°N, {target_lon}°E:")
        print(f"U component (east-west): {u_value:.2f} m/s")
        print(f"V component (north-south): {v_value:.2f} m/s")
        print(f"Horizontal wind speed: {wind_speed_value:.2f} m/s")
        print(f"Wind direction: {wind_direction:.1f}° (from north)")
        print("="*60 + "\n")
        
        
    # 绘制风场图
    plt.figure(figsize=(12, 6))
    
    # 填充背景颜色
    contour = plt.contourf(lon_grid, lat_grid, wind_speed, cmap='plasma', 
                          levels=np.linspace(0, 16, 17), extend='both', 
                          vmin=0, vmax=16)
    cbar = plt.colorbar(contour)
    cbar.set_label('Horizontal Wind Speed (m/s)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # 绘制风矢量图（每2个点取一个）
    stride = 2
    quiver = plt.quiver(lon_grid[::stride, ::stride], lat_grid[::stride, ::stride],
                       u_interp[::stride, ::stride], v_interp[::stride, ::stride],
                       color='cyan', scale=scale_value)
    
    # 添加矢量比例说明
    scale_str = f"Scale: {quiver.scale:.2f}"
    
    # 添加地形等高线
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = surface_dataset.variables['longitude'][:]
        elevation = surface_dataset.variables['zMOL'][:]
        
        # 调整经度到-180~180范围
        lons_surface = np.where(lons_surface > 180, lons_surface - 360, lons_surface)
        lon_surface, lat_surface = np.meshgrid(lons_surface, lats_surface)
        
        # 绘制等高线
        CS = plt.contour(lon_surface, lat_surface, elevation, 
                        colors=[(0, 0, 0, 0.5)], levels=np.arange(-10, 8.1, 2))
        plt.clabel(CS, inline=True, fontsize=5)
    
    # 标记InSight位置 - 使用白色背景红色文字
    plt.annotate(
        'InSight',
        xy=(135.6, 4.5),
        xytext=(135.6 - 30, 4.5 + 15),  # 调整文本位置
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    insight_scatter = plt.scatter(135.6, 4.5, color='red', s=100, 
                             edgecolors='white', linewidth=2, label='InSight Location')
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
      # 例如 16
    #plt.title(f'Global Wind Field at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # 保存图片
    filename = Path(__file__).resolve().parent / f'mcd_wind_field_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Wind field saved to {filename}")

# 示例调用
if __name__ == '__main__':
    figure2_dir = Path(__file__).resolve().parents[1]
    plot_wind_field_mcd(
        figure2_dir.parent / 'Data_MCD' / 'mcd_data.nc',
        ls_value=90,
        local_hour=12,
        surface_file=figure2_dir / 'surface.nc',
    )
