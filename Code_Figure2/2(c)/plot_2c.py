import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
scale = 16
scale_value = 400
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
    # ... 保留原有实现 ...
    pass

def adjust_longitude(lons):
    """Adjust longitude values from [-180, 180] to [0, 360]."""
    lons = lons.copy()
    lons[lons < 0] += 360
    return lons

def save_wind_speed_to_txt(wind_speed, lon_subset, lat_subset, filename='wind_speed.txt'):
    """
    Save the wind speed data to a text file.

    Parameters:
    - wind_speed: 2D numpy array of wind speed values.
    - lon_subset: 2D numpy array of longitudes corresponding to wind_speed.
    - lat_subset: 2D numpy array of latitudes corresponding to wind_speed.
    - filename: String, name of the output text file.
    """
    with open(filename, 'w') as f:
        # Write the actual data
        for i in range(len(lat_subset)):
            line = " ".join(f"{ws:.4f}" for ws in wind_speed[i, :])
            f.write(line + "\n")

def plot_local_wind_field_mcd(file_path, ls_value, local_hour, 
                            lat_min=-35, lat_max=45, lon_min=80, lon_max=200,
                            surface_file='surface.nc'):
    """
    绘制局部风场图，严格保持80°E到200°E的范围
    
    参数:
        file_path (str): NetCDF文件路径
        ls_value (float): 太阳经度 (Ls)
        local_hour (int): 地方时
        lat_min (float): 最小纬度
        lat_max (float): 最大纬度
        lon_min (float): 最小经度
        lon_max (float): 最大经度
        surface_file (str): 地形文件路径
    """
    # 扩大数据读取范围，确保不留白
    lat_min_extended = lat_min - 5
    lat_max_extended = lat_max + 5
    lon_min_extended = lon_min - 10
    lon_max_extended = lon_max + 10
    
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
        
        # 计算每个经度对应的世界时
        ut_target = (local_hour + 9 - lons/15) % 24
        
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
            u_extended_data =np.concatenate([u_slice, u_slice[:, 0:1]], axis=1)
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
        
        # 定义子集索引 - 使用扩展的范围
        lat_indices = np.where((lats >= lat_min_extended) & (lats <= lat_max_extended))[0]
        
        # 处理经度环绕问题（使用扩展的范围）
        # 第一部分：扩展的最小经度到180°E
        lon_indices1 = np.where((lons >= lon_min_extended) & (lons <= 180))[0]
        # 第二部分：-180°W到扩展的最大经度（转换为东经）
        lon_indices2 = np.where((lons >= -180) & (lons <= (lon_max_extended - 360)))[0]
        
        # 合并经度索引
        lon_indices = np.concatenate([lon_indices1, lon_indices2])
        
        # 裁剪数据到扩展区域
        u_subset = u_interp[lon_indices, :][:, lat_indices]
        v_subset = v_interp[lon_indices, :][:, lat_indices]
        wind_speed_subset = wind_speed[lon_indices, :][:, lat_indices]
        
        # 调整经度值，使200°E正确显示
        lon_subset = lon_grid[lon_indices, :][:, lat_indices]
        lon_subset[lon_subset < 0] += 360  # 将负经度转换为正东经
        
        lat_subset = lat_grid[lon_indices, :][:, lat_indices]
    
    # 绘制局部风场图
    plt.figure(figsize=(10, 5))
    
    # 填充背景颜色 - 使用扩展的数据范围
    contour = plt.contourf(lon_subset, lat_subset, wind_speed_subset, cmap='plasma', 
                          levels=np.linspace(0, 16, 17), extend='both', 
                          vmin=0, vmax=16)
    cbar = plt.colorbar(contour)
    cbar.set_label('Horizontal Wind Speed (m/s)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # 绘制风矢量图 - 缩短箭头（增大scale值）
    stride = 1
    quiver = plt.quiver(lon_subset[::stride, ::stride], lat_subset[::stride, ::stride],
                       u_subset[::stride, ::stride], v_subset[::stride, ::stride],
                       color='cyan', scale=scale_value)  # 增大scale值以缩短箭头
    
    # 打开地形文件并读取海拔数据
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = adjust_longitude(surface_dataset.variables['longitude'][:])
        elevation = surface_dataset.variables['zMOL'][:]
        
        # 分割地形数据为两部分
        mask_lon_part1 = (lons_surface >= lon_min) & (lons_surface <= 180)
        mask_lon_part2 = (lons_surface > 180) & (lons_surface <= lon_max)

        # 对于每个部分，计算其对应的纬度和经度索引
        lat_indices = np.where((lats_surface >= lat_min) & (lats_surface <= lat_max))[0]
        lon_indices_part1 = np.where(mask_lon_part1)[0]
        lon_indices_part2 = np.where(mask_lon_part2)[0]

        # 创建独立的网格点用于海拔数据
        lon_surface_part1, lat_surface_part1 = np.meshgrid(lons_surface[lon_indices_part1], lats_surface[lat_indices])
        lon_surface_part2, lat_surface_part2 = np.meshgrid(lons_surface[lon_indices_part2], lats_surface[lat_indices])

        # 裁剪海拔数据至指定区域
        elevation_subset_part1 = elevation[lat_indices, :][:, lon_indices_part1]
        elevation_subset_part2 = elevation[lat_indices, :][:, lon_indices_part2] if len(lon_indices_part2) > 0 else np.array([])

        # 合并两部分海拔数据（如果有的话）
        if len(elevation_subset_part2) > 0:
            lons_combined_elev = np.concatenate([lons_surface[lon_indices_part1], lons_surface[lon_indices_part2]])
            elevation_data_combined = np.hstack([elevation_subset_part1, elevation_subset_part2])
        else:
            lons_combined_elev = lons_surface[lon_indices_part1]
            elevation_data_combined = elevation_subset_part1

        # 保存合并后的海拔数据到txt文件
        elevation_filename = f'elevation_ls_{ls_value}_local_hour_{local_hour}.txt'
        save_wind_speed_to_txt(elevation_data_combined, lon_surface_part1, lat_surface_part1, filename=elevation_filename)

    # 从文本文件中读取合并后的海拔数据并绘制等高线图
    elevation_data_combined = np.loadtxt(elevation_filename)

    # 因为纬度是倒序的，我们也需要反转海拔数据矩阵的纬度方向
    elevation_data_combined = elevation_data_combined[:, :]

    # 绘制等高线
    CS = plt.contour(lons_combined_elev, lats_surface[lat_indices], elevation_data_combined, colors=[(0, 0, 0, 0.5)], levels=np.arange(-10, 8.1, 2))
    plt.clabel(CS, inline=True, fontsize=8)  # 添加等高线标签 (可选)
    
    # 标记InSight位置 - 保留红点
    insight_lon, insight_lat = 135.6, 4.5
    plt.scatter(135.6, 4.5, color='red', s=100, 
                             edgecolors='white', linewidth=2, label='InSight Location')
    
    # 为InSight位置添加白色背景红色文字的标签
    plt.annotate(
        'InSight',
        xy=(insight_lon, insight_lat),
        xytext=(insight_lon - 30, insight_lat + 10),  # 调整文本位置
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    
    #plt.title(f'Local Wind Field at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # 设置坐标轴范围 - 确保显示我们想要的范围，而不是扩展的范围
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])
    
    # 保存图片
    filename = Path(__file__).resolve().parent / f'mcd_local_wind_field_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Local wind field saved to {filename}")

# 示例调用
if __name__ == '__main__':
    figure2_dir = Path(__file__).resolve().parents[1]
    plot_local_wind_field_mcd(
        figure2_dir.parent / 'Data_MCD' / 'mcd_data.nc',
        ls_value=90,
        local_hour=12,
        lat_min=-35,
        lat_max=45,
        lon_min=80,
        lon_max=200,
        surface_file=figure2_dir / 'surface.nc',
    )
