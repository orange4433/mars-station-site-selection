import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# 设置图形参数
scale = 16
plt.rcParams['font.family'] = 'Arial'  # 设置全局字体

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
    """
    hour_angle = (local_hour - 12) * 15
    epsilon = math.radians(25.19)
    delta = math.asin(math.sin(math.radians(Ls)) * math.sin(epsilon))
    latitude = math.degrees(delta)
    sub_solar_longitude = longitude - hour_angle
    
    return latitude, sub_solar_longitude

def adjust_longitude(lons):
    """将经度从[-180, 180]调整到[0, 360]"""
    lons = lons.copy()
    lons[lons < 0] += 360
    return lons

def save_data_to_txt(data, filename='data.txt'):
    """将数据保存为文本文件"""
    with open(filename, 'w') as f:
        for i in range(data.shape[0]):
            line = " ".join(f"{val:.4f}" for val in data[i, :])
            f.write(line + "\n")

def plot_local_rho_v2_field_mcd(file_path, ls_value, local_hour, 
                               lat_min=-35, lat_max=45, lon_min=80, lon_max=200,
                               surface_file='surface.nc', vmax=1.8):
    """
    绘制局部动压分布图
    
    参数:
        file_path: 动压数据文件路径
        ls_value: 太阳经度 (Ls)
        local_hour: 火星本地时
        lat_min, lat_max: 纬度范围
        lon_min, lon_max: 经度范围
        surface_file: 地形文件路径
        vmax: 动压最大值，用于颜色条范围
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
        # Read only the requested season; loading all 721 Ls slices wastes ~0.9 GB.
        rho_v2_data = 0.5 * dataset.variables['rho_v2'][:, :, :, ls_idx]
        print(f"Ls value: {ls_array[ls_idx]:.2f}")
        
        # 计算每个经度对应的世界时
        ut_target = (local_hour + 9 - lons/15) % 24
        
        # 初始化插值后的动压数据
        rho_v2_interp = np.zeros((len(lons), len(lats)))
        
        # 对每个经度点进行时间插值
        for i, lon in enumerate(lons):
            # 当前经度对应的目标世界时
            current_ut = ut_target[i]
            
            # 扩展UT数组以处理周期边界
            ut_extended = np.append(ut_array, [24])
            
            # 获取当前经度的动压切片
            rho_v2_slice = rho_v2_data[i, :, :]  # shape: (latitude, UT)
            
            # 添加最后一个时间步的数据以处理周期性
            rho_v2_extended_data = np.concatenate([rho_v2_slice, rho_v2_slice[:, 0:1]], axis=1)
            
            # 创建插值函数
            f_rho_v2 = interp1d(ut_extended, rho_v2_extended_data, axis=1, kind='linear', 
                               bounds_error=False, fill_value="extrapolate")
            
            # 执行插值
            rho_v2_interp[i, :] = f_rho_v2(current_ut)
        
        # 创建网格
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
        
        # 计算直射点位置
        subsolar_lat, subsolar_lon = calculate_insolation_point(ls_value, local_hour, 135)
        if subsolar_lon > 180:
            subsolar_lon -= 360
        elif subsolar_lon < -180:
            subsolar_lon += 360
        
        # 定义子集索引 - 使用扩展的范围
        lat_indices = np.where((lats >= lat_min_extended) & (lats <= lat_max_extended))[0]
        
        # 处理经度环绕问题
        # 第一部分：扩展的最小经度到180°E
        lon_indices1 = np.where((lons >= lon_min_extended) & (lons <= 180))[0]
        # 第二部分：-180°W到扩展的最大经度（转换为东经）
        lon_indices2 = np.where((lons >= -180) & (lons <= (lon_max_extended - 360)))[0]
        
        # 合并经度索引
        lon_indices = np.concatenate([lon_indices1, lon_indices2])
        
        # 裁剪数据到扩展区域
        rho_v2_subset = rho_v2_interp[lon_indices, :][:, lat_indices]
        
        # 调整经度值，使200°E正确显示
        lon_subset = lon_grid[lon_indices, :][:, lat_indices]
        lon_subset[lon_subset < 0] += 360  # 将负经度转换为正东经
        
        lat_subset = lat_grid[lon_indices, :][:, lat_indices]
        
        # 输出特定位置信息
        print("\n" + "="*60)
        print(f"动压范围: [{rho_v2_subset.min():.6f}, {rho_v2_subset.max():.6f}] Pa")
        print(f"局部平均动压: {np.nanmean(rho_v2_subset):.6f} Pa")
        print("="*60)
    
    # 绘制局部动压分布图
    plt.figure(figsize=(10, 5))
    
    # 计算动压分位数，用于等值线级别
    rho_v2_min = np.nanmin(rho_v2_subset)
    rho_v2_max = np.nanmax(rho_v2_subset)
    
    # 创建20个等间距的等值线级别
    levels = np.linspace(0, vmax, 21)
    
    # 使用plasma颜色映射填充动压等值线
    contour = plt.contourf(lon_subset, lat_subset, rho_v2_subset, 
                          cmap='plasma', 
                          levels=levels, 
                          extend='both',
                          vmin=0, vmax=vmax)
    
    # 添加颜色条
    cbar = plt.colorbar(contour)
    cbar.set_label('Dynamic Pressure (Pa)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # 打开地形文件并读取海拔数据
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = adjust_longitude(surface_dataset.variables['longitude'][:])
        elevation = surface_dataset.variables['zMOL'][:]
        
        # 分割地形数据为两部分
        mask_lon_part1 = (lons_surface >= lon_min) & (lons_surface <= 180)
        mask_lon_part2 = (lons_surface > 180) & (lons_surface <= lon_max)

        # 对于每个部分，计算其对应的纬度和经度索引
        lat_indices_surface = np.where((lats_surface >= lat_min) & (lats_surface <= lat_max))[0]
        lon_indices_part1 = np.where(mask_lon_part1)[0]
        lon_indices_part2 = np.where(mask_lon_part2)[0]

        # 创建独立的网格点用于海拔数据
        lon_surface_part1, lat_surface_part1 = np.meshgrid(lons_surface[lon_indices_part1], lats_surface[lat_indices_surface])
        lon_surface_part2, lat_surface_part2 = np.meshgrid(lons_surface[lon_indices_part2], lats_surface[lat_indices_surface])

        # 裁剪海拔数据至指定区域
        elevation_subset_part1 = elevation[lat_indices_surface, :][:, lon_indices_part1]
        elevation_subset_part2 = elevation[lat_indices_surface, :][:, lon_indices_part2] if len(lon_indices_part2) > 0 else np.array([])

        # 合并两部分海拔数据
        if len(elevation_subset_part2) > 0:
            lons_combined_elev = np.concatenate([lons_surface[lon_indices_part1], lons_surface[lon_indices_part2]])
            elevation_data_combined = np.hstack([elevation_subset_part1, elevation_subset_part2])
        else:
            lons_combined_elev = lons_surface[lon_indices_part1]
            elevation_data_combined = elevation_subset_part1

        # 保存地形数据到txt文件
        elevation_filename = f'local_elevation_ls_{ls_value}_local_hour_{local_hour}.txt'
        save_data_to_txt(elevation_data_combined, filename=elevation_filename)
        
        # 从文本文件中读取地形数据并绘制等高线图
        elevation_data_combined = np.loadtxt(elevation_filename)
        
        # 绘制等高线
        CS = plt.contour(lons_combined_elev, lats_surface[lat_indices_surface], elevation_data_combined, 
                        colors=[(0, 0, 0, 0.5)], levels=np.arange(-10, 8.1, 2))
        plt.clabel(CS, inline=True, fontsize=8)  # 添加等高线标签
    
    # 标记InSight位置
    insight_lon, insight_lat = 135.6, 4.5
    plt.scatter(insight_lon, insight_lat, color='red', s=100)
    
    # 为InSight位置添加标签
    plt.annotate(
        'InSight',
        xy=(insight_lon, insight_lat),
        xytext=(insight_lon - 30, insight_lat + 10),  # 调整文本位置
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    
    # 设置坐标轴
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    
    # 应用自定义格式化函数
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    #plt.title(f'Local ρv² Distribution at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # 设置坐标轴范围
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])
    
    # 保存图片
    filename = Path(__file__).resolve().parent / f'local_rho_v2_field_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存动压数据到txt文件
    rho_v2_filename = Path(__file__).resolve().parent / f'local_rho_v2_ls_{ls_value}_local_hour_{local_hour}.txt'
    save_data_to_txt(rho_v2_subset, filename=rho_v2_filename)
    
    print(f"局部动压分布图已保存到: {filename}")
    print(f"动压数据已保存到: {rho_v2_filename}")
    print(f"地形数据已保存到: {elevation_filename}")
    
    return rho_v2_subset, lon_subset, lat_subset

# 示例调用
if __name__ == "__main__":
    # 参数说明:
    # 第一个参数: 动压数据文件路径
    # 第二个参数: Ls值 (太阳经度, 0-360度)
    # 第三个参数: 火星本地时 (0-23)
    # 第四到七个参数: 纬度最小值, 纬度最大值, 经度最小值, 经度最大值
    # 第八个参数: 地形文件路径
    # 第九个参数: 动压最大值，用于颜色条范围
    
    # 示例: 绘制Ls=90°, 当地正午(12:00)，纬度-35°到45°，经度80°到200°的动压分布
    rho_v2_data, lon_grid, lat_grid = plot_local_rho_v2_field_mcd(
        Path(__file__).resolve().parents[1] / 'rho_v2_output.nc',
        ls_value=90,            # 太阳经度
        local_hour=12,          # 本地时
        lat_min=-35,            # 最小纬度
        lat_max=45,             # 最大纬度
        lon_min=80,             # 最小经度
        lon_max=200,            # 最大经度
        surface_file=Path(__file__).resolve().parents[1] / 'surface.nc',
        vmax=1.8                # final Figure 2 dynamic-pressure range
    )
    
    # 其他示例调用:
    
    # 1. 绘制北半球夏季(Ls=90°)黄昏(18:00)的动压分布
    # rho_v2_data, lon_grid, lat_grid = plot_local_rho_v2_field_mcd(
    #     '../rho_v2_output.nc',
    #     ls_value=90,
    #     local_hour=18,
    #     lat_min=-35,
    #     lat_max=45,
    #     lon_min=80,
    #     lon_max=200,
    #     surface_file='surface.nc',
    #     vmax=2
    # )
    
    # 2. 绘制北半球冬季(Ls=270°)正午(12:00)的动压分布
    # rho_v2_data, lon_grid, lat_grid = plot_local_rho_v2_field_mcd(
    #     '../rho_v2_output.nc',
    #     ls_value=270,
    #     local_hour=12,
    #     lat_min=-35,
    #     lat_max=45,
    #     lon_min=80,
    #     lon_max=200,
    #     surface_file='surface.nc',
    #     vmax=2
    # )
    
    # 3. 绘制更小范围的动压分布（如洞察号附近）
    # rho_v2_data, lon_grid, lat_grid = plot_local_rho_v2_field_mcd(
    #     '../rho_v2_output.nc',
    #     ls_value=90,
    #     local_hour=12,
    #     lat_min=0,      # 最小纬度
    #     lat_max=20,     # 最大纬度
    #     lon_min=120,    # 最小经度
    #     lon_max=150,    # 最大经度
    #     surface_file='surface.nc',
    #     vmax=2
    # )
