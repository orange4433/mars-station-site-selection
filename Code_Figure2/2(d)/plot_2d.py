import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 设置图形参数
scale = 18
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
    sub_solar_longitude = longitude - hour_angle
    
    return latitude, sub_solar_longitude

def plot_rho_v2_field_mcd(file_path, ls_value, local_hour, surface_file='surface.nc'):
    """
    绘制火星全球ρv²分布图
    
    参数:
        file_path: ρv²数据文件路径
        ls_value: 太阳经度 (Ls) 值
        local_hour: 火星本地时
        surface_file: 地形数据文件路径
    """
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
        # 注意：原始数据使用世界时(UT)，需要转换为火星本地时(LT)
        ut_target = (local_hour + 9 - lons/15) % 24
        
        # 初始化插值后的ρv²数据
        rho_v2_interp = np.zeros((len(lons), len(lats)))
        
        # 对每个经度点进行时间插值
        for i, lon in enumerate(lons):
            # 当前经度对应的目标世界时
            current_ut = ut_target[i]
            
            # 扩展UT数组以处理周期边界
            ut_extended = np.append(ut_array, [24])
            
            # 获取当前经度的ρv²切片 (纬度, 时间)
            rho_v2_slice = rho_v2_data[i, :, :]  # shape: (latitude, UT)
            
            # 添加最后一个时间步的数据（即第一个时间步的数据）以处理周期性
            rho_v2_extended_data = np.concatenate([rho_v2_slice, rho_v2_slice[:, 0:1]], axis=1)
            
            # 创建插值函数
            f_rho_v2 = interp1d(ut_extended, rho_v2_extended_data, axis=1, kind='linear', 
                               bounds_error=False, fill_value="extrapolate")
            
            # 执行插值
            rho_v2_interp[i, :] = f_rho_v2(current_ut)
        
        # 创建网格
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
        
        # 将经度调整到[-180, 180]范围
        lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
        
        # 计算直射点位置
        subsolar_lat, subsolar_lon = calculate_insolation_point(ls_value, local_hour, 135)
        # 将经度调整到[-180, 180]范围
        if subsolar_lon > 180:
            subsolar_lon -= 360
        elif subsolar_lon < -180:
            subsolar_lon += 360
        
        # ==============================================
        # 输出特定位置的信息
        # ==============================================
        # 1. 输出InSight位置 (4.5°N, 135.6°E) 的ρv²
        target_lat_insight = 4.5
        target_lon_insight = 135.6
        
        # 找到最接近的经度和纬度索引
        lon_idx_insight = np.argmin(np.abs(lon_grid[:, 0] - target_lon_insight))
        lat_idx_insight = np.argmin(np.abs(lat_grid[0, :] - target_lat_insight))
        
        # 获取该位置的ρv²
        rho_v2_insight = rho_v2_interp[lon_idx_insight, lat_idx_insight]
        
        # 2. 输出直射点位置的ρv²
        lon_idx_subsolar = np.argmin(np.abs(lon_grid[:, 0] - subsolar_lon))
        lat_idx_subsolar = np.argmin(np.abs(lat_grid[0, :] - subsolar_lat))
        rho_v2_subsolar = rho_v2_interp[lon_idx_subsolar, lat_idx_subsolar]
        
        # 输出结果
        print("\n" + "="*60)
        print(f"太阳直射点位置: {subsolar_lat:.2f}°N, {subsolar_lon:.2f}°E")
        print(f"直射点动压: {rho_v2_subsolar:.6f} Pa")
        print(f"InSight位置 ({target_lat_insight}°N, {target_lon_insight}°E) 动压: {rho_v2_insight:.6f} Pa")
        print(f"全球平均动压: {np.nanmean(rho_v2_interp):.6f} Pa")
        print(f"全球最大动压: {np.nanmax(rho_v2_interp):.6f} Pa at index {np.unravel_index(np.nanargmax(rho_v2_interp), rho_v2_interp.shape)}")
        print(f"全球最小动压: {np.nanmin(rho_v2_interp):.6f} Pa at index {np.unravel_index(np.nanargmin(rho_v2_interp), rho_v2_interp.shape)}")
        print("="*60 + "\n")
    
    # 绘制ρv²分布图
    plt.figure(figsize=(12, 6))
    
    # 计算ρv²分位数，用于等值线级别
    rho_v2_min = np.nanmin(rho_v2_interp)
    rho_v2_max = np.nanmax(rho_v2_interp)
    
    # 创建20个等间距的等值线级别
    levels = np.linspace(0, 1.8, 19)
    
    # 使用plasma颜色映射填充ρv²等值线
    contour = plt.contourf(lon_grid, lat_grid, rho_v2_interp, 
                          cmap='plasma', 
                          levels=levels, 
                          extend='both',
                          vmin=0, vmax=1.8)  # final Figure 2 dynamic-pressure range
    
    # 添加颜色条
    cbar = plt.colorbar(contour)
    cbar.set_label('Dynamic Pressure (Pa)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # 添加地形等高线
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = surface_dataset.variables['longitude'][:]
        elevation = surface_dataset.variables['zMOL'][:]
        
        # 调整经度到-180~180范围
        lons_surface = np.where(lons_surface > 180, lons_surface - 360, lons_surface)
        lon_surface, lat_surface = np.meshgrid(lons_surface, lats_surface)
        
        # 绘制等高线 - 使用参考代码中的设置
        CS = plt.contour(lon_surface, lat_surface, elevation, 
                        colors=[(0, 0, 0, 0.5)],  # 黑色，50%透明度
                        levels=np.arange(-10, 8.1, 2))  # 固定等高线级别
        plt.clabel(CS, inline=True, fontsize=5)
    
    # 标记InSight位置
    plt.annotate(
        'InSight',
        xy=(135.6, 4.5),
        xytext=(135.6 - 30, 4.5 + 15),  # 调整文本位置
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    insight_scatter = plt.scatter(135.6, 4.5, color='red', s=100, label='InSight Location')
    
    # 设置坐标轴
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    
    # 应用自定义格式化函数
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    
    # 设置标题
    #plt.title(f'Global ρv² Distribution at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # 保存图片
    filename = Path(__file__).resolve().parent / f'rho_v2_distribution_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dynamic-pressure distribution saved to {filename}")
    print(f"Dynamic-pressure range: {rho_v2_min:.6f} to {rho_v2_max:.6f} Pa")
    
    # 返回插值后的数据
    return rho_v2_interp, lon_grid, lat_grid

# 示例调用
if __name__ == "__main__":
    # 示例: 绘制Ls=90°, 当地正午(12:00)的ρv²分布
    rho_v2_data, lon_grid, lat_grid = plot_rho_v2_field_mcd(
        Path(__file__).resolve().parents[1] / 'rho_v2_output.nc',
        ls_value=90,           # 太阳经度
        local_hour=12,
        surface_file=Path(__file__).resolve().parents[1] / 'surface.nc',
    )
