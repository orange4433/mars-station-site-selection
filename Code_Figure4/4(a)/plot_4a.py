import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_vendor"))
# 需要安装 Cartopy 库来支持罗宾逊投影
# pip install cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 设置全局字体为 Arial ---
plt.rcParams['font.family'] = 'Arial'
scale = 18
def normalize_longitude(lon):
    """将经度标准化到[-180, 180]范围"""
    lon = lon % 360
    return lon - 360 if lon > 180 else lon

def calculate_circle_points(center_lat, center_lon, angular_distance_deg, num_points=360):
    angular_distance_rad = math.radians(angular_distance_deg)
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)
    
    lats = []
    lons = []
    
    for azimuth in np.linspace(0, 360, num_points):
        azimuth_rad = math.radians(azimuth)
        
        # 计算纬度
        lat = math.asin(math.sin(center_lat_rad)*math.cos(angular_distance_rad) + 
                      math.cos(center_lat_rad)*math.sin(angular_distance_rad)*math.cos(azimuth_rad))
        
        # 计算经度
        lon = center_lon_rad + math.atan2(
            math.sin(azimuth_rad)*math.sin(angular_distance_rad)*math.cos(center_lat_rad),
            math.cos(angular_distance_rad) - math.sin(center_lat_rad)*math.sin(lat)
        )
        
        # 标准化经度
        lon_deg = math.degrees(lon)
        lon_deg = normalize_longitude(lon_deg)
        
        lats.append(math.degrees(lat))
        lons.append(lon_deg)
    
    return np.array(lats), np.array(lons)

def plot_circle_with_adaptive_labels(ax, lons, lats, label_text, transform, **kwargs):
    """
    智能绘制圆周线并在最佳位置添加标签 (修改以支持 Cartopy)

    参数:
        ax: matplotlib轴对象 (具有 Cartopy 投影)
        lons: 经度数组
        lats: 纬度数组
        label_text: 标签文本
        transform: 数据的坐标系 (例如 ccrs.PlateCarree())
        **kwargs: 传递给plot的绘图参数
    """
    # 1. 检测线段分割点 (基于经度跳跃)
    segments = []
    current_segment_lons = []
    current_segment_lats = []

    for i in range(len(lons)):
        if not current_segment_lons:
            current_segment_lons.append(lons[i])
            current_segment_lats.append(lats[i])
            continue

        prev_lon = current_segment_lons[-1]
        lon_diff = abs(lons[i] - prev_lon)

        # 如果经度跳跃超过180度，认为是跨越了日期线，开始新线段
        if lon_diff > 180:
            segments.append((np.array(current_segment_lons), np.array(current_segment_lats)))
            current_segment_lons = [lons[i]]
            current_segment_lats = [lats[i]]
        else:
            current_segment_lons.append(lons[i])
            current_segment_lats.append(lats[i])

    if current_segment_lons:
        segments.append((np.array(current_segment_lons), np.array(current_segment_lats)))

    # 2. 绘制所有线段
    for seg_lons, seg_lats in segments:
        # 注意：传递 transform 参数给 plot，告诉 Cartopy 数据的原始坐标系
        ax.plot(seg_lons, seg_lats, transform=transform, **kwargs)

    # --- 标签部分根据需求可能需要调整，暂时保留核心逻辑 ---
    # 3. 为每个主要段添加标签 (示例：只给第一个段加标签)
    if segments and label_text:
        seg_lons, seg_lats = segments[0]
        if len(seg_lons) > 1:
             # 寻找一个不太可能被数据或其它元素遮挡的位置放置标签
             # 这里简单地选择线的中点，并稍微偏移
             mid_idx = len(seg_lons) // 2
             mid_lon = seg_lons[mid_idx]
             mid_lat = seg_lats[mid_idx]
             
             # 尝试偏移到线的“外部”
             dx = seg_lons[mid_idx] - seg_lons[mid_idx-1] if mid_idx > 0 else 0
             dy = seg_lats[mid_idx] - seg_lats[mid_idx-1] if mid_idx > 0 else 1
             length = np.sqrt(dx**2 + dy**2)
             if length > 0:
                 dx, dy = dx/length, dy/length
             else:
                 dx, dy = 0, 1
             # 垂直偏移
             offset_x, offset_y = -dy * 2.0, dx * 2.0 # 偏移 2 度
             label_lon = mid_lon + offset_x
             label_lat = mid_lat + offset_y

             # 确保标签在合理范围内
             label_lon = normalize_longitude(label_lon)
             label_lat = np.clip(label_lat, -90, 90)

             # 简单放置标签 (Cartopy 会处理投影)
             ax.text(label_lon, label_lat, label_text,
                     transform=transform, # 标签坐标也是 PlateCarree
                     fontsize=12, color=kwargs.get('color', 'black'),
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.8),
                     zorder=10)


# === 主程序 ===

# 打开NetCDF文件
file_path = ROOT / 'data' / 'seismic' / 'surface.nc'
try:
    dataset = nc.Dataset(file_path)
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}。请检查文件路径。")
    # 如果文件不存在，创建一个示例数据集用于演示
    print("正在创建示例数据进行演示...")
    import numpy as np
    lats_demo = np.linspace(-89.5, 89.5, 180)
    lons_demo = np.linspace(-179.5, 179.5, 360)
    lon_grid_demo, lat_grid_demo = np.meshgrid(lons_demo, lats_demo)
    # 创建一个简单的模拟地形 (例如，中心高四周低)
    elevation_demo = np.exp(-((lon_grid_demo / 50.0) ** 2 + (lat_grid_demo / 30.0) ** 2)) * 5000
    # 模拟变量
    class DummyDataset:
        pass
    class DummyVar:
        def __init__(self, data):
            self.data = data
        def __getitem__(self, key):
            return self.data
    dataset = DummyDataset()
    dataset.variables = {
        'latitude': DummyVar(lats_demo),
        'longitude': DummyVar(lons_demo),
        'zMOL': DummyVar(elevation_demo)
    }


# 获取经纬度数据
lats = dataset.variables['latitude'][:]
lons = dataset.variables['longitude'][:]

# 获取海拔数据
elevation = dataset.variables['zMOL'][:]

# 创建网格
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --- 设置罗宾逊投影 ---
# 创建具有罗宾逊投影的图形和轴
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
# 设置绘图范围为全球
ax.set_global()

# 设置地图背景色为白色
ax.set_facecolor('white') # 修复 AttributeError

# 添加自然地理要素 (可选，增加背景细节)
# ax.add_feature(cfeature.LAND, facecolor='lightgray')
# ax.add_feature(cfeature.OCEAN, facecolor='white')
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)


# 绘制地形 - 使用灰度色图 ('gray') 从黑到白
cs_global = ax.contourf(lon_grid, lat_grid, elevation, cmap='gray', levels=30, transform=ccrs.PlateCarree())

# 修改颜色条位置到左侧
cbar = fig.colorbar(cs_global, ax=ax, orientation='vertical', pad=0.05, location='left', shrink=0.7, aspect=20)
cbar.set_label('Elevation (km)', fontsize=scale, labelpad=10) # 增加labelpad使标签离颜色条更远
cbar.ax.tick_params(labelsize=scale-2)

# 添加经纬度网格线 - 60度间隔标签
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# 获取投影边界点
# 对于纬度标签，沿着左侧边界放置
left_boundary = []
for lat in [-60, -30, 0, 30, 60]:  # 60度间隔，不包含90°
    # 获取左侧边界点的投影坐标
    x, y = ccrs.Robinson().transform_point(-180, lat, ccrs.PlateCarree())
    left_boundary.append((x, y, lat))

# 对于经度标签，沿着底部边界放置
bottom_boundary = []
for lon in [-180, -120, -60, 0, 60, 120, 180]:  # 60度间隔
    # 获取底部边界点的投影坐标
    x, y = ccrs.Robinson().transform_point(lon, -90, ccrs.PlateCarree())
    bottom_boundary.append((x, y, lon))

# 手动添加纬度标签（左侧边界）
for x, y, lat in left_boundary:
    # 格式化标签文本
    if lat == 0:
        label = "0°"
    else:
        label = f"{abs(lat)}°{'S' if lat < 0 else 'N'}"
    
    # 向左调整位置（x坐标减小）
    x_adjusted = x - 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    
    # 添加标签，颜色改为黑色
    ax.text(x_adjusted, y, label, 
            ha='right', va='center', fontsize=scale-4, color='black',
            transform=ax.transData)

# 手动添加经度标签（底部边界）
for x, y, lon in bottom_boundary:
    # 格式化标签文本
    if lon == 0:
        label = "0°"
    elif abs(lon) == 180:  # 180°不写方向
        label = "180°"
    else:
        label = f"{abs(lon)}°{'W' if lon < 0 else 'E'}"
# 向下调整位置（y坐标减小）
    y_adjusted = y - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])        


    # 添加标签，颜色改为黑色
    ax.text(x, y_adjusted, label, 
            ha='center', va='top', fontsize=scale-4, color='black',
            transform=ax.transData)


# 定义坐标变换
data_crs = ccrs.PlateCarree() # NetCDF 数据的坐标系

# === 修改地震点标注部分 ===

# 地震点及其震级数据（保持不变）
earthquake_magnitudes = {
    (35, 90): (3.3, "4.7"), (15, 60): (3.3, "4.7"), (-15, 110): (3.4, "4.8"),
    (10, 160): (3.4, "4.8"), (-20, 150): (3.5, "4.8"), (18.4, -134): (3.0, "4.5"),
    (10, -105): (2.9, "4.4"),
    (-40, 60): (3.3, "4.7"), (-50, -45): (3.0, "4.4"), (-10, -75): (2.9, "4.3")
}
#,  (-10, -120): (3.0, "4.5"),
# 标注地震点及其震级 - 修改为红色
marsquake_lons = [p[1] for p in earthquake_magnitudes.keys()]
marsquake_lats = [p[0] for p in earthquake_magnitudes.keys()]
if earthquake_magnitudes:
    # 创建大小列表：震级大于3.6的用大尺寸(600)，否则用小尺寸(400)
    sizes = [650 if mag[0] > 3.6 else 350 for mag in earthquake_magnitudes.values()]
    
    # 使用红色五角星
    ax.scatter(marsquake_lons, marsquake_lats, color='cyan', s=sizes, marker='*', 
               zorder=5, transform=data_crs, label='Marsquake')
    
    # 添加文本标签 - 修改格式和位置
    for (lat, lon), magnitude in earthquake_magnitudes.items():
        main_mag = f"{magnitude[0]:.1f}  "
        slash = "  /  "  # 中间的斜杠
        extra_mag = f"  {magnitude[1]}"
        
        # 获取文本渲染器用于测量宽度（可选，但更精确）
        renderer = ax.figure.canvas.get_renderer()
        
        # 先绘制主文本（黄色）
        t1 = ax.text(lon, lat - 6, main_mag,
                     fontsize=scale-2, ha='right', va='top',  # 注意: 使用'right'对齐
                     color='orange',weight='bold', transform=data_crs, zorder=6)
        
        # 绘制中间的斜杠（白色）
        t2 = ax.text(t1.get_position()[0], lat - 6, slash,
                      fontsize=scale-2, ha='center', va='top',
                      color='white',weight='bold', transform=data_crs, zorder=6)
        
        # 第二部分（粉色）放在斜杠右侧，左对齐
        ax.text(t2.get_position()[0], lat - 6, extra_mag,
                fontsize=scale-2, ha='left', va='top',
                color='#0099ff',weight='bold', transform=data_crs, zorder=6)

# === 修改地震仪点颜色 ===
seismometer_points = [(21.1, -43.2)]
seismometer_lons = [p[1] for p in seismometer_points]
seismometer_lats = [p[0] for p in seismometer_points]
if seismometer_points:
    # 使用红色表示地震仪
    ax.scatter(seismometer_lons, seismometer_lats, color='white', s=200, 
               marker='o', zorder=4, transform=data_crs, linewidth=2, edgecolor='white')
    ax.scatter(seismometer_lons, seismometer_lats, color='red', s=120, 
               marker='o', zorder=5, transform=data_crs, label='Seismometer')
    for lat, lon in seismometer_points:
        ax.text(lon, lat + 5, 'Seismometer', 
                fontsize=scale, ha='center', va='bottom', 
                color='red', weight='bold',  # 改为红色
                transform=data_crs, zorder=6)

# === 添加圆和阴影区域 ===
arc_shadow = 1.6 # 修改为 1.6
angle_deg_shadow = arc_shadow * 180 / np.pi
center_point = (21.1, -43.2)

# 1. 绘制弧度为 1.6 的圆
circle_lats, circle_lons = calculate_circle_points(center_point[0], center_point[1], angle_deg_shadow, num_points=720)
plot_circle_with_adaptive_labels(ax, circle_lons, circle_lats, '', data_crs, color='yellow', linewidth=2, linestyle='-')

# 2. 绘制阴影区域 (弧度大于 1.6)
# 创建一个较密的网格用于计算阴影掩膜
lats_mask = np.linspace(-90, 90, 720)  # 增加分辨率以获得更平滑的边缘
lons_mask = np.linspace(-180, 180, 1440)
lon_grid_mask, lat_grid_mask = np.meshgrid(lons_mask, lats_mask)

# 计算网格上每个点到中心点的球面距离 (使用 Haversine 公式，结果为弧度)
def haversine_distance_vectorized(lat_grid, lon_grid, center_lat, center_lon):
    """
    向量化计算网格点到单点的球面距离
    输入单位：度
    输出单位：弧度
    """
    lat1_rad = np.radians(lat_grid)
    lon1_rad = np.radians(lon_grid)
    lat2_rad = np.radians(center_lat)
    lon2_rad = np.radians(center_lon)
    
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c # 返回弧度

distances_mask = haversine_distance_vectorized(lat_grid_mask, lon_grid_mask, center_point[0], center_point[1])

# 创建阴影掩膜 (距离大于 arc_shadow 的区域为 1，否则为 0)
shadow_mask = np.where(distances_mask > arc_shadow, 1, 0)

# 绘制阴影区域 - 使用 contourf 绘制二值掩膜
# levels=[0.5, 1.5] 意味着值在 (0.5, 1.5] 范围内的区域会被填充
cs_shadow = ax.contourf(lon_grid_mask, lat_grid_mask, shadow_mask, levels=[0.5, 1.5], colors=['pink'], alpha=0.2, transform=data_crs, zorder=2)


# 3. 在阴影区域添加标签 "Shadow Zone"
# 选择一个合适的位置放置标签，例如在圆的对面方向
label_angle_rad = np.radians(180 + angle_deg_shadow / 2) # 大致在阴影区中心
label_lat_rad = np.radians(center_point[0])
# 简单近似计算标签位置
label_zone_lat = np.degrees(np.arcsin(np.sin(label_lat_rad)*np.cos(np.radians(angle_deg_shadow + 30)) +
                           np.cos(label_lat_rad)*np.sin(np.radians(angle_deg_shadow + 30))*np.cos(label_angle_rad))+0.02)
label_zone_lon = center_point[1] + np.degrees(np.arctan2(np.sin(label_angle_rad)*np.sin(np.radians(angle_deg_shadow + 30))*np.cos(label_lat_rad),
                                                np.cos(np.radians(angle_deg_shadow + 30)) - np.sin(label_lat_rad)*np.sin(np.radians(label_zone_lat)))+0.1)
label_zone_lon = normalize_longitude(label_zone_lon)
'''
# 添加文本标签
ax.text(label_zone_lon, label_zone_lat, 'Shadow Zone', fontsize=16, ha='center', va='center',
        color='darkred', weight='bold', transform=data_crs, zorder=4,
        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='none', alpha=0.7))

'''
# 标注地震仪可取点
# 统一使用一个图例项 "Seismometer"


'''
# 标注地震点及其震级
# 统一使用一个图例项 "Marsquake"
marsquake_lons = [p[1] for p in earthquake_magnitudes.keys()]
marsquake_lats = [p[0] for p in earthquake_magnitudes.keys()]
if earthquake_magnitudes: # 只有在有点时才绘制和添加图例
    ax.scatter(marsquake_lons, marsquake_lats, color='red', s=150, marker='*', zorder=5, transform=data_crs, label='Marsquake')
    # 添加文本标签，修改颜色、大小和字体 (字体已在全局设置)
    for (lat, lon), magnitude in earthquake_magnitudes.items():
        ax.text(lon, lat, f'{magnitude[0]:.1f}({float(magnitude[1]):.1f})',
                fontsize=14, ha='right', va='bottom', color='yellow', # 修改颜色和大小
                transform=data_crs, zorder=6) # 设置 zorder
'''
# 添加图例 (放在右上角)
from matplotlib.lines import Line2D

# 创建自定义图例句柄
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan', markersize=22, label='Marsquake'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='white', markeredgewidth=2.5, 
           markerfacecolor='red', markersize=15, label='Seismometer')
]

# 添加自定义图例
legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=scale-2, 
                   framealpha=0.9, edgecolor='#606060', facecolor='#f8f8f8')
# 图例字体已在全局设置


# 显示图形
plt.tight_layout()
output_dir = ROOT / 'output'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / '4a_global_map.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '4a_global_map.pdf', bbox_inches='tight')
plt.close()
if hasattr(dataset, 'close'):
    dataset.close()
print(f"Saved: {output_dir / '4a_global_map.png'}")




