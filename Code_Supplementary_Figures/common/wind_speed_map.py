import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Code_Figure4' / '_vendor'))
import cartopy.crs as ccrs
from netCDF4 import Dataset
from scipy.interpolate import griddata
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
import numpy as np
scale = 18
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
# 或者更完整地设置：
plt.rcParams['font.family'] = 'sans-serif'
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--surface', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 优先使用 Arial

def read_data(file_path):
    """读取txt文件并返回经纬度及百分比"""
    latitudes = []
    longitudes = []
    percentages = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            lat, lon, percentage = map(float, line.strip().split(','))
            latitudes.append(lat)
            longitudes.append(lon)
            percentages.append(percentage)
    return np.array(latitudes), np.array(longitudes), np.array(percentages)

# 读取数据并计算极值
latitudes, longitudes, percentages = read_data(args.data)
max_percent = np.ceil(np.max(percentages))  # 向上取整到最近的整数

# 创建新的非均匀颜色映射
def create_custom_cmap():
    """创建自定义非连续颜色映射"""
    # 定义关键节点位置（0-16%和16%-max）
    segments = {
        'red':   [(0, 0.698, 0.698),   # 深蓝起始
                 (50/100, 1.0, 1.0),     # 白色中点
                 (1, 0.129, 0.129)],   # 深红结束
        
        'green': [(0, 0.094, 0.094),   # 深蓝起始
                 (50/100, 1.0, 1.0),     # 白色中点
                 (1, 0.400, 0.400)],   # 深红结束
        
        'blue':  [(0, 0.168, 0.168),   # 深蓝起始
                 (50/100, 1.0, 1.0),     # 白色中点
                 (1, 0.667, 0.667)]    # 深红结束
    }
    return LinearSegmentedColormap('custom_rdbu', segments)

# 创建颜色标准化对象（重点修改）
cmap = create_custom_cmap()
boundaries = [0, 10,20,30,40,50,60, 70]
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

# 加载火星表面数据
surface_file = args.surface
with Dataset(surface_file, 'r') as surface_dataset:
    lats_surface = surface_dataset.variables['latitude'][:]
    lons_surface = surface_dataset.variables['longitude'][:]
    elevation = surface_dataset.variables['zMOL'][:]
    
    lon_surface, lat_surface = np.meshgrid(lons_surface, lats_surface)
    elevation_masked = np.ma.masked_where(elevation > 8, elevation)

    # 创建高分辨率网格
    lon_highres = np.linspace(lons_surface.min(), lons_surface.max(), 300)
    lat_highres = np.linspace(lats_surface.min(), lats_surface.max(), 300)
    lon_grid_highres, lat_grid_highres = np.meshgrid(lon_highres, lat_highres)

    # 插值百分比数据
    points = np.column_stack((latitudes, longitudes))
    grid_percentages_highres = griddata(points, percentages, 
                                      (lat_grid_highres, lon_grid_highres), 
                                      method='linear')

    # 创建绘图对象
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())  # 使用罗宾逊投影

    # 绘制颜色填充图 - 保持transform参数为PlateCarree
    mesh = ax.pcolormesh(lon_grid_highres, lat_grid_highres, grid_percentages_highres,
                        cmap=cmap, norm=norm,
                        transform=ccrs.PlateCarree(),  # 保持坐标系不变
                        shading='gouraud')
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.04, shrink=0.6,
                        ticks=boundaries)

    # 设置 colorbar 标题字体为 scale = 18
    cbar.set_label('Low Wind Speed Percentage (%)', 
                   fontsize=scale, 
                   labelpad=10, 
                   fontname='Arial')

    # 设置 colorbar 刻度字体为 scale-2 = 16
    cbar.ax.tick_params(labelsize=scale-2)
    
    # 突出显示16%刻度
    cbar.ax.axhline(y=16/max_percent, color='k', linewidth=1.2, linestyle='--')

    # ===== 添加等高线 - 关键修改部分 =====
    # 在绘制等高线时添加zorder参数解决标注问题
    CS = ax.contour(lon_surface, lat_surface, elevation_masked, 
                   colors='#404040', linewidths=0.8,
                   levels=np.arange(-10, 8.1, 2),
                   transform=ccrs.PlateCarree(),
                   zorder=4)  # 高于网格层但低于点标记层
    ax.clabel(CS, inline=True, fontsize=8, fmt='%d km')
    
   
    # 对背景数据进行高斯平滑处理
    smoothed_percentages = gaussian_filter(grid_percentages_highres, sigma=3)
    
    # ===== 添加查询点(4.5,135.6)的值 =====
    # 查询指定坐标点的值
    target_lat = 4.5
    target_lon =135.6

    # 查找最近网格点的索引
    lat_idx = np.abs(lat_highres - target_lat).argmin()
    lon_idx = np.abs(lon_highres - target_lon).argmin()

    # 获取值
    point_value = grid_percentages_highres[lat_idx, lon_idx]

    # 打印结果
    print(f"经纬度({target_lat:.1f}°, {target_lon:.1f}°) 位置的风速<3m/s时间占比: {point_value:.1f}%")
    
    '''
    J06	23.7336	-98.9964
    G03	20.7525	-72.6525
    G04	22.5875	-68.4528
    J01	29.5283	-70.3183
    B06	25.9986	-62.0725
    K04	25.2186	-59.2964
    H07	25.6869	-55.3706
    E05	10.8556	-30.5306
    E04	7.5050	-34.8528
    E01	8.3647	-49.4397
    G01	11.9789	-45.2353
    G10	11.9361	-46.5636
    D05	22.6008	-18.4167
    D09	21.4183	-8.4919
    G08	27.7992	11.3875
    G11	32.6081	21.8875
    J09	16.3108	13.0042
    E03	21.0414	33.2756
    D07	17.4767	77.9706
    B04	19.9992	79.3828
    E02	29.3889	73.2444
    H10	17.6417	77.0047
    H09	20.3489	79.4072
    K05	20.0678	79.3261
    G09	7.0861	150.7350
    D10	27.7992	134.3103
    '''
    # ============= 添加候选着陆点 =============
    # 从文档中读取候选着陆点数据
    candidate_points = """
    A01	21.9083	-22.4314
    A02	23.9136	-20.0325
    A03	24.8497	-19.2969
    A05	27.8111	-52.2319
    A06	26.2425	-56.2006
    B01	23.4239	-20.1278
    B02	21.8842	-22.4803
    B03	24.2978	-18.9283
    B05	25.2117	-34.1139
    C01	26.2544	-56.0175
    C02	26.8744	-53.3208
    C03	27.8242	-53.4792
    C04	29.5033	-52.6386
    C05	23.2000	-20.2900
    C06	21.7300	-22.2900
    C07	22.9633	-33.6528
    C08	23.8031	-31.1833
    C09	24.3203	-32.6953
    C10	17.1961	-31.8325
    D01	21.8975	-22.3989
    D02	24.5897	-21.3092
    D03	24.1925	-19.0442
    D04	23.3858	-19.9794
    D06	18.3242	-24.6647
    D08	19.1283	-35.3975
    D11 22.1269 -8.2633
    F01	24.7719	109.9681
    F02	20.3733	109.7394
    F03	22.7614	129.7258
    F04	28.8617	-48.7525
    F05	17.7456	-35.3000
    F06	21.7058	89.4972
    F07	17.9225	93.1706
    G02	23.4675	-19.8900
    G05	28.6175	91.0125
    G06	27.8806	111.4372
    G07	21.5322	123.4467
    H01	24.0258	-18.8514
    H02	24.4414	-18.8694
    H03	24.8319	-19.2928
    H04	21.7956	-22.3120
    H05	17.9953	-24.1903
    H06	18.4183	-24.4658
    H08	25.7978	-54.8372
    I01	24.2464	109.9786
    I02	25.8936	-40.2253
    I04	26.4147	-45.2856
    I05	27.1497	-22.4481
    I06	21.8072	-22.6036
    I07	23.4069	-20.3486
    J02	29.6350	133.7297
    J03	23.3544	121.7467
    J04	20.7642	132.9056
    J05	17.3467	-31.3431
    J07	19.6706	24.7650
    J08	19.2706	24.9697
    K01	23.7911	-152.0442
    K02	19.1767	-37.4906
    K03	21.8192	121.9775
    K06 21.1142 -43.1592
    """
    
    # 解析候选着陆点
    candidate_lats = []
    candidate_lons = []
    # 修改1: 将Viking点的颜色改为黑色
    viking_lats = [22.487, 48.269]
    viking_lons = [-48.041, 134.01]
    
    # 修改2: 将候选点分为两类: 大于40%的黄色点和小于等于40%的灰色点
    yellow_points_lats = []
    yellow_points_lons = []
    gray_points_lats = []
    gray_points_lons = []
    
    # 统计点数量
    total_points = 0
    yellow_count = 0
    gray_count = 0
    
    for line in candidate_points.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3:
            try:
                lat = float(parts[1])
                lon = float(parts[2])
                candidate_lats.append(lat)
                candidate_lons.append(lon)
                total_points += 1
                
                # 找到这个点在网格中的值
                lat_idx = np.abs(lat_highres - lat).argmin()
                lon_idx = np.abs(lon_highres - lon).argmin()
                point_value = grid_percentages_highres[lat_idx, lon_idx]
                
                # 根据值判断颜色
                if point_value > 40:
                    yellow_points_lats.append(lat)
                    yellow_points_lons.append(lon)
                    yellow_count += 1
                else:
                    gray_points_lats.append(lat)
                    gray_points_lons.append(lon)
                    gray_count += 1
                    
            except ValueError:
                continue
    
    # 统计结果输出
    print(f"\n候选着陆点统计:")
    print(f"  总点数: {total_points}")
    print(f"  黄色点(>40%): {yellow_count} 个")
    print(f"  灰色点(≤40%): {gray_count} 个")
    print(f"  黄色点占比: {yellow_count/total_points*100:.1f}%")
    print(f"  灰色点占比: {gray_count/total_points*100:.1f}%")
    
    # 绘制黄色候选着陆点（大于40%）
    if yellow_points_lats:
        ax.scatter(yellow_points_lons, yellow_points_lats, 
                   transform=ccrs.PlateCarree(),
                   c='yellow', s=40,  # 点的大小
                   alpha=0.8,       # 点的透明度
                   edgecolor='k',   # 点边缘颜色
                   linewidth=0.5,   # 点边缘线宽
                   zorder=5,       # 确保点在最上层
                   label=f'Tianwen-3 Candidates (>40%)')
    
    # 绘制灰色候选着陆点（小于等于40%）
    if gray_points_lats:
        ax.scatter(gray_points_lons, gray_points_lats, 
                   transform=ccrs.PlateCarree(),
                   c='gray', s=40,  # 点的大小
                   alpha=0.8,       # 点的透明度
                   edgecolor='k',   # 点边缘颜色
                   linewidth=0.5,   # 点边缘线宽
                   zorder=5,       # 确保点在最上层
                   label=f'Tianwen-3 Candidates (≤40%)')
    
    # 添加InSight点
    ax.scatter([135.6], [4.5], 
               transform=ccrs.PlateCarree(),
               c='white',       # 亮绿色
               s=100,           # 更大的点
               alpha=0.9,       # 更高的透明度
               edgecolor='k',   # 黑色边缘
               linewidth=1.5,   # 更粗的边缘
               zorder=10,       # 确保在最上层
               label='Insight Site')
    
    # 修改1: Viking点改为黑色
    ax.scatter(viking_lons, viking_lats, 
               transform=ccrs.PlateCarree(),
               c='black',       # 修改为黑色
               s=80,           # 更大的点
               alpha=0.9,       # 更高的透明度
               edgecolor='k',   # 黑色边缘
               linewidth=1.5,   # 更粗的边缘
               zorder=10,       # 确保在最上层
               label='Viking 1 & 2 Sites')
    
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,
            fontsize=scale, ha='center', va='center')
    ax.text(-0.05, 0.5, 'Latitude', transform=ax.transAxes,
            fontsize=scale, rotation=90, ha='center', va='center')
    
    # 创建图例
    legend_elements = []
    
    # 添加黄色点图例
    if yellow_points_lats:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                      markerfacecolor='yellow', markeredgecolor='k',
                      label=f'Tianwen-3 (>40%)')
        )
    
    # 添加灰色点图例
    if gray_points_lats:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                      markerfacecolor='gray', markeredgecolor='k',
                      label=f'Tianwen-3 (≤40%)')
        )
    
    # 添加其他点图例
    legend_elements.extend([
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                  markerfacecolor='white', markeredgecolor='k',
                  label='Insight'),
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                  markerfacecolor='black', markeredgecolor='k',  # 修改为黑色
                  label='Viking 1 & 2')
    ])
    
    legend = ax.legend(handles=legend_elements, loc='upper left',
         fontsize=scale-4,                  # 图例文字大小
         title='',                           # 设置为空字符串
         framealpha=0.9,
         edgecolor='#606060', 
         facecolor='#f8f8f8',
         bbox_to_anchor=(-0.07, 1.0)) 
    legend.set_zorder(10)
    
    # 设置网格线和经纬度标签
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.6, color='gray', alpha=0.4,
                     linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True          # 明确开启左边纬度标签
    gl.xformatter = LongitudeFormatter(zero_direction_label=True)
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': scale-2, 'color': '#404040'}
    gl.ylabel_style = {'size': scale-2, 'color': '#404040'}

    for label in gl.label_artists:
        if isinstance(label, plt.Text):
            text = label.get_text()
            # 直接使用 Unicode 字符 °，或用 \u00B0 写在 raw string 外
            if text == '60°N' or text == '60°S':
                label.set_visible(False)

    plt.tight_layout()
    
    # 计算全球符合条件的区域比例
    def calculate_global_percentage(grid_data, threshold=20):
        """
        计算全球网格数据中超过阈值的比例
        """
        # 创建火星表面有效区域掩码（排除NaN和无效值）
        valid_mask = ~np.isnan(grid_data) & (grid_data >= 0)
        
        # 计算纬度权重（cosine纬度校正）
        lat_rad = np.deg2rad(lat_grid_highres)
        weights = np.cos(lat_rad)  # 等经纬度网格的面积权重
        
        # 计算超过阈值的区域
        threshold_mask = (grid_data > threshold) & valid_mask
        
        # 计算加权比例
        total_weighted_area = np.sum(weights[valid_mask])
        threshold_weighted_area = np.sum(weights[threshold_mask])
        
        return (threshold_weighted_area / total_weighted_area) * 100

    # 执行计算
    global_percentage = calculate_global_percentage(grid_percentages_highres, 27.8)
    print(f"全球风速<3m/s时间占比>27.8%的区域比例: {global_percentage:.2f}%")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=100, bbox_inches='tight')
    plt.close(fig)
