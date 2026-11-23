import netCDF4 as nc
import numpy as np
import os

# Threshold parameter; change it here.
THRESHOLD = 0.1  # Units: Pa

def calculate_rho_v2_percentage(nc_file_path, threshold=THRESHOLD):
    """
    Calculate the time fraction with dynamic pressure at or below the threshold and save it as text.
    """
    # Check the file.
    if not os.path.exists(nc_file_path):
        print(f"错误: 文件 '{nc_file_path}' 不存在!")
        return
    
    print(f"计算动压 ≤ {threshold} Pa 的时间占比...")
    
    # Read data.
    with nc.Dataset(nc_file_path, 'r') as ds:
        rho_v2 = (1/2)*ds.variables['rho_v2'][:]  # (65, 49, 48, 721)
        lons = ds.variables['longitude'][:]  # (65,)
        lats = ds.variables['latitude'][:]   # (49,)
        ut = ds.variables['ut'][:]           # (48,)
        Ls = ds.variables['Ls'][:]           # (721,)
    
    # Calculate the time fraction.
    below_threshold_count = np.sum(rho_v2 <= threshold, axis=(2, 3))  # (65, 49)
    total_time_points = len(ut) * len(Ls)  # 48 * 721
    percentage = (below_threshold_count / total_time_points) * 100  # (65, 49)
    
    # Save the results.
    output_dir = f"rho_v2_threshold_{threshold}"
    os.makedirs(output_dir, exist_ok=True)
    
    txt_file = os.path.join(output_dir, f"rho_v2_percentage_threshold_{threshold}.txt")
    
    with open(txt_file, 'w') as f:
        f.write("Latitude, Longitude, Percentage\n")
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                perc = percentage[j, i]  # j: longitude, i: latitude
                f.write(f"{lat:.2f}, {lon:.2f}, {perc:.2f}\n")
    
    # Statistics at InSight
    insight_lat = 4.5
    insight_lon = 135.6
    
    lat_idx = np.argmin(np.abs(lats - insight_lat))
    lon_idx = np.argmin(np.abs(lons - insight_lon))
    
    insight_percentage = percentage[lon_idx, lat_idx]
    actual_lat = lats[lat_idx]
    actual_lon = lons[lon_idx]
    
    # Calculate the area fraction with a time fraction greater than InSight's.
    # Calculate grid-cell area weights, accounting for latitude.
    # Planetary radius: Mars radius = 3396.2 km.
    mars_radius = 3396.2  # Units: km
    dlon = np.abs(lons[1] - lons[0])  # Longitude spacing
    dlat = np.abs(lats[1] - lats[0])  # Latitude spacing
    
    # Latitude weights account for shorter zonal distances at higher latitudes.
    lat_weights = np.cos(np.deg2rad(lats))
    
    # Calculate each grid cell's area weight.
    area_weights = np.outer(np.ones_like(lons), lat_weights)  # Shape: (65, 49)
    
    # Find cells whose time fraction exceeds InSight's.
    above_insight_mask = percentage > insight_percentage
    
    # Count cells above InSight and sum their area weights.
    num_total_grids = np.prod(percentage.shape)
    num_above_insight = np.sum(above_insight_mask)
    grid_percentage_above = (num_above_insight / num_total_grids) * 100
    
    # Calculate the area-weighted fraction above InSight.
    total_area_weight = np.sum(area_weights)
    above_insight_area_weight = np.sum(area_weights[above_insight_mask])
    area_percentage_above = (above_insight_area_weight / total_area_weight) * 100
    
    # Print statistics.
    print(f"\n动压 ≤ {threshold} Pa 统计:")
    print(f"  InSight ({actual_lon:.1f}°E, {actual_lat:.1f}°N): {insight_percentage:.1f}%")
    print(f"  全球平均: {np.nanmean(percentage):.1f}%")
    print(f"  全球最小: {np.nanmin(percentage):.1f}%")
    print(f"  全球最大: {np.nanmax(percentage):.1f}%")
    print(f"  中位数: {np.nanmedian(percentage):.1f}%")
    print(f"\n时间占比超过InSight的区域:")
    print(f"  格点数量占比: {grid_percentage_above:.1f}% ({num_above_insight}/{num_total_grids})")
    print(f"  面积权重占比: {area_percentage_above:.1f}%")
    
    # Save details of regions above InSight.
    above_insight_file = os.path.join(output_dir, f"above_insight_threshold_{threshold}.txt")
    with open(above_insight_file, 'w') as f:
        f.write(f"InSight位置 ({actual_lon:.1f}°E, {actual_lat:.1f}°N) 时间占比: {insight_percentage:.2f}%\n")
        f.write(f"阈值: {threshold} Pa\n")
        f.write(f"超过InSight的格点数量: {num_above_insight}/{num_total_grids} ({grid_percentage_above:.2f}%)\n")
        f.write(f"超过InSight的面积权重占比: {area_percentage_above:.2f}%\n\n")
        f.write("超过InSight的格点详情:\n")
        f.write("Latitude, Longitude, Percentage\n")
        
        # Find all cells above InSight and sort from highest to lowest.
        above_indices = np.where(above_insight_mask)
        above_values = percentage[above_indices]
        above_lats = lats[above_indices[1]]
        above_lons = lons[above_indices[0]]
        
        # Sort by percentage.
        sorted_indices = np.argsort(above_values)[::-1]  # Descending order
        for idx in sorted_indices:
            lat = above_lats[idx]
            lon = above_lons[idx]
            perc = above_values[idx]
            f.write(f"{lat:.2f}, {lon:.2f}, {perc:.2f}\n")
    
    print(f"\n结果已保存到: {txt_file}")
    print(f"超过InSight区域的详细信息已保存到: {above_insight_file}")
    
    return percentage

if __name__ == "__main__":
    # File path
    nc_file = "rho_v2_output.nc" if os.path.exists("rho_v2_output.nc") else "../rho_v2_output.nc"
    
    if not os.path.exists(nc_file):
        print(f"错误: 找不到文件 '{nc_file}'")
        print("请将rho_v2_output.nc放在当前目录或上级目录")
    else:
        calculate_rho_v2_percentage(nc_file, THRESHOLD)