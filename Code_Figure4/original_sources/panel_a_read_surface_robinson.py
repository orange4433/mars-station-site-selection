import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import math
# Install Cartopy for the Robinson projection.
# pip install cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Set the global font to Arial ---
plt.rcParams['font.family'] = 'Arial'
scale = 18
def normalize_longitude(lon):
    """Normalize longitude to [-180, 180] degrees."""
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
        
        # Calculate latitude.
        lat = math.asin(math.sin(center_lat_rad)*math.cos(angular_distance_rad) + 
                      math.cos(center_lat_rad)*math.sin(angular_distance_rad)*math.cos(azimuth_rad))
        
        # Calculate longitude.
        lon = center_lon_rad + math.atan2(
            math.sin(azimuth_rad)*math.sin(angular_distance_rad)*math.cos(center_lat_rad),
            math.cos(angular_distance_rad) - math.sin(center_lat_rad)*math.sin(lat)
        )
        
        # Normalize longitude.
        lon_deg = math.degrees(lon)
        lon_deg = normalize_longitude(lon_deg)
        
        lats.append(math.degrees(lat))
        lons.append(lon_deg)
    
    return np.array(lats), np.array(lons)

def plot_circle_with_adaptive_labels(ax, lons, lats, label_text, transform, **kwargs):
    """
    Draw a circle and place its label at a suitable location using Cartopy.

    Parameters:
        ax: Matplotlib axes with a Cartopy projection.
        lons: Longitude array.
        lats: Latitude array.
        label_text: Label text.
        transform: Data coordinate system, such as ccrs.PlateCarree().
        **kwargs: Plotting arguments passed to plot.
    """
    # 1. Split the line where longitude jumps.
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

        # A jump greater than 180 degrees starts a new segment across the dateline.
        if lon_diff > 180:
            segments.append((np.array(current_segment_lons), np.array(current_segment_lats)))
            current_segment_lons = [lons[i]]
            current_segment_lats = [lats[i]]
        else:
            current_segment_lons.append(lons[i])
            current_segment_lats.append(lats[i])

    if current_segment_lons:
        segments.append((np.array(current_segment_lons), np.array(current_segment_lats)))

    # 2. Draw all line segments.
    for seg_lons, seg_lats in segments:
        # Pass transform to plot so Cartopy knows the source coordinate system.
        ax.plot(seg_lons, seg_lats, transform=transform, **kwargs)

    # --- Adjust label placement as needed; retain the core logic ---
    # 3. Label each main segment; here, label only the first.
    if segments and label_text:
        seg_lons, seg_lats = segments[0]
        if len(seg_lons) > 1:
             # Choose a label position unlikely to be obscured by other elements.
             # Use the line midpoint with a small offset.
             mid_idx = len(seg_lons) // 2
             mid_lon = seg_lons[mid_idx]
             mid_lat = seg_lats[mid_idx]
             
             # Try offsetting toward the outside of the line.
             dx = seg_lons[mid_idx] - seg_lons[mid_idx-1] if mid_idx > 0 else 0
             dy = seg_lats[mid_idx] - seg_lats[mid_idx-1] if mid_idx > 0 else 1
             length = np.sqrt(dx**2 + dy**2)
             if length > 0:
                 dx, dy = dx/length, dy/length
             else:
                 dx, dy = 0, 1
             # Vertical offset
             offset_x, offset_y = -dy * 2.0, dx * 2.0 # Offset by 2 degrees.
             label_lon = mid_lon + offset_x
             label_lat = mid_lat + offset_y

             # Keep the label within valid bounds.
             label_lon = normalize_longitude(label_lon)
             label_lat = np.clip(label_lat, -90, 90)

             # Place the label; Cartopy handles the projection.
             ax.text(label_lon, label_lat, label_text,
                     transform=transform, # Label coordinates are also in PlateCarree.
                     fontsize=12, color=kwargs.get('color', 'black'),
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.8),
                     zorder=10)


# === Main program ===

# Open the NetCDF file.
file_path = 'surface.nc' # Verify the file path.
try:
    dataset = nc.Dataset(file_path)
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}。请检查文件路径。")
    # Create an example dataset for demonstration if the file is missing.
    print("正在创建示例数据进行演示...")
    import numpy as np
    lats_demo = np.linspace(-89.5, 89.5, 180)
    lons_demo = np.linspace(-179.5, 179.5, 360)
    lon_grid_demo, lat_grid_demo = np.meshgrid(lons_demo, lats_demo)
    # Create simple synthetic topography with a high center and low edges.
    elevation_demo = np.exp(-((lon_grid_demo / 50.0) ** 2 + (lat_grid_demo / 30.0) ** 2)) * 5000
    # Synthetic variables
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


# Extract longitude and latitude.
lats = dataset.variables['latitude'][:]
lons = dataset.variables['longitude'][:]

# Extract elevation.
elevation = dataset.variables['zMOL'][:]

# Create the grid.
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --- Set up the Robinson projection ---
# Create a figure and axes with a Robinson projection.
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
# Set the map extent to global.
ax.set_global()

# Use a white map background.
ax.set_facecolor('white') # Fix AttributeError.

# Add geographic features (optional background detail).
# ax.add_feature(cfeature.LAND, facecolor='lightgray')
# ax.add_feature(cfeature.OCEAN, facecolor='white')
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)


# Draw topography with the black-to-white gray colormap.
cs_global = ax.contourf(lon_grid, lat_grid, elevation, cmap='gray', levels=30, transform=ccrs.PlateCarree())

# Move the colorbar to the left.
cbar = fig.colorbar(cs_global, ax=ax, orientation='vertical', pad=0.05, location='left', shrink=0.7, aspect=20)
cbar.set_label('Elevation (km)', fontsize=scale, labelpad=10) # Increase labelpad to move the label farther from the colorbar.
cbar.ax.tick_params(labelsize=scale-2)

# Add longitude/latitude gridlines with labels every 60 degrees.
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# Get projection boundary points.
# Place latitude labels along the left boundary.
left_boundary = []
for lat in [-60, -30, 0, 30, 60]:  # Every 60 degrees, excluding 90 degrees
    # Get projected coordinates on the left boundary.
    x, y = ccrs.Robinson().transform_point(-180, lat, ccrs.PlateCarree())
    left_boundary.append((x, y, lat))

# Place longitude labels along the bottom boundary.
bottom_boundary = []
for lon in [-180, -120, -60, 0, 60, 120, 180]:  # Every 60 degrees
    # Get projected coordinates on the bottom boundary.
    x, y = ccrs.Robinson().transform_point(lon, -90, ccrs.PlateCarree())
    bottom_boundary.append((x, y, lon))

# Add latitude labels manually along the left boundary.
for x, y, lat in left_boundary:
    # Format the label text.
    if lat == 0:
        label = "0°"
    else:
        label = f"{abs(lat)}°{'S' if lat < 0 else 'N'}"
    
    # Move left by decreasing x.
    x_adjusted = x - 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    
    # Add labels in black.
    ax.text(x_adjusted, y, label, 
            ha='right', va='center', fontsize=scale-4, color='black',
            transform=ax.transData)

# Add longitude labels manually along the bottom boundary.
for x, y, lon in bottom_boundary:
    # Format the label text.
    if lon == 0:
        label = "0°"
    elif abs(lon) == 180:  # Omit the direction at 180 degrees.
        label = "180°"
    else:
        label = f"{abs(lon)}°{'W' if lon < 0 else 'E'}"
# Move down by decreasing y.
    y_adjusted = y - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])        


    # Add labels in black.
    ax.text(x, y_adjusted, label, 
            ha='center', va='top', fontsize=scale-4, color='black',
            transform=ax.transData)


# Define the coordinate transform.
data_crs = ccrs.PlateCarree() # Coordinate system of the NetCDF data

# === Marsquake annotations ===

# Marsquake locations and magnitudes (unchanged)
earthquake_magnitudes = {
    (35, 90): (3.3, "4.7"), (15, 60): (3.3, "4.7"), (-15, 110): (3.4, "4.8"),
    (10, 160): (3.4, "4.8"), (-20, 150): (3.5, "4.8"), (18.4, -134): (3.0, "4.5"),
    (10, -105): (2.9, "4.4"),
    (-40, 60): (3.3, "4.7"), (-50, -45): (3.0, "4.4"), (-10, -75): (2.9, "4.3")
}
#,  (-10, -120): (3.0, "4.5"),
# Mark marsquakes and their magnitudes in red.
marsquake_lons = [p[1] for p in earthquake_magnitudes.keys()]
marsquake_lats = [p[0] for p in earthquake_magnitudes.keys()]
if earthquake_magnitudes:
    # Use marker size 600 for magnitudes above 3.6, otherwise 400.
    sizes = [650 if mag[0] > 3.6 else 350 for mag in earthquake_magnitudes.values()]
    
    # Use red stars.
    ax.scatter(marsquake_lons, marsquake_lats, color='cyan', s=sizes, marker='*', 
               zorder=5, transform=data_crs, label='Marsquake')
    
    # Add text labels with the revised format and position.
    for (lat, lon), magnitude in earthquake_magnitudes.items():
        main_mag = f"{magnitude[0]:.1f}  "
        slash = "  /  "  # Separator slash
        extra_mag = f"  {magnitude[1]}"
        
        # Get the text renderer to measure widths (optional, more accurate).
        renderer = ax.figure.canvas.get_renderer()
        
        # Draw the main text in yellow first.
        t1 = ax.text(lon, lat - 6, main_mag,
                     fontsize=scale-2, ha='right', va='top',  # Use right alignment.
                     color='orange',weight='bold', transform=data_crs, zorder=6)
        
        # Draw the separator slash in white.
        t2 = ax.text(t1.get_position()[0], lat - 6, slash,
                      fontsize=scale-2, ha='center', va='top',
                      color='white',weight='bold', transform=data_crs, zorder=6)
        
        # Place the pink second part to the right of the slash, left-aligned.
        ax.text(t2.get_position()[0], lat - 6, extra_mag,
                fontsize=scale-2, ha='left', va='top',
                color='#0099ff',weight='bold', transform=data_crs, zorder=6)

# === Seismometer marker color ===
seismometer_points = [(21.1, -43.2)]
seismometer_lons = [p[1] for p in seismometer_points]
seismometer_lats = [p[0] for p in seismometer_points]
if seismometer_points:
    # Draw seismometers in red.
    ax.scatter(seismometer_lons, seismometer_lats, color='white', s=200, 
               marker='o', zorder=4, transform=data_crs, linewidth=2, edgecolor='white')
    ax.scatter(seismometer_lons, seismometer_lats, color='red', s=120, 
               marker='o', zorder=5, transform=data_crs, label='Seismometer')
    for lat, lon in seismometer_points:
        ax.text(lon, lat + 5, 'Seismometer', 
                fontsize=scale, ha='center', va='bottom', 
                color='red', weight='bold',  # Use red.
                transform=data_crs, zorder=6)

# === Add the circle and shadow zone ===
arc_shadow = 1.6 # Set to 1.6.
angle_deg_shadow = arc_shadow * 180 / np.pi
center_point = (21.1, -43.2)

# 1. Draw a circle with angular radius 1.6 radians.
circle_lats, circle_lons = calculate_circle_points(center_point[0], center_point[1], angle_deg_shadow, num_points=720)
plot_circle_with_adaptive_labels(ax, circle_lons, circle_lats, '', data_crs, color='yellow', linewidth=2, linestyle='-')

# 2. Shade the region beyond 1.6 radians.
# Create a dense grid for the shadow mask.
lats_mask = np.linspace(-90, 90, 720)  # Increase resolution for smoother edges.
lons_mask = np.linspace(-180, 180, 1440)
lon_grid_mask, lat_grid_mask = np.meshgrid(lons_mask, lats_mask)

# Calculate grid-to-center great-circle distances with the Haversine formula, in radians.
def haversine_distance_vectorized(lat_grid, lon_grid, center_lat, center_lon):
    """
    Calculate spherical distances from grid points to one point using vectorized operations.
    Input units: degrees.
    Output units: radians.
    """
    lat1_rad = np.radians(lat_grid)
    lon1_rad = np.radians(lon_grid)
    lat2_rad = np.radians(center_lat)
    lon2_rad = np.radians(center_lon)
    
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c # Return radians.

distances_mask = haversine_distance_vectorized(lat_grid_mask, lon_grid_mask, center_point[0], center_point[1])

# Set the shadow mask to 1 beyond arc_shadow, otherwise 0.
shadow_mask = np.where(distances_mask > arc_shadow, 1, 0)

# Draw the shadow zone with contourf on the binary mask.
# levels=[0.5, 1.5] fills cells with values in (0.5, 1.5].
cs_shadow = ax.contourf(lon_grid_mask, lat_grid_mask, shadow_mask, levels=[0.5, 1.5], colors=['pink'], alpha=0.2, transform=data_crs, zorder=2)


# 3. Add the "Shadow Zone" label.
# Choose a suitable label position, such as the opposite side of the circle.
label_angle_rad = np.radians(180 + angle_deg_shadow / 2) # Approximately the center of the shadow zone
label_lat_rad = np.radians(center_point[0])
# Approximate the label position.
label_zone_lat = np.degrees(np.arcsin(np.sin(label_lat_rad)*np.cos(np.radians(angle_deg_shadow + 30)) +
                           np.cos(label_lat_rad)*np.sin(np.radians(angle_deg_shadow + 30))*np.cos(label_angle_rad))+0.02)
label_zone_lon = center_point[1] + np.degrees(np.arctan2(np.sin(label_angle_rad)*np.sin(np.radians(angle_deg_shadow + 30))*np.cos(label_lat_rad),
                                                np.cos(np.radians(angle_deg_shadow + 30)) - np.sin(label_lat_rad)*np.sin(np.radians(label_zone_lat)))+0.1)
label_zone_lon = normalize_longitude(label_zone_lon)
'''
# Add the text label.
ax.text(label_zone_lon, label_zone_lat, 'Shadow Zone', fontsize=16, ha='center', va='center',
        color='darkred', weight='bold', transform=data_crs, zorder=4,
        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='none', alpha=0.7))

'''
# Mark candidate seismometer locations.
# Use one legend entry named "Seismometer".


'''
# Mark marsquake locations and magnitudes.
# Use one legend entry named "Marsquake".
marsquake_lons = [p[1] for p in earthquake_magnitudes.keys()]
marsquake_lats = [p[0] for p in earthquake_magnitudes.keys()]
if earthquake_magnitudes: # Draw markers and add a legend only if points exist.
    ax.scatter(marsquake_lons, marsquake_lats, color='red', s=150, marker='*', zorder=5, transform=data_crs, label='Marsquake')
    # Add labels with the revised color and size; the font is set globally.
    for (lat, lon), magnitude in earthquake_magnitudes.items():
        ax.text(lon, lat, f'{magnitude[0]:.1f}({float(magnitude[1]):.1f})',
                fontsize=14, ha='right', va='bottom', color='yellow', # Update color and size.
                transform=data_crs, zorder=6) # Set zorder.
'''
# Add the legend in the upper right.
from matplotlib.lines import Line2D

# Create custom legend handles.
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan', markersize=22, label='Marsquake'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='white', markeredgewidth=2.5, 
           markerfacecolor='red', markersize=15, label='Seismometer')
]

# Add the custom legend.
legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=scale-2, 
                   framealpha=0.9, edgecolor='#606060', facecolor='#f8f8f8')
# The legend font is set globally.


# Display the figure.
plt.tight_layout()
try:
    plt.show()
except KeyboardInterrupt:
    print("\n绘图被用户中断。")
finally:
    # Close the NetCDF file.
    if hasattr(dataset, 'close'):
        dataset.close()
    print("脚本执行完毕。")




