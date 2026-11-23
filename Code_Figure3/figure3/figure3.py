from pathlib import Path
import sys

# Use the bundled Windows Cartopy runtime when Cartopy is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Code_Figure4' / '_vendor'))

import matplotlib.pyplot as plt
import numpy as np
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
# Alternatively, specify the complete configuration:
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # Prefer Arial.

def read_data(file_path):
    """Read longitude, latitude, and percentage values from a text file."""
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

# Read data and calculate extrema.
latitudes, longitudes, percentages = read_data('rho_v2_percentage_threshold_0.1.txt')
max_percent = np.ceil(np.max(percentages))  # Round up to the nearest integer.

# Create a nonuniform colormap.
def create_custom_cmap():
    """Create a custom discontinuous colormap."""
    # Define color stops for 0-16% and 16%-maximum.
    segments = {
        'red':   [(0, 0.698, 0.698),   # Start with dark blue.
                 (50/100, 1.0, 1.0),     # White midpoint
                 (1, 0.129, 0.129)],   # End with dark red.
        
        'green': [(0, 0.094, 0.094),   # Start with dark blue.
                 (50/100, 1.0, 1.0),     # White midpoint
                 (1, 0.400, 0.400)],   # End with dark red.
        
        'blue':  [(0, 0.168, 0.168),   # Start with dark blue.
                 (50/100, 1.0, 1.0),     # White midpoint
                 (1, 0.667, 0.667)]    # End with dark red.
    }
    return LinearSegmentedColormap('custom_rdbu', segments)

# Create the color normalization.
cmap = create_custom_cmap()
boundaries = [0, 10,20,30,40,50,60, 70]
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

# Load Mars surface data.
surface_file = 'surface.nc'
with Dataset(surface_file, 'r') as surface_dataset:
    lats_surface = surface_dataset.variables['latitude'][:]
    lons_surface = surface_dataset.variables['longitude'][:]
    elevation = surface_dataset.variables['zMOL'][:]
    
    lon_surface, lat_surface = np.meshgrid(lons_surface, lats_surface)
    elevation_masked = np.ma.masked_where(elevation > 8, elevation)

    # Create a high-resolution grid.
    lon_highres = np.linspace(lons_surface.min(), lons_surface.max(), 300)
    lat_highres = np.linspace(lats_surface.min(), lats_surface.max(), 300)
    lon_grid_highres, lat_grid_highres = np.meshgrid(lon_highres, lat_highres)

    # Interpolate the percentage data.
    points = np.column_stack((latitudes, longitudes))
    grid_percentages_highres = griddata(points, percentages, 
                                      (lat_grid_highres, lon_grid_highres), 
                                      method='linear')

    # Create the figure.
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())  # Use a Robinson projection.

    # Draw filled contours with a PlateCarree data transform.
    mesh = ax.pcolormesh(lon_grid_highres, lat_grid_highres, grid_percentages_highres,
                        cmap=cmap, norm=norm,
                        transform=ccrs.PlateCarree(),  # Keep the coordinate system unchanged.
                        shading='gouraud')
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.04, shrink=0.6,
                        ticks=boundaries)

    # Set the colorbar title font size to scale = 18.
    cbar.set_label('Low Dynamic Pressure Percentage (%)', 
                   fontsize=scale, 
                   labelpad=10, 
                   fontname='Arial')

    # Set the colorbar tick font size to scale - 2 = 16.
    cbar.ax.tick_params(labelsize=scale-2)
    
    # Highlight the 16% tick.
    cbar.ax.axhline(y=16/max_percent, color='k', linewidth=1.2, linestyle='--')

    # ===== Add topographic contours =====
    # Set contour zorder to control label overlap.
    CS = ax.contour(lon_surface, lat_surface, elevation_masked, 
                   colors='#404040', linewidths=0.8,
                   levels=np.arange(-10, 8.1, 2),
                   transform=ccrs.PlateCarree(),
                   zorder=4)  # Above the grid but below point markers
    ax.clabel(CS, inline=True, fontsize=8, fmt='%d km')
    
   
    # Apply Gaussian smoothing to the background data.
    smoothed_percentages = gaussian_filter(grid_percentages_highres, sigma=3)
    
    # ===== Query the value at (4.5, 135.6) =====
    # Query the value at the specified coordinates.
    target_lat = 4.5
    target_lon =135.6

    # Find the nearest grid-point indices.
    lat_idx = np.abs(lat_highres - target_lat).argmin()
    lon_idx = np.abs(lon_highres - target_lon).argmin()

    # Get the value.
    point_value = grid_percentages_highres[lat_idx, lon_idx]

    # Print the result.
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
    # ============= Add candidate landing sites =============
    # Read candidate landing-site data from the document.
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
    
    # Parse the candidate landing sites.
    candidate_lats = []
    candidate_lons = []
    # Change 1: draw the Viking marker in black.
    viking_lats = [22.487, 48.269]
    viking_lons = [-48.041, 134.01]
    
    # Change 2: use yellow for values above 40% and gray for values at or below 40%.
    yellow_points_lats = []
    yellow_points_lons = []
    gray_points_lats = []
    gray_points_lons = []
    
    # Count the points.
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
                
                # Find the grid value at this point.
                lat_idx = np.abs(lat_highres - lat).argmin()
                lon_idx = np.abs(lon_highres - lon).argmin()
                point_value = grid_percentages_highres[lat_idx, lon_idx]
                
                # Select the color based on the value.
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
    
    # Print the counts.
    print(f"\n候选着陆点统计:")
    print(f"  总点数: {total_points}")
    print(f"  黄色点(>40%): {yellow_count} 个")
    print(f"  灰色点(≤40%): {gray_count} 个")
    print(f"  黄色点占比: {yellow_count/total_points*100:.1f}%")
    print(f"  灰色点占比: {gray_count/total_points*100:.1f}%")
    
    # Plot yellow candidate landing sites (above 40%).
    if yellow_points_lats:
        ax.scatter(yellow_points_lons, yellow_points_lats, 
                   transform=ccrs.PlateCarree(),
                   c='yellow', s=40,  # Marker size
                   alpha=0.8,       # Marker opacity
                   edgecolor='k',   # Marker edge color
                   linewidth=0.5,   # Marker edge width
                   zorder=5,       # Keep markers on top.
                   label=f'Tianwen-3 Candidates (>40%)')
    
    # Plot gray candidate landing sites (at or below 40%).
    if gray_points_lats:
        ax.scatter(gray_points_lons, gray_points_lats, 
                   transform=ccrs.PlateCarree(),
                   c='gray', s=40,  # Marker size
                   alpha=0.8,       # Marker opacity
                   edgecolor='k',   # Marker edge color
                   linewidth=0.5,   # Marker edge width
                   zorder=5,       # Keep markers on top.
                   label=f'Tianwen-3 Candidates (≤40%)')
    
    # Add the InSight marker.
    ax.scatter([135.6], [4.5], 
               transform=ccrs.PlateCarree(),
               c='white',       # Bright green
               s=100,           # Larger marker
               alpha=0.9,       # Higher opacity
               edgecolor='k',   # Black edge
               linewidth=1.5,   # Thicker edge
               zorder=10,       # Keep on top.
               label='Insight Site')
    
    # Change 1: draw the Viking marker in black.
    ax.scatter(viking_lons, viking_lats, 
               transform=ccrs.PlateCarree(),
               c='black',       # Use black.
               s=80,           # Larger marker
               alpha=0.9,       # Higher opacity
               edgecolor='k',   # Black edge
               linewidth=1.5,   # Thicker edge
               zorder=10,       # Keep on top.
               label='Viking 1 & 2 Sites')
    
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,
            fontsize=scale, ha='center', va='center')
    ax.text(-0.05, 0.5, 'Latitude', transform=ax.transAxes,
            fontsize=scale, rotation=90, ha='center', va='center')
    
    # Create the legend.
    legend_elements = []
    
    # Add a legend entry for yellow markers.
    if yellow_points_lats:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                      markerfacecolor='yellow', markeredgecolor='k',
                      label=f'Tianwen-3 (>40%)')
        )
    
    # Add a legend entry for gray markers.
    if gray_points_lats:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                      markerfacecolor='gray', markeredgecolor='k',
                      label=f'Tianwen-3 (≤40%)')
        )
    
    # Add legend entries for other markers.
    legend_elements.extend([
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                  markerfacecolor='white', markeredgecolor='k',
                  label='Insight'),
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                  markerfacecolor='black', markeredgecolor='k',  # Use black.
                  label='Viking 1 & 2')
    ])
    
    legend = ax.legend(handles=legend_elements, loc='upper left',
         fontsize=scale-4,                  # Legend font size
         title='',                           # Use an empty string.
         framealpha=0.9,
         edgecolor='#606060', 
         facecolor='#f8f8f8',
         bbox_to_anchor=(-0.07, 1.0)) 
    legend.set_zorder(10)
    
    # Configure gridlines and longitude/latitude labels.
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.6, color='gray', alpha=0.4,
                     linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True          # Enable latitude labels on the left.
    gl.xformatter = LongitudeFormatter(zero_direction_label=True)
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': scale-2, 'color': '#404040'}
    gl.ylabel_style = {'size': scale-2, 'color': '#404040'}

    for label in gl.label_artists:
        if isinstance(label, plt.Text):
            text = label.get_text()
            # Use the Unicode degree symbol directly, or put \u00B0 outside a raw string.
            if text == '60°N' or text == '60°S':
                label.set_visible(False)

    plt.tight_layout()
    
    # Calculate the fraction of the global area that meets the criterion.
    def calculate_global_percentage(grid_data, threshold=20):
        """
        Calculate the fraction of the global grid above the threshold.
        """
        # Mask valid Mars surface cells, excluding NaN and invalid values.
        valid_mask = ~np.isnan(grid_data) & (grid_data >= 0)
        
        # Calculate latitude weights using the cosine correction.
        lat_rad = np.deg2rad(lat_grid_highres)
        weights = np.cos(lat_rad)  # Area weights for a regular longitude-latitude grid
        
        # Identify cells above the threshold.
        threshold_mask = (grid_data > threshold) & valid_mask
        
        # Calculate the weighted fraction.
        total_weighted_area = np.sum(weights[valid_mask])
        threshold_weighted_area = np.sum(weights[threshold_mask])
        
        return (threshold_weighted_area / total_weighted_area) * 100

    # Run the calculation.
    global_percentage = calculate_global_percentage(grid_percentages_highres, 27.8)
    print(f"全球风速<3m/s时间占比>27.8%的区域比例: {global_percentage:.2f}%")
    
    plt.show()
