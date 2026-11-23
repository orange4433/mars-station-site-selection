import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Figure settings
scale = 16
plt.rcParams['font.family'] = 'Arial'  # Set the global font.

# Custom longitude label formatter
def lon_formatter(x, pos):
    if x < 0:
        return f"{-x:.0f}°W"
    elif x > 0:
        return f"{x:.0f}°E"
    else:
        return "0"

# Custom latitude label formatter
def lat_formatter(y, pos):
    if y > 0:
        return f"{y:.0f}°N"
    elif y < 0:
        return f"{-y:.0f}°S"
    else:
        return "0"

def calculate_insolation_point(Ls, local_hour, longitude):
    """
    Calculate the subsolar latitude and longitude for a local time on Mars.
    """
    hour_angle = (local_hour - 12) * 15
    epsilon = math.radians(25.19)
    delta = math.asin(math.sin(math.radians(Ls)) * math.sin(epsilon))
    latitude = math.degrees(delta)
    sub_solar_longitude = longitude - hour_angle
    
    return latitude, sub_solar_longitude

def adjust_longitude(lons):
    """Convert longitude from [-180, 180] to [0, 360] degrees."""
    lons = lons.copy()
    lons[lons < 0] += 360
    return lons

def save_data_to_txt(data, filename='data.txt'):
    """Save data to a text file."""
    with open(filename, 'w') as f:
        for i in range(data.shape[0]):
            line = " ".join(f"{val:.4f}" for val in data[i, :])
            f.write(line + "\n")

def plot_local_rho_v2_field_mcd(file_path, ls_value, local_hour, 
                               lat_min=-35, lat_max=45, lon_min=80, lon_max=200,
                               surface_file='surface.nc', vmax=1.8):
    """
    Plot the regional dynamic-pressure distribution.

    Parameters:
        file_path: Dynamic-pressure data file path.
        ls_value: Solar longitude (Ls).
        local_hour: Mars local time.
        lat_min, lat_max: Latitude limits.
        lon_min, lon_max: Longitude limits.
        surface_file: Topography file path.
        vmax: Maximum dynamic pressure for the colorbar.
    """
    # Read an extended region to avoid blank edges.
    lat_min_extended = lat_min - 5
    lat_max_extended = lat_max + 5
    lon_min_extended = lon_min - 10
    lon_max_extended = lon_max + 10
    
    # Open the NetCDF file.
    with nc.Dataset(file_path, 'r') as dataset:
        lons = dataset.variables['longitude'][:]  # (65,)
        lats = dataset.variables['latitude'][:]   # (49,)
        ut_array = dataset.variables['ut'][:]     # (48,) universal time
        ls_array = dataset.variables['Ls'][:]      # (721,) solar longitude
        # Find the nearest Ls index.
        ls_idx = np.argmin(np.abs(ls_array - ls_value))
        # Read only the requested season; loading all 721 Ls slices wastes ~0.9 GB.
        rho_v2_data = 0.5 * dataset.variables['rho_v2'][:, :, :, ls_idx]
        print(f"Ls value: {ls_array[ls_idx]:.2f}")
        
        # Calculate universal time at each longitude.
        ut_target = (local_hour + 9 - lons/15) % 24
        
        # Initialize the interpolated dynamic-pressure data.
        rho_v2_interp = np.zeros((len(lons), len(lats)))
        
        # Interpolate in time at each longitude.
        for i, lon in enumerate(lons):
            # Target universal time at this longitude
            current_ut = ut_target[i]
            
            # Extend the UT array across the periodic boundary.
            ut_extended = np.append(ut_array, [24])
            
            # Extract the dynamic-pressure slice at this longitude.
            rho_v2_slice = rho_v2_data[i, :, :]  # shape: (latitude, UT)
            
            # Append the final time step to handle periodicity.
            rho_v2_extended_data = np.concatenate([rho_v2_slice, rho_v2_slice[:, 0:1]], axis=1)
            
            # Create the interpolation function.
            f_rho_v2 = interp1d(ut_extended, rho_v2_extended_data, axis=1, kind='linear', 
                               bounds_error=False, fill_value="extrapolate")
            
            # Interpolate.
            rho_v2_interp[i, :] = f_rho_v2(current_ut)
        
        # Create the grid.
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
        
        # Calculate the subsolar point.
        subsolar_lat, subsolar_lon = calculate_insolation_point(ls_value, local_hour, 135)
        if subsolar_lon > 180:
            subsolar_lon -= 360
        elif subsolar_lon < -180:
            subsolar_lon += 360
        
        # Define subset indices over the extended region.
        lat_indices = np.where((lats >= lat_min_extended) & (lats <= lat_max_extended))[0]
        
        # Handle longitude wrapping.
        # First part: extended minimum longitude to 180 degrees E.
        lon_indices1 = np.where((lons >= lon_min_extended) & (lons <= 180))[0]
        # Second part: -180 degrees to the extended maximum, expressed as east longitude.
        lon_indices2 = np.where((lons >= -180) & (lons <= (lon_max_extended - 360)))[0]
        
        # Combine the longitude indices.
        lon_indices = np.concatenate([lon_indices1, lon_indices2])
        
        # Crop data to the extended region.
        rho_v2_subset = rho_v2_interp[lon_indices, :][:, lat_indices]
        
        # Adjust longitude values so 200 degrees E is displayed correctly.
        lon_subset = lon_grid[lon_indices, :][:, lat_indices]
        lon_subset[lon_subset < 0] += 360  # Convert negative longitude to positive east longitude.
        
        lat_subset = lat_grid[lon_indices, :][:, lat_indices]
        
        # Report values at specific locations.
        print("\n" + "="*60)
        print(f"动压范围: [{rho_v2_subset.min():.6f}, {rho_v2_subset.max():.6f}] Pa")
        print(f"局部平均动压: {np.nanmean(rho_v2_subset):.6f} Pa")
        print("="*60)
    
    # Plot the regional dynamic-pressure distribution.
    plt.figure(figsize=(10, 5))
    
    # Calculate dynamic-pressure percentiles for contour levels.
    rho_v2_min = np.nanmin(rho_v2_subset)
    rho_v2_max = np.nanmax(rho_v2_subset)
    
    # Create 20 evenly spaced contour levels.
    levels = np.linspace(0, vmax, 21)
    
    # Fill dynamic-pressure contours using the plasma colormap.
    contour = plt.contourf(lon_subset, lat_subset, rho_v2_subset, 
                          cmap='plasma', 
                          levels=levels, 
                          extend='both',
                          vmin=0, vmax=vmax)
    
    # Add a colorbar.
    cbar = plt.colorbar(contour)
    cbar.set_label('Dynamic Pressure (Pa)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # Open the topography file and read elevation.
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = adjust_longitude(surface_dataset.variables['longitude'][:])
        elevation = surface_dataset.variables['zMOL'][:]
        
        # Split the topographic data into two parts.
        mask_lon_part1 = (lons_surface >= lon_min) & (lons_surface <= 180)
        mask_lon_part2 = (lons_surface > 180) & (lons_surface <= lon_max)

        # Calculate latitude and longitude indices for each part.
        lat_indices_surface = np.where((lats_surface >= lat_min) & (lats_surface <= lat_max))[0]
        lon_indices_part1 = np.where(mask_lon_part1)[0]
        lon_indices_part2 = np.where(mask_lon_part2)[0]

        # Create a separate grid for elevation.
        lon_surface_part1, lat_surface_part1 = np.meshgrid(lons_surface[lon_indices_part1], lats_surface[lat_indices_surface])
        lon_surface_part2, lat_surface_part2 = np.meshgrid(lons_surface[lon_indices_part2], lats_surface[lat_indices_surface])

        # Crop elevation to the specified region.
        elevation_subset_part1 = elevation[lat_indices_surface, :][:, lon_indices_part1]
        elevation_subset_part2 = elevation[lat_indices_surface, :][:, lon_indices_part2] if len(lon_indices_part2) > 0 else np.array([])

        # Merge the two elevation subsets.
        if len(elevation_subset_part2) > 0:
            lons_combined_elev = np.concatenate([lons_surface[lon_indices_part1], lons_surface[lon_indices_part2]])
            elevation_data_combined = np.hstack([elevation_subset_part1, elevation_subset_part2])
        else:
            lons_combined_elev = lons_surface[lon_indices_part1]
            elevation_data_combined = elevation_subset_part1

        # Save topographic data to a text file.
        elevation_filename = f'local_elevation_ls_{ls_value}_local_hour_{local_hour}.txt'
        save_data_to_txt(elevation_data_combined, filename=elevation_filename)
        
        # Read the topography text file and draw contours.
        elevation_data_combined = np.loadtxt(elevation_filename)
        
        # Draw contours.
        CS = plt.contour(lons_combined_elev, lats_surface[lat_indices_surface], elevation_data_combined, 
                        colors=[(0, 0, 0, 0.5)], levels=np.arange(-10, 8.1, 2))
        plt.clabel(CS, inline=True, fontsize=8)  # Add contour labels.
    
    # Mark the InSight location.
    insight_lon, insight_lat = 135.6, 4.5
    plt.scatter(insight_lon, insight_lat, color='red', s=100)
    
    # Add the InSight label.
    plt.annotate(
        'InSight',
        xy=(insight_lon, insight_lat),
        xytext=(insight_lon - 30, insight_lat + 10),  # Adjust the text position.
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    
    # Configure the axes.
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    
    # Apply the custom formatter.
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    #plt.title(f'Local ρv² Distribution at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # Set the axis limits.
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])
    
    # Save the figure.
    filename = Path(__file__).resolve().parent / f'local_rho_v2_field_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save dynamic-pressure data to a text file.
    rho_v2_filename = Path(__file__).resolve().parent / f'local_rho_v2_ls_{ls_value}_local_hour_{local_hour}.txt'
    save_data_to_txt(rho_v2_subset, filename=rho_v2_filename)
    
    print(f"局部动压分布图已保存到: {filename}")
    print(f"动压数据已保存到: {rho_v2_filename}")
    print(f"地形数据已保存到: {elevation_filename}")
    
    return rho_v2_subset, lon_subset, lat_subset

# Example call
if __name__ == "__main__":
    # Parameters:
    # First argument: dynamic-pressure data file.
    # Second argument: solar longitude Ls, 0 to 360 degrees.
    # Third argument: Mars local time, 0 to 23 hours.
    # Fourth through seventh arguments: minimum/maximum latitude and longitude.
    # Eighth argument: topography file.
    # Ninth argument: maximum dynamic pressure for the colorbar.
    
    # Example: Ls = 90 degrees, local noon, latitude -35 to 45, longitude 80 to 200.
    rho_v2_data, lon_grid, lat_grid = plot_local_rho_v2_field_mcd(
        Path(__file__).resolve().parents[1] / 'rho_v2_output.nc',
        ls_value=90,            # Solar longitude
        local_hour=12,          # Local time
        lat_min=-35,            # Minimum latitude
        lat_max=45,             # Maximum latitude
        lon_min=80,             # Minimum longitude
        lon_max=200,            # Maximum longitude
        surface_file=Path(__file__).resolve().parents[1] / 'surface.nc',
        vmax=1.8                # final Figure 2 dynamic-pressure range
    )
    
    # Additional example calls:
    
    # 1. Dynamic pressure at northern summer (Ls = 90 degrees), dusk (18:00).
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
    
    # 2. Dynamic pressure at northern winter (Ls = 270 degrees), noon (12:00).
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
    
    # 3. Dynamic pressure over a smaller region, such as around InSight.
    # rho_v2_data, lon_grid, lat_grid = plot_local_rho_v2_field_mcd(
    #     '../rho_v2_output.nc',
    #     ls_value=90,
    #     local_hour=12,
    #     lat_min=0,      # Minimum latitude
    #     lat_max=20,     # Maximum latitude
    #     lon_min=120,    # Minimum longitude
    #     lon_max=150,    # Maximum longitude
    #     surface_file='surface.nc',
    #     vmax=2
    # )
