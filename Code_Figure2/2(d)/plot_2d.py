import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Figure settings
scale = 18
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

    Parameters:
        Ls (float): Mars solar longitude in degrees.
        local_hour (int): Local time on a 24-hour clock.
        longitude (float): Longitude in degrees, positive eastward.

    Returns:
        tuple: Subsolar latitude and longitude in degrees.
    """
    # Convert local time to solar hour angle.
    hour_angle = (local_hour - 12) * 15  # Each hour corresponds to 15 degrees.
    
    # Mars's axial tilt (obliquity)
    epsilon = math.radians(25.19)  # Approximately 25.19 degrees
    
    # Solar declination
    delta = math.asin(math.sin(math.radians(Ls)) * math.sin(epsilon))
    
    # Latitude of the subsolar point
    latitude = math.degrees(delta)
    
    # Calculate the hour-angle contribution to subsolar longitude.
    sub_solar_longitude = longitude - hour_angle
    
    return latitude, sub_solar_longitude

def plot_rho_v2_field_mcd(file_path, ls_value, local_hour, surface_file='surface.nc'):
    """
    Plot the global rho v^2 distribution on Mars.

    Parameters:
        file_path: Path to the rho v^2 data file.
        ls_value: Solar longitude (Ls).
        local_hour: Mars local time.
        surface_file: Topography data file path.
    """
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
        # Convert the original universal time (UT) to Mars local time (LT).
        ut_target = (local_hour + 9 - lons/15) % 24
        
        # Initialize the interpolated rho v^2 data.
        rho_v2_interp = np.zeros((len(lons), len(lats)))
        
        # Interpolate in time at each longitude.
        for i, lon in enumerate(lons):
            # Target universal time at this longitude
            current_ut = ut_target[i]
            
            # Extend the UT array across the periodic boundary.
            ut_extended = np.append(ut_array, [24])
            
            # Extract the rho v^2 slice at this longitude (latitude, time).
            rho_v2_slice = rho_v2_data[i, :, :]  # shape: (latitude, UT)
            
            # Append the first time step at the end to handle periodicity.
            rho_v2_extended_data = np.concatenate([rho_v2_slice, rho_v2_slice[:, 0:1]], axis=1)
            
            # Create the interpolation function.
            f_rho_v2 = interp1d(ut_extended, rho_v2_extended_data, axis=1, kind='linear', 
                               bounds_error=False, fill_value="extrapolate")
            
            # Interpolate.
            rho_v2_interp[i, :] = f_rho_v2(current_ut)
        
        # Create the grid.
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
        
        # Wrap longitude to [-180, 180] degrees.
        lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
        
        # Calculate the subsolar point.
        subsolar_lat, subsolar_lon = calculate_insolation_point(ls_value, local_hour, 135)
        # Wrap longitude to [-180, 180] degrees.
        if subsolar_lon > 180:
            subsolar_lon -= 360
        elif subsolar_lon < -180:
            subsolar_lon += 360
        
        # ==============================================
        # Report values at selected locations.
        # ==============================================
        # 1. Report rho v^2 at InSight (4.5 degrees N, 135.6 degrees E).
        target_lat_insight = 4.5
        target_lon_insight = 135.6
        
        # Find the nearest longitude and latitude indices.
        lon_idx_insight = np.argmin(np.abs(lon_grid[:, 0] - target_lon_insight))
        lat_idx_insight = np.argmin(np.abs(lat_grid[0, :] - target_lat_insight))
        
        # Extract rho v^2 at this location.
        rho_v2_insight = rho_v2_interp[lon_idx_insight, lat_idx_insight]
        
        # 2. Report rho v^2 at the subsolar point.
        lon_idx_subsolar = np.argmin(np.abs(lon_grid[:, 0] - subsolar_lon))
        lat_idx_subsolar = np.argmin(np.abs(lat_grid[0, :] - subsolar_lat))
        rho_v2_subsolar = rho_v2_interp[lon_idx_subsolar, lat_idx_subsolar]
        
        # Print results.
        print("\n" + "="*60)
        print(f"太阳直射点位置: {subsolar_lat:.2f}°N, {subsolar_lon:.2f}°E")
        print(f"直射点动压: {rho_v2_subsolar:.6f} Pa")
        print(f"InSight位置 ({target_lat_insight}°N, {target_lon_insight}°E) 动压: {rho_v2_insight:.6f} Pa")
        print(f"全球平均动压: {np.nanmean(rho_v2_interp):.6f} Pa")
        print(f"全球最大动压: {np.nanmax(rho_v2_interp):.6f} Pa at index {np.unravel_index(np.nanargmax(rho_v2_interp), rho_v2_interp.shape)}")
        print(f"全球最小动压: {np.nanmin(rho_v2_interp):.6f} Pa at index {np.unravel_index(np.nanargmin(rho_v2_interp), rho_v2_interp.shape)}")
        print("="*60 + "\n")
    
    # Plot the rho v^2 distribution.
    plt.figure(figsize=(12, 6))
    
    # Calculate rho v^2 percentiles for contour levels.
    rho_v2_min = np.nanmin(rho_v2_interp)
    rho_v2_max = np.nanmax(rho_v2_interp)
    
    # Create 20 evenly spaced contour levels.
    levels = np.linspace(0, 1.8, 19)
    
    # Fill rho v^2 contours using the plasma colormap.
    contour = plt.contourf(lon_grid, lat_grid, rho_v2_interp, 
                          cmap='plasma', 
                          levels=levels, 
                          extend='both',
                          vmin=0, vmax=1.8)  # final Figure 2 dynamic-pressure range
    
    # Add a colorbar.
    cbar = plt.colorbar(contour)
    cbar.set_label('Dynamic Pressure (Pa)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # Add topographic contours.
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = surface_dataset.variables['longitude'][:]
        elevation = surface_dataset.variables['zMOL'][:]
        
        # Wrap longitude to -180 to 180 degrees.
        lons_surface = np.where(lons_surface > 180, lons_surface - 360, lons_surface)
        lon_surface, lat_surface = np.meshgrid(lons_surface, lats_surface)
        
        # Draw topographic contours using the reference settings.
        CS = plt.contour(lon_surface, lat_surface, elevation, 
                        colors=[(0, 0, 0, 0.5)],  # Black, 50% opacity
                        levels=np.arange(-10, 8.1, 2))  # Fixed contour levels
        plt.clabel(CS, inline=True, fontsize=5)
    
    # Mark the InSight location.
    plt.annotate(
        'InSight',
        xy=(135.6, 4.5),
        xytext=(135.6 - 30, 4.5 + 15),  # Adjust the text position.
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    insight_scatter = plt.scatter(135.6, 4.5, color='red', s=100, label='InSight Location')
    
    # Configure the axes.
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    
    # Apply the custom formatter.
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    
    # Set the title.
    #plt.title(f'Global ρv² Distribution at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # Save the figure.
    filename = Path(__file__).resolve().parent / f'rho_v2_distribution_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dynamic-pressure distribution saved to {filename}")
    print(f"Dynamic-pressure range: {rho_v2_min:.6f} to {rho_v2_max:.6f} Pa")
    
    # Return the interpolated data.
    return rho_v2_interp, lon_grid, lat_grid

# Example call
if __name__ == "__main__":
    # Example: plot rho v^2 at Ls = 90 degrees and local noon (12:00).
    rho_v2_data, lon_grid, lat_grid = plot_rho_v2_field_mcd(
        Path(__file__).resolve().parents[1] / 'rho_v2_output.nc',
        ls_value=90,           # Solar longitude
        local_hour=12,
        surface_file=Path(__file__).resolve().parents[1] / 'surface.nc',
    )
