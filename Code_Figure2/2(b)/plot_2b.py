import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
scale = 18
scale_value = 300
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'  # Set the global font.


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ... Leave the remaining code unchanged ...

# Replace the following plotting section:

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
    # This approximation ignores differences in rotation rate and orbital eccentricity.
    sub_solar_longitude = longitude - hour_angle
    
    return latitude, sub_solar_longitude

def plot_wind_field_mcd(file_path, ls_value, local_hour, surface_file='surface.nc'):
    # Open the NetCDF file.
    with nc.Dataset(file_path, 'r') as dataset:
        lons = dataset.variables['longitude'][:]  # (65,)
        lats = dataset.variables['latitude'][:]   # (49,)
        ut_array = dataset.variables['ut'][:]     # (48,) universal time
        ls_array = dataset.variables['Ls'][:]      # (721,) solar longitude
        # Find the nearest Ls index.
        ls_idx = np.argmin(np.abs(ls_array - ls_value))
        u_data = dataset.variables['U'][:, :, :, ls_idx]
        v_data = dataset.variables['V'][:, :, :, ls_idx]
        print(ls_idx)
        # Calculate universal time at each longitude.
        ut_target = (local_hour +9 - lons/15) % 24
        
        # Initialize the interpolated wind field.
        u_interp = np.zeros((len(lons), len(lats)))
        v_interp = np.zeros((len(lons), len(lats)))
        
        # Interpolate at each longitude.
        for i, lon in enumerate(lons):
            # Target universal time at this longitude
            current_ut = ut_target[i]
            
            # Extend the UT array across the periodic boundary.
            ut_extended = np.append(ut_array, [24])
            
            # Ensure compatible array shapes.
            u_slice = u_data[i, :, :]  # (49, 48)
            v_slice = v_data[i, :, :]  # (49, 48)
            
            # Append the first time step at the end.
            u_extended_data = np.concatenate([u_slice, u_slice[:, 0:1]], axis=1)
            v_extended_data = np.concatenate([v_slice, v_slice[:, 0:1]], axis=1)
            
            # Create the interpolation function.
            f_u = interp1d(ut_extended, u_extended_data, axis=1, kind='linear', 
                          bounds_error=False, fill_value="extrapolate")
            f_v = interp1d(ut_extended, v_extended_data, axis=1, kind='linear', 
                          bounds_error=False, fill_value="extrapolate")
            
            # Interpolate.
            u_interp[i, :] = f_u(current_ut)
            v_interp[i, :] = f_v(current_ut)
        
        # Create the grid.
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
        
        # Calculate wind-speed magnitude.
        wind_speed = np.sqrt(u_interp**2 + v_interp**2)
        
        # ==============================================
        # Report wind information at 3.75 degrees N, 135 degrees E.
        # ==============================================
        target_lat = 3.75  # 3.75 degrees N
        target_lon = 135.0  # 135 degrees E
        
        # Find the nearest longitude and latitude indices.
        lon_idx = np.argmin(np.abs(lons - target_lon))
        lat_idx = np.argmin(np.abs(lats - target_lat))
        
        # Get wind information at this location.
        u_value = u_interp[lon_idx, lat_idx]
        v_value = v_interp[lon_idx, lat_idx]
        wind_speed_value = wind_speed[lon_idx, lat_idx]
        
        # Calculate wind direction in degrees, clockwise from north.
        wind_direction = np.degrees(np.arctan2(-u_value, -v_value)) % 360
        
        # Print results.
        print("\n" + "="*60)
        print(f"Wind information at {target_lat}°N, {target_lon}°E:")
        print(f"U component (east-west): {u_value:.2f} m/s")
        print(f"V component (north-south): {v_value:.2f} m/s")
        print(f"Horizontal wind speed: {wind_speed_value:.2f} m/s")
        print(f"Wind direction: {wind_direction:.1f}° (from north)")
        print("="*60 + "\n")
        
        
    # Plot the wind field.
    plt.figure(figsize=(12, 6))
    
    # Fill the background.
    contour = plt.contourf(lon_grid, lat_grid, wind_speed, cmap='plasma', 
                          levels=np.linspace(0, 16, 17), extend='both', 
                          vmin=0, vmax=16)
    cbar = plt.colorbar(contour)
    cbar.set_label('Horizontal Wind Speed (m/s)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # Plot wind vectors at every second grid point.
    stride = 2
    quiver = plt.quiver(lon_grid[::stride, ::stride], lat_grid[::stride, ::stride],
                       u_interp[::stride, ::stride], v_interp[::stride, ::stride],
                       color='cyan', scale=scale_value)
    
    # Add the vector scale key.
    scale_str = f"Scale: {quiver.scale:.2f}"
    
    # Add topographic contours.
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = surface_dataset.variables['longitude'][:]
        elevation = surface_dataset.variables['zMOL'][:]
        
        # Wrap longitude to -180 to 180 degrees.
        lons_surface = np.where(lons_surface > 180, lons_surface - 360, lons_surface)
        lon_surface, lat_surface = np.meshgrid(lons_surface, lats_surface)
        
        # Draw contours.
        CS = plt.contour(lon_surface, lat_surface, elevation, 
                        colors=[(0, 0, 0, 0.5)], levels=np.arange(-10, 8.1, 2))
        plt.clabel(CS, inline=True, fontsize=5)
    
    # Label InSight in red with a white background.
    plt.annotate(
        'InSight',
        xy=(135.6, 4.5),
        xytext=(135.6 - 30, 4.5 + 15),  # Adjust the text position.
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    insight_scatter = plt.scatter(135.6, 4.5, color='red', s=100, 
                             edgecolors='white', linewidth=2, label='InSight Location')
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
      # For example, 16
    #plt.title(f'Global Wind Field at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # Save the figure.
    filename = Path(__file__).resolve().parent / f'mcd_wind_field_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Wind field saved to {filename}")

# Example call
if __name__ == '__main__':
    figure2_dir = Path(__file__).resolve().parents[1]
    plot_wind_field_mcd(
        figure2_dir.parent / 'Data_MCD' / 'mcd_data.nc',
        ls_value=90,
        local_hour=12,
        surface_file=figure2_dir / 'surface.nc',
    )
