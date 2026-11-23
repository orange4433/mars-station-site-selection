import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from pathlib import Path
scale = 16
scale_value = 400
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
    # ... Retain the existing implementation ...
    pass

def adjust_longitude(lons):
    """Adjust longitude values from [-180, 180] to [0, 360]."""
    lons = lons.copy()
    lons[lons < 0] += 360
    return lons

def save_wind_speed_to_txt(wind_speed, lon_subset, lat_subset, filename='wind_speed.txt'):
    """
    Save the wind speed data to a text file.

    Parameters:
    - wind_speed: 2D numpy array of wind speed values.
    - lon_subset: 2D numpy array of longitudes corresponding to wind_speed.
    - lat_subset: 2D numpy array of latitudes corresponding to wind_speed.
    - filename: String, name of the output text file.
    """
    with open(filename, 'w') as f:
        # Write the actual data
        for i in range(len(lat_subset)):
            line = " ".join(f"{ws:.4f}" for ws in wind_speed[i, :])
            f.write(line + "\n")

def plot_local_wind_field_mcd(file_path, ls_value, local_hour, 
                            lat_min=-35, lat_max=45, lon_min=80, lon_max=200,
                            surface_file='surface.nc'):
    """
    Plot the regional wind field within exactly 80 to 200 degrees E.

    Parameters:
        file_path (str): NetCDF file path.
        ls_value (float): Solar longitude (Ls).
        local_hour (int): Local time.
        lat_min (float): Minimum latitude.
        lat_max (float): Maximum latitude.
        lon_min (float): Minimum longitude.
        lon_max (float): Maximum longitude.
        surface_file (str): Topography file path.
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
        u_data = dataset.variables['U'][:, :, :, ls_idx]
        v_data = dataset.variables['V'][:, :, :, ls_idx]
        
        # Calculate universal time at each longitude.
        ut_target = (local_hour + 9 - lons/15) % 24
        
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
            u_extended_data =np.concatenate([u_slice, u_slice[:, 0:1]], axis=1)
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
        
        # Define subset indices over the extended region.
        lat_indices = np.where((lats >= lat_min_extended) & (lats <= lat_max_extended))[0]
        
        # Handle longitude wrapping over the extended region.
        # First part: extended minimum longitude to 180 degrees E.
        lon_indices1 = np.where((lons >= lon_min_extended) & (lons <= 180))[0]
        # Second part: -180 degrees to the extended maximum, expressed as east longitude.
        lon_indices2 = np.where((lons >= -180) & (lons <= (lon_max_extended - 360)))[0]
        
        # Combine the longitude indices.
        lon_indices = np.concatenate([lon_indices1, lon_indices2])
        
        # Crop data to the extended region.
        u_subset = u_interp[lon_indices, :][:, lat_indices]
        v_subset = v_interp[lon_indices, :][:, lat_indices]
        wind_speed_subset = wind_speed[lon_indices, :][:, lat_indices]
        
        # Adjust longitude values so 200 degrees E is displayed correctly.
        lon_subset = lon_grid[lon_indices, :][:, lat_indices]
        lon_subset[lon_subset < 0] += 360  # Convert negative longitude to positive east longitude.
        
        lat_subset = lat_grid[lon_indices, :][:, lat_indices]
    
    # Plot the regional wind field.
    plt.figure(figsize=(10, 5))
    
    # Fill the background using the extended data region.
    contour = plt.contourf(lon_subset, lat_subset, wind_speed_subset, cmap='plasma', 
                          levels=np.linspace(0, 16, 17), extend='both', 
                          vmin=0, vmax=16)
    cbar = plt.colorbar(contour)
    cbar.set_label('Horizontal Wind Speed (m/s)', fontsize=scale)
    cbar.ax.tick_params(labelsize=scale-2)
    
    # Shorten wind arrows by increasing scale.
    stride = 1
    quiver = plt.quiver(lon_subset[::stride, ::stride], lat_subset[::stride, ::stride],
                       u_subset[::stride, ::stride], v_subset[::stride, ::stride],
                       color='cyan', scale=scale_value)  # Increase scale to shorten the arrows.
    
    # Open the topography file and read elevation.
    with nc.Dataset(surface_file, 'r') as surface_dataset:
        lats_surface = surface_dataset.variables['latitude'][:]
        lons_surface = adjust_longitude(surface_dataset.variables['longitude'][:])
        elevation = surface_dataset.variables['zMOL'][:]
        
        # Split the topographic data into two parts.
        mask_lon_part1 = (lons_surface >= lon_min) & (lons_surface <= 180)
        mask_lon_part2 = (lons_surface > 180) & (lons_surface <= lon_max)

        # Calculate latitude and longitude indices for each part.
        lat_indices = np.where((lats_surface >= lat_min) & (lats_surface <= lat_max))[0]
        lon_indices_part1 = np.where(mask_lon_part1)[0]
        lon_indices_part2 = np.where(mask_lon_part2)[0]

        # Create a separate grid for elevation.
        lon_surface_part1, lat_surface_part1 = np.meshgrid(lons_surface[lon_indices_part1], lats_surface[lat_indices])
        lon_surface_part2, lat_surface_part2 = np.meshgrid(lons_surface[lon_indices_part2], lats_surface[lat_indices])

        # Crop elevation to the specified region.
        elevation_subset_part1 = elevation[lat_indices, :][:, lon_indices_part1]
        elevation_subset_part2 = elevation[lat_indices, :][:, lon_indices_part2] if len(lon_indices_part2) > 0 else np.array([])

        # Merge the two elevation subsets, if present.
        if len(elevation_subset_part2) > 0:
            lons_combined_elev = np.concatenate([lons_surface[lon_indices_part1], lons_surface[lon_indices_part2]])
            elevation_data_combined = np.hstack([elevation_subset_part1, elevation_subset_part2])
        else:
            lons_combined_elev = lons_surface[lon_indices_part1]
            elevation_data_combined = elevation_subset_part1

        # Save the merged elevation data to a text file.
        elevation_filename = f'elevation_ls_{ls_value}_local_hour_{local_hour}.txt'
        save_wind_speed_to_txt(elevation_data_combined, lon_surface_part1, lat_surface_part1, filename=elevation_filename)

    # Read the merged elevation text file and draw contours.
    elevation_data_combined = np.loadtxt(elevation_filename)

    # Reverse the elevation latitude dimension because latitude is descending.
    elevation_data_combined = elevation_data_combined[:, :]

    # Draw contours.
    CS = plt.contour(lons_combined_elev, lats_surface[lat_indices], elevation_data_combined, colors=[(0, 0, 0, 0.5)], levels=np.arange(-10, 8.1, 2))
    plt.clabel(CS, inline=True, fontsize=8)  # Add contour labels (optional).
    
    # Keep the red marker at InSight.
    insight_lon, insight_lat = 135.6, 4.5
    plt.scatter(135.6, 4.5, color='red', s=100, 
                             edgecolors='white', linewidth=2, label='InSight Location')
    
    # Add an InSight label in red with a white background.
    plt.annotate(
        'InSight',
        xy=(insight_lon, insight_lat),
        xytext=(insight_lon - 30, insight_lat + 10),  # Adjust the text position.
        fontsize=scale,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='red', alpha=0.9),
        arrowprops=dict(facecolor='red', edgecolor='white', arrowstyle="->", linewidth=2),
    )
    
    plt.xlabel('Longitude', fontsize=scale)
    plt.ylabel('Latitude', fontsize=scale)
    plt.tick_params(axis='both', which='major', labelsize=scale-2)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    
    #plt.title(f'Local Wind Field at Ls={ls_value:.2f} Local Hour={local_hour}', fontsize=scale+2)
    plt.grid(True)
    
    # Display the requested region rather than the extended data region.
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])
    
    # Save the figure.
    filename = Path(__file__).resolve().parent / f'mcd_local_wind_field_ls_{ls_value}_local_hour_{local_hour}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Local wind field saved to {filename}")

# Example call
if __name__ == '__main__':
    figure2_dir = Path(__file__).resolve().parents[1]
    plot_local_wind_field_mcd(
        figure2_dir.parent / 'Data_MCD' / 'mcd_data.nc',
        ls_value=90,
        local_hour=12,
        lat_min=-35,
        lat_max=45,
        lon_min=80,
        lon_max=200,
        surface_file=figure2_dir / 'surface.nc',
    )
