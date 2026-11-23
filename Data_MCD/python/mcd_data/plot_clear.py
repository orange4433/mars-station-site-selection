import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import xarray as xr
import os
import math

def adjust_data_for_local_time(u_data, shift_hours=9):
    """
    Shift the ut axis (0-24 hours) forward by shift_hours at each Ls for local-time correction.
    shift_hours=9 represents UTC+9 at 135 degrees E.
    u_data: shape (n_Ls, n_ut) = (721, 48).
    """
    shift_points = shift_hours * 2  # One point per half hour
    # Roll left by shift_points along the ut dimension (axis=1).
    return np.roll(u_data, shift_points, axis=1)

def mars_sol2ls(sol):
    """
    Convert Mars sol to solar longitude (Ls).
    Use precise Mars orbital parameters.
    """
    # Mars orbital parameters
    year_day = 668.6  # Length of a Mars year in sols
    peri_day = 485.35  # Sol at perihelion
    e_elips = 0.09340  # Orbital eccentricity
    timeperi = 1.90258341759902  # Longitude of perihelion in radians
    
    # Calculate the mean anomaly.
    M = 2 * np.pi * (sol - peri_day) / year_day
    M = M % (2 * np.pi)
    
    # Solve Kepler's equation with Newton iteration.
    E = M
    for _ in range(10):
        delta_E = (E - e_elips * np.sin(E) - M) / (1 - e_elips * np.cos(E))
        E -= delta_E
        if abs(delta_E) < 1e-7:
            break
    
    # Calculate the true anomaly.
    true_anomaly = 2 * np.arctan(np.sqrt((1 + e_elips) / (1 - e_elips)) * np.tan(E / 2))
    
    # Calculate solar longitude Ls.
    Ls = (true_anomaly - timeperi) % (2 * np.pi)
    return np.degrees(Ls)

def plot_wind_speed_seasonal_variation(nc_file_path, lat=None, lon=None, min_value=0, max_value=16):
    """
    Plot wind-speed magnitude as a heatmap over local time and season.

    Parameters:
        nc_file_path: NetCDF file path.
        lat: Selected latitude (None for a global mean).
        lon: Selected longitude (None for a global mean).
        min_value: Colorbar minimum.
        max_value: Colorbar maximum.
    """
    # 1. Read the NetCDF file.
    ds = xr.open_dataset(nc_file_path)
    
    # 2. Calculate wind-speed magnitude sqrt(U^2 + V^2).
    wind_speed = np.sqrt(ds.U**2 + ds.V**2)
    
    # 3. Select a location or calculate the global mean.
    if lat is not None and lon is not None:
        # Select a location.
        data = wind_speed.interp(latitude=lat, longitude=lon, method='linear')
    else:
        # Calculate the latitude-weighted global mean.
        weights = np.cos(np.deg2rad(ds.latitude))
        data = wind_speed.weighted(weights).mean(dim=('longitude', 'latitude'))
    
    # 4. Prepare the plotting data.
    # Dimensions: (Ls, ut)
    plot_data = data.transpose('Ls', 'ut').values
    plot_data = adjust_data_for_local_time(plot_data)
    
    # 5. Create the heatmap.
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the colormap.
    norm = colors.Normalize(vmin=min_value, vmax=max_value, clip=True)
    cmap = plt.get_cmap('plasma')
    
    # Create the heatmap with an inverted y-axis.
    im = ax.imshow(plot_data, norm=norm, cmap=cmap, aspect='auto', 
                   extent=[0, 24, 360, 0], origin='upper')
    
    # 6. Add a colorbar.
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Wind Speed (m/s)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    
    # 7. Configure the axes.
    # X-axis: local mean solar time (LMST)
    ax.set_xlabel('Local Mean Solar Time (LMST)', 
                  fontdict={'family': 'Times New Roman', 'fontsize': 14})
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f'{int(h)}:00' for h in np.arange(0, 25, 3)])
    
    # Y-axis: solar longitude (Ls)
    ax.set_ylabel('Solar Longitude (Ls)', 
                  fontdict={'family': 'Times New Roman', 'fontsize': 14})
    
    # Set the y-axis range in descending order.
    ax.set_ylim(360, 0)
    
    # Set Ls ticks to 0, 45, 90, ..., 360 degrees.
    ax.set_yticks(np.arange(0, 361, 45))
    
    # 8. Add a sol axis on the right.
    # Create the right-hand axis.
    ax2 = ax.twinx()
    
    # Define the sol range, 114 to 794.
    sol_min = 114
    sol_max = 794
    
    # Set the sol-axis limits.
    ax2.set_ylim(sol_min, sol_max)
    
    # Calculate the relation between Ls and sol.
    ls_values = np.arange(0, 361, 45)
    sol_values = []
    
    # Find the sol for each Ls by iteration.
    for ls in ls_values:
        # Use bisection to find the corresponding sol.
        low, high = sol_min, sol_max
        for _ in range(20):  # At most 20 iterations
            mid = (low + high) / 2
            mid_ls = mars_sol2ls(mid)
            if abs(mid_ls - ls) < 0.1:
                sol_values.append(mid)
                break
            elif mid_ls < ls:
                low = mid
            else:
                high = mid
        else:
            sol_values.append((low + high) / 2)  # Use the midpoint if no exact match is found.
    
    # Set the right-hand axis ticks.
    ax2.set_yticks(sol_values)
    ax2.set_yticklabels([f"{int(round(s))}" for s in sol_values])
    ax2.set_ylabel('Sol', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    
    # 9. Add a title.
    if lat is not None and lon is not None:
        title = f'Wind Speed at Latitude: {lat}°, Longitude: {lon}°'
    else:
        title = 'Global Average Wind Speed'
    
    ax.set_title(title, fontdict={'family': 'Times New Roman', 'fontsize': 16})
    
    # 10. Adjust the layout.
    plt.tight_layout()
    plt.show()
    
    return plot_data

# Usage example
if __name__ == "__main__":
    # Specify the NetCDF file path.
    nc_file = "./surface_winds_ds_1_LS_0.00-360.00_rev1.1.nc"
    
    # Plot wind speed at a specific location, such as latitude 0, longitude 0.
    print("\nPlotting wind speed at specific location (0°, 0°)...")
    plot_wind_speed_seasonal_variation(nc_file, lat=4.5, lon=135.6)