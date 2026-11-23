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
    
    '''
    # 3. Select a location or calculate the global mean.
    if lat is not None and lon is not None:
        # Select a location.
        data = wind_speed.interp(latitude=lat, longitude=lon, method='linear')
    else:
        # Calculate the latitude-weighted global mean.
        weights = np.cos(np.deg2rad(ds.latitude))
        data = wind_speed.weighted(weights).mean(dim=('longitude', 'latitude'))
    '''
    u_data = ds.U.sel(latitude=3.75, longitude=135, method='nearest')
    v_data = ds.V.sel(latitude=3.75, longitude=135, method='nearest')
    data = wind_speed.sel(latitude=3.75, longitude=135, method='nearest')
    # 4. Prepare the plotting data.
    # Dimensions: (Ls, ut)
    plot_data = data.transpose('Ls', 'ut').values
    u_data = u_data.transpose('Ls', 'ut').values
    v_data = v_data.transpose('Ls', 'ut').values
    #plot_data = adjust_data_for_local_time(plot_data)
    #u_data =adjust_data_for_local_time(u_data)
    #v_data = adjust_data_for_local_time(v_data)
    print(plot_data[180][22])
    print(u_data[180][22])
    print(v_data[180][22])
    # 5. Create the heatmap.
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the colormap.
    norm = colors.Normalize(vmin=min_value, vmax=max_value, clip=True)
    cmap = plt.get_cmap('plasma')
    
    # Create the heatmap.
    im = ax.imshow(plot_data, norm=norm, cmap=cmap, aspect='auto', 
                   extent=[0, 24, ds.Ls.min(), ds.Ls.max()], origin='lower')
    
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
    
    # Add season markers.
    seasons = {
        0: 'Vernal Equinox',
        90: 'Summer Solstice',
        180: 'Autumnal Equinox',
        270: 'Winter Solstice'
    }
    
    for ls, season in seasons.items():
        if ls <= ds.Ls.max():
            ax.axhline(y=ls, color='gray', linestyle='--', alpha=0.5)
            ax.text(24.5, ls, season, ha='left', va='center', fontsize=10)
    
    # 8. Add a title.
    if lat is not None and lon is not None:
        title = f'Wind Speed at Latitude: {lat}°, Longitude: {lon}°'
    else:
        title = 'Global Average Wind Speed'
    
    ax.set_title(title, fontdict={'family': 'Times New Roman', 'fontsize': 16})
    
    # 9. Adjust the layout.
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
    