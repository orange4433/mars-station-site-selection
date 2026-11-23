import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import math
from pathlib import Path
from matplotlib import patches
from scipy.ndimage import gaussian_filter1d

def insight_sol_to_mars_day(insight_sol):
    mars_day = insight_sol + 555
    mars_year_length = 668.6
    if mars_day > mars_year_length:
        mars_day = mars_day % mars_year_length
    return mars_day

def mars_sol2ls(sol):
    year_day = 668.6
    peri_day = 485.35
    e_elips = 0.09340
    timeperi = 1.90258341759902
    
    M = 2 * np.pi * (sol - peri_day) / year_day
    M = M % (2 * np.pi)
    
    E = M
    for _ in range(10):
        delta_E = (E - e_elips * np.sin(E) - M) / (1 - e_elips * np.cos(E))
        E -= delta_E
        if abs(delta_E) < 1e-7:
            break
    
    true_anomaly = 2 * np.arctan(np.sqrt((1 + e_elips) / (1 - e_elips)) * np.tan(E / 2))
    Ls = (true_anomaly - timeperi) % (2 * np.pi)
    return np.degrees(Ls)

def calculate_daily_mean_wind_speed(nc_file_path, lat=None, lon=None):
    nc_file_path = Path(nc_file_path).resolve()
    original = Path.cwd()
    try:
        # netCDF4/HDF5 on Windows can fail on the Unicode parent folder name.
        os.chdir(nc_file_path.parent)
        with xr.open_dataset(nc_file_path.name) as ds:
            wind_speed = np.sqrt(ds.U**2 + ds.V**2)
            if lat is not None and lon is not None:
                data = wind_speed.interp(latitude=lat, longitude=lon, method='linear')
            else:
                weights = np.cos(np.deg2rad(ds.latitude))
                data = wind_speed.weighted(weights).mean(dim=('longitude', 'latitude'))
            daily_mean = data.mean(dim='ut').load()
            return daily_mean.values, daily_mean.Ls.values
    finally:
        os.chdir(original)

def plot_daily_mean_wind_speed(nc_file_path, lat=None, lon=None, 
                              wind_east_file='wind_east.npy', 
                              wind_north_file='wind_north.npy'):
    # Calculate daily mean MCD wind speed and the corresponding Ls values.
    mcd_daily_mean, mcd_Ls_values = calculate_daily_mean_wind_speed(nc_file_path, lat, lon)
    smoothed_mcd = gaussian_filter1d(mcd_daily_mean, sigma=1.5)
    
    # Keep the portrait aspect ratio used by Figure 2a, while reserving room
    # outside the left axis for the Figure 1-style seasonal scale.
    fig, ax = plt.subplots(figsize=(4.45, 9.07))
    # Exact axes box measured from the original 445 x 907 px Figure 2a:
    # x = 106..403 px and y = 17..816 px.
    fig.subplots_adjust(
        left=106 / 445, right=403 / 445,
        top=1 - 17 / 907, bottom=1 - 816 / 907,
    )
    
    # Plot the smoothed MCD curve with Ls on the y-axis and wind speed on the x-axis.
    ax.plot(smoothed_mcd, mcd_Ls_values, 'b-', linewidth=3, label='MCD Model')
    
    # Load observed wind-speed data.
    if os.path.exists(wind_east_file) and os.path.exists(wind_north_file):
        wind_east = np.load(wind_east_file)
        wind_north = np.load(wind_north_file)
        wind_east_subset = wind_east[114:779]
        wind_north_subset = wind_north[114:779]
        wind_speed = np.sqrt(wind_east_subset**2 + wind_north_subset**2)
        
        obs_daily_mean = np.zeros(wind_speed.shape[0]) * np.nan
        for i in range(wind_speed.shape[0]):
            day_data = wind_speed[i, :]
            nan_count = np.isnan(day_data).sum()
            nan_ratio = nan_count / len(day_data)
            if nan_ratio > 0.6:
                obs_daily_mean[i] = np.nan
            else:
                obs_daily_mean[i] = np.nanmean(day_data)
        
        sol_values = np.arange(115, 780)
        mars_days = np.array([insight_sol_to_mars_day(sol) for sol in sol_values])
        obs_Ls_values = np.array([mars_sol2ls(md) for md in mars_days])
        
        # Find contiguous segments of valid data.
        valid_indices = ~np.isnan(obs_daily_mean)
        segments = []
        current_segment = []
        
        for i in range(len(obs_daily_mean)):
            if valid_indices[i]:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        
        if current_segment:
            segments.append(current_segment)
        
        # Track whether a legend label has already been added.
        obs_label_added = False
        
        # Plot only contiguous segments with at least 3 points.
        for segment in segments:
            if len(segment) >= 3:
                segment_data = obs_daily_mean[segment]
                segment_Ls = obs_Ls_values[segment]
                
                # Check whether Ls lies between 225 and 315 degrees.
                in_special_range = np.any((segment_Ls >= 225) & (segment_Ls <= 315))
                
                # Select smoothing parameters for the interval.
                sigma = 2.5 if in_special_range else 1
                smoothed_segment = gaussian_filter1d(segment_data, sigma=sigma)
                
                # Plot the contiguous segment with Ls on the y-axis and wind speed on the x-axis.
                # Add the legend label only once.
                if not obs_label_added:
                    ax.plot(smoothed_segment, segment_Ls, 'r-', linewidth=4, label='InSight Observation')
                    obs_label_added = True
                else:
                    ax.plot(smoothed_segment, segment_Ls, 'r-', linewidth=4)
    
    # Configure the axes.
    ax.set_xlabel('Daily Mean Wind Speed (m/s)', fontdict={'family': 'Arial', 'fontsize': 16})
    ax.set_ylabel('Solar Longitude (Ls)', fontdict={'family': 'Arial', 'fontsize': 16}, labelpad=42)
    
    # Move the y-axis to the left.
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position('left')
    
    # Figure 1 seasonal-bar code, mirrored to the left of Figure 2a.  Because
    # this panel is already plotted directly against Ls, the seasonal
    # boundaries are exactly 0, 90, 180, 270, and 360 degrees.
    season_bounds = np.array([0, 90, 180, 270, 360])
    season_colors = ['green', 'red', 'orange', 'b']
    season_names = ['Spring', 'Summer', 'Fall', 'Winter']
    for start, stop, color, season in zip(
            season_bounds[:-1], season_bounds[1:], season_colors, season_names):
        y_bottom = 1.0 - stop / 360.0
        height = (stop - start) / 360.0
        fig.add_artist(patches.Rectangle(
            (-0.04, y_bottom), width=0.04, height=height,
            alpha=0.3, facecolor=color, edgecolor='none',
            transform=ax.transAxes, clip_on=False))
        ax.text(-0.02, y_bottom + height / 2.0, season,
                transform=ax.transAxes, rotation=90,
                ha='center', va='center', fontfamily='Arial', fontsize=12)

    # Replace the original 45-degree Ls ticks with only the four Figure 1
    # boundary labels.  Their 14-pt Arial font matches the original Figure 2a.
    ax.set_ylim(360, 0)
    ax.set_yticks([])
    for ls_value in season_bounds[:-1]:
        ax.text(-0.055, 1.0 - ls_value / 360.0, f'{ls_value}\N{DEGREE SIGN}',
                transform=ax.transAxes, ha='right', va='center',
                fontfamily='Arial', fontsize=14)
    
    # Set the wind-speed x-axis range to 0 to 11.
    ax.set_xlim(0, 11)
    ax.set_xticks(np.arange(0, 12, 1))
    ax.set_xticklabels(np.arange(0, 12, 1), fontname='Arial', fontsize=14)
    
    # Add a grid.
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.legend(loc='upper right', fontsize=13, framealpha=1,
              facecolor='white', edgecolor='black')
    
    # Adjust the layout so the x-axis label is visible.
    output_file = Path(wind_east_file).resolve().parent / '2(a)' / 'Figure2a.png'
    fig.savefig(output_file, dpi=300, facecolor='white')
    fig.savefig(output_file.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    print(f'Saved: {output_file}')

# Usage example
if __name__ == "__main__":
    figure2_dir = Path(__file__).resolve().parents[1]
    archive_root = figure2_dir.parent
    nc_file = archive_root / 'Data_MCD' / 'mcd_data.nc'
    plot_daily_mean_wind_speed(
        nc_file, lat=4.5, lon=135.6, 
        wind_east_file=figure2_dir / 'wind_east.npy',
        wind_north_file=figure2_dir / 'wind_north.npy'
    )
