# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:56:10 2024

@author: 19011
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

plt.rcParams['font.family'] = 'Arial'

# Parameter settings
sol_range = [113, 781]
dt = 0.01
hours = np.arange(0, 24, dt)
size = 20

# Load density data.
file_path = 'surface_density_ds_1_LS_0.00-360.00_rev1.1.nc'
nc_file = Dataset(file_path, 'r')

# Extract data.
dens_data = nc_file.variables['dens'][:]  # Shape: (65, 49, 48, 721)
longitude = nc_file.variables['longitude'][:]  # Shape: (65,)
latitude = nc_file.variables['latitude'][:]  # Shape: (49,)
ut = nc_file.variables['ut'][:]  # Shape: (48,)
Ls = nc_file.variables['Ls'][:]  # Shape: (721,)

# Close the file.
nc_file.close()

# Find the indices nearest InSight (135.6 degrees E, 4.5 degrees N).
lon_idx = np.argmin(np.abs(longitude - 135.6))
lat_idx = np.argmin(np.abs(latitude - 4.5))

# Extract density data at InSight.
dens_insight = dens_data[lon_idx, lat_idx, :, :]  # Shape: (48, 721)

# Define the conversion from solar longitude (Ls) to sol.
def mars_sol2ls(soltabin, forcecontinuity=True):
    year_day = 668.6
    peri_day = 485.35
    e_elips = 0.09340
    radtodeg = 57.2957795130823
    timeperi = 1.90258341759902

    if type(soltabin).__name__ in ['int', 'float', 'float32', 'float64','numpy.int64']:
        soltab = [soltabin]
        solout = np.zeros([1])
    else:
        soltab = soltabin
        solout = np.zeros([len(soltab)])
    
    for i, sol in enumerate(soltab):
        zz = (sol - peri_day) / year_day
        zanom = 2. * np.pi * (zz - np.floor(zz))
        xref = np.abs(zanom)
        
        # Solve zx0 - e * sin(zx0) = xref.
        zx0 = xref + e_elips * np.sin(xref)
        iter_count = 0
        while iter_count <= 10:
            iter_count += 1
            zdx = -(zx0 - e_elips * np.sin(zx0) - xref) / (1. - e_elips * np.cos(zx0))
            if np.abs(zdx) <= 1.e-7:
                break
            zx0 = zx0 + zdx
        
        if zanom < 0.: 
            zx0 = -zx0
        
        # Calculate the true anomaly.
        zteta = 2. * np.arctan(np.sqrt((1. + e_elips) / (1. - e_elips)) * np.tan(zx0 / 2.))
        
        # Calculate Ls.
        ls = zteta - timeperi
        if ls < 0.: 
            ls = ls + 2. * np.pi
        if ls > 2. * np.pi: 
            ls = ls - 2. * np.pi
        
        # Convert to degrees.
        ls = radtodeg * ls
        solout[i] = ls
    
    if forcecontinuity:
        for iii in range(len(soltab) - 1):
            while solout[iii + 1] - solout[iii] > 180.:  
                solout[iii + 1] = solout[iii + 1] - 360.
            while solout[iii] - solout[iii + 1] > 180.:  
                solout[iii + 1] = solout[iii + 1] + 360.
    
    return solout

def insight_sol2ls(soltabin, forcecontinuity=True):
    solzero = 555
    return mars_sol2ls(soltabin + solzero, forcecontinuity=forcecontinuity)

# Create the density matrix.
sols = np.arange(sol_range[0], sol_range[1])
num_sols = len(sols)
num_hours = len(hours)

# Initialize the density matrix.
density_matrix = np.zeros((num_sols, num_hours))

# Calculate density for each sol and hour.
for i, sol in enumerate(sols):
    # Calculate Ls for the current sol.
    sol = float(sol)
    ls_value = insight_sol2ls(sol)
    
    # Find the nearest Ls index.
    ls_idx = np.argmin(np.abs(Ls - ls_value))
    
    # Keep the index within valid bounds.
    if ls_idx >= len(Ls):
        ls_idx = len(Ls) - 1
    
    # Get the 24-hour density profile for this Ls.
    dens_at_ls = dens_insight[:, ls_idx]  # Shape: (48,)
    
    # Create the interpolation function.
    # ut spans 0 to 23.5 hours with 48 samples.
    f = RegularGridInterpolator((ut,), dens_at_ls, bounds_error=False, fill_value=None)
    
    # Interpolate to the required time resolution.
    hours_normalized = hours % 24
    # Map hour 24 to hour 0.
    hours_normalized[hours_normalized == 24] = 0
    
    # Interpolate.
    dens_interpolated = f(hours_normalized.reshape(-1, 1)).flatten()
    
    # Store the result in the matrix.
    density_matrix[i, :] = dens_interpolated

# Plot settings
fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.get_cmap('plasma')
vmin = 0.0  # Minimum density
vmax = 0.025  # Maximum density; adjust to the data range.

# Create the heatmap.
mesh = ax.pcolormesh(
    hours, sols, density_matrix,
    cmap=cmap,
    norm=colors.Normalize(vmin=vmin, vmax=vmax),
    shading='auto'
)

# Seasonal bands
dayleng = np.array([0, 193, 372, 515, 668])
scaleleng = dayleng / dayleng[-1]
lengthscale = np.diff(dayleng) / dayleng[-1]
c = ['green', 'red', 'orange', 'b']
season = ['Spring', 'Summer', 'Fall', 'Winter']
for i in range(len(c)):
    p1 = patches.Rectangle((1., 1 - scaleleng[i + 1]),  # Lower-left corner of the rectangle
                          width=0.035,  # Rectangle width
                          height=lengthscale[i],  # Rectangle height
                          alpha=0.3,  # Rectangle opacity
                          facecolor=c[i],  # Whether to fill the rectangle (green)
                          transform=ax.transAxes)  # Keep the rectangle above the image.
    fig.add_artist(p1)
    plt.text(1.005, 1 - scaleleng[i + 1] + lengthscale[i] / 3, season[i], 
             transform=ax.transAxes, rotation=90, fontsize=14)

# Ls ticks
ls_values = ['0°', '90°', '180°', '270°']  # Ls values
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())

# Place Ls ticks between the seasonal bands.
ls_ticks = [(dayleng[i] + 113) for i in range(len(dayleng) - 1)]
ax2.set_yticks(ls_ticks)
ax2.set_yticklabels(ls_values, fontsize=14)
ax2.invert_yaxis()
plt.text(0.98, 1.04, 'Ls', transform=ax.transAxes,
         fontsize=16, verticalalignment='top', rotation=0)
# Position the Ls label to avoid overlap with the season labels.
ax2.yaxis.set_label_coords(1.08, 0.5)

# Sunrise and sunset curves
SunRiseSetTime = np.load('./SunRiseSetTime.npy')
sols_sun = SunRiseSetTime[:668, 0]
SunRise = SunRiseSetTime[:668, 1]
SunSet = SunRiseSetTime[:668, 2]
plt.plot(SunRise, sols_sun, color='orange', linestyle='--')
plt.plot(SunSet, sols_sun, color='orange', linestyle='--')
plt.text(0.25, 1.01, "Sunrise", transform=ax.transAxes, fontsize=12)
plt.text(0.75, 1.01, "Sunset", transform=ax.transAxes, fontsize=12)

# Axis settings
ax.set_xlim(0, 24)
ax.set_ylim(sol_range[0], sol_range[1])
ax.invert_yaxis()
ax.set_xlabel('Local Mean Solar Time (hour)', fontfamily='Times New Roman', fontsize=16)
ax.set_ylabel('Sol (After InSight landing)', fontfamily='Times New Roman', fontsize=16)
ax.set_xticks(np.arange(0, 25, 2))
ax.set_xticklabels([f'{int(x)}' for x in np.arange(0, 25, 2)], 
                   fontfamily='Times New Roman', fontsize=size - 2)

# Set y-axis tick positions and labels manually.
y_ticks = np.arange(sol_range[0] + 87, sol_range[1] + 1, 100)  # Place a tick every 100 sols.
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks.astype(int), fontfamily='Times New Roman', fontsize=size - 2)

# Add a colorbar.
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap('plasma')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(density_matrix)
cbar = fig.colorbar(sm, cax=ax.inset_axes((0., -0.12, 0.4, 0.02)), 
                    orientation='horizontal', extend='both')
plt.text(0.10, -0.18, 'Density (kg/m³)', transform=ax.transAxes, fontsize=12)

plt.text(0.02, 0.98, 'Surface Atmospheric Density', transform=ax.transAxes,
         fontsize=16, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))

plt.tight_layout()
plt.show()

# Optional: print summary statistics.
print(f"密度数据范围: {density_matrix.min():.6f} 到 {density_matrix.max():.6f} kg/m³")
print(f"密度平均值: {density_matrix.mean():.6f} kg/m³")
print(f"密度矩阵形状: {density_matrix.shape}")
print(f"Sol范围: {sol_range[0]} 到 {sol_range[1]}")
print(f"小时分辨率: {dt} 小时")