import netCDF4 as nc
import numpy as np

# File path
density_file = '../surface_density_ds_1_LS_0.00-360.00_rev1.1.nc'
wind_file = '../mcd_data.nc'
output_file = 'rho_v2_output.nc'

print("读取数据...")
# Read density data.
with nc.Dataset(density_file, 'r') as dens_ds:
    dens = dens_ds.variables['dens'][:]  # Density
    lons = dens_ds.variables['longitude'][:]
    lats = dens_ds.variables['latitude'][:]
    uts = dens_ds.variables['ut'][:]
    Ls = dens_ds.variables['Ls'][:]

# Read wind-speed data.
with nc.Dataset(wind_file, 'r') as wind_ds:
    U = wind_ds.variables['U'][:]  # East-west wind speed
    V = wind_ds.variables['V'][:]  # North-south wind speed

print("计算ρv²...")
# Calculate squared wind-speed magnitude.
v_sq = U**2 + V**2

# Calculate rho v^2.
rho_v2 = dens * v_sq

# Print basic statistics.
print("\n统计信息:")
print(f"最小: {np.nanmin(rho_v2):.6e} Pa")
print(f"最大: {np.nanmax(rho_v2):.6e} Pa")
print(f"中位数: {np.nanmedian(rho_v2):.6e} Pa")
print(f"平均值: {np.nanmean(rho_v2):.6e} Pa")

print("\n保存到文件...")
# Create the output file.
with nc.Dataset(output_file, 'w') as out_ds:
    # Create dimensions.
    out_ds.createDimension('longitude', len(lons))
    out_ds.createDimension('latitude', len(lats))
    out_ds.createDimension('ut', len(uts))
    out_ds.createDimension('Ls', len(Ls))
    
    # Create coordinate variables.
    lon_var = out_ds.createVariable('longitude', 'f8', ('longitude',))
    lon_var[:] = lons
    lon_var.units = 'degrees_east'
    
    lat_var = out_ds.createVariable('latitude', 'f8', ('latitude',))
    lat_var[:] = lats
    lat_var.units = 'degrees_north'
    
    ut_var = out_ds.createVariable('ut', 'f8', ('ut',))
    ut_var[:] = uts
    ut_var.units = 'hours'
    
    Ls_var = out_ds.createVariable('Ls', 'f8', ('Ls',))
    Ls_var[:] = Ls
    Ls_var.units = 'degrees'
    
    # Create the rho v^2 variable.
    rho_v2_var = out_ds.createVariable('rho_v2', 'f8', ('longitude', 'latitude', 'ut', 'Ls'))
    rho_v2_var[:] = rho_v2
    rho_v2_var.units = 'Pa'
    rho_v2_var.long_name = 'Momentum flux term (ρv²)'
    rho_v2_var.description = 'Density times wind speed squared'

print(f"完成！保存到: {output_file}")