import xarray as xr
import numpy as np

# Open the NetCDF file.
ds = xr.open_dataset("./surface_winds_ds_1_LS_0.00-180.00_rev1.1.nc")

# Check the data statistics.
print("U statistics:")
print(f"Min: {ds.U.min().values:.6f}")
print(f"Max: {ds.U.max().values:.6f}")
print(f"Mean: {ds.U.mean().values:.6f}")
print(f"Std: {ds.U.std().values:.6f}")

print("\nV statistics:")
print(f"Min: {ds.V.min().values:.6f}")
print(f"Max: {ds.V.max().values:.6f}")
print(f"Mean: {ds.V.mean().values:.6f}")
print(f"Std: {ds.V.std().values:.6f}")

# Check whether all values are very small.
all_small = np.all(np.abs(ds.U.values) < 0.01) and np.all(np.abs(ds.V.values) < 0.01)
print(f"\nAll values are very small (< 0.01 m/s): {all_small}")