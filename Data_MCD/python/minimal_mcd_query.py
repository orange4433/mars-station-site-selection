"""
Minimal one-off query to MCD v6.1 via f2py (fmcd)
"""

import os
import sys
import numpy as np


def resolve_dset() -> str:
    '''
    # 1) Environment variable path
    env_path = os.environ.get("MCD_DATA", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path if env_path.endswith("/") else env_path + "/"

    # 2) Local link/folder named MCD_DATA
    if os.path.isdir("MCD_DATA"):
        return "MCD_DATA/"
    '''
    # 3) Common default used in prior scripts
    default_path = "/mnt/d/pku/science/ljq/data/model data/MCDv6.1_python_interface/python/data/"
    if os.path.isdir(default_path):
        return default_path if default_path.endswith("/") else default_path + "/"

    # 4) Nothing found
    raise RuntimeError(
        "MCD data path not found. Set env var MCD_DATA to your MCD 'data' directory, "
        "or create a local 'MCD_DATA' symlink/folder pointing to it."
    )


def main():
    try:
        import sys
        sys.path.append("/mnt/d/pku/science/ljq/data/model data/MCDv6.1_python_interface/python/data/")
        from fmcd import mcd  # f2py-built module; ensure fmcd.so is present in this directory
    except Exception as e:
        print(
            "ERROR: Could not import 'fmcd'. Ensure the f2py extension is built (fmcd.so) "
            "in interfaces/python. You can use interfaces/python/fmcd_gfortran.sh to build it.\n" 
            f"Import error: {e}"
        )
        sys.exit(1)

    # Exact user-specified configuration
    lon = 135.0            # deg East
    lat = 3.75             # deg North
    datekey = 1            # Mars date
    xdate = 225.0          # Ls in degrees
    loct = 10#135/360*24#10.0            # local time (martian hours)
    zkey = 3               # height above surface (m) key,
    xz = 1               # 1 m above surface
    hrkey = 1              # high-resolution mode
    dust = 1               # climatology, average solar EUV

    # No perturbations; no extra variables requested
    perturkey = 1
    seedin = 0
    gwlength = 0.0
    extvarkeys = np.zeros(100, dtype=np.int32)
    extvarkeys[19] = 1
    # Data path
    dset = resolve_dset()

    # Call MCD
    pres, dens, temp, zonwind, merwind, meanvar, extvar, seedout, ierr = mcd.call_mcd(
        zkey, xz, lon, lat, hrkey,
        datekey, xdate, loct, dset, dust,
        perturkey, seedin, gwlength, extvarkeys,
    )

    # Basic error check (ierr=0 is OK; 17 indicates underground when zkey=3)
    if ierr != 0:
        print(f"Warning: MCD returned ierr={ierr} (non-zero). Values may be invalid.")

    u = float(zonwind)  # W-E wind component (m/s), positive eastward
    v = float(merwind)  # S-N wind component (m/s), positive northward
    hws = float(np.hypot(u, v))  # horizontal wind speed magnitude (m/s)

    print("Query summary:")
    print(f"  Lon (E): {lon}")
    print(f"  Lat (N): {lat}")
    print(f"  Ls: {xdate} deg, Local Time: {loct} h")
    print(f"  zkey=3 (height above surface), z= {xz} m, HR={hrkey}")
    print(f"  Scenario (dust): {dust} -> Climatology average solar, dset: {dset}")
    print("Outputs:")
    print(f"  U (W-E zonal): {u:.6f} m/s")
    print(f"  V (S-N merid.): {v:.6f} m/s")
    print(f"  Horizontal speed: {hws:.6f} m/s")

    # print(f"  extvar[18]: {extvar[19]:.6f}")


if __name__ == "__main__":
    main()
