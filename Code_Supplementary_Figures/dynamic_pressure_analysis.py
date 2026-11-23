"""Dynamic-pressure version of the 17-band InSight wind-noise figure.

The script reads the existing source data in this workspace, computes

    q = 0.5 * rho * (u**2 + v**2)

at the InSight site, and writes all derived data and figures beside this file.
No source data are copied or modified.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline


# ------------------------------ configuration ------------------------------

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
OUTPUT = HERE / "output"

DENSITY_FILE = (
    WORKSPACE / "Code_Figure1" / "surface_density_ds_1_LS_0.00-360.00_rev1.1.nc"
)
WIND_EAST_FILE = WORKSPACE / "Code_Figure2" / "wind_east.npy"
WIND_NORTH_FILE = WORKSPACE / "Code_Figure2" / "wind_north.npy"
RMS_DIR = HERE / "data" / "dynamic_pressure_transfer" / "rms"
SUN_FILE = WORKSPACE / "Code_Figure1" / "SunRiseSetTime.npy"

FREQUENCY_EDGES = np.array(
    [0.05, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8]
)
TIME_PERIODS = {
    "00-06": (0.0, 6.0),
    "06-17": (6.0, 17.0),
    "17-24": (17.0, 24.0),
}

INSIGHT_LAT = 4.5
INSIGHT_LON = 135.6
DT_HOUR = 0.01  # 36 s, matching the wind and seismic RMS arrays
FIRST_WIND_SOL = 1
FIRST_RMS_SOL = 113
MAP_START_SOL = 113
MAP_END_SOL = 781  # exclusive
FIT_START_SOL = 220
FIT_END_SOL = 700  # exclusive
Q_BIN_WIDTH_PA = 0.01
Q_FIT_MAX_PA = 2.0
MIN_POINTS_PER_Q_BIN = 100
SPLINE_SMOOTHING = 1.5


# ---------------------------- physical preparation -------------------------

def insight_sol_to_ls(sols: np.ndarray) -> np.ndarray:
    """Convert InSight mission sol to areocentric solar longitude in degrees."""
    mars_sols = np.asarray(sols, dtype=float) + 555.0
    eccentricity = 0.09340
    mean_anomaly = 2.0 * np.pi * np.mod((mars_sols - 485.35) / 668.6, 1.0)

    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(10):
        eccentric_anomaly -= (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))

    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
    )
    return np.degrees(np.mod(true_anomaly - 1.90258341759902, 2.0 * np.pi))


def load_density_at_insight(sols: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Interpolate MCD 1-m density to each requested InSight sol and LMST."""
    # Some Windows netCDF4/HDF5 builds cannot open a Unicode absolute path.
    # Opening the same file through an ASCII relative path avoids that library bug.
    original_directory = Path.cwd()
    try:
        os.chdir(WORKSPACE)
        with Dataset(DENSITY_FILE.relative_to(WORKSPACE)) as dataset:
            longitudes = np.asarray(dataset["longitude"][:])
            latitudes = np.asarray(dataset["latitude"][:])
            mcd_hours = np.asarray(dataset["ut"][:])
            mcd_ls = np.asarray(dataset["Ls"][:])

            lon_index = int(np.abs(longitudes - INSIGHT_LON).argmin())
            lat_index = int(np.abs(latitudes - INSIGHT_LAT).argmin())
            density = np.asarray(dataset["dens"][lon_index, lat_index, :, :])
    finally:
        os.chdir(original_directory)

    # The Figure-1 archive treats this 0--23.5 coordinate as InSight LMST.
    # Add the first half-hour at 24:00 so interpolation is continuous at midnight.
    mcd_hours_cyclic = np.append(mcd_hours, 24.0)
    density_cyclic = np.vstack((density, density[0:1, :]))
    interpolator = RegularGridInterpolator(
        (mcd_hours_cyclic, mcd_ls), density_cyclic, bounds_error=False
    )

    ls_by_sol = insight_sol_to_ls(sols)
    query_hours, query_ls = np.meshgrid(hours, ls_by_sol)
    points = np.column_stack((query_hours.ravel(), query_ls.ravel()))
    density_lmst = interpolator(points).reshape(len(sols), len(hours))
    return density_lmst, float(longitudes[lon_index]), float(latitudes[lat_index])


def build_dynamic_pressure() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build and save q(Sol, LMST) from MCD density and measured InSight wind."""
    sols = np.arange(MAP_START_SOL, MAP_END_SOL)
    hours = np.arange(0.0, 24.0, DT_HOUR)
    density, grid_lon, grid_lat = load_density_at_insight(sols, hours)

    east = np.load(WIND_EAST_FILE, mmap_mode="r")
    north = np.load(WIND_NORTH_FILE, mmap_mode="r")
    start = MAP_START_SOL - FIRST_WIND_SOL
    stop = MAP_END_SOL - FIRST_WIND_SOL
    speed_squared = np.asarray(east[start:stop, : len(hours)]) ** 2
    speed_squared += np.asarray(north[start:stop, : len(hours)]) ** 2

    if density.shape != speed_squared.shape:
        raise ValueError(
            f"Density shape {density.shape} does not match wind shape {speed_squared.shape}."
        )

    dynamic_pressure = 0.5 * density * speed_squared
    OUTPUT.mkdir(exist_ok=True)
    np.save(OUTPUT / "density_insight_1m.npy", density)
    np.save(OUTPUT / "dynamic_pressure.npy", dynamic_pressure)
    np.savez(
        OUTPUT / "dynamic_pressure_coordinates.npz",
        sols=sols,
        lmst_hours=hours,
        ls=insight_sol_to_ls(sols),
        mcd_grid_longitude=grid_lon,
        mcd_grid_latitude=grid_lat,
    )
    return dynamic_pressure, sols, hours, grid_lon, grid_lat


# ------------------------------- map figure --------------------------------

def plot_dynamic_pressure_map(
    dynamic_pressure: np.ndarray,
    sols: np.ndarray,
    hours: np.ndarray,
    grid_lon: float,
    grid_lat: float,
) -> None:
    """Plot the Sol--LMST dynamic-pressure panel corresponding to Figure 1d."""
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    mesh = ax.pcolormesh(
        hours,
        sols,
        dynamic_pressure,
        cmap="plasma",
        norm=colors.Normalize(0.0, 1.0),
        shading="auto",
        rasterized=True,
    )

    sunrise_sunset = np.load(SUN_FILE)
    mask = (sunrise_sunset[:, 0] >= sols[0]) & (sunrise_sunset[:, 0] < sols[-1] + 1)
    sun = sunrise_sunset[mask]
    ax.plot(sun[:, 1], sun[:, 0], "--", color="orange", lw=1.2)
    ax.plot(sun[:, 2], sun[:, 0], "--", color="orange", lw=1.2)
    ax.text(0.25, 1.01, "Sunrise", transform=ax.transAxes, ha="center")
    ax.text(0.75, 1.01, "Sunset", transform=ax.transAxes, ha="center")

    season_offsets = np.array([0, 193, 372, 515, 668])
    season_names = ["Spring", "Summer", "Fall", "Winter"]
    season_colors = ["green", "red", "orange", "royalblue"]
    total = season_offsets[-1]
    for start, stop, name, color in zip(
        season_offsets[:-1], season_offsets[1:], season_names, season_colors
    ):
        y0 = 1.0 - stop / total
        height = (stop - start) / total
        fig.add_artist(
            patches.Rectangle(
                (1.0, y0), 0.035, height, transform=ax.transAxes,
                facecolor=color, alpha=0.3, clip_on=False
            )
        )
        ax.text(1.008, y0 + height / 2.0, name, transform=ax.transAxes,
                rotation=90, va="center", fontsize=11)

    ls_axis = ax.twinx()
    ls_axis.set_ylim(ax.get_ylim())
    ls_axis.set_yticks(MAP_START_SOL + season_offsets[:-1])
    ls_axis.set_yticklabels(["0°", "90°", "180°", "270°"])
    ls_axis.invert_yaxis()
    ax.text(0.99, 1.04, "Ls", transform=ax.transAxes, ha="right", fontsize=12)

    ax.set_xlim(0, 24)
    ax.set_ylim(MAP_START_SOL, MAP_END_SOL)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_yticks(np.arange(200, MAP_END_SOL, 100))
    ax.set_xlabel("Local Mean Solar Time (hour)")
    ax.set_ylabel("Sol (after InSight landing)")
    ax.text(
        0.02, 0.98, r"Dynamic Pressure ($q=\frac{1}{2}\rho v^2$)",
        transform=ax.transAxes, va="top", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", edgecolor="black"),
    )
    ax.text(
        0.02, 0.02, f"MCD 1 m density grid: {grid_lat:.2f}°N, {grid_lon:.2f}°E",
        transform=ax.transAxes, color="white", fontsize=8,
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"),
    )

    cax = ax.inset_axes((0.0, -0.12, 0.42, 0.025))
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    colorbar.set_label("Dynamic pressure (Pa)")
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.18, top=0.94)
    fig.savefig(OUTPUT / "dynamic_pressure_sol_lmst.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "dynamic_pressure_sol_lmst.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------- 17-band relationship ---------------------------

def frequency_label(low: float, high: float) -> str:
    return f"{low:g}-{high:g} Hz"


def modal_curve(q: np.ndarray, log_rms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q-bin centres, modal log10(RMS), and sample counts."""
    edges = np.arange(0.0, Q_FIT_MAX_PA + Q_BIN_WIDTH_PA, Q_BIN_WIDTH_PA)
    centres = 0.5 * (edges[:-1] + edges[1:])
    modes = np.full(centres.shape, np.nan)
    counts = np.zeros(centres.shape, dtype=int)

    q_bin = np.digitize(q, edges) - 1
    for index in range(len(centres)):
        values = log_rms[q_bin == index]
        counts[index] = values.size
        if values.size < MIN_POINTS_PER_Q_BIN:
            continue
        histogram, rms_edges = np.histogram(values, bins=50)
        peak = int(histogram.argmax())
        modes[index] = 0.5 * (rms_edges[peak] + rms_edges[peak + 1])
    return centres, modes, counts


def smooth_modes(q: np.ndarray, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth reliable modal points without extrapolating beyond their q range."""
    valid = np.isfinite(modes)
    if valid.sum() < 4:
        return q[valid], modes[valid]
    q_valid = q[valid]
    rms_valid = modes[valid]
    spline = UnivariateSpline(q_valid, rms_valid, s=SPLINE_SMOOTHING)
    q_smooth = np.linspace(q_valid[0], q_valid[-1], 300)
    return q_smooth, spline(q_smooth)


def plot_period_relationship(
    dynamic_pressure: np.ndarray,
    hours: np.ndarray,
    period_name: str,
    start_hour: float,
    end_hour: float,
) -> None:
    """Create one 17-panel q--noise figure and save its numerical curves."""
    sol_slice = slice(FIT_START_SOL - MAP_START_SOL, FIT_END_SOL - MAP_START_SOL)
    hour_mask = (hours >= start_hour) & (hours < end_hour)
    q_period = dynamic_pressure[sol_slice][:, hour_mask].ravel()

    fig, axes = plt.subplots(6, 3, figsize=(15, 18), constrained_layout=True)
    axes = axes.ravel()
    modal_rows: list[tuple[str, float, float, int]] = []
    curve_columns: dict[str, np.ndarray] = {}
    curve_q = np.linspace(0.0, Q_FIT_MAX_PA, 401)

    for band_index, (low, high) in enumerate(zip(FREQUENCY_EDGES[:-1], FREQUENCY_EDGES[1:])):
        rms_path = RMS_DIR / f"RMSmap_BHZ_{low:g}-{high:g}.npy"
        rms = np.load(rms_path, mmap_mode="r")
        rms_rows = slice(FIT_START_SOL - FIRST_RMS_SOL, FIT_END_SOL - FIRST_RMS_SOL)
        log_rms = np.asarray(rms[rms_rows, : len(hours)][:, hour_mask]).ravel()

        valid = np.isfinite(q_period) & np.isfinite(log_rms) & (log_rms != 0.0)
        q_valid = q_period[valid]
        rms_valid = log_rms[valid]
        in_plot_range = q_valid <= Q_FIT_MAX_PA

        ax = axes[band_index]
        ax.set_facecolor("#050505")
        ax.hexbin(
            q_valid[in_plot_range], rms_valid[in_plot_range], gridsize=55,
            cmap="inferno", mincnt=2, norm=colors.PowerNorm(gamma=1.35),
            rasterized=True
        )

        q_centres, modes, counts = modal_curve(q_valid, rms_valid)
        q_smooth, rms_smooth = smooth_modes(q_centres, modes)
        ax.plot(q_smooth, rms_smooth, color="#28a745", lw=2.0,
                label=frequency_label(low, high))

        column_name = f"log10_rms_{low:g}_{high:g}_hz"
        curve = np.full(curve_q.shape, np.nan)
        if q_smooth.size >= 2:
            inside = (curve_q >= q_smooth[0]) & (curve_q <= q_smooth[-1])
            curve[inside] = np.interp(curve_q[inside], q_smooth, rms_smooth)
        curve_columns[column_name] = curve
        for centre, mode, count in zip(q_centres, modes, counts):
            modal_rows.append((frequency_label(low, high), centre, mode, int(count)))

        ax.set_xlim(0.0, Q_FIT_MAX_PA)
        ax.grid(color="white", alpha=0.25, lw=0.6)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        if band_index % 3 == 0:
            ax.set_ylabel(r"BHZ $\log_{10}$(RMS)")
        if band_index >= 14:
            ax.set_xlabel("Dynamic pressure (Pa)")

    for ax in axes[len(FREQUENCY_EDGES) - 1 :]:
        fig.delaxes(ax)
    fig.suptitle(
        f"InSight dynamic pressure vs seismic noise: {period_name} LMST, "
        f"Sols {FIT_START_SOL}-{FIT_END_SOL - 1}",
        fontsize=15,
    )

    stem = f"dynamic_pressure_noise_17bands_{period_name}"
    fig.savefig(OUTPUT / f"{stem}.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    header = "frequency_band,q_bin_center_pa,modal_log10_rms,count"
    modal_table = np.array(modal_rows, dtype=object)
    np.savetxt(
        OUTPUT / f"modal_points_{period_name}.csv", modal_table,
        fmt=["%s", "%.5f", "%.10g", "%d"], delimiter=",", header=header, comments=""
    )

    curve_table = np.column_stack([curve_q, *curve_columns.values()])
    np.savetxt(
        OUTPUT / f"smoothed_curves_{period_name}.csv", curve_table,
        delimiter=",", header=",".join(["dynamic_pressure_pa", *curve_columns]),
        comments="", fmt="%.10g"
    )


def main() -> None:
    for path in (DENSITY_FILE, WIND_EAST_FILE, WIND_NORTH_FILE, RMS_DIR, SUN_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    dynamic_pressure, sols, hours, grid_lon, grid_lat = build_dynamic_pressure()
    assert dynamic_pressure.shape == (len(sols), len(hours))
    assert np.nanmin(dynamic_pressure) >= 0.0
    assert np.isclose(insight_sol_to_ls(np.array([0.0]))[0], 295.6512, atol=0.01)
    plot_dynamic_pressure_map(dynamic_pressure, sols, hours, grid_lon, grid_lat)
    for name, (start, end) in TIME_PERIODS.items():
        plot_period_relationship(dynamic_pressure, hours, name, start, end)

    finite = dynamic_pressure[np.isfinite(dynamic_pressure)]
    print(f"Saved results to: {OUTPUT}")
    print(f"Dynamic-pressure shape: {dynamic_pressure.shape}")
    print(
        "Dynamic-pressure percentiles (50, 95, 99.5%): "
        f"{np.percentile(finite, [50, 95, 99.5])} Pa"
    )


if __name__ == "__main__":
    main()
