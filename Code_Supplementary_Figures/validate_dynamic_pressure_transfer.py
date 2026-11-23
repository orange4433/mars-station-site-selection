"""Validate the 17-band noise-transfer model with dynamic pressure.

This generates the dynamic-pressure transfer validations for Supplementary Figures 9 and 10.
It transfers the measured Sol 641 BHZ waveform to Sols 600 and 300 using

    gain = f_i(q_target) / f_i(q_reference),

where f_i(q) returns the empirical RMS amplitude for one frequency band and
one of the three LMST periods. Internally, f_i converts the fitted log10(RMS)
curve back to linear RMS amplitude.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
from netCDF4 import Dataset
from obspy import UTCDateTime, read
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, sosfiltfilt, spectrogram, windows

from dynamic_pressure_analysis import (
    DT_HOUR,
    FREQUENCY_EDGES,
    INSIGHT_LAT,
    INSIGHT_LON,
    MAP_END_SOL,
    MAP_START_SOL,
    OUTPUT,
    TIME_PERIODS,
    WORKSPACE,
    insight_sol_to_ls,
)


REFERENCE_SOL = 641
TARGET_SOLS = (600, 300)
SAMPLE_RATE = 20.0
MARS_SOL_SECONDS = 88775.243971
SOL_ZERO_UTC = UTCDateTime("2018-11-26T05:10:50.336037Z")
SPEC_WINDOW = 512
SPEC_OVERLAP = 256
COLOR_LIMITS_DB = (-220.0, -160.0)
FREQUENCY_TICKS = (0.1, 0.3, 1.0, 3.0, 8.0)
GAIN_LIMITS = (0.01, 100.0)

MCD_WIND_FILE = WORKSPACE / "Data_MCD" / "mcd_data.nc"
MCD_DENSITY_FILE = (
    WORKSPACE / "Code_Figure1" / "surface_density_ds_1_LS_0.00-360.00_rev1.1.nc"
)
SEISMIC_DIR = WORKSPACE / "Data_Insight_Seismic" / "Data_Step5_Rotate_DISP"


def read_site_field(nc_path: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one MCD (LMST, Ls) field at the nearest InSight grid point."""
    original_directory = Path.cwd()
    try:
        os.chdir(WORKSPACE)
        with Dataset(nc_path.relative_to(WORKSPACE)) as dataset:
            longitudes = np.asarray(dataset["longitude"][:])
            latitudes = np.asarray(dataset["latitude"][:])
            lon_index = int(np.abs(longitudes - INSIGHT_LON).argmin())
            lat_index = int(np.abs(latitudes - INSIGHT_LAT).argmin())
            field = np.asarray(dataset[variable][lon_index, lat_index, :, :])
            hours = np.asarray(dataset["ut"][:])
            solar_longitudes = np.asarray(dataset["Ls"][:])
    finally:
        os.chdir(original_directory)
    return field, hours, solar_longitudes


def interpolate_mcd_field(field: np.ndarray, mcd_hours: np.ndarray,
                          mcd_ls: np.ndarray, sols: np.ndarray,
                          output_hours: np.ndarray) -> np.ndarray:
    """Interpolate a site field cyclically in LMST and linearly in Ls."""
    hours_cyclic = np.append(mcd_hours, 24.0)
    field_cyclic = np.vstack((field, field[0:1, :]))
    interpolation = RegularGridInterpolator(
        (hours_cyclic, mcd_ls), field_cyclic, bounds_error=False
    )
    ls_by_sol = insight_sol_to_ls(sols)
    query_hour, query_ls = np.meshgrid(output_hours, ls_by_sol)
    points = np.column_stack((query_hour.ravel(), query_ls.ravel()))
    return interpolation(points).reshape(len(sols), len(output_hours))


def load_mcd_dynamic_pressure(sols: np.ndarray, hours: np.ndarray) -> dict[int, np.ndarray]:
    """Calculate MCD-only q at InSight for each requested sol."""
    density, density_hours, density_ls = read_site_field(MCD_DENSITY_FILE, "dens")
    east, wind_hours, wind_ls = read_site_field(MCD_WIND_FILE, "U")
    north, _, _ = read_site_field(MCD_WIND_FILE, "V")

    rho = interpolate_mcd_field(density, density_hours, density_ls, sols, hours)
    u = interpolate_mcd_field(east, wind_hours, wind_ls, sols, hours)
    v = interpolate_mcd_field(north, wind_hours, wind_ls, sols, hours)
    dynamic_pressure = 0.5 * rho * (u**2 + v**2)
    return {int(sol): dynamic_pressure[index] for index, sol in enumerate(sols)}


def load_measured_dynamic_pressure(sols: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Read q made from measured InSight wind and MCD density."""
    data = np.load(OUTPUT / "dynamic_pressure.npy", mmap_mode="r")
    result = {}
    for sol in sols:
        if not MAP_START_SOL <= sol < MAP_END_SOL:
            raise ValueError(f"Sol {sol} is outside the saved q range.")
        result[sol] = np.asarray(data[sol - MAP_START_SOL])
    return result


def load_transfer_curves() -> dict[str, np.ndarray]:
    """Load q and the 17 smoothed log10(RMS) curves for each LMST period."""
    curves = {}
    for period in TIME_PERIODS:
        path = OUTPUT / f"smoothed_curves_{period}.csv"
        curves[period] = np.loadtxt(path, delimiter=",", skiprows=1)
    return curves


def log_rms_from_q(q: np.ndarray, hours: np.ndarray, band_index: int,
                   curves: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate the correct time-period curve, clamping only at supported ends."""
    result = np.full(q.shape, np.nan)
    for period, (start_hour, end_hour) in TIME_PERIODS.items():
        time_mask = (hours >= start_hour) & (hours < end_hour)
        table = curves[period]
        curve_q = table[:, 0]
        curve_rms = table[:, band_index + 1]
        curve_valid = np.isfinite(curve_rms)
        q_valid = time_mask & np.isfinite(q)
        if curve_valid.sum() < 2 or not q_valid.any():
            continue
        result[q_valid] = np.interp(
            q[q_valid], curve_q[curve_valid], curve_rms[curve_valid],
            left=curve_rms[curve_valid][0], right=curve_rms[curve_valid][-1]
        )
    return result


def gain_at_q_resolution(reference_q: np.ndarray, target_q: np.ndarray,
                         hours: np.ndarray, band_index: int,
                         curves: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate f(q_target) / f(q_reference) at 0.01-hour resolution."""
    reference_amplitude = noise_amplitude_from_q(
        reference_q, hours, band_index, curves
    )
    target_amplitude = noise_amplitude_from_q(
        target_q, hours, band_index, curves
    )
    gain = target_amplitude / reference_amplitude
    finite = np.isfinite(gain)
    gain[finite] = np.clip(gain[finite], *GAIN_LIMITS)
    return gain


def noise_amplitude_from_q(q: np.ndarray, hours: np.ndarray, band_index: int,
                           curves: dict[str, np.ndarray]) -> np.ndarray:
    """Return f_i(q): linear RMS amplitude from the fitted log10(RMS) curve."""
    return 10.0 ** log_rms_from_q(q, hours, band_index, curves)


def upsample_gain(gain: np.ndarray, sample_hours: np.ndarray) -> np.ndarray:
    """Linearly upsample each continuous valid block while preserving data gaps."""
    output = np.full(sample_hours.shape, np.nan)
    bin_hours = np.arange(gain.size) * DT_HOUR
    valid_indices = np.flatnonzero(np.isfinite(gain))
    if valid_indices.size == 0:
        return output

    runs = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1)
    for run in runs:
        left = bin_hours[run[0]]
        right = min(24.0, bin_hours[run[-1]] + DT_HOUR)
        sample_mask = (sample_hours >= left) & (sample_hours < right)
        if run.size == 1:
            output[sample_mask] = gain[run[0]]
        else:
            output[sample_mask] = np.interp(
                sample_hours[sample_mask], bin_hours[run], gain[run]
            )
    return output


def read_daily_bhz(sol: int) -> tuple[np.ndarray, np.ndarray]:
    """Read exactly one InSight sol of 20-Hz BHZ displacement data."""
    stream = read(str(SEISMIC_DIR / f"LMST_{sol:04d}.mseed"))
    trace = stream.select(channel="BHZ")[0]
    sol_start = SOL_ZERO_UTC + sol * MARS_SOL_SECONDS
    first = int(round((sol_start - trace.stats.starttime) * SAMPLE_RATE))
    count = int(round(MARS_SOL_SECONDS * SAMPLE_RATE))
    data = np.asarray(trace.data[first:first + count], dtype=float)
    if data.size != count:
        raise ValueError(f"Sol {sol} contains {data.size} BHZ samples; expected {count}.")
    local_hours = np.arange(count) / SAMPLE_RATE / MARS_SOL_SECONDS * 24.0
    return data, local_hours


def compute_spectrogram(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return spectrogram(
        data,
        fs=SAMPLE_RATE,
        window=windows.hamming(SPEC_WINDOW),
        nperseg=SPEC_WINDOW,
        noverlap=SPEC_OVERLAP,
        scaling="density",
    )


def merge_band_spectra(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.concatenate([part[0] for part in parts])
    spectra = np.vstack([part[1] for part in parts])
    order = np.argsort(frequencies)
    return frequencies[order], spectra[order]


def plot_spectrum(ax: plt.Axes, time_seconds: np.ndarray, frequency: np.ndarray,
                  power: np.ndarray, title: str) -> None:
    mask = (frequency >= FREQUENCY_TICKS[0]) & (frequency <= FREQUENCY_TICKS[-1])
    db = 10.0 * np.log10(np.maximum(power[mask], np.finfo(float).tiny))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("white")
    ax.pcolormesh(
        time_seconds / 3600.0, frequency[mask], db,
        cmap=cmap, shading="auto", vmin=COLOR_LIMITS_DB[0], vmax=COLOR_LIMITS_DB[1],
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 24)
    ax.set_ylim(FREQUENCY_TICKS[0], FREQUENCY_TICKS[-1])
    ax.set_xticks(np.arange(0, 25, 1))
    ax.yaxis.set_major_locator(FixedLocator(FREQUENCY_TICKS))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Local hour")
    ax.set_title(title, fontsize=10)


def make_validation_figures() -> None:
    required_sols = np.array([REFERENCE_SOL, *TARGET_SOLS])
    q_hours = np.arange(0.0, 24.0, DT_HOUR)
    mcd_q = load_mcd_dynamic_pressure(required_sols, q_hours)
    measured_q = load_measured_dynamic_pressure(tuple(required_sols))
    curves = load_transfer_curves()
    reference_signal, sample_hours = read_daily_bhz(REFERENCE_SOL)

    cases = {
        (target, source): []
        for target in TARGET_SOLS for source in ("mcd", "measured")
    }
    spectrum_time = None

    for band_index, (low, high) in enumerate(zip(FREQUENCY_EDGES[:-1], FREQUENCY_EDGES[1:])):
        sos = butter(2, [low, high], btype="bandpass", fs=SAMPLE_RATE, output="sos")
        reference_band = sosfiltfilt(sos, reference_signal)

        for target_sol in TARGET_SOLS:
            for source, q_data in (("mcd", mcd_q), ("measured", measured_q)):
                gain_bins = gain_at_q_resolution(
                    q_data[REFERENCE_SOL], q_data[target_sol], q_hours,
                    band_index, curves
                )
                gain_samples = upsample_gain(gain_bins, sample_hours)
                frequency, spectrum_time, power = compute_spectrogram(
                    reference_band * gain_samples
                )
                band_mask = (frequency >= low) & (
                    frequency <= high if high == FREQUENCY_EDGES[-1] else frequency < high
                )
                cases[(target_sol, source)].append((frequency[band_mask], power[band_mask]))

    reference_frequency, reference_time, reference_power = compute_spectrogram(reference_signal)

    for target_sol in TARGET_SOLS:
        mcd_frequency, mcd_power = merge_band_spectra(cases[(target_sol, "mcd")])
        measured_frequency, measured_power = merge_band_spectra(
            cases[(target_sol, "measured")]
        )
        target_signal, _ = read_daily_bhz(target_sol)
        target_frequency, target_time, target_power = compute_spectrogram(target_signal)

        fig, axes = plt.subplots(4, 1, figsize=(15, 10), constrained_layout=True)
        plot_spectrum(
            axes[0], spectrum_time, mcd_frequency, mcd_power,
            f"Adjusted signal using MCD dynamic pressure (Sol {target_sol})"
        )
        plot_spectrum(
            axes[1], spectrum_time, measured_frequency, measured_power,
            f"Adjusted signal using measured dynamic pressure (Sol {target_sol})"
        )
        plot_spectrum(
            axes[2], reference_time, reference_frequency, reference_power,
            f"Reference signal (Sol {REFERENCE_SOL})"
        )
        plot_spectrum(
            axes[3], target_time, target_frequency, target_power,
            f"Raw target signal (Sol {target_sol})"
        )
        for extension, dpi in (("png", 250), ("pdf", 300)):
            fig.savefig(
                OUTPUT / f"dynamic_pressure_validation_sol{target_sol}.{extension}",
                dpi=dpi, bbox_inches="tight"
            )
        plt.close(fig)


def main() -> None:
    required = [
        MCD_WIND_FILE,
        MCD_DENSITY_FILE,
        OUTPUT / "dynamic_pressure.npy",
        *(SEISMIC_DIR / f"LMST_{sol:04d}.mseed" for sol in (REFERENCE_SOL, *TARGET_SOLS)),
        *(OUTPUT / f"smoothed_curves_{period}.csv" for period in TIME_PERIODS),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))

    make_validation_figures()
    print(f"Saved Sol 600 and Sol 300 validation figures to: {OUTPUT}")


if __name__ == "__main__":
    main()
