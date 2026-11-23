"""Plot the original surface waveform/spectrogram layout with q scaling."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.signal import butter, sosfiltfilt, spectrogram, windows

from dynamic_pressure_common import (
    EARTHQUAKE_POINTS, FREQUENCY_EDGES, OUTPUT, REFERENCE_DISP, REFERENCE_VEL,
    SAMPLE_RATE, SOURCE_DIR, TARGET_SITE, angular_distance_radians, check_inputs,
    earthquake_ratio, load_reference_day, scaled_broadband_window, scaled_windows,
    target_gains,
)
from run_settings import EARTHQUAKE_INDEX, EVENT_MW, TARGET_HOUR, TARGET_SOL

PANEL_SECONDS = 20 * 60
EVENT_DELAY_SECONDS = 50
P_ARRIVAL_SECONDS = 229 + EVENT_DELAY_SECONDS
S_ARRIVAL_SECONDS = 405 + EVENT_DELAY_SECONDS


def add_event(background: np.ndarray, event: np.ndarray, ratio: float) -> np.ndarray:
    result = background.copy()
    count = min(result.size, event.size)
    result[:count] += event[:count] * ratio
    return result


def time_label(hour: float) -> str:
    minutes = int(round((hour % 24) * 60))
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def build_waveform(reference: np.ndarray, gains: np.ndarray, event: np.ndarray, ratio: float) -> np.ndarray:
    shifted = scaled_broadband_window(
        reference, TARGET_HOUR + EVENT_DELAY_SECONDS / 3600, PANEL_SECONDS, gains
    )
    shifted = add_event(shifted, event, ratio)
    prefix = scaled_broadband_window(reference, TARGET_HOUR, EVENT_DELAY_SECONDS, gains)
    waveform = np.concatenate((prefix, shifted))[:int(PANEL_SECONDS * SAMPLE_RATE)]
    # Panel c waveform and detection band are both 0.2--1 Hz.
    bandpass = butter(4, (0.2, 1.0), btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(bandpass, waveform)


def build_spectrum(reference: np.ndarray, gains: np.ndarray, event: np.ndarray, ratio: float):
    shifted = scaled_windows(
        reference, TARGET_HOUR + EVENT_DELAY_SECONDS / 3600, PANEL_SECONDS, gains
    )
    prefix50 = scaled_windows(reference, TARGET_HOUR, EVENT_DELAY_SECONDS, gains)
    prefix20 = scaled_windows(reference, TARGET_HOUR - 20 / 3600, 20, gains)
    suffix20 = scaled_windows(reference, TARGET_HOUR + PANEL_SECONDS / 3600, 20, gains)

    frequency_parts, power_parts = [], []
    time = None
    for band in range(17):
        centre = add_event(shifted[band], event, ratio)
        centre = np.concatenate((prefix50[band], centre))[:int(PANEL_SECONDS * SAMPLE_RATE)]
        signal = np.concatenate((prefix20[band], centre, suffix20[band]))
        frequency, time, power = spectrogram(
            signal, fs=SAMPLE_RATE, window=windows.hamming(512),
            nperseg=512, noverlap=256, scaling="density",
        )
        mask = ((frequency >= 0.9 * FREQUENCY_EDGES[band])
                & (frequency < 1.1 * FREQUENCY_EDGES[band + 1]))
        frequency_parts.append(frequency[mask])
        power_parts.append(power[mask])

    frequency = np.concatenate(frequency_parts)
    power = np.vstack(power_parts)
    order = np.argsort(frequency)
    return TARGET_HOUR * 3600 - 20 + time, frequency[order], power[order]


def plot(waveform: np.ndarray, spectrum: tuple[np.ndarray, np.ndarray, np.ndarray], event_mw: float) -> Path:
    scale = 20
    fig = plt.figure(figsize=(14, 12))
    grid = GridSpec(2, 2, figure=fig, width_ratios=[5, 1], height_ratios=[1, 2])
    wave_ax = fig.add_subplot(grid[0, 0])
    spec_ax = fig.add_subplot(grid[1, :])

    start, end = TARGET_HOUR, TARGET_HOUR + PANEL_SECONDS / 3600
    wave_time = np.linspace(start, end, waveform.size)
    ticks = np.arange(start, end + 0.01, 10 / 60)
    wave_ax.set_xlim(start, end)
    wave_ax.set_xticks(ticks)
    wave_ax.set_xticklabels([time_label(value) for value in ticks])
    wave_ax.plot(wave_time, waveform, color="b")
    wave_ax.axvline(TARGET_HOUR + P_ARRIVAL_SECONDS / 3600, color="g", linestyle=":", linewidth=2)
    wave_ax.axvline(TARGET_HOUR + S_ARRIVAL_SECONDS / 3600, color="orange", linestyle=":", linewidth=2)
    wave_ax.text(TARGET_HOUR + P_ARRIVAL_SECONDS / 3600, 1.01, "P",
                 transform=wave_ax.get_xaxis_transform(), ha="center", fontsize=scale)
    wave_ax.text(TARGET_HOUR + S_ARRIVAL_SECONDS / 3600, 1.01, "S",
                 transform=wave_ax.get_xaxis_transform(), ha="center", fontsize=scale)
    wave_ax.set_ylabel("Amplitude(m/s)", fontsize=scale)
    wave_ax.yaxis.offsetText.set_fontsize(scale - 2)
    wave_ax.tick_params(axis="both", which="major", labelsize=scale - 2)

    time, frequency, power = spectrum
    image = spec_ax.pcolormesh(
        time / 3600, np.log10(frequency[1:]), 10 * np.log10(power[1:]),
        cmap="plasma", shading="auto", vmin=-220, vmax=-170,
    )
    spec_ax.set_xlim(start, end)
    spec_ax.set_xticks(ticks)
    spec_ax.set_xticklabels([time_label(value) for value in ticks])
    freq_ticks = np.array([0.1, 0.3, 1, 3, 8])
    spec_ax.set_ylim(np.log10([freq_ticks[0], freq_ticks[-1]]))
    spec_ax.set_yticks(np.log10(freq_ticks))
    spec_ax.set_yticklabels([str(value) for value in freq_ticks])
    spec_ax.set_xlabel("Local Hour (h)", fontsize=scale)
    spec_ax.set_ylabel("Frequency (Hz)", fontsize=scale)
    spec_ax.tick_params(axis="both", which="major", labelsize=scale - 2)
    # Detection window used for the minimum-magnitude test: 0.2--1 Hz.
    box_start = start + 4 / 60
    box_end = start + 15 / 60
    spec_ax.add_patch(Rectangle(
        (box_start, np.log10(0.2)), box_end - box_start,
        np.log10(1.0) - np.log10(0.2), fill=False,
        edgecolor="white", linewidth=2.5, linestyle=(0, (6, 4)),
    ))
    spec_ax.text(
        0.99, 0.96, "Surface", transform=spec_ax.transAxes,
        ha="right", va="top", fontsize=scale - 2, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.5"),
    )
    spec_ax.text(
        0.01, 0.02, f"Mw {event_mw:.1f}", transform=spec_ax.transAxes,
        ha="left", va="bottom", fontsize=scale - 2, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.5"),
    )
    colorbar = plt.colorbar(image, ax=spec_ax, orientation="vertical", pad=0.04, shrink=0.6)
    colorbar.set_label("Power Spectral Density (dB)", fontsize=scale, labelpad=15)
    colorbar.ax.tick_params(labelsize=scale - 2)

    plt.tight_layout()
    hour_tag = f"{TARGET_HOUR:g}".replace(".", "p")
    output = OUTPUT / f"4c_surface_subplot_sol{TARGET_SOL}_hour{hour_tag}_mw{event_mw:.1f}.png"
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return output


def minimum_mw_from_csv() -> float:
    csv_file = OUTPUT / "minimum_magnitudes_dynamic_pressure.csv"
    if not csv_file.exists():
        raise FileNotFoundError("Run 02_secure_minimum_magnitude.py before script 03")
    target = EARTHQUAKE_POINTS[EARTHQUAKE_INDEX]
    with csv_file.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            csv_moment = (int(row.get("background_sol", -1)),
                          float(row.get("background_lmst_hour", "nan")))
            if csv_moment[0] != int(TARGET_SOL) or not np.isclose(csv_moment[1], TARGET_HOUR):
                raise ValueError("The CSV belongs to another Sol/LMST; rerun script 02 first")
            point = (float(row["earthquake_latitude"]), float(row["earthquake_longitude"]))
            if np.allclose(point, target):
                return float(row["minimum_mw"])
    raise ValueError(f"No minimum Mw found for earthquake {target}")


def main() -> None:
    if not 0 <= EARTHQUAKE_INDEX < len(EARTHQUAKE_POINTS):
        raise ValueError(f"EARTHQUAKE_INDEX must be 0..{len(EARTHQUAKE_POINTS) - 1}")
    check_inputs(
        REFERENCE_DISP, REFERENCE_VEL,
        SOURCE_DIR / "data_detrended_1133.npy",
        SOURCE_DIR / "data_detrended_1133_disp.npy",
    )
    gains, broadband_gains, target_q = target_gains(TARGET_SOL)
    displacement, fs_disp = load_reference_day(REFERENCE_DISP)
    velocity, fs_vel = load_reference_day(REFERENCE_VEL)
    if fs_disp != SAMPLE_RATE or fs_vel != SAMPLE_RATE:
        raise ValueError(f"Expected 20 Hz templates, got {fs_disp:g} and {fs_vel:g} Hz")

    angle = angular_distance_radians(EARTHQUAKE_POINTS[EARTHQUAKE_INDEX], TARGET_SITE)
    event_mw = minimum_mw_from_csv() if EVENT_MW is None else float(EVENT_MW)
    ratio = earthquake_ratio(event_mw, angle)
    waveform = build_waveform(
        velocity, broadband_gains, np.load(SOURCE_DIR / "data_detrended_1133.npy"), ratio
    )
    spectrum = build_spectrum(
        displacement, gains, np.load(SOURCE_DIR / "data_detrended_1133_disp.npy"), ratio
    )
    output = plot(waveform, spectrum, event_mw)
    q_index = int(round((TARGET_HOUR % 24) * 2)) % 48
    print(f"Sol={TARGET_SOL}, LMST={TARGET_HOUR:g}, Mw={event_mw:g}, "
          f"earthquake={EARTHQUAKE_POINTS[EARTHQUAKE_INDEX]}, q={target_q[q_index]:.4f} Pa")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
