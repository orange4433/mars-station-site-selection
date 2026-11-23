"""Shared data preparation for the three dynamic-pressure scripts."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from obspy import UTCDateTime, read
from scipy.signal import welch


HERE = Path(__file__).resolve().parent
DATA_AND_CODE_ROOT = HERE.parent
OUTPUT = HERE / "output"
CURVE_DIR = HERE / "data" / "transfer_curves"

WIND_FILE = DATA_AND_CODE_ROOT / "Data_MCD" / "mcd_data.nc"
DENSITY_FILE = (
    DATA_AND_CODE_ROOT / "Code_Figure1"
    / "surface_density_ds_1_LS_0.00-360.00_rev1.1.nc"
)
SOURCE_DIR = HERE / "data" / "seismic"
SEISMIC_DATA_DIR = DATA_AND_CODE_ROOT / "Data_Insight_Seismic"
REFERENCE_DISP = SEISMIC_DATA_DIR / "Data_Step5_Rotate_DISP" / "LMST_0641.mseed"
REFERENCE_VEL = SEISMIC_DATA_DIR / "Data_Step5_Rotate" / "LMST_0641.mseed"

FREQUENCY_EDGES = np.array(
    [0.05, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
     0.9, 1, 2, 3, 4, 5, 6, 7, 8]
)
PERIODS = (("00-06", 0, 6), ("06-17", 6, 17), ("17-24", 17, 24))
SOL_RANGE = np.arange(114, 795)
REFERENCE_SOL = 641
INSIGHT_SITE = (4.5, 135.6)
TARGET_SITE = (21.1, -43.2)
SAMPLE_RATE = 20.0

EARTHQUAKE_POINTS = [
    (10, -105), (35, 90), (15, 60), (-15, 110),
    (10, 160), (-20, 150), (18.4, -134), (0, -115),
    (-10, -120), (-40, 60), (-50, -45), (-10, -75),
]


def check_inputs(*paths: Path) -> None:
    root = DATA_AND_CODE_ROOT.resolve()
    outside = [str(path) for path in paths if not path.resolve().is_relative_to(root)]
    if outside:
        raise ValueError("Input outside data-and-code archive:\n" + "\n".join(outside))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input(s):\n" + "\n".join(missing))
    OUTPUT.mkdir(exist_ok=True)


def insight_sol_to_ls(sols: np.ndarray | float) -> np.ndarray:
    mars_sols = np.asarray(sols, dtype=float) + 555.0
    e = 0.09340
    mean = 2 * np.pi * np.mod((mars_sols - 485.35) / 668.6, 1)
    eccentric = mean.copy()
    for _ in range(10):
        eccentric -= (eccentric - e * np.sin(eccentric) - mean) / (1 - e * np.cos(eccentric))
    true = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(eccentric / 2),
        np.sqrt(1 - e) * np.cos(eccentric / 2),
    )
    return np.degrees(np.mod(true - 1.90258341759902, 2 * np.pi))


def _read_site(path: Path, variables: tuple[str, ...], site: tuple[float, float]):
    """Read through an ASCII filename to avoid the Windows netCDF Unicode bug."""
    original = Path.cwd()
    try:
        os.chdir(path.parent)
        with Dataset(path.name) as ds:
            latitudes = np.asarray(ds["latitude"][:])
            longitudes = np.asarray(ds["longitude"][:])
            lat_i = int(np.abs(latitudes - site[0]).argmin())
            lon_i = int(np.abs(longitudes - site[1]).argmin())
            values = {
                name: np.ma.filled(ds[name][lon_i, lat_i, :, :], np.nan)
                for name in variables
            }
            return (
                np.asarray(ds["Ls"][:]), np.asarray(ds["ut"][:]), values,
                float(latitudes[lat_i]), float(longitudes[lon_i]),
            )
    finally:
        os.chdir(original)


def dynamic_pressure(site: tuple[float, float], sols: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Return q(Sol, LMST)=rho*(U**2+V**2)/2 at the nearest MCD grid point."""
    wind_ls, hours, wind, lat, lon = _read_site(WIND_FILE, ("U", "V"), site)
    rho_ls, rho_hours, density, rho_lat, rho_lon = _read_site(DENSITY_FILE, ("dens",), site)
    if not (np.allclose(wind_ls, rho_ls) and np.allclose(hours, rho_hours)):
        raise ValueError("Wind and density coordinates do not match.")
    if (lat, lon) != (rho_lat, rho_lon):
        raise ValueError("Wind and density use different nearest grid points.")

    target_ls = insight_sol_to_ls(sols)
    interpolate = lambda a: np.vstack([np.interp(target_ls, wind_ls, row) for row in a]).T
    u, v, rho = interpolate(wind["U"]), interpolate(wind["V"]), interpolate(density["dens"])
    return 0.5 * rho * (u**2 + v**2), hours, (lat, lon)


class TransferCurves:
    """The 17 fitted dynamic-pressure transfer curves bundled with this folder."""

    def __init__(self):
        self.tables = {
            name: np.loadtxt(CURVE_DIR / f"smoothed_curves_{name}.csv", delimiter=",", skiprows=1)
            for name, _, _ in PERIODS
        }

    @staticmethod
    def period(hour: float) -> str:
        hour %= 24
        return next(name for name, start, end in PERIODS if start <= hour < end)

    def amplitude(self, q: float, band: int, hour: float) -> float:
        table = self.tables[self.period(hour)]
        valid = np.isfinite(table[:, 0]) & np.isfinite(table[:, band + 1])
        return float(10 ** np.interp(q, table[valid, 0], table[valid, band + 1]))

    def gains(self, reference_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        result = np.empty((17, 48))
        for band in range(17):
            for half_hour in range(48):
                hour = half_hour / 2
                result[band, half_hour] = (
                    self.amplitude(target_q[half_hour], band, hour)
                    / self.amplitude(reference_q[half_hour], band, hour)
                )
        return np.clip(result, 0.01, 100)

    def broadband_gains(self, reference_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        """RSS combination of the fitted 0.2--1 Hz bands used by the waveform."""
        result = np.empty(48)
        bands = range(2, 10)
        for half_hour in range(48):
            hour = half_hour / 2
            ref = [self.amplitude(reference_q[half_hour], i, hour) for i in bands]
            target = [self.amplitude(target_q[half_hour], i, hour) for i in bands]
            result[half_hour] = np.linalg.norm(target) / np.linalg.norm(ref)
        return np.clip(result, 0.01, 100)


def target_gains(sol: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    check_inputs(WIND_FILE, DENSITY_FILE, *(CURVE_DIR / f"smoothed_curves_{name}.csv" for name, _, _ in PERIODS))
    reference_q, _, _ = dynamic_pressure(INSIGHT_SITE, np.array([REFERENCE_SOL]))
    target_q, _, _ = dynamic_pressure(TARGET_SITE, np.array([sol]))
    curves = TransferCurves()
    return curves.gains(reference_q[0], target_q[0]), curves.broadband_gains(reference_q[0], target_q[0]), target_q[0]


def lmst_start_utc(sol: int) -> UTCDateTime:
    reference = UTCDateTime("2018-11-26T05:10:50.336037Z")
    sol1 = UTCDateTime("2018-11-27T05:50:25.580014Z")
    sol2 = UTCDateTime("2018-11-28T06:30:00.823990Z")
    return reference + sol * (sol2 - sol1 - 0.000005)


def load_reference_day(path: Path) -> tuple[np.ndarray, float]:
    trace = read(str(path)).select(channel="BHZ")[0]
    fs = float(trace.stats.sampling_rate)
    start = int(round((lmst_start_utc(REFERENCE_SOL) - trace.stats.starttime) * fs))
    count = int(24 * 3600 * fs)
    data = np.asarray(trace.data[start:start + count], dtype=float)
    if data.size != count:
        raise ValueError(f"Sol {REFERENCE_SOL} is incomplete in {path}")
    return data, fs


def scaled_windows(reference: np.ndarray, start_hour: float, seconds: float, gains: np.ndarray, fs: float = SAMPLE_RATE) -> list[np.ndarray]:
    """Old adjuster logic: 17 copies of one template, each scaled by its f(q) gain."""
    count = int(round(seconds * fs))
    absolute = start_hour * 3600 * fs + np.arange(count)
    indices = np.mod(np.rint(absolute).astype(np.int64), reference.size)
    half_hours = np.floor(np.mod(absolute / fs / 3600, 24) * 2).astype(int)
    raw = reference[indices]
    return [raw * gains[band, half_hours] for band in range(17)]


def scaled_broadband_window(reference: np.ndarray, start_hour: float, seconds: float, gains: np.ndarray, fs: float = SAMPLE_RATE) -> np.ndarray:
    count = int(round(seconds * fs))
    absolute = start_hour * 3600 * fs + np.arange(count)
    indices = np.mod(np.rint(absolute).astype(np.int64), reference.size)
    half_hours = np.floor(np.mod(absolute / fs / 3600, 24) * 2).astype(int)
    return reference[indices] * gains[half_hours]


def old_style_noise_asd(reference: np.ndarray, target_hour: float, gains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the old one-hour, 17-band Welch PSD assembly."""
    windows = scaled_windows(reference, target_hour - 0.5, 3600, gains)
    frequency_pieces, pieces = [], []
    for band, signal in enumerate(windows):
        frequency, psd = welch(signal, fs=SAMPLE_RATE, window="hann", nperseg=512, noverlap=256, scaling="density")
        mask = (frequency >= FREQUENCY_EDGES[band]) & (frequency < FREQUENCY_EDGES[band + 1])
        pieces.append(psd[mask])
        frequency_pieces.append(frequency[mask])
    return np.concatenate(frequency_pieces), np.sqrt(np.concatenate(pieces))


def angular_distance_radians(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def earthquake_ratio(mw: float, angle_radians: float) -> float:
    return (30.2 / math.degrees(angle_radians)) * 10 ** (1.5 * (mw - 3.41))


def self_check() -> None:
    assert abs(math.degrees(angular_distance_radians((0, 0), (0, 90))) - 90) < 1e-10


if __name__ == "__main__":
    self_check()
    print("dynamic_pressure_common self-check passed")
