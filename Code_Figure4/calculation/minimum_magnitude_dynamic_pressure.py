"""Search every site's minimum Mw with the original detection judgment."""

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dynamic_pressure_common import (
    EARTHQUAKE_POINTS, OUTPUT, REFERENCE_DISP, SOURCE_DIR, TARGET_SITE,
    angular_distance_radians, check_inputs, earthquake_ratio, load_reference_day,
    old_style_noise_asd, target_gains,
)
from run_settings import TARGET_HOUR, TARGET_SOL


MODEL_FILE = SOURCE_DIR / "model_asd_data_final.txt"


def is_visible(mw: float, angle: float, noise_frequency: np.ndarray, noise_asd: np.ndarray) -> tuple[bool, float, float]:
    frequencies, base_log10_asd = np.loadtxt(MODEL_FILE)
    scaled_log10_asd = base_log10_asd + np.log10(earthquake_ratio(mw, angle))

    # These masks and the 50% test are deliberately unchanged from the old code.
    valid = ((frequencies >= 0.1) & (frequencies <= 10)
             & (scaled_log10_asd >= -15) & (scaled_log10_asd <= -6))
    seismic_band = valid & (frequencies >= 0.2) & (frequencies <= 1)
    noise_band = (noise_frequency >= 0.2) & (noise_frequency <= 1)
    mean_seismic = float(np.mean(10 ** scaled_log10_asd[seismic_band]))
    mean_noise = float(np.mean(noise_asd[noise_band]))
    return mean_seismic * 0.5 > mean_noise, mean_seismic, mean_noise


def minimum_magnitude(point: tuple[float, float], noise_frequency: np.ndarray, noise_asd: np.ndarray) -> dict:
    angle = angular_distance_radians(point, TARGET_SITE)
    mw = 3.3
    visible = True
    attempts = 0
    while visible:  # Original search: start 3.3 and decrease by 0.01.
        visible, mean_seismic, mean_noise = is_visible(mw, angle, noise_frequency, noise_asd)
        mw -= 0.01
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Minimum-Mw loop did not converge")

    mw += 0.02  # Original recovery of the last detectable 0.01 step.
    uncorrected_mw = mw
    if angle > 1.65:  # Original shadow-zone correction, unchanged.
        ratio = 2 * np.sin(angle / 4) / np.sin(angle / 2)
        mw += (2 / 3) * np.log10(ratio)
        mw -= (2 / 3) * np.log10(0.6)

    return {
        "earthquake_latitude": point[0],
        "earthquake_longitude": point[1],
        "distance_deg": math.degrees(angle),
        "minimum_mw_before_shadow_correction": uncorrected_mw,
        "minimum_mw": mw,
        "shadow_zone_corrected": angle > 1.65,
        "mean_noise_asd_0p2_1hz": mean_noise,
        "last_tested_seismic_asd_0p2_1hz": mean_seismic,
    }


def main() -> None:
    check_inputs(REFERENCE_DISP, MODEL_FILE)
    sol, hour = int(TARGET_SOL), float(TARGET_HOUR)
    gains, _, _ = target_gains(sol)
    reference, fs = load_reference_day(REFERENCE_DISP)
    if fs != 20:
        raise ValueError(f"Expected 20 Hz Sol-641 data, got {fs:g} Hz")
    noise_frequency, noise_asd = old_style_noise_asd(reference, hour, gains)
    rows = [minimum_magnitude(point, noise_frequency, noise_asd) for point in EARTHQUAKE_POINTS]
    for row in rows:
        row["background_sol"] = sol
        row["background_lmst_hour"] = hour

    output = OUTPUT / "minimum_magnitudes_dynamic_pressure.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dynamic-pressure background: Sol {sol}, LMST {hour:g} h")
    for row in rows:
        print(f"({row['earthquake_latitude']:5g}, {row['earthquake_longitude']:5g})  "
              f"distance={row['distance_deg']:6.2f} deg  minimum Mw={row['minimum_mw']:.3f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
