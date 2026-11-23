"""Bootstrap background noise and keep S1133c magnitude uncertainty separate.

This deliberately preserves the Major2 detection rule: mean signal ASD in
0.2--1 Hz must be twice the mean background ASD. M0=3.41 is the fitted
S1133c reference magnitude. The event-specific uncertainty is taken from
the MQS V14 catalogue and propagated separately from the noise bootstrap.
"""

from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
MAJOR2 = WORKSPACE / "Code_Figure4"
sys.path.insert(0, str(MAJOR2))

from dynamic_pressure_common import (  # noqa: E402
    EARTHQUAKE_POINTS, FREQUENCY_EDGES, REFERENCE_DISP, SAMPLE_RATE,
    SOURCE_DIR, TARGET_SITE, angular_distance_radians, load_reference_day,
    old_style_noise_asd, scaled_windows, target_gains,
)
from run_settings import EARTHQUAKE_INDEX, TARGET_HOUR, TARGET_SOL  # noqa: E402


OUTPUT = HERE / "output"
MODEL_FILE = SOURCE_DIR / "model_asd_data_final.txt"
N_BOOTSTRAP = 5000
WELCH_BLOCK_SEGMENTS = 5  # ~64 s with 512-sample windows and 50% overlap
SEED = 20260928

# The bundled S1133c template calibration gives M0=3.41. MQS V14 lists
# S1133c as MW=3.8 +/- 0.3.
# We retain the internally fitted centre for amplitude normalization and
# propagate the catalogue's event-specific +/-0.3 magnitude uncertainty.
REFERENCE_MW = 3.41
MQS_CATALOG_MW = 3.8
REFERENCE_MW_UNCERTAINTY = 0.3
REFERENCE_MW_LOW = REFERENCE_MW - REFERENCE_MW_UNCERTAINTY
REFERENCE_MW_HIGH = REFERENCE_MW + REFERENCE_MW_UNCERTAINTY
REFERENCE_DISTANCE_DEG = 30.2


def circular_block_counts(n_samples, block_size, repeats, rng):
    n_blocks = int(np.ceil(n_samples / block_size))
    starts = rng.integers(0, n_samples, size=(repeats, n_blocks))
    indices = (starts[:, :, None] + np.arange(block_size)) % n_samples
    indices = indices.reshape(repeats, -1)[:, :n_samples]
    counts = np.zeros((repeats, n_samples), np.float64)
    np.add.at(counts, (np.repeat(np.arange(repeats), n_samples), indices.ravel()), 1)
    return counts


def bootstrap_nbar(reference, hour, gains, block_size=WELCH_BLOCK_SEGMENTS):
    """Bootstrap Welch segments and return central and sampled mean ASD."""
    windows = scaled_windows(reference, hour - 0.5, 3600, gains)
    segment_psds = []
    for band, signal in enumerate(windows):
        frequency, _, psd = spectrogram(
            signal, fs=SAMPLE_RATE, window="hann", nperseg=512, noverlap=256,
            detrend="constant", scaling="density", mode="psd",
        )
        mask = ((frequency >= FREQUENCY_EDGES[band])
                & (frequency < FREQUENCY_EDGES[band + 1])
                & (frequency >= 0.2) & (frequency <= 1.0))
        if np.any(mask):
            segment_psds.append(psd[mask])

    n_segments = segment_psds[0].shape[1]
    if any(piece.shape[1] != n_segments for piece in segment_psds):
        raise ValueError("Welch segment counts differ between frequency bands")
    counts = circular_block_counts(
        n_segments, block_size, N_BOOTSTRAP, np.random.default_rng(SEED + block_size)
    )
    total_asd = np.zeros(N_BOOTSTRAP)
    n_frequencies = 0
    for psd in segment_psds:
        sampled_mean_psd = counts @ psd.T / n_segments
        total_asd += np.sqrt(sampled_mean_psd).sum(axis=1)
        n_frequencies += psd.shape[0]
    return np.mean(np.sqrt(np.concatenate([p.mean(axis=1) for p in segment_psds]))), total_asd / n_frequencies


def analytic_minimum_mw(nbar, distance_deg, template_mean_asd):
    """Solve Sbar(M, distance)=2*Nbar without the old 0.01-Mw grid."""
    return REFERENCE_MW + (2 / 3) * np.log10(
        2 * np.asarray(nbar) / template_mean_asd * distance_deg / REFERENCE_DISTANCE_DEG
    )


def main():
    for path in (REFERENCE_DISP, MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(exist_ok=True)

    gains, _, _ = target_gains(int(TARGET_SOL))
    reference, fs = load_reference_day(REFERENCE_DISP)
    if fs != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE:g} Hz, got {fs:g} Hz")
    nbar, nbar_boot = bootstrap_nbar(reference, float(TARGET_HOUR), gains)

    # Confirm the segment implementation reproduces the original Welch estimate.
    frequency, original_asd = old_style_noise_asd(reference, float(TARGET_HOUR), gains)
    original_nbar = np.mean(original_asd[(frequency >= 0.2) & (frequency <= 1)])
    if not np.isclose(nbar, original_nbar, rtol=1e-12):
        raise AssertionError(f"Welch mismatch: {nbar} versus {original_nbar}")

    model_frequency, model_log10_asd = np.loadtxt(MODEL_FILE)
    mask = ((model_frequency >= 0.2) & (model_frequency <= 1)
            & (model_frequency >= 0.1) & (model_frequency <= 10)
            & (model_log10_asd >= -15) & (model_log10_asd <= -6))
    template_mean_asd = np.mean(10 ** model_log10_asd[mask])
    point = EARTHQUAKE_POINTS[int(EARTHQUAKE_INDEX)]
    distance_deg = np.degrees(angular_distance_radians(point, TARGET_SITE))

    mw = float(analytic_minimum_mw(nbar, distance_deg, template_mean_asd))
    mw_noise = analytic_minimum_mw(nbar_boot, distance_deg, template_mean_asd)
    noise_low, noise_high = np.percentile(mw_noise, [2.5, 97.5])
    n_low, n_high = np.percentile(nbar_boot, [2.5, 97.5])
    sigma_log10_nbar = float(np.std(np.log10(nbar_boot), ddof=1))
    sigma_mw_noise = (2 / 3) * sigma_log10_nbar
    envelope_low = noise_low + REFERENCE_MW_LOW - REFERENCE_MW
    envelope_high = noise_high + REFERENCE_MW_HIGH - REFERENCE_MW

    result = {
        "background_sol": int(TARGET_SOL),
        "background_lmst_hour": float(TARGET_HOUR),
        "earthquake_latitude": point[0],
        "earthquake_longitude": point[1],
        "distance_deg": distance_deg,
        "reference_mw": REFERENCE_MW,
        "mqs_catalog_mw": MQS_CATALOG_MW,
        "mqs_catalog_mw_uncertainty": REFERENCE_MW_UNCERTAINTY,
        "reference_mw_uncertainty_low": REFERENCE_MW_LOW,
        "reference_mw_uncertainty_high": REFERENCE_MW_HIGH,
        "mean_noise_asd": nbar,
        "noise_asd_bootstrap_95_low": n_low,
        "noise_asd_bootstrap_95_high": n_high,
        "sigma_log10_mean_noise_asd": sigma_log10_nbar,
        "minimum_mw_sigma_from_noise": sigma_mw_noise,
        "minimum_mw_continuous": mw,
        "minimum_mw_noise_only_95_low": noise_low,
        "minimum_mw_noise_only_95_high": noise_high,
        "minimum_mw_conservative_envelope_low": envelope_low,
        "minimum_mw_conservative_envelope_high": envelope_high,
    }
    csv_path = OUTPUT / "02_minimum_magnitude_uncertainty.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    sensitivity_path = OUTPUT / "02_noise_bootstrap_block_sensitivity.csv"
    with sensitivity_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["welch_block_segments", "approx_block_seconds", "mw_95_low", "mw_95_high"])
        for block in (1, 5, 10):
            _, samples = bootstrap_nbar(reference, float(TARGET_HOUR), gains, block)
            bounds = np.percentile(analytic_minimum_mw(samples, distance_deg, template_mean_asd), [2.5, 97.5])
            writer.writerow([block, block * 256 / SAMPLE_RATE, *bounds])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(mw_noise, bins=45, color="#4c78a8", alpha=0.8, density=True)
    ax.axvline(mw, color="black", linewidth=1.5, label=f"Estimate: Mw {mw:.2f}")
    ax.axvspan(noise_low, noise_high, color="#72b7b2", alpha=0.35, label="Noise-only 95% CI")
    ax.axvspan(envelope_low, envelope_high, color="#f58518", alpha=0.12,
               label="MQS M0 uncertainty + noise envelope")
    ax.set(xlabel="Minimum detectable magnitude (Mw)", ylabel="Bootstrap density")
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = OUTPUT / "02_minimum_magnitude_uncertainty.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(f"Nbar = {nbar:.6e} ({n_low:.6e}-{n_high:.6e}, bootstrap 95%)")
    print(f"Minimum Mw = {mw:.3f} ({noise_low:.3f}-{noise_high:.3f}, noise-only 95%)")
    print(f"MQS S1133c MW = {MQS_CATALOG_MW:.1f} +/- {REFERENCE_MW_UNCERTAINTY:.1f}")
    print(f"Internal M0 uncertainty range = {REFERENCE_MW_LOW:.2f}-{REFERENCE_MW_HIGH:.2f}")
    print(f"MQS M0 uncertainty + noise envelope = {envelope_low:.3f}-{envelope_high:.3f}")
    print(f"Saved: {csv_path}\nSaved: {sensitivity_path}\nSaved: {figure_path}")


def self_check():
    counts = circular_block_counts(20, 5, 9, np.random.default_rng(1))
    assert counts.shape == (9, 20) and np.all(counts.sum(axis=1) == 20)
    assert np.isclose(analytic_minimum_mw(1, REFERENCE_DISTANCE_DEG, 2), REFERENCE_MW)


if __name__ == "__main__":
    self_check()
    main()
