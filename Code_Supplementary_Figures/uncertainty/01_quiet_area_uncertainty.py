"""Area quieter than InSight and its temporal-sampling sensitivity.

The source stores rho*v^2, while dynamic pressure is q=rho*v^2/2. Every
sampling variant still covers a complete Martian year or a complete sol;
only resolution and starting phase change. This avoids reweighting seasons.
"""

from pathlib import Path
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DATA = WORKSPACE / "Code_Figure2" / "rho_v2_output.nc"
OUTPUT = HERE / "output"

Q_THRESHOLDS = np.round(np.arange(0.05, 0.151, 0.01), 2)
LS_STEPS = (2, 4, 10)  # 1, 2, and 5 degrees at native 0.5-degree spacing
LMST_STEPS = (2, 4)    # 1 and 2 hours at native 0.5-hour spacing
INSIGHT = (4.5, 135.6)  # latitude, longitude


def read_quiet_counts():
    """Count quiet LMST samples per Ls and quiet Ls samples per LMST."""
    old_cwd = Path.cwd()
    try:  # ASCII filename avoids the netCDF Windows Unicode-path problem.
        os.chdir(DATA.parent)
        with Dataset(DATA.name) as ds:
            lons = np.asarray(ds["longitude"][:])
            lats = np.asarray(ds["latitude"][:])
            ls = np.asarray(ds["Ls"][:])
            lmst = np.asarray(ds["ut"][:])
            if np.isclose(ls[-1] - ls[0], 360):
                ls = ls[:-1]  # 360 degrees repeats the season at 0 degrees.

            source = ds["rho_v2"]
            n_grid = len(lons) * len(lats)
            by_ls = np.empty((len(Q_THRESHOLDS), len(ls), n_grid), np.uint8)
            by_lmst = np.zeros((len(Q_THRESHOLDS), len(lmst), n_grid), np.uint16)

            for start in range(0, len(ls), 30):
                stop = min(start + 30, len(ls))
                q = 0.5 * np.asarray(source[:, :, :, start:stop], dtype=np.float32)
                for i, threshold in enumerate(Q_THRESHOLDS):
                    quiet = q <= threshold
                    by_ls[i, start:stop] = quiet.sum(axis=2).transpose(2, 0, 1).reshape(stop - start, -1)
                    by_lmst[i] += quiet.sum(axis=3, dtype=np.uint16).transpose(2, 0, 1).reshape(len(lmst), -1)
    finally:
        os.chdir(old_cwd)
    return lons, lats, ls, lmst, by_ls, by_lmst


def area_quieter_than_insight(quiet, weights, insight_index):
    """Area-weighted percentage whose quiet fraction exceeds InSight's."""
    insight_quiet = quiet[..., insight_index]
    return 100 * ((quiet > insight_quiet[..., None]) * weights).sum(axis=-1) / weights.sum()


def sampling_variants(by_ls, by_lmst, weights, insight_index):
    """Return area percentages for all coarser-resolution phase offsets."""
    n_ls, n_lmst = by_ls.shape[1], by_lmst.shape[1]
    samples, labels = [], []

    for step in LS_STEPS:
        for offset in range(step):
            quiet = by_ls[:, offset::step].sum(axis=1) / (len(range(offset, n_ls, step)) * n_lmst)
            samples.append(area_quieter_than_insight(quiet, weights, insight_index))
            labels.append(("Ls", step * 0.5, offset * 0.5))

    for step in LMST_STEPS:
        for offset in range(step):
            quiet = by_lmst[:, offset::step].sum(axis=1) / (len(range(offset, n_lmst, step)) * n_ls)
            samples.append(area_quieter_than_insight(quiet, weights, insight_index))
            labels.append(("LMST", step * 0.5, offset * 0.5))

    return np.asarray(samples), labels


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    OUTPUT.mkdir(exist_ok=True)
    lons, lats, ls, lmst, by_ls, by_lmst = read_quiet_counts()
    lon_i = int(np.abs(lons - INSIGHT[1]).argmin())
    lat_i = int(np.abs(lats - INSIGHT[0]).argmin())
    insight_index = lon_i * len(lats) + lat_i
    weights = np.broadcast_to(np.cos(np.deg2rad(lats)), (len(lons), len(lats))).ravel()

    quiet = by_ls.sum(axis=1) / (len(ls) * len(lmst))
    central = area_quieter_than_insight(quiet, weights, insight_index)
    insight = 100 * quiet[:, insight_index]
    samples, labels = sampling_variants(by_ls, by_lmst, weights, insight_index)
    low, high = samples.min(axis=0), samples.max(axis=0)

    csv_path = OUTPUT / "01_quieter_area_vs_threshold.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_threshold_pa", "insight_quiet_percent", "mars_area_quieter_percent",
                         "sampling_envelope_low", "sampling_envelope_high"])
        writer.writerows(zip(Q_THRESHOLDS, insight, central, low, high))

    target_i = int(np.where(np.isclose(Q_THRESHOLDS, 0.1))[0][0])
    sensitivity_path = OUTPUT / "01_temporal_sampling_sensitivity_q0p1.csv"
    with sensitivity_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["varied_axis", "sampling_interval", "starting_phase", "area_quieter_percent"])
        for label, values in zip(labels, samples):
            writer.writerow([*label, values[target_i]])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(Q_THRESHOLDS, low, high, color="#4c78a8", alpha=0.25,
                    label="Temporal-sampling envelope")
    ax.plot(Q_THRESHOLDS, central, "o-", color="#1f5a94", label="Area quieter than InSight")
    ax.axvline(0.1, color="black", linestyle="--", linewidth=1, label="Paper threshold (0.1 Pa)")
    ax.set(xlabel="Dynamic-pressure threshold, q (Pa)",
           ylabel="Martian surface quieter than InSight (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = OUTPUT / "01_quieter_area_threshold_sensitivity.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(f"MCD grid used for InSight: {lats[lat_i]:g} N, {lons[lon_i]:g} E")
    print(f"q=0.10 Pa: InSight quiet fraction = {insight[target_i]:.2f}%")
    print(f"q=0.10 Pa: quieter area = {central[target_i]:.2f}% "
          f"(sampling envelope {low[target_i]:.2f}-{high[target_i]:.2f}%)")
    print(f"Saved: {csv_path}\nSaved: {sensitivity_path}\nSaved: {figure_path}")


def self_check():
    quiet = np.array([[0.2, 0.3, 0.1]])
    assert area_quieter_than_insight(quiet, np.ones(3), 0)[0] == 100 / 3


if __name__ == "__main__":
    self_check()
    main()
