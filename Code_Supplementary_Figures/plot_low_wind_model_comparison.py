"""Plot the three low-wind occurrence maps used in the original SI."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "low_wind_models"
OUTPUT = HERE / "output"
PANELS = (
    ("wind_speed_percentage.txt", "MCD, 1.2 m: wind speed < 3 m/s"),
    ("mcd_100m_percentage.txt", "MCD, 100 m: wind speed < 7 m/s"),
    ("pcm_100m_percentage.txt", "PCM, 100 m: wind speed < 7 m/s"),
)


def load_grid(path: Path):
    data = np.loadtxt(path, delimiter=",")
    lon = np.linspace(-180, 180, 361)
    lat = np.linspace(-90, 90, 181)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    values = griddata(data[:, :2][:, ::-1], data[:, 2], (lon_grid, lat_grid), method="linear")
    return lon_grid, lat_grid, values


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    image = None
    for label, (filename, title) in zip("abc", PANELS):
        ax = axes[ord(label) - ord("a")]
        lon, lat, values = load_grid(DATA / filename)
        image = ax.pcolormesh(lon, lat, values, cmap="RdBu_r", vmin=0, vmax=100, shading="auto")
        ax.set(xlim=(-180, 180), ylim=(-90, 90), ylabel="Latitude", title=title)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.text(-0.055, 1.02, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
    axes[-1].set_xlabel("Longitude")
    bar = fig.colorbar(image, ax=axes, pad=0.02, shrink=0.85)
    bar.set_label("Low-wind occurrence (%)")
    fig.savefig(OUTPUT / "SI_low_wind_model_comparison.png", dpi=300)
    fig.savefig(OUTPUT / "SI_low_wind_model_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
