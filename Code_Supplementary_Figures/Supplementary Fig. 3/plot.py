from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from si_tools import resize_like_si

DATA = ROOT / "data" / "viking2_pressure"
OUTPUT = ROOT / "output"


def load_table(path):
    return np.loadtxt(path, skiprows=1)


def main():
    observations = load_table(DATA / "viking2_observations.txt")
    model = load_table(DATA / "mcd_daily_pressure.txt")
    ls, pressure = model[:, 0], model[:, 1]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(observations[:, 0], observations[:, 1], s=10, c="blue", alpha=0.7,
               label="Viking 2 Observations")
    first = ls <= 360
    before = first & (ls <= 275)
    after = first & (ls >= 320)
    second = ls > 360
    ax.plot(ls[before], pressure[before], "r-", linewidth=2.5, label="MCD Model")
    ax.plot(ls[after], pressure[after], "r-", linewidth=2.5)
    ax.plot(ls[second], pressure[second], "r-", linewidth=2.5)
    ax.set_xlabel("Solar Longitude (deg)", fontfamily="Times New Roman", fontsize=16)
    ax.set_ylabel("Pressure (Pa)", fontfamily="Times New Roman", fontsize=16)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(100, 720)
    ax.set_xticks(np.arange(100, 721, 100))
    fig.tight_layout()

    OUTPUT.mkdir(exist_ok=True)
    raw = OUTPUT / "_Supplementary Fig. 3.raw.png"
    fig.savefig(raw, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "Supplementary Fig. 3.pdf", bbox_inches="tight")
    plt.close(fig)
    resize_like_si(raw, 3, OUTPUT / "Supplementary Fig. 3.png")
    raw.unlink()


if __name__ == "__main__":
    main()
