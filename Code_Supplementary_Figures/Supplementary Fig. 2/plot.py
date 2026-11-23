from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from si_tools import OUTPUT


def main():
    data = ROOT / "data" / "pressure_validation"
    observations = np.genfromtxt(data / "viking1_observations_daily.csv", delimiter=",", skip_header=1)
    model = np.genfromtxt(data / "viking1_mcd61_daily_climatology.csv", delimiter=",", skip_header=1)
    ls, pressure, rms = model.T

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    ax.scatter(observations[:, 2], observations[:, 3], s=7, color="red", alpha=0.7,
               linewidths=0, label="Viking Lander 1 observations", zorder=2)
    ax.plot(ls, pressure, color="green", lw=1.7,
            label=r"MCD6.1 climatology $\pm\sqrt{2}$ RMS")
    spread = np.sqrt(2.0) * rms
    ax.plot(ls, pressure + spread, "--", color="green", lw=1.0)
    ax.plot(ls, pressure - spread, "--", color="green", lw=1.0)
    ax.set(title="MCD6.1 Surface Pressure at Viking Lander 1 Site",
           xlabel="Solar longitude (deg)", ylabel="Surface pressure (Pa)", xlim=(-2, 362))
    ax.set_xticks(np.arange(0, 361, 50))
    ax.grid(ls=":", color="0.45", alpha=0.8)
    ax.legend(loc="upper right")
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "Supplementary Fig. 2.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "Supplementary Fig. 2.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
