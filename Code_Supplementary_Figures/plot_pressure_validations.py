"""Replot the Perseverance/MEDA and Viking 1 pressure validations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "pressure_validation"
OUTPUT = HERE / "output"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    meda = np.genfromtxt(DATA / "meda_mcd_daily.csv", delimiter=",", names=True)
    valid = np.isfinite(meda["MEDA_daily_mean_Pa"]) & np.isfinite(meda["MCD_daily_mean_Pa"])
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    ax.plot(meda["Sol"][valid], meda["MEDA_daily_mean_Pa"][valid], color="red", lw=1.2, label="Perseverance/MEDA")
    ax.plot(meda["Sol"][valid], meda["MCD_daily_mean_Pa"][valid], color="blue", lw=1.2, label="MCD6.1")
    ax.set(xlabel="Mission sol", ylabel="Daily-mean surface pressure (Pa)")
    ax.grid(ls=":", alpha=0.5)
    ax.legend()
    fig.savefig(OUTPUT / "SI_perseverance_meda_mcd_pressure.png", dpi=300)
    fig.savefig(OUTPUT / "SI_perseverance_meda_mcd_pressure.pdf")
    plt.close(fig)

    obs = np.genfromtxt(DATA / "viking1_observations_daily.csv", delimiter=",", names=True)
    model = np.genfromtxt(DATA / "viking1_mcd61_daily_climatology.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    ax.scatter(obs["Ls_deg"], obs["VL1_daily_mean_pressure_Pa"], s=7, color="red", alpha=0.65,
               linewidths=0, label="Viking Lander 1 observations")
    centre = model["MCD61_daily_mean_pressure_Pa"]
    spread = np.sqrt(2) * model["MCD61_surface_pressure_RMS_Pa"]
    ax.plot(model["Ls_deg"], centre, color="green", lw=1.7, label="MCD6.1 climatology")
    ax.fill_between(model["Ls_deg"], centre - spread, centre + spread, color="green", alpha=0.15)
    ax.set(xlabel="Solar longitude (deg)", ylabel="Surface pressure (Pa)", xlim=(-2, 362))
    ax.grid(ls=":", alpha=0.5)
    ax.legend()
    fig.savefig(OUTPUT / "SI_viking1_mcd_pressure.png", dpi=300)
    fig.savefig(OUTPUT / "SI_viking1_mcd_pressure.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
