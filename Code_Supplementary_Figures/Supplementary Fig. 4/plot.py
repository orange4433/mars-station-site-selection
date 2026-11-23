from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from si_tools import OUTPUT


def main():
    data = np.genfromtxt(ROOT / "data" / "pressure_validation" / "meda_mcd_daily.csv",
                         delimiter=",", names=True)
    valid = np.isfinite(data["MEDA_daily_mean_Pa"]) & (data["MEDA_sample_count"] >= 10_000)
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    ax.plot(data["Sol"], data["MCD_daily_mean_Pa"], color="red", lw=1.6,
            label="MCD6.1 (daily mean)")
    ax.plot(data["Sol"][valid], data["MEDA_daily_mean_Pa"][valid], "+", color="black",
            ms=4, mew=0.7, label="Perseverance MEDA (daily mean)")
    ax.set(title="MCD6.1 Surface Pressure compared with Perseverance",
           xlabel="Mission sol", ylabel="Surface pressure (Pa)", xlim=(0, 300))
    ax.grid(ls=":", color="0.45", alpha=0.8)
    ax.legend()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "Supplementary Fig. 4.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "Supplementary Fig. 4.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
