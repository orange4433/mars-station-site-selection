"""Plot the InSight on-deck SP noise distribution and its 25th percentile."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "on_deck" / "RMSmap_EHZ.npy"
OUTPUT = HERE / "output"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rms = np.load(DATA)
    values = rms[np.isfinite(rms) & (rms != 0)]
    percentile = float(np.percentile(values, 25))
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.hist(values, bins=80, density=True, color="#4c78a8", alpha=0.85)
    ax.axvline(percentile, color="red", ls="--", lw=1.8,
               label=f"25th percentile = {percentile:.2f}")
    ax.set(xlabel=r"On-deck SP noise, $\log_{10}$(RMS)", ylabel="Probability density")
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "SI_on_deck_noise_distribution.png", dpi=300)
    fig.savefig(OUTPUT / "SI_on_deck_noise_distribution.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
