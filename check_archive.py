"""Verify that the reorganized archive has every required local input."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "Code_Figure1/surface_density_ds_1_LS_0.00-360.00_rev1.1.nc",
    "Data_MCD/mcd_data.nc",
    "Code_Figure2/rho_v2_output.nc",
    "Code_Figure2/wind_east.npy",
    "Code_Figure2/wind_north.npy",
    "Data_Insight_Seismic/Data_Step5_Rotate/LMST_0641.mseed",
    "Data_Insight_Seismic/Data_Step5_Rotate_DISP/LMST_0641.mseed",
    "Code_Figure4/data/seismic/data_detrended_1133.npy",
    "Code_Figure4/data/seismic/data_detrended_1133_disp.npy",
    "Code_Figure4/data/on_deck/data_detrended_1222.npy",
    "Code_Figure4/data/on_deck/data_detrended_1222_lvbo.npy",
    "Code_Figure4/data/on_deck/event_1133_spectrogram.npz",
    "Code_Figure4/data/on_deck/LMST_020.mseed",
    "Code_Figure4/4(b)/Raypath.pdf",
    "Code_Supplementary_Figures/data/on_deck/RMSmap_EHZ.npy",
)


def main() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing archive input(s):\n" + "\n".join(missing))
    print(f"Archive check passed: {len(REQUIRED)} required inputs found under {ROOT}")


if __name__ == "__main__":
    main()
