from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from si_tools import OUTPUT, compose


def main():
    data = ROOT / "data" / "low_wind_models"
    renderer = ROOT / "common" / "dynamic_pressure_map.py"
    surface = ROOT.parent / "Code_Figure3" / "figure3" / "surface.nc"
    inputs = ["dynamic_pressure_1p2m_percentage.txt", "mcd_100m_percentage.txt",
              "pcm_100m_percentage.txt"]
    with TemporaryDirectory() as temporary:
        panels = []
        for index, name in enumerate(inputs):
            panel = Path(temporary) / f"panel_{index}.png"
            subprocess.run([sys.executable, str(renderer), "--data", str(data / name),
                            "--surface", str(surface), "--output", str(panel)], check=True)
            panels.append(panel)
        compose(panels, 5, 3, 1, OUTPUT / "Supplementary Fig. 5.png",
                scale=0.96, vertical="bottom",
                offsets=[(3, 8), (-1, -3), (0, -10)])


if __name__ == "__main__":
    main()
