from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from si_tools import OUTPUT, compose


def main():
    data = ROOT / "data" / "low_wind_models"
    surface = ROOT.parent / "Code_Figure3" / "figure3" / "surface.nc"
    calls = [
        (ROOT / "common" / "dynamic_pressure_map.py", data / "dynamic_pressure_1p2m_percentage.txt"),
        (ROOT / "common" / "wind_speed_map.py", data / "wind_speed_percentage.txt"),
    ]
    with TemporaryDirectory() as temporary:
        panels = []
        for index, (renderer, source) in enumerate(calls):
            panel = Path(temporary) / f"panel_{index}.png"
            subprocess.run([sys.executable, str(renderer), "--data", str(source),
                            "--surface", str(surface), "--output", str(panel)], check=True)
            panels.append(panel)
        compose(panels, 6, 2, 1, OUTPUT / "Supplementary Fig. 6.png",
                scale=0.96, vertical="bottom", offsets=[(2, 8), (3, -4)])


if __name__ == "__main__":
    main()
