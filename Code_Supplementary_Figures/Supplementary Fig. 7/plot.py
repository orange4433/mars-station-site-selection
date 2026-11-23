from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from si_tools import resize_like_si


def main():
    script = ROOT / "uncertainty" / "01_quiet_area_uncertainty.py"
    subprocess.run([sys.executable, str(script)], check=True)
    source = ROOT / "uncertainty" / "output" / "01_quieter_area_threshold_sensitivity.png"
    resize_like_si(source, 7, ROOT / "output" / "Supplementary Fig. 7.png")


if __name__ == "__main__":
    main()
