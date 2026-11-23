from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / "dynamic_pressure_analysis.py")], check=True)
    source = ROOT / "output" / "dynamic_pressure_noise_17bands_06-17.png"
    shutil.copy2(source, ROOT / "output" / "Supplementary Fig. 8.png")


if __name__ == "__main__":
    main()
