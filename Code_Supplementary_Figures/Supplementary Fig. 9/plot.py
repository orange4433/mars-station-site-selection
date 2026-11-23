from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "output" / "dynamic_pressure_validation_sol600.png"
    if not source.exists():
        subprocess.run([sys.executable, str(ROOT / "validate_dynamic_pressure_transfer.py")], check=True)
    shutil.copy2(source, ROOT / "output" / "Supplementary Fig. 9.png")


if __name__ == "__main__":
    main()
