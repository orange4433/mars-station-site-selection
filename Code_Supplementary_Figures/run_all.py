from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="*", type=int, default=range(1, 12))
    args = parser.parse_args()
    for number in args.figures:
        script = ROOT / f"Supplementary Fig. {number}" / "plot.py"
        print(f"Generating Supplementary Fig. {number} ...", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
