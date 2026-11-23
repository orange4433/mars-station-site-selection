"""Reproduce the five Figure 2 panels."""

from pathlib import Path
import os
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    HERE / "2(a)" / "plot_2a.py",
    HERE / "2(b)" / "plot_2b.py",
    HERE / "2(c)" / "plot_2c.py",
    HERE / "2(d)" / "plot_2d.py",
    HERE / "2(e)" / "plot_2e.py",
]


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    for script in SCRIPTS:
        print(f">>> {script.relative_to(HERE)}", flush=True)
        subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
