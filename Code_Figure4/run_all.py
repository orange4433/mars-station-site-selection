"""Generate the code-based source panels for Figure 4."""

from pathlib import Path
import os
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    HERE / "4(a)" / "plot_4a.py",
    HERE / "4(c)" / "plot_4c.py",
    HERE / "4(d)" / "plot_4d.py",
    HERE / "4(e)" / "plot_4e.py",
]


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = str(HERE / "_vendor") + os.pathsep + env.get("PYTHONPATH", "")
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
