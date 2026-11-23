from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from si_tools import OUTPUT, capture_legacy_script, compose


def main():
    source = ROOT.parent / "Code_Figure1"
    scripts = ["plot_rms.py", "plot_rms_v^2.py", "plot_rms_rou.py", "plot_rms_rouv^2.py"]
    with TemporaryDirectory() as temporary:
        panels = []
        for index, name in enumerate(scripts):
            panel = Path(temporary) / f"panel_{index}.png"
            capture_legacy_script(source / name, panel, source)
            panels.append(panel)
        compose(panels, 1, 2, 2, OUTPUT / "Supplementary Fig. 1.png",
                offsets=[(-18, -4), (20, -4), (-14, 5), (17, 5)])


if __name__ == "__main__":
    main()
