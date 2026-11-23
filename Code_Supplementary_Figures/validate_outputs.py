from pathlib import Path
import hashlib

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    missing = []
    for number in range(1, 12):
        output = ROOT / "output" / f"Supplementary Fig. {number}.png"
        reference = ROOT / "reference" / f"image{number}.png"
        if not output.exists():
            missing.append(str(output))
            continue
        with Image.open(output) as actual, Image.open(reference) as expected:
            same_size = actual.size == expected.size
            a = np.asarray(actual.convert("RGB"), dtype=np.float32) / 255.0
            b = np.asarray(expected.convert("RGB"), dtype=np.float32) / 255.0
        mae = float(np.mean(np.abs(a - b))) if same_size else float("nan")
        exact = digest(output) == digest(reference)
        print(f"Fig. {number:2d}: size={same_size}, exact={exact}, mean_abs_error={mae:.6f}")
    if missing:
        raise FileNotFoundError("Missing outputs:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
