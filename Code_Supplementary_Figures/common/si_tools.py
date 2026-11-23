from pathlib import Path
import runpy
import contextlib
import io

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REFERENCE = ROOT / "reference"


def reference_size(number):
    with Image.open(REFERENCE / f"image{number}.png") as image:
        return image.size


def resize_like_si(source, number, destination):
    with Image.open(source) as image:
        result = image.convert("RGB").resize(reference_size(number), Image.Resampling.LANCZOS)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, optimize=True)


def capture_legacy_script(script, destination, working_directory, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(destination)
    old_show = plt.show
    old_cwd = Path.cwd()

    def save_current_figure():
        destination.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(destination, dpi=dpi, bbox_inches="tight")

    try:
        plt.show = save_current_figure
        import os
        os.chdir(working_directory)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        plt.show = old_show
        import os
        os.chdir(old_cwd)
        plt.close("all")


def _font(size):
    path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def compose(panels, number, rows, columns, destination, scale=1.0,
            vertical="center", offsets=None):
    width, height = reference_size(number)
    cell_width = width // columns
    cell_height = height // rows
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    offsets = offsets or [(0, 0)] * len(panels)

    for index, panel_path in enumerate(panels):
        offset_x, offset_y = offsets[index]
        row, column = divmod(index, columns)
        left = column * cell_width
        top = row * cell_height
        right = width if column == columns - 1 else left + cell_width
        bottom = height if row == rows - 1 else top + cell_height
        with Image.open(panel_path) as panel:
            panel = ImageOps.contain(
                panel.convert("RGB"),
                (round((right - left) * scale), round((bottom - top) * scale)),
                Image.Resampling.LANCZOS,
            )
        panel_left = left + (right - left - panel.width) // 2 + offset_x
        panel_top = ((bottom - panel.height) if vertical == "bottom"
                     else top + (bottom - top - panel.height) // 2) + offset_y
        canvas.paste(panel, (panel_left, panel_top))
        draw.text((left + 16 + offset_x, top + 4 + offset_y), chr(97 + index),
                  fill="black", font=_font(32))

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
