"""Create an explicitly labelled composite scene from validation images only."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from model_config import MANIFEST_PATH, PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "data" / "demo" / "composite_validation_scene.png"
PREFERRED_CLASSES = ("cardboard", "metal", "plastic")


def font(size: int, bold: bool = False):
    names = ("arialbd.ttf", "Arial Bold.ttf") if bold else ("arial.ttf", "Arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def validation_sources() -> list[tuple[str, Path]]:
    selected: dict[str, Path] = {}
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            class_label = row["class_label"]
            if (
                row["split"] == "validation"
                and class_label in PREFERRED_CLASSES
                and class_label not in selected
            ):
                selected[class_label] = PROJECT_ROOT / Path(row["relative_path"])
    missing = [name for name in PREFERRED_CLASSES if name not in selected]
    if missing:
        raise RuntimeError(f"No validation image found for: {', '.join(missing)}")
    return [(name, selected[name]) for name in PREFERRED_CLASSES]


def main() -> None:
    sources = validation_sources()
    canvas = Image.new("RGB", (1400, 720), "#FAFCFB")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (54, 30),
        "COMPOSITE DEMONSTRATION · VALIDATION-SPLIT PANELS · NOT A FACTORY PHOTO",
        fill="#BEC5C3",
        font=font(25, bold=True),
    )
    draw.text(
        (54, 68),
        "Built locally for a controlled-background smoke test; no test-split image is used.",
        fill="#C5CCCA",
        font=font(18),
    )

    card_colors = ("#DDECE8", "#E7E4DA", "#DCE7EF")
    card_width, card_height = 392, 520
    start_x, gap, top = 72, 38, 140
    for index, ((_, source_path), card_color) in enumerate(
        zip(sources, card_colors), start=1
    ):
        x = start_x + (index - 1) * (card_width + gap)
        draw.rounded_rectangle(
            (x, top, x + card_width, top + card_height),
            radius=28,
            fill=card_color,
        )
        with Image.open(source_path) as source:
            source.load()
            panel = ImageOps.fit(
                source.convert("RGB"),
                (336, 390),
                method=Image.Resampling.LANCZOS,
            )
        canvas.paste(panel, (x + 28, top + 34))
        draw.text(
            (x + 28, top + 452),
            f"Panel {index}",
            fill="#173F3B",
            font=font(23, bold=True),
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"Saved composite demonstration: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    for class_label, source_path in sources:
        print(
            f"  validation source ({class_label}): "
            f"{source_path.relative_to(PROJECT_ROOT).as_posix()}"
        )


if __name__ == "__main__":
    main()
