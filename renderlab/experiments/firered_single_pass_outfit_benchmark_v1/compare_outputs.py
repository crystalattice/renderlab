"""Create local review sheets without changing source or native output bytes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
CASES = ["shoes_to_stilettos", "dress_to_top_and_miniskirt", "clothing_to_bikini"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/firered_single_pass_results")
    )
    args = parser.parse_args()
    assert not args.output.resolve().is_relative_to(ROOT.parents[2]), (
        "Keep review images outside Git"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (3600, 1040), "#202020")
    draw = ImageDraw.Draw(sheet)
    records = {}
    for index, case in enumerate(CASES):
        manifest = json.loads((ROOT / case / "manifest.json").read_text())
        source = Path(manifest["source"]["original_path"])
        output = ROOT / case / "native_output.png"
        pair = Image.new("RGB", (1200, 1000), "#202020")
        pd = ImageDraw.Draw(pair)
        records[case] = {}
        for column, (label, path) in enumerate(
            [("SOURCE", source), ("NATIVE OUTPUT", output)]
        ):
            with Image.open(path) as image:
                image.load()
                records[case][label] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "dimensions": list(image.size),
                    "mode": image.mode,
                    "bytes": path.stat().st_size,
                }
                preview = ImageOps.contain(image.convert("RGB"), (580, 870))
                pair.paste(preview, (column * 600 + (600 - preview.width) // 2, 70))
                pd.text((column * 600 + 15, 15), label, fill="white", font_size=24)
        pd.text((15, 960), case, fill="white", font_size=22)
        pair.save(args.output / (case + ".jpg"), quality=95)
        preview = ImageOps.contain(pair, (1200, 1000))
        sheet.paste(preview, (index * 1200, 40))
        draw.text((index * 1200 + 10, 10), case, fill="white", font_size=19)
    sheet.save(args.output / "comparison.jpg", quality=95)
    (args.output / "image_metrics.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    sys.stdout.write(str(args.output / "comparison.jpg") + "\n")


if __name__ == "__main__":
    main()
