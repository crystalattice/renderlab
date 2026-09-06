"""Inventory local corpus bytes and render review sheets outside the repository."""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

ROOT = Path("/home/codyjackson/Datasets/renderlab-source")
HERE = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def image_bytes(location):
    path = Path(location["path"])
    if "member" in location:
        with zipfile.ZipFile(path) as archive:
            return archive.read(location["member"])
    return path.read_bytes()


def inventory():
    groups = {}
    errors = []
    files = sorted(path for path in ROOT.rglob("*") if path.is_file())

    def add(data, location):
        digest = hashlib.sha256(data).hexdigest()
        if digest in groups:
            groups[digest]["locations"].append(location)
            return
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height = image.size
                oriented = ImageOps.exif_transpose(image)
                groups[digest] = {
                    "sha256": digest,
                    "bytes": len(data),
                    "dimensions": [width, height],
                    "display_dimensions": list(oriented.size),
                    "mode": image.mode,
                    "format": image.format,
                    "frames": getattr(image, "n_frames", 1),
                    "locations": [location],
                }
        except (UnidentifiedImageError, OSError, ValueError) as error:
            errors.append({"location": location, "error": str(error)})

    # Filesystem copies take precedence over identical archive members.
    for path in files:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            add(path.read_bytes(), {"path": str(path)})
    archives = []
    for path in files:
        if path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                member
                for member in archive.namelist()
                if Path(member).suffix.lower() in IMAGE_SUFFIXES
            )
            archives.append({"path": str(path), "image_members": len(members)})
            for member in members:
                add(archive.read(member), {"path": str(path), "member": member})
    rows = sorted(
        groups.values(),
        key=lambda row: (
            row["locations"][0]["path"],
            row["locations"][0].get("member", ""),
        ),
    )
    for index, row in enumerate(rows, 1):
        row["review_id"] = f"S{index:04d}"
        row["canonical_location"] = row["locations"][0]
        row["duplicate_occurrences"] = len(row["locations"]) - 1
    (HERE / "inventory.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "root": str(ROOT),
        "scope": "Recursive filesystem images and image members of every discovered ZIP; SHA-256 deduplication over complete image-file bytes.",
        "files_by_suffix": dict(Counter(path.suffix.lower() for path in files)),
        "unique_images": len(rows),
        "image_occurrences": sum(len(row["locations"]) for row in rows),
        "duplicate_occurrences": sum(row["duplicate_occurrences"] for row in rows),
        "duplicate_groups": sum(row["duplicate_occurrences"] > 0 for row in rows),
        "archives": archives,
        "errors": errors,
        "deduplication_mutated_sources": False,
        "semantic_deduplication": False,
    }
    (HERE / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rows, summary


def sheets(rows, output, prefix, columns, page_size, tile_w, tile_h):
    repo = HERE.parents[3]
    assert not output.resolve().is_relative_to(repo), (
        "Contact sheets must be outside the repository"
    )
    output.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 24 if tile_w >= 400 else 17)
    paths = []
    for start in range(0, len(rows), page_size):
        page = rows[start : start + page_size]
        canvas = Image.new(
            "RGB",
            (columns * tile_w, ((len(page) + columns - 1) // columns) * tile_h),
            "#20242a",
        )
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(page):
            data = image_bytes(row["canonical_location"])
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
            with Image.open(io.BytesIO(data)) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((tile_w - 12, tile_h - 48), Image.Resampling.LANCZOS)
                x = (index % columns) * tile_w
                y = (index // columns) * tile_h
                canvas.paste(image, (x + (tile_w - image.width) // 2, y + 38))
                width, height = row["dimensions"]
                draw.text(
                    (x + 6, y + 5),
                    f"{row['review_id']}  {width}x{height}",
                    font=font,
                    fill="white",
                )
        path = output / f"{prefix}_{start // page_size + 1:02d}.jpg"
        canvas.save(path, quality=92)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--prefix", default="screening")
    parser.add_argument("--path-contains")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=220)
    parser.add_argument("--tile-height", type=int, default=310)
    parser.add_argument("--page-size", type=int, default=48)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/firered_single_pass_source_review")
    )
    args = parser.parse_args()
    if args.inventory:
        rows, summary = inventory()
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    else:
        rows = [
            json.loads(line)
            for line in (HERE / "inventory.jsonl").read_text().splitlines()
        ]
    if args.ids:
        wanted = set(args.ids)
        indexed = {row["review_id"]: row for row in rows}
        rows = [indexed[review_id] for review_id in args.ids]
        assert {row["review_id"] for row in rows} == wanted
    if args.path_contains:
        rows = [
            row
            for row in rows
            if args.path_contains in row["canonical_location"]["path"]
        ]
    paths = sheets(
        rows,
        args.output,
        args.prefix,
        args.columns,
        args.page_size,
        args.tile_width,
        args.tile_height,
    )
    sys.stdout.write(
        json.dumps({"sheets": paths, "review_images": len(rows)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
