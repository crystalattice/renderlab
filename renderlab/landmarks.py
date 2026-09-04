from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from .corpus import file_sha256


CHANNELS = ("left_nipple", "right_nipple", "navel")
PROVENANCE = {"observed", "canon_inferred", "morphology_estimated"}
VISIBILITY = {"exposed", "covered", "partially_visible", "unknown"}


class LandmarkError(RuntimeError):
    pass


def _load_spec(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LandmarkError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LandmarkError("landmark spec root must be an object")
    return value


def validate_landmark_spec(spec: dict[str, Any]) -> None:
    errors = []
    if spec.get("schema") != "renderlab.landmark-map.v1":
        errors.append("schema must be renderlab.landmark-map.v1")
    for name in ("width", "height"):
        if not isinstance(spec.get(name), int) or isinstance(spec.get(name), bool) or spec.get(name, 0) <= 0:
            errors.append(f"{name} must be a positive integer")
    channels = spec.get("channels")
    if not isinstance(channels, dict):
        errors.append("channels must be an object")
        channels = {}
    unknown = sorted(set(channels) - set(CHANNELS))
    if unknown:
        errors.append(f"unknown channels: {', '.join(unknown)}")
    for channel in CHANNELS:
        point = channels.get(channel)
        if not isinstance(point, dict):
            errors.append(f"channels.{channel} must be an object")
            continue
        for field in ("center_x", "center_y", "sigma_x", "sigma_y", "confidence"):
            value = point.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                errors.append(f"channels.{channel}.{field} must be a finite number")
        if all(isinstance(point.get(field), (int, float)) and not isinstance(point.get(field), bool) for field in ("center_x", "center_y", "confidence")):
            for field in ("center_x", "center_y", "confidence"):
                if not 0 <= point[field] <= 1:
                    errors.append(f"channels.{channel}.{field} must be between 0 and 1")
        if all(isinstance(point.get(field), (int, float)) and not isinstance(point.get(field), bool) for field in ("sigma_x", "sigma_y")):
            for field in ("sigma_x", "sigma_y"):
                if not 0 < point[field] <= 1:
                    errors.append(f"channels.{channel}.{field} must be greater than 0 and at most 1")
        if point.get("provenance") not in PROVENANCE:
            errors.append(f"channels.{channel}.provenance is invalid")
        if point.get("visibility") not in VISIBILITY:
            errors.append(f"channels.{channel}.visibility is invalid")
    if errors:
        raise LandmarkError("landmark validation failed:\n" + "\n".join(errors))


def _render_channel(width: int, height: int, point: dict[str, Any]) -> Image.Image:
    center_x = point["center_x"] * (width - 1)
    center_y = point["center_y"] * (height - 1)
    sigma_x = point["sigma_x"] * width
    sigma_y = point["sigma_y"] * height
    confidence = point["confidence"]
    pixels = bytearray(width * height)
    for y in range(height):
        dy = ((y - center_y) / sigma_y) ** 2
        for x in range(width):
            dx = ((x - center_x) / sigma_x) ** 2
            pixels[y * width + x] = round(255 * confidence * math.exp(-0.5 * (dx + dy)))
    return Image.frombytes("L", (width, height), bytes(pixels))


def render_landmark_maps(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    spec = _load_spec(spec_path)
    validate_landmark_spec(spec)
    output_dir.mkdir(parents=True, exist_ok=False)
    images = {}
    rendered = []
    for channel in CHANNELS:
        image = _render_channel(spec["width"], spec["height"], spec["channels"][channel])
        path = output_dir / f"{channel}.png"
        image.save(path)
        images[channel] = image
        rendered.append({"channel": channel, "path": str(path.resolve()), "sha256": file_sha256(path)})
    preview = Image.merge("RGB", tuple(images[channel] for channel in CHANNELS))
    preview_path = output_dir / "landmarks_rgb.png"
    preview.save(preview_path)
    result = {
        "schema": "renderlab.landmark-render.v1",
        "source": {"path": str(spec_path), "sha256": file_sha256(spec_path)},
        "width": spec["width"],
        "height": spec["height"],
        "channels": rendered,
        "preview": {"path": str(preview_path.resolve()), "sha256": file_sha256(preview_path)},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
