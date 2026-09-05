import colorsys
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
from comfy_extras.nodes_mask import ImageCompositeMasked, MaskComposite, SolidMask

p = ROOT / "renderlab/experiments/firered_controlled_inpaint_v1"
case = json.loads((p / "experiment.json").read_text())
src_path = Path(case["source"]["path"])
raw_path = p / "raw_output_1bdff9f9.png"
assert (
    hashlib.sha256(raw_path.read_bytes()).hexdigest()
    == "1ce239ac042100aac328dae6463c741a4536b36b01a52352e36476a14368c35c"
)
assert hashlib.sha256(src_path.read_bytes()).hexdigest() == case["source"]["sha256"]
source_image = Image.open(src_path)
raw_image = Image.open(raw_path)
assert source_image.mode == raw_image.mode == "RGB"
assert source_image.size == raw_image.size == (1160, 896)
src = np.array(source_image)
raw = np.array(raw_image)
mask = MaskComposite.execute(
    SolidMask.execute(0, 1160, 896)[0],
    SolidMask.execute(1, 128, 128)[0],
    552,
    424,
    "add",
)[0]
inside = mask.numpy()[0].astype(bool)
assert inside.sum() == 16384
result = ImageCompositeMasked.execute(
    torch.from_numpy(src.astype(np.float32) / 255)[None],
    torch.from_numpy(raw.astype(np.float32) / 255)[None],
    0,
    0,
    False,
    mask,
)[0]
production = np.clip(result.numpy()[0] * 255, 0, 255).astype(np.uint8)
assert np.array_equal(production[~inside], src[~inside])
assert np.array_equal(production[inside], raw[inside])
Image.fromarray(production).save("/tmp/firered-production.png")
with Image.open("/tmp/firered-production.png") as saved:
    assert saved.mode == "RGB" and saved.size == (1160, 896)
    assert saved.tobytes() == production.tobytes()


def metrics(a):
    delta = np.abs(a.astype(np.int16) - src.astype(np.int16))
    outside = delta[~inside]
    hsv = np.array(
        [colorsys.rgb_to_hsv(*pixel) for pixel in a[inside].astype(np.float64) / 255]
    )
    coverage = (
        (hsv[:, 0] * 360 >= 200)
        & (hsv[:, 0] * 360 <= 250)
        & (hsv[:, 1] >= 0.35)
        & (hsv[:, 2] >= 0.15)
    ).mean()
    inner = np.zeros(inside.shape, bool)
    inner[424:552, 552:680] = True
    inner[432:544, 560:672] = False
    outer = np.zeros(inside.shape, bool)
    outer[416:560, 544:688] = True
    outer[424:552, 552:680] = False
    return {
        "outside_changed_pixels": int(np.any(outside != 0, axis=1).sum()),
        "outside_total_pixels": int((~inside).sum()),
        "outside_changed_pixel_fraction": float(np.any(outside != 0, axis=1).mean()),
        "outside_mae_rgb": float(outside.mean()),
        "outside_max_channel_error": int(outside.max()),
        "inside_blue_coverage_fraction": float(coverage),
        "inside_pixels_identical_to_raw": bool(np.array_equal(a[inside], raw[inside])),
        "outer_ring_changed_fraction": float(np.any(delta[outer] != 0, axis=1).mean()),
        "inner_ring_changed_fraction": float(np.any(delta[inner] != 0, axis=1).mean()),
    }


m = {
    "source_sha256": case["source"]["sha256"],
    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    "production_sha256": hashlib.sha256(
        Path("/tmp/firered-production.png").read_bytes()
    ).hexdigest(),
    "dimensions": [1160, 896],
    "source_mode": "RGB",
    "raw_mode": "RGB",
    "production_mode": "RGB",
    "alpha_added": False,
    "color_mode_conversion": False,
    "editable_pixels": 16384,
    "raw": metrics(raw),
    "production": metrics(production),
    "native_node": "ImageCompositeMasked",
    "native_settings": {"x": 0, "y": 0, "resize_source": False},
    "native_impl_sha256": hashlib.sha256(
        (ROOT / "comfy_extras/nodes_mask.py").read_bytes()
    ).hexdigest(),
    "comparison": "Native CPU compositor; saved PNG decoded and verified pixel-exact against direct source/raw selection.",
}
assert m["production"]["inside_blue_coverage_fraction"] > 0.8
Path("/tmp/firered-composite-metrics.json").write_text(json.dumps(m, indent=2) + "\n")
# Unmodified crops enlarged by nearest-neighbor solely for inspection, not production.
panel = Image.new("RGB", (3 * 352, 352))
for i, im in enumerate([source_image, raw_image, Image.fromarray(production)]):
    panel.paste(
        im.crop((528, 400, 704, 576)).resize((352, 352), Image.Resampling.NEAREST),
        (i * 352, 0),
    )
panel.save("/tmp/firered-boundaries.png")
sys.stdout.write(json.dumps(m, indent=2) + "\n")
