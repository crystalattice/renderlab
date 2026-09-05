"""Local geometry-only mask candidates. No model inference or network access."""

import sys
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import distance_transform_edt

SOURCE = Path(
    "/home/codyjackson/Downloads/img_00562__firered-image-edit-1.1__single-pass__621146938678618_00001_.png"
)
OUT = Path(__file__).resolve().parent
EXPECTED = "da52612e6f04b4f030e4413d54f49246a2ea34e30a4f0ec93c9641cee4f84b75"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
source = Image.open(SOURCE)
assert source.size == (1160, 896) and source.mode == "RGB"
W, H = source.size
OUT.mkdir(exist_ok=True)
# Pixel coordinates traced from the full-resolution source and enlarged grid views.
core_polygon = [
    (614, 283),
    (618, 307),
    (627, 314),
    (641, 316),
    (657, 312),
    (679, 305),
    (701, 297),
    (719, 288),
    (725, 284),
    (739, 291),
    (755, 298),
    (771, 294),
    (785, 292),
    (798, 298),
    (810, 307),
    (830, 314),
    (849, 325),
    (866, 335),
    (871, 340),
    (862, 348),
    (854, 361),
    (847, 380),
    (840, 400),
    (836, 411),
    (846, 415),
    (860, 420),
    (851, 430),
    (839, 443),
    (827, 450),
    (816, 450),
    (802, 442),
    (790, 436),
    (782, 440),
    (774, 461),
    (764, 486),
    (752, 520),
    (740, 557),
    (728, 590),
    (716, 626),
    (707, 655),
    (700, 684),
    (695, 708),
    (691, 731),
    (692, 748),
    (692, 765),
    (685, 790),
    (678, 813),
    (674, 833),
    (670, 811),
    (663, 797),
    (649, 783),
    (634, 770),
    (617, 760),
    (598, 753),
    (578, 748),
    (560, 747),
    (541, 751),
    (525, 758),
    (511, 766),
    (501, 769),
    (489, 764),
    (477, 757),
    (464, 746),
    (450, 729),
    (436, 711),
    (424, 694),
    (411, 679),
    (427, 673),
    (438, 665),
    (448, 651),
    (458, 643),
    (466, 625),
    (473, 603),
    (478, 580),
    (481, 550),
    (484, 520),
    (487, 488),
    (489, 459),
    (489, 436),
    (492, 420),
    (501, 403),
    (514, 386),
    (526, 375),
    (533, 357),
    (538, 333),
    (545, 315),
    (555, 305),
    (571, 296),
    (590, 288),
    (603, 283),
]
shoulder_left = [
    (515, 292),
    (535, 283),
    (566, 284),
    (598, 273),
    (614, 274),
    (623, 294),
    (624, 323),
    (563, 366),
    (541, 378),
    (520, 356),
    (511, 327),
]
shoulder_right = [
    (722, 272),
    (747, 277),
    (770, 277),
    (801, 283),
    (823, 296),
    (844, 305),
    (873, 320),
    (891, 335),
    (903, 357),
    (903, 410),
    (882, 432),
    (852, 454),
    (826, 469),
    (800, 457),
    (774, 449),
    (774, 411),
    (797, 367),
    (764, 327),
]
neckline_margin = [
    (602, 274),
    (616, 272),
    (627, 295),
    (645, 304),
    (670, 298),
    (697, 288),
    (718, 272),
    (730, 280),
    (720, 304),
    (667, 326),
    (620, 329),
]


def polygon(points):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(points, fill=255)
    return np.asarray(im) > 0


core = polygon(core_polygon)
upper = core.copy()
upper[469:] = False
expanded = (
    np.asarray(
        Image.fromarray((upper * 255).astype("uint8")).filter(ImageFilter.MaxFilter(37))
    )
    > 0
)
# 18-pixel Chebyshev dilation only around the upper garment, plus explicit reconstruction patches.
generation = (
    core
    | expanded
    | polygon(shoulder_left)
    | polygon(shoulder_right)
    | polygon(neckline_margin)
)
protected_polygons = {
    "face_and_hair": [
        (522, 66),
        (718, 66),
        (744, 133),
        (743, 219),
        (724, 259),
        (711, 274),
        (686, 264),
        (651, 279),
        (610, 280),
        (586, 266),
        (551, 240),
        (524, 192),
    ],
    "left_hand_and_forearm": [
        (366, 134),
        (446, 129),
        (465, 207),
        (523, 277),
        (515, 299),
        (477, 291),
        (428, 256),
        (389, 221),
        (372, 183),
    ],
    "right_hand_and_forearm": [
        (875, 129),
        (979, 129),
        (988, 244),
        (970, 334),
        (947, 372),
        (907, 369),
        (905, 337),
        (927, 298),
        (917, 253),
        (884, 223),
    ],
    "lower_clothing_and_thighs": [
        (0, 674),
        (404, 674),
        (419, 689),
        (434, 709),
        (449, 729),
        (463, 746),
        (477, 758),
        (489, 767),
        (501, 772),
        (517, 765),
        (536, 755),
        (560, 750),
        (578, 751),
        (598, 756),
        (617, 763),
        (633, 774),
        (648, 787),
        (660, 800),
        (668, 813),
        (672, 834),
        (669, 850),
        (726, 896),
        (0, 896),
    ],
    "left_gym_pad_and_frame": [
        (0, 266),
        (432, 266),
        (466, 286),
        (488, 342),
        (491, 376),
        (475, 438),
        (461, 487),
        (0, 487),
    ],
    "right_gym_pad_and_frame": [
        (912, 314),
        (1160, 303),
        (1160, 896),
        (920, 896),
        (907, 486),
        (898, 441),
        (904, 405),
    ],
}
protected = {name: polygon(points) for name, points in protected_polygons.items()}
# Seat protection is a color selection inside its explicit spatial envelope, not a general red-pixel mask.
seat_roi = polygon(
    [
        (704, 0),
        (863, 0),
        (872, 124),
        (839, 318),
        (805, 414),
        (792, 473),
        (743, 896),
        (554, 896),
        (663, 787),
        (706, 586),
        (744, 436),
        (724, 327),
    ]
)
a = np.asarray(source).astype(float)
seat_red = (
    seat_roi
    & (a[:, :, 0] > 100)
    & (a[:, :, 0] > 1.45 * a[:, :, 1])
    & (a[:, :, 0] > 1.35 * a[:, :, 2])
)
# Only a narrow seat rim within 12 pixels of upper sleeve/shoulder fabric may be editable.
seat_exception = (
    seat_red
    & (distance_transform_edt(~upper) <= 12)
    & polygon([(705, 275), (845, 275), (870, 453), (772, 470), (729, 339)])
)
protected["red_seat_outside_sleeve_exception"] = seat_red & ~seat_exception
protected_union = np.logical_or.reduce(list(protected.values()))
core_before = core.copy()
generation_before = generation.copy()
core &= ~protected_union & ~seat_red
generation &= ~protected_union
assert np.all(~core | generation)
# Grayscale blend is one inside after 10 pixels, ramps inward, and is rigorously zero outside generation.
distance = distance_transform_edt(generation)
weights = np.zeros((H, W), dtype=np.uint8)
weights[generation] = np.rint(
    255 * np.clip((distance[generation] - 1) / 10, 0, 1)
).astype(np.uint8)
assert np.count_nonzero(weights[~generation]) == 0
masks = {
    "shirt_core_mask": (core * 255).astype(np.uint8),
    "shirt_generation_mask": (generation * 255).astype(np.uint8),
    "shirt_composite_mask": weights,
}
colors = {
    "shirt_core_mask": (0, 225, 235),
    "shirt_generation_mask": (255, 95, 10),
    "shirt_composite_mask": (215, 60, 245),
}
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = (
    ImageFont.truetype(font_path, 28)
    if Path(font_path).exists()
    else ImageFont.load_default()
)


def bbox(mask):
    y, x = np.where(mask)
    return (
        None
        if not len(x)
        else [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]
    )


report = {
    "status": "CANDIDATES_REQUIRE_USER_VISUAL_APPROVAL",
    "automatic_suitability_claim": False,
    "source": {
        "path": str(SOURCE),
        "sha256": EXPECTED,
        "dimensions": [W, H],
        "mode": source.mode,
    },
    "construction": {
        "libraries": ["Pillow", "NumPy", "SciPy distance_transform_edt"],
        "inference_used": False,
        "geometry": {
            "core_polygon": core_polygon,
            "left_shoulder_expansion": shoulder_left,
            "right_shoulder_expansion": shoulder_right,
            "neckline_expansion": neckline_margin,
            "protected_polygons": protected_polygons,
        },
        "method": "Hand-traced pixel-space garment polygon; upper-only 18px square dilation plus reconstruction polygons; subtract explicit protection masks. No resizing of mask canvas.",
        "composite_method": "Inward-only Euclidean-distance ramp: round(255*clip((distance_inside-1)/10,0,1)); forced zero outside binary generation boundary.",
        "feather_width_pixels": 10,
        "red_seat_policy": "Core excludes all selected seat pixels. Generation/composite protect red-color-selected pixels inside explicit seat ROI except <=12px from upper garment within sleeve/shoulder exception polygon.",
        "core_pixels_removed_by_protection": int((core_before & ~core).sum()),
        "generation_pixels_removed_by_protection": int(
            (generation_before & protected_union).sum()
        ),
    },
    "protected_regions": {},
    "masks": {},
    "review_cautions": [
        "Hand-traced candidates, not automatically validated segmentation. Inspect sleeve cuffs, collar, long hem and all skin boundaries at full resolution.",
        "Protected-region intersections are measured against explicitly declared geometry/color masks, not independently verified semantic segmentation. Zero intersections do not prove perfect anatomy protection.",
        "Core trace may require pixel-level corrections at anti-aliased fabric/skin boundaries.",
        "Generation margins authorize limited proximal upper-arm and neckline reconstruction; hands and distal forearms remain excluded.",
        "Inward feather at non-expanded hem may preserve a narrow old-shirt edge; review whether additional safe margin or a different composite ramp is needed.",
        "No suitability or execution-readiness claim is made.",
    ],
}
for name, m in masks.items():
    Image.fromarray(m).save(OUT / (name + ".png"))
    alpha = (m.astype(float) / 255 * 0.42)[:, :, None]
    overlay = (
        np.rint(a * (1 - alpha) + np.array(colors[name])[None, None, :] * alpha)
        .clip(0, 255)
        .astype(np.uint8)
    )
    oi = Image.fromarray(overlay)
    oi.save(OUT / (name + "_overlay.png"))
    nz = m > 0
    report["masks"][name] = {
        "filename": name + ".png",
        "overlay": name + "_overlay.png",
        "sha256": hashlib.sha256((OUT / (name + ".png")).read_bytes()).hexdigest(),
        "dimensions": [W, H],
        "mode": "L",
        "bbox_xyxy_exclusive": bbox(nz),
        "nonzero_pixels": int(nz.sum()),
        "white_255_pixels": int((m == 255).sum()),
        "black_0_pixels": int((m == 0).sum()),
        "blend_1_to_254_pixels": int(((m > 0) & (m < 255)).sum()),
        "coverage_fraction": float(nz.mean()),
        "unique_values": list(map(int, np.unique(m))),
        "protected_region_intersections": {
            key: int((nz & value).sum()) for key, value in protected.items()
        },
        "editable_red_seat_exception_pixels": int((nz & seat_exception).sum()),
    }
for name, m in protected.items():
    report["protected_regions"][name] = {
        "pixels": int(m.sum()),
        "bbox_xyxy_exclusive": bbox(m),
    }
report["checks"] = {
    "core_binary": set(np.unique(masks["shirt_core_mask"])) == {0, 255},
    "generation_binary": set(np.unique(masks["shirt_generation_mask"])) == {0, 255},
    "core_subset_generation": bool(np.all(~core | generation)),
    "composite_nonzero_outside_generation": int(np.count_nonzero(weights[~generation])),
    "all_declared_protected_intersections_zero": all(
        not np.any((m > 0) & protected_union) for m in masks.values()
    ),
    "source_unchanged": hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED,
}
(OUT / "mask_candidates_report.json").write_text(json.dumps(report, indent=2) + "\n")
# Full-resolution tiles: no inspection panel downsampling needed.
items = [
    ("SOURCE — UNCHANGED", source),
    ("CORE — CYAN OVERLAY", Image.open(OUT / "shirt_core_mask_overlay.png")),
    ("CORE — BINARY", Image.open(OUT / "shirt_core_mask.png").convert("RGB")),
    (
        "GENERATION — BINARY",
        Image.open(OUT / "shirt_generation_mask.png").convert("RGB"),
    ),
    (
        "GENERATION — ORANGE OVERLAY",
        Image.open(OUT / "shirt_generation_mask_overlay.png"),
    ),
    (
        "COMPOSITE — GRAYSCALE WEIGHTS",
        Image.open(OUT / "shirt_composite_mask.png").convert("RGB"),
    ),
    (
        "COMPOSITE — MAGENTA OVERLAY",
        Image.open(OUT / "shirt_composite_mask_overlay.png"),
    ),
]
sheet = Image.new("RGB", (W * 3, (H + 58) * 3), (27, 29, 34))
d = ImageDraw.Draw(sheet)
for i, (label, im) in enumerate(items):
    x = (i % 3) * W
    y = (i // 3) * (H + 58)
    d.text((x + 16, y + 13), label, font=font, fill="white")
    sheet.paste(im, (x, y + 58))
d.text(
    (W + 24, 2 * (H + 58) + 50),
    "CANDIDATES — NOT APPROVED",
    font=font,
    fill=(255, 190, 70),
)
d.text(
    (W + 24, 2 * (H + 58) + 100),
    "Every tile uses original 1160 x 896 pixels.",
    font=font,
    fill="white",
)
d.text(
    (W + 24, 2 * (H + 58) + 150),
    "Inspect boundaries before authorizing use.",
    font=font,
    fill="white",
)
sheet.save(OUT / "mask_candidates_contact_sheet.png")
sys.stdout.write(
    json.dumps(
        {
            "output_directory": str(OUT),
            "checks": report["checks"],
            "mask_summary": {
                k: {
                    f: v[f]
                    for f in [
                        "bbox_xyxy_exclusive",
                        "nonzero_pixels",
                        "white_255_pixels",
                        "blend_1_to_254_pixels",
                        "editable_red_seat_exception_pixels",
                    ]
                }
                for k, v in report["masks"].items()
            },
        },
        indent=2,
    )
)
