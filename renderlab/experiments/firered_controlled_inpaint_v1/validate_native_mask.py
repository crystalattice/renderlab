"""Validate the native mask on CPU without loading diffusion models."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Native ComfyUI imports must use CPU; only mask tensor operations are executed.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
from comfy_extras.nodes_mask import MaskComposite, SolidMask

P = Path(__file__).parent
BASE = "3289dfb2"


def validate():
    graph = json.loads((P / "api.cloud.resolved.json").read_text())
    prepared = json.loads((P / "api.prepared.json").read_text())
    baseline = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                f"{BASE}:renderlab/experiments/firered_controlled_inpaint_v1/api.cloud.resolved.json",
            ],
            cwd=ROOT,
            text=True,
        )
    )
    assert set(graph) == set(baseline) | {"17", "18"}
    assert all(graph[k] == baseline[k] for k in baseline)
    assert graph["2"] == {
        "class_type": "MaskComposite",
        "inputs": {
            "destination": ["15", 0],
            "source": ["16", 0],
            "x": 552,
            "y": 424,
            "operation": "add",
        },
    }
    assert graph["15"] == {
        "class_type": "SolidMask",
        "inputs": {"value": 0.0, "width": 1160, "height": 896},
    }
    assert graph["16"] == {
        "class_type": "SolidMask",
        "inputs": {"value": 1.0, "width": 128, "height": 128},
    }
    assert graph["17"] == {
        "class_type": "ImageCompositeMasked",
        "inputs": {
            "destination": ["1", 0],
            "source": ["13", 0],
            "x": 0,
            "y": 0,
            "resize_source": False,
            "mask": ["2", 0],
        },
    }
    assert graph["18"] == {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["17", 0],
            "filename_prefix": "renderlab_firered_controlled_inpaint_v1_s3407_production",
        },
    }
    assert graph["11"]["inputs"]["mask"] == ["2", 0]
    assert all(prepared[k] == graph[k] for k in graph if k != "1")
    assert graph["1"]["class_type"] == prepared["1"]["class_type"] == "LoadImage"
    assert set(prepared["1"]["inputs"]) == set(graph["1"]["inputs"]) == {"image"}
    assert not any(n["class_type"] == "LoadImageMask" for n in graph.values())
    canvas = SolidMask.execute(**graph["15"]["inputs"])[0]
    rectangle = SolidMask.execute(**graph["16"]["inputs"])[0]
    generated = (
        MaskComposite.execute(canvas, rectangle, 552, 424, "add")[0].cpu().numpy()[0]
    )
    raw = (P / "mask.png").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "1a6430abf64aa59c03d06acee164075f6481c7520aaa6a47e4bab9c15a3c86c7"
    )
    with Image.open(P / "mask.png") as image:
        assert image.mode == "RGB" and image.size == (1160, 896)
        reference = np.asarray(image)
    assert generated.shape == (896, 1160)
    assert np.array_equal(generated, reference[:, :, 0].astype(np.float32) / 255)
    rgb = np.repeat((generated * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    assert rgb.tobytes() == reference.tobytes()
    assert np.count_nonzero(generated) == 16384
    assert set(np.unique(generated)) == {0.0, 1.0}
    editor = json.loads((P / "workflow.json").read_text())
    specs = json.loads((P / "cloud_availability.json").read_text())["node_schemas"]
    links = {edge[0]: edge for edge in editor["links"]}
    nodes = {n["id"]: n for n in editor["nodes"]}
    reconstructed = {}
    for node in editor["nodes"]:
        values = iter(node["widgets_values"])
        inputs = {}
        for field in specs[node["type"]]["input_details"]:
            socket = next(
                (s for s in node["inputs"] if s["name"] == field["name"]), None
            )
            if socket is not None:
                if socket["link"] is not None:
                    edge = links[socket["link"]]
                    assert edge[3] == node["id"] and node["inputs"][edge[4]] == socket
                    origin = nodes[edge[1]]["outputs"][edge[2]]
                    assert (
                        edge[0] in origin["links"]
                        and edge[5] == origin["type"] == socket["type"]
                    )
                    inputs[field["name"]] = [str(edge[1]), edge[2]]
            elif field["name"] in graph[str(node["id"])]["inputs"]:
                inputs[field["name"]] = next(values)
                if node["type"] == "KSampler" and field["name"] == "seed":
                    assert next(values) == "fixed"
        if node["type"] == "LoadImage":
            assert next(values) == "image"
        assert list(values) == []
        reconstructed[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    assert reconstructed == graph
    visited = set()
    active = set()

    def visit(key):
        assert key not in active, "Cycle in graph"
        if key in visited:
            return
        active.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list):
                visit(value[0])
        active.remove(key)
        visited.add(key)

    visit("14")
    visit("18")
    assert visited == set(graph)
    return {
        "status": "PASS",
        "base_commit": BASE,
        "node_count": len(graph),
        "changed_existing_nodes": [],
        "added_nodes": ["17", "18"],
        "all_preexisting_nodes_and_inputs_unchanged": True,
        "source_binding_unchanged": True,
        "set_latent_noise_mask_connection_unchanged": True,
        "mask_dimensions": [1160, 896],
        "editable_pixels": int(np.count_nonzero(generated)),
        "normalized_mask_pixel_equivalent": True,
        "decoded_rgb_bytes_identical": True,
        "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "reference_png_sha256": hashlib.sha256(raw).hexdigest(),
        "comparison_scope": "Decoded pixels and RGB bytes, not PNG compressed container bytes.",
        "native_implementation": "comfy_extras/nodes_mask.py: SolidMask.execute, MaskComposite.execute; CPU",
        "native_implementation_sha256": hashlib.sha256(
            (ROOT / "comfy_extras/nodes_mask.py").read_bytes()
        ).hexdigest(),
        "editor_roundtrip_equal": True,
        "acyclic_all_nodes_reach_raw_or_production_save": True,
        "resolved_sha256": hashlib.sha256(
            (P / "api.cloud.resolved.json").read_bytes()
        ).hexdigest(),
        "generation_executed": False,
    }


if __name__ == "__main__":
    sys.stdout.write(json.dumps(validate(), indent=2) + "\n")
