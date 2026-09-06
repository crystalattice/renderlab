"""Validate preparation and CPU compositing; never loads a diffusion model or calls Cloud."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

P = Path(__file__).resolve().parent
ROOT = Path("/home/codyjackson/PycharmProjects/renderlab")
BASE = "1a50854cc0824d24178c7c1d03f9ab185d0e7eb0"


def read(name):
    return json.loads((P / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_array(path, mode):
    with Image.open(path) as image:
        assert image.mode == mode and image.size == (1160, 896), (
            path,
            image.mode,
            image.size,
        )
        return np.array(image)


def mask_checks(core, generation, composite):
    assert set(np.unique(core)) == {0, 255}
    assert set(np.unique(generation)) == {0, 255}
    counts = {
        "core_outside_generation": int(((core == 255) & (generation != 255)).sum()),
        "core_underweight_pixels": int(((core == 255) & (composite != 255)).sum()),
        "feather_outside_margin": int(
            (
                (composite > 0)
                & (composite < 255)
                & ~((generation == 255) & (core == 0))
            ).sum()
        ),
        "nonzero_outside_generation": int(((generation == 0) & (composite != 0)).sum()),
    }
    assert all(v == 0 for v in counts.values()), counts
    return counts


def output_metrics(source, raw, production, core, generation):
    assert source.shape == raw.shape == production.shape == (896, 1160, 3)
    assert source.dtype == raw.dtype == production.dtype == np.uint8
    delta = np.abs(production.astype(np.int16) - source.astype(np.int16))
    outside = generation == 0
    metrics = {
        "outside_generation_pixels": int(outside.sum()),
        "outside_changed_pixels": int(np.any(delta[outside] != 0, axis=1).sum()),
        "outside_max_channel_error": int(delta[outside].max()),
        "core_pixels_different_from_raw": int(
            np.any(production[core == 255] != raw[core == 255], axis=1).sum()
        ),
        "dimensions": [1160, 896],
        "mode": "RGB",
        "alpha_added": False,
    }
    assert (
        metrics["outside_changed_pixels"]
        == metrics["outside_max_channel_error"]
        == metrics["core_pixels_different_from_raw"]
        == 0
    ), metrics
    return metrics


def validate_graph(graph, editor, specs):
    links = {edge[0]: edge for edge in editor["links"]}
    assert len(links) == len(editor["links"])
    nodes = {node["id"]: node for node in editor["nodes"]}
    assert set(map(str, nodes)) == set(graph)
    reconstructed = {}
    used_links = set()
    for node in editor["nodes"]:
        values = iter(node["widgets_values"])
        inputs = {}
        spec = specs[node["type"]]
        expected = graph[str(node["id"])]
        assert node["type"] == expected["class_type"]
        for field in spec["input_details"]:
            name = field["name"]
            socket = next((s for s in node["inputs"] if s["name"] == name), None)
            if socket is not None:
                if socket["link"] is None:
                    assert not field["required"]
                    continue
                edge = links[socket["link"]]
                used_links.add(edge[0])
                assert edge[3] == node["id"] and node["inputs"][edge[4]] == socket
                origin = nodes[edge[1]]["outputs"][edge[2]]
                assert (
                    edge[0] in origin["links"]
                    and edge[5] == origin["type"] == socket["type"]
                )
                inputs[name] = [str(edge[1]), edge[2]]
            elif name in expected["inputs"]:
                inputs[name] = next(values)
                if node["type"] == "KSampler" and name == "seed":
                    assert next(values) == "fixed"
            elif "default" in field:
                assert next(values) == field["default"]
        if node["type"] in ("LoadImage", "LoadImageMask"):
            assert next(values) == "image"
        assert list(values) == []
        reconstructed[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    assert reconstructed == graph and used_links == set(links)
    visited, active = set(), set()

    def visit(key):
        assert key not in active, "Graph cycle"
        if key in visited:
            return
        active.add(key)
        for name, value in graph[key]["inputs"].items():
            field = next(
                f
                for f in specs[graph[key]["class_type"]]["input_details"]
                if f["name"] == name
            )
            if isinstance(value, list):
                assert value[0] in graph
                assert (
                    specs[graph[value[0]]["class_type"]]["outputs"][value[1]]
                    == field["type"]
                )
                visit(value[0])
            elif field["type"] == "COMBO" and name != "image":
                assert value in field["options"], (key, name, value)
        active.remove(key)
        visited.add(key)

    visit("14")
    visit("18")
    assert visited == set(graph)


def validate(raw_path=None, production_path=None):
    manifest = read("manifest.json")
    masks = P / "mask_candidates_v1"
    core, generation, composite = [
        image_array(masks / f"shirt_{kind}_mask.png", "L")
        for kind in ("core", "generation", "composite")
    ]
    for kind, expected in manifest["approved_mask_hashes"].items():
        assert digest(masks / f"shirt_{kind}_mask.png") == expected
    assert (
        digest(masks / "shirt_composite_mask.png")
        == manifest["composite_reference_sha256"]
    )
    for asset in manifest["mask_assets"]:
        assert digest(P / asset["path"]) == asset["sha256"]
        rgb = image_array(P / asset["path"], "RGB")
        reference = image_array(P / asset["derived_from"], "L")
        assert all(
            np.array_equal(rgb[:, :, channel], reference) for channel in range(3)
        )
    result = {
        "status": "PASS",
        "mask_invariants": mask_checks(core, generation, composite),
    }
    source_path = Path(manifest["source"]["path"])
    assert digest(source_path) == manifest["source"]["sha256"]
    source = image_array(source_path, "RGB")
    graph = read("api.cloud.resolved.json")
    prepared = read("api.prepared.json")
    for key in graph:
        for field, value in graph[key]["inputs"].items():
            if key in ("1", "2", "19") and field == "image":
                continue
            assert prepared[key]["inputs"][field] == value
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
    unchanged = [str(n) for n in (1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13)]
    for key in unchanged:
        assert graph[key] == baseline[key], key
    assert set(graph) == (set(baseline) - {"15", "16"}) | {"19"}
    assert graph["8"]["inputs"]["prompt"] == manifest["prompt"]
    assert {k: v for k, v in graph["8"]["inputs"].items() if k != "prompt"} == {
        k: v for k, v in baseline["8"]["inputs"].items() if k != "prompt"
    }
    upload_evidence = {
        u["node"]: u for u in read("cloud_upload_evidence.json")["uploads"]
    }
    pending = read("api.cloud.pending.json")
    expected_resolved = json.loads(json.dumps(pending))
    for key, evidence in upload_evidence.items():
        assert (
            evidence["put_exit_code"] == 0
            and evidence["put_attempts"] == 1
            and evidence["retry_attempts"] == 0
        )
        assert (
            evidence["sha256_before_upload"]
            == evidence["sha256_after_upload"]
            == digest(Path(evidence["path"]))
        )
        assert (
            evidence["put_response"]["subfolder"] == ""
            and evidence["put_response"]["type"] == "input"
        )
        expected_resolved[key]["inputs"]["image"] = evidence["put_response"]["name"]
    assert graph == expected_resolved
    for key, kind in (("2", "generation"), ("19", "composite")):
        assert graph[key] == {
            "class_type": "LoadImageMask",
            "inputs": {
                "image": upload_evidence[key]["put_response"]["name"],
                "channel": "red",
            },
        }
    assert graph["17"] == {
        "class_type": "ImageCompositeMasked",
        "inputs": {**baseline["17"]["inputs"], "mask": ["19", 0]},
    }
    for key, kind, origin in (("14", "raw", "13"), ("18", "production", "17")):
        assert graph[key]["inputs"] == {
            "images": [origin, 0],
            "filename_prefix": f"renderlab_firered_outfit_change_v1_s3407_{kind}",
        }
    specs = read("cloud_availability.json")["node_schemas"]
    validate_graph(graph, read("workflow.json"), specs)
    result["structural_comparison"] = {
        "base_commit": BASE,
        "unchanged_nodes": unchanged,
        "changed_nodes": ["2", "8", "14", "17", "18"],
        "removed_nodes": ["15", "16"],
        "added_nodes": ["19"],
        "permitted_changes": "mask source and composite mask edge, outfit prompt, output prefixes only",
        "editor_roundtrip": True,
        "acyclic_all_nodes_reach_save": True,
    }
    assert read("cloud_dry_run_request.json") == {"workflow": graph, "dry_run": True}
    verdict = read("cloud_dry_run_response.json")["structuredContent"]["result"]
    assert verdict["status"] == "validated" and verdict["submitted"] is False
    assert len(verdict["warnings"]) == 2
    for warning, key in zip(verdict["warnings"], ("2", "19")):
        assert (
            warning["code"] == "input_validation"
            and f"Node #{key} (LoadImageMask)" in warning["detail"]
            and graph[key]["inputs"]["image"] in warning["detail"]
        )
    result["cloud"] = {
        "status": "validated",
        "submitted": False,
        "filename_advisories": 2,
        "runtime_file_resolution_verified": False,
        "execution_ready": False,
    }
    return result, source, core, generation, composite


def native_composite(source, raw, weights):
    # CPU image arithmetic only: no sampler, VAE, text encoder or diffusion model executes.
    from comfy_extras.nodes_mask import ImageCompositeMasked

    result = ImageCompositeMasked.execute(
        torch.from_numpy(source.astype(np.float32) / 255)[None],
        torch.from_numpy(raw.astype(np.float32) / 255)[None],
        0,
        0,
        False,
        torch.from_numpy(weights.astype(np.float32) / 255)[None],
    )[0]
    return np.clip(result.numpy()[0] * 255, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--production", type=Path)
    args = parser.parse_args()
    assert bool(args.raw) == bool(args.production), (
        "Supply both raw and production for output validation"
    )
    sys.path.insert(0, str(ROOT))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    result, source, core, generation, composite = validate()
    import nodes
    import folder_paths

    folder_paths.set_input_directory(str(P / "cloud_assets"))
    for kind, expected in [("generation", generation), ("composite", composite)]:
        loaded = (
            nodes.LoadImageMask()
            .load_image_mask(f"shirt_{kind}_mask.rgb.png", "red")[0]
            .cpu()
            .numpy()[0]
        )
        assert np.array_equal(loaded, expected.astype(np.float32) / 255)
    result["native_load_image_mask_red_channel_equivalent"] = True
    # Adversarial full-frame difference, not an outfit-generation result.
    raw = 255 - source
    production = native_composite(source, raw, composite)
    result["synthetic_native_cpu_containment"] = output_metrics(
        source, raw, production, core, generation
    )
    result["synthetic_fixture_is_inference"] = False
    if args.raw:
        raw = image_array(args.raw, "RGB")
        production = image_array(args.production, "RGB")
        expected = native_composite(source, raw, composite)
        assert np.array_equal(expected, production), (
            "Production differs from native full composite"
        )
        result["actual_output"] = output_metrics(
            source, raw, production, core, generation
        )
        result["actual_output"]["raw_sha256"] = digest(args.raw)
        result["actual_output"]["production_sha256"] = digest(args.production)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
