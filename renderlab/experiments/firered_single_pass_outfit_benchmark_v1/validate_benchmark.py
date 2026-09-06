"""Local validation for three separately gated, unmasked FireRed cases."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
PROMPT_COMMIT = "d422bf22"
CASES = {
    "shoes_to_stilettos": "S0040",
    "dress_to_top_and_miniskirt": "S0534",
    "clothing_to_bikini": "S0040",
}
CLASSES = {
    "1": "LoadImage",
    "3": "UNETLoader",
    "4": "CLIPLoader",
    "5": "VAELoader",
    "6": "ModelSamplingAuraFlow",
    "7": "CFGNorm",
    "8": "TextEncodeQwenImageEditPlus",
    "9": "TextEncodeQwenImageEditPlus",
    "10": "VAEEncode",
    "12": "KSampler",
    "13": "VAEDecode",
    "14": "SaveImage",
}
SAMPLER = {
    "model": ["7", 0],
    "positive": ["8", 0],
    "negative": ["9", 0],
    "latent_image": ["10", 0],
    "seed": 3407,
    "steps": 40,
    "cfg": 4,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": 1,
}
MODEL_INPUTS = {
    "3": {
        "unet_name": "FireRed-Image-Edit-1.1-transformer.safetensors",
        "weight_dtype": "default",
    },
    "4": {
        "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "type": "qwen_image",
        "device": "default",
    },
    "5": {"vae_name": "qwen_image_vae.safetensors"},
}


def read(name):
    return json.loads((P / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_graph(graph, case, specs):
    assert {key: node["class_type"] for key, node in graph.items()} == CLASSES
    assert graph["1"]["inputs"] == {
        "image": read(case + "/cloud_bindings.json")["cloud_filename"]
    }
    for key, inputs in MODEL_INPUTS.items():
        assert graph[key]["inputs"] == inputs
    assert graph["6"]["inputs"] == {"model": ["3", 0], "shift": 3.1}
    assert graph["7"]["inputs"] == {"model": ["6", 0], "strength": 1}
    context = {"clip": ["4", 0], "vae": ["5", 0], "image1": ["1", 0]}
    prompt = (P / case / "prompt.txt").read_text().removesuffix("\n")
    assert graph["8"]["inputs"] == {**context, "prompt": prompt}
    assert graph["9"]["inputs"] == {**context, "prompt": ""}
    assert graph["10"]["inputs"] == {"pixels": ["1", 0], "vae": ["5", 0]}
    assert graph["12"]["inputs"] == SAMPLER
    assert graph["13"]["inputs"] == {"samples": ["12", 0], "vae": ["5", 0]}
    assert graph["14"]["inputs"] == {
        "images": ["13", 0],
        "filename_prefix": f"renderlab_firered_single_pass_outfit_benchmark_v1_{case}_s3407_native",
    }
    visited, active = set(), set()

    def visit(key):
        assert key not in active, "Cycle"
        if key in visited:
            return
        active.add(key)
        node = graph[key]
        fields = {
            field["name"]: field for field in specs[node["class_type"]]["input_details"]
        }
        for name, value in node["inputs"].items():
            field = fields[name]
            if isinstance(value, list):
                assert value[0] in graph
                assert (
                    specs[graph[value[0]]["class_type"]]["outputs"][value[1]]
                    == field["type"]
                )
                visit(value[0])
            elif field["type"] == "COMBO" and name != "image":
                assert value in field["options"], (key, name, value)
        assert all(
            field["name"] in node["inputs"]
            for field in fields.values()
            if field["required"]
        )
        active.remove(key)
        visited.add(key)

    visit("14")
    assert visited == set(graph)


def validate_editor(graph, editor, specs):
    nodes = {node["id"]: node for node in editor["nodes"]}
    links = {edge[0]: edge for edge in editor["links"]}
    assert len(links) == len(editor["links"]) and set(map(str, nodes)) == set(graph)
    restored, used = {}, set()
    for node in editor["nodes"]:
        expected = graph[str(node["id"])]
        assert node["type"] == expected["class_type"] and node["mode"] == 0
        values = iter(node["widgets_values"])
        inputs = {}
        for field in specs[node["type"]]["input_details"]:
            name = field["name"]
            socket = next((s for s in node["inputs"] if s["name"] == name), None)
            if socket is not None:
                if socket["link"] is None:
                    assert not field["required"] and name not in expected["inputs"]
                    continue
                edge = links[socket["link"]]
                used.add(edge[0])
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
        if node["type"] == "LoadImage":
            assert next(values) == "image"
        assert list(values) == []
        restored[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    assert restored == graph and used == set(links)


def validate_same_source(a, b):
    differences = []
    assert set(a) == set(b)
    for key in a:
        assert a[key]["class_type"] == b[key]["class_type"]
        assert set(a[key]["inputs"]) == set(b[key]["inputs"])
        differences.extend(
            f"{key}.{name}"
            for name in a[key]["inputs"]
            if a[key]["inputs"][name] != b[key]["inputs"][name]
        )
    assert differences == ["8.prompt", "14.filename_prefix"], differences
    return differences


def validate():
    specs = read("cloud_availability.json")["node_schemas"]
    models = [
        row["model_name"]
        for result in read("cloud_availability.json")["model_searches"]
        for row in result["data"]
    ]
    assert all(
        name in models
        for name in [
            MODEL_INPUTS["3"]["unet_name"],
            MODEL_INPUTS["4"]["clip_name"],
            MODEL_INPUTS["5"]["vae_name"],
        ]
    )
    bindings = read("cloud_bindings.json")
    assert bindings["unique_upload_count"] == 2 and not bindings["uploads_authorized"]
    assert {a["source_id"] for a in bindings["source_assets"]} == {"S0040", "S0534"}
    assets = {a["source_id"]: a for a in bindings["source_assets"]}
    evidence = read("cloud_upload_evidence.json")
    assert evidence["upload_count"] == 2 and evidence["retry_count"] == 0
    for source_id, asset in assets.items():
        upload = evidence["uploads"][source_id]
        assert upload["exit_code"] == 0 and upload["upload_attempts"] == 1
        assert upload["retries"] == 0
        assert upload["put_response"]["name"] == asset["cloud_filename"]
        assert upload["put_response"]["type"] == "input"
        assert upload["put_response"]["subfolder"] == ""
        assert (
            upload["sha256_before_upload"]
            == upload["sha256_after_upload"]
            == asset["sha256"]
        )
    report = {
        "status": "PASS_PREPARATION_ONLY",
        "execution_ready": False,
        "case_results": {},
        "unique_sources": 2,
        "uploads": 2,
        "inference_jobs": 0,
    }
    for case, source_id in CASES.items():
        m = read(case + "/manifest.json")
        src = m["source"]
        path = Path(src["original_path"])
        assert src["review_id"] == source_id and src["approval_received"] is True
        assert sha(path) == src["sha256"] == assets[source_id]["sha256"]
        assert src["original_path"] == assets[source_id]["original_path"]
        with Image.open(path) as image:
            assert (
                list(image.size) == src["dimensions"] == assets[source_id]["dimensions"]
            )
            assert image.mode == src["mode"] == "RGB"
            assert image.getexif().get(274, 1) == 1 and all(
                d % 8 == 0 for d in image.size
            )
        assert src["result_fits_existing_canvas"] and not any(
            src[key]
            for key in [
                "outpainting_required",
                "reframing_required",
                "off_frame_anatomy_required",
            ]
        )
        prompt_path = P / case / "prompt.txt"
        baseline_prompt = subprocess.check_output(
            ["git", "show", f"{PROMPT_COMMIT}:{prompt_path.relative_to(ROOT)}"],
            cwd=ROOT,
        )
        assert prompt_path.read_bytes() == baseline_prompt
        assert sha(prompt_path) == m["prompt_sha256"]
        assert sha(P / "settings.json") == m["settings_sha256"]
        graph = read(case + "/api.cloud.resolved.json")
        prepared = read(case + "/api.prepared.json")
        prepared["1"]["inputs"]["image"] = assets[source_id]["cloud_filename"]
        assert prepared == graph
        baseline = json.loads(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"142c5df9:{(P / case / 'api.cloud.pending.json').relative_to(ROOT)}",
                ],
                cwd=ROOT,
            )
        )
        baseline["1"]["inputs"]["image"] = assets[source_id]["cloud_filename"]
        assert baseline == graph
        validate_graph(graph, case, specs)
        validate_editor(graph, read(case + "/workflow.json"), specs)
        gate = read(case + "/cloud_submit.gated.json")
        assert gate["call_template"] == {
            "tool": "submit_workflow",
            "arguments": {"workflow": graph, "dry_run": False},
        }
        assert gate["max_jobs"] == gate["max_submission_attempts"] == 1
        assert not any(
            gate[key]
            for key in [
                "execution_ready",
                "uploads_authorized",
                "inference_authorized",
                "retries_authorized",
                "batch_allowed",
            ]
        )
        assert read(case + "/cloud_dry_run_request.json") == {
            "workflow": graph,
            "dry_run": True,
        }
        dry = read(case + "/cloud_dry_run_response.json")["structuredContent"]["result"]
        assert (
            dry["status"] == "validated"
            and dry["submitted"] is False
            and dry["warnings"] == []
        )
        rubric = read(case + "/evaluation_rubric.json")
        assert (
            len(rubric["independent_scores"]) == 12
            and len(rubric["hard_fail_conditions"]) == 9
        )
        assert all(
            v["score"] is None and v["range"] == [1, 5]
            for v in rubric["independent_scores"].values()
        )
        assert (
            rubric["independent_scores"]["artifact_severity"]["direction"]
            == "lower_is_better"
        )
        report["case_results"][case] = {
            "source_id": source_id,
            "source_sha256": src["sha256"],
            "dimensions": src["dimensions"],
            "source_hash_dimensions_mode": "PASS",
            "source_selection_user_approved": True,
            "adult_status_basis": src["adult_status"],
            "node_count": len(graph),
            "unique_node_types": len({n["class_type"] for n in graph.values()}),
            "typed_links_acyclic_all_nodes_reach_native_save": True,
            "editor_api_equivalent": True,
            "only_change_from_committed_pending_graph": "1.inputs.image",
            "settings_and_prompt_frozen": True,
            "no_mask_composite_outpaint_reframe_repair_nodes": True,
            "dry_run": dry,
            "file_uploaded": True,
            "payload_sha256": sha(P / case / "api.cloud.resolved.json"),
        }
    a = read("shoes_to_stilettos/api.cloud.resolved.json")
    b = read("clothing_to_bikini/api.cloud.resolved.json")
    report["same_source_comparison"] = {
        "status": "PASS",
        "only_differences": validate_same_source(a, b),
        "shared_source_sha256": assets["S0040"]["sha256"],
        "unique_uploads_for_pair": 1,
    }
    assert {use["case"] for use in assets["S0040"]["uses"]} == {
        "shoes_to_stilettos",
        "clothing_to_bikini",
    }
    assert not any(
        path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        for path in P.rglob("*")
        if path.is_file()
    )
    for prior in ["firered_controlled_inpaint_v1", "firered_outfit_change_v1"]:
        assert not subprocess.check_output(
            ["git", "diff", "2e37059c", "--", f"renderlab/experiments/{prior}"],
            cwd=ROOT,
        ), prior
    for path in P.rglob("*.json"):
        json.loads(path.read_text())
    report["previous_experiments_unchanged"] = True
    report["corpus_images_or_contact_sheets_in_repository"] = 0
    return report


if __name__ == "__main__":
    sys.stdout.write(json.dumps(validate(), indent=2) + "\n")
