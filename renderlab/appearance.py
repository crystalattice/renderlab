from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PRESETS = PACKAGE_DIR / "presets" / "appearance_v1.json"

OPERATIONS = {"inpaint", "outpaint", "outfit_change", "unclothe", "reclothe", "face_swap"}
DEFAULT_PRESERVE = ["identity", "morphology", "pose", "camera", "lighting", "environment"]
BACKENDS = {
    "inpaint": "qwen-image-inpaint",
    "outpaint": "qwen-image-outpaint",
    "outfit_change": "qwen-image-edit",
    "unclothe": "qwen-image-edit",
    "reclothe": "qwen-image-edit",
    "face_swap": "unselected-face-swap",
}
BACKEND_BLUEPRINTS = {
    "qwen-image-edit": "blueprints/Image Edit (Qwen 2509).json",
    "qwen-image-inpaint": "blueprints/Image Inpainting (Qwen-image).json",
    "qwen-image-outpaint": "blueprints/Image Outpainting (Qwen-Image).json",
}
DEFAULT_CONTROLS = {
    "outpaint": {"left": 0, "top": 0, "right": 0, "bottom": 512},
}


class AppearanceError(RuntimeError):
    pass


def load_presets(path: Path = DEFAULT_PRESETS) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "renderlab.appearance-presets.v1":
        raise AppearanceError("appearance preset schema must be renderlab.appearance-presets.v1")
    presets = document.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise AppearanceError("appearance presets must be a non-empty object")
    return presets


def list_presets(path: Path = DEFAULT_PRESETS) -> list[dict[str, Any]]:
    presets = load_presets(path)
    return [
        {"id": preset_id, **preset}
        for preset_id, preset in sorted(presets.items())
    ]


def _merge_target(preset: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    target = dict(preset.get("target", {}))
    override = request.get("target", {})
    if not isinstance(override, dict):
        raise AppearanceError("target must be an object")
    target.update(override)
    return target


def plan_appearance(request_path: Path, preset_path: Path = DEFAULT_PRESETS) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema") != "renderlab.appearance-request.v1":
        raise AppearanceError("appearance request schema must be renderlab.appearance-request.v1")
    preset_id = request.get("preset")
    presets = load_presets(preset_path)
    if preset_id not in presets:
        raise AppearanceError(f"unknown appearance preset: {preset_id}")
    preset = presets[preset_id]
    operation = request.get("operation", preset.get("operation"))
    if operation not in OPERATIONS:
        raise AppearanceError(f"unsupported appearance operation: {operation}")
    source = request.get("source")
    if not isinstance(source, dict) or not source.get("path"):
        raise AppearanceError("source.path is required")
    mask = request.get("mask")
    if operation == "inpaint" and (not isinstance(mask, dict) or not mask.get("path")):
        raise AppearanceError("mask.path is required for inpaint")
    subject = request.get("subject", "subject:primary")
    preserve = request.get("preserve", DEFAULT_PRESERVE)
    if not isinstance(preserve, list) or not all(isinstance(item, str) and item for item in preserve):
        raise AppearanceError("preserve must be a list of non-empty strings")
    acceptance_override = request.get("acceptance", {})
    if not isinstance(acceptance_override, dict):
        raise AppearanceError("acceptance must be an object")
    acceptance = {
        "identity_preservation": 4,
        "morphology_preservation": 4,
        "pose_preservation": 4,
        "instruction_adherence": 4,
        "artifact_severity_max": 2,
        **acceptance_override,
    }
    target = _merge_target(preset, request)
    controls = {**DEFAULT_CONTROLS.get(operation, {})}
    control_override = request.get("controls", {})
    if not isinstance(control_override, dict):
        raise AppearanceError("controls must be an object")
    controls.update(control_override)
    backend_id = request.get("backend", BACKENDS[operation])
    transform = {
        "id": "appearance_transform",
        "type": operation,
        "backend": {
            "id": backend_id,
            "available": backend_id in BACKEND_BLUEPRINTS,
            "workflow_blueprint": BACKEND_BLUEPRINTS.get(backend_id),
        },
        "input": {
            "source": source,
            "mask": mask,
            "references": request.get("references", []),
        },
        "subject": subject,
        "target": target,
        "controls": controls,
        "preserves": preserve,
        "owns": ["appearance"],
        "depends_on": [],
        "acceptance": acceptance,
    }
    stages = [transform]
    if operation != "face_swap" and "identity" in preserve:
        identity_backend = request.get("identity_backend", "unselected-face-swap")
        stages.append({
            "id": "identity_repair",
            "type": "face_swap",
            "backend": {
                "id": identity_backend,
                "available": identity_backend in BACKEND_BLUEPRINTS,
                "workflow_blueprint": BACKEND_BLUEPRINTS.get(identity_backend),
            },
            "subject": subject,
            "runs_when": "identity_preservation < acceptance.identity_preservation",
            "owns": ["face_identity", "hairline", "apparent_age"],
            "preserves": [item for item in preserve if item != "identity"] + ["appearance"],
            "depends_on": ["appearance_transform"],
        })
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "renderlab.render-plan.v1",
        "request_sha256": hashlib.sha256(canonical).hexdigest(),
        "intent": {"preset": preset_id, "operation": operation, "target": target},
        "source_semantics": request.get("source_semantics", "editable_canvas"),
        "stages": stages,
        "result_contract": {
            "record": ["input_hashes", "output_hash", "backend", "seed", "prompt", "metrics", "parent_stage"],
            "acceptance": acceptance,
        },
    }
