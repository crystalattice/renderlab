from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import file_sha256


STAGE_TYPES = {"environment", "body", "garment", "identity", "surface", "repair", "integration"}
STAGE_STATUSES = {"pending", "running", "completed", "failed", "skipped", "escalated"}
RESULT_STATUSES = STAGE_STATUSES - {"pending"}
VALIDATION_DECISIONS = {"accept", "retry", "escalate", "reject"}


class RenderRunError(RuntimeError):
    pass


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderRunError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RenderRunError(f"{path}: root must be a JSON object")
    return value


def _write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_render_spec(spec: dict[str, Any]) -> None:
    errors = []
    if spec.get("schema") != "renderlab.render-spec.v1":
        errors.append("schema must be renderlab.render-spec.v1")
    if not isinstance(spec.get("run_id"), str) or not spec.get("run_id", "").strip():
        errors.append("run_id must be a non-empty string")
    if not isinstance(spec.get("scene"), dict):
        errors.append("scene must be an object")
    if not isinstance(spec.get("canon"), dict):
        errors.append("canon must be an object")
    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty array")
        stages = []
    ids = set()
    previous_ids = set()
    for index, stage in enumerate(stages, 1):
        if not isinstance(stage, dict):
            errors.append(f"stage {index} must be an object")
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            errors.append(f"stage {index}: id must be a non-empty string")
        elif stage_id in ids:
            errors.append(f"stage {index}: duplicate id {stage_id}")
        else:
            ids.add(stage_id)
        if stage.get("type") not in STAGE_TYPES:
            errors.append(f"stage {index}: invalid type {stage.get('type')}")
        if not isinstance(stage.get("backend"), dict) or not stage["backend"].get("id"):
            errors.append(f"stage {index}: backend.id is required")
        owns = stage.get("owns")
        preserves = stage.get("preserves")
        if not isinstance(owns, list) or not owns:
            errors.append(f"stage {index}: owns must be a non-empty array")
        if not isinstance(preserves, list):
            errors.append(f"stage {index}: preserves must be an array")
        elif isinstance(owns, list) and set(owns) & set(preserves):
            errors.append(f"stage {index}: owns and preserves must not overlap")
        depends_on = stage.get("depends_on", [])
        if not isinstance(depends_on, list):
            errors.append(f"stage {index}: depends_on must be an array")
        else:
            unknown = [value for value in depends_on if value not in previous_ids]
            if unknown:
                errors.append(f"stage {index}: dependencies must reference earlier stages: {', '.join(unknown)}")
        if isinstance(stage_id, str):
            previous_ids.add(stage_id)
    if errors:
        raise RenderRunError("render spec validation failed:\n" + "\n".join(errors))


def initialize_render_run(spec_path: Path, run_dir: Path) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    spec = _load_object(spec_path)
    validate_render_spec(spec)
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_spec = {**spec, "source": {"path": str(spec_path), "sha256": file_sha256(spec_path)}}
    _write_object(run_dir / "spec.json", resolved_spec)
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "schema": "renderlab.render-run.v1",
        "run_id": spec["run_id"],
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "stages": [
            {"id": stage["id"], "type": stage["type"], "status": "pending", "attempts": []}
            for stage in spec["stages"]
        ],
    }
    _write_object(run_dir / "run.json", state)
    return state


def record_stage_result(run_dir: Path, stage_id: str, result_path: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    state = _load_object(run_path)
    spec = _load_object(run_dir / "spec.json")
    result = _load_object(result_path)
    if result.get("schema") != "renderlab.stage-result.v1":
        raise RenderRunError("result schema must be renderlab.stage-result.v1")
    if result.get("stage_id") != stage_id:
        raise RenderRunError(f"result stage_id must be {stage_id}")
    if result.get("status") not in RESULT_STATUSES:
        raise RenderRunError(f"invalid result status {result.get('status')}")
    decision = result.get("validation", {}).get("decision")
    if decision not in VALIDATION_DECISIONS:
        raise RenderRunError("validation.decision must be accept, retry, escalate, or reject")
    spec_stage = next((stage for stage in spec["stages"] if stage["id"] == stage_id), None)
    state_stage = next((stage for stage in state["stages"] if stage["id"] == stage_id), None)
    if spec_stage is None or state_stage is None:
        raise RenderRunError(f"unknown stage {stage_id}")
    for dependency in spec_stage.get("depends_on", []):
        dependency_state = next(stage for stage in state["stages"] if stage["id"] == dependency)
        if dependency_state["status"] not in {"completed", "skipped"}:
            raise RenderRunError(f"stage {stage_id} dependency {dependency} is not complete")
    output = result.get("output")
    if result["status"] == "completed":
        if not isinstance(output, dict) or not output.get("path"):
            raise RenderRunError("completed result requires output.path")
        output_path = Path(output["path"]).expanduser().resolve()
        if not output_path.is_file():
            raise RenderRunError(f"stage output does not exist: {output_path}")
        actual_hash = file_sha256(output_path)
        if output.get("sha256") not in {None, actual_hash}:
            raise RenderRunError("stage output sha256 does not match the file")
        result["output"] = {**output, "path": str(output_path), "sha256": actual_hash}
    attempt = len(state_stage["attempts"]) + 1
    recorded_path = run_dir / "stages" / stage_id / f"attempt-{attempt:03d}.json"
    recorded = {
        **result,
        "attempt": attempt,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(result_path.resolve()), "sha256": file_sha256(result_path)},
    }
    _write_object(recorded_path, recorded)
    state_stage["attempts"].append(str(recorded_path.relative_to(run_dir)))
    state_stage["status"] = (
        "pending" if decision == "retry" else
        "escalated" if decision == "escalate" else
        result["status"]
    )
    statuses = {stage["status"] for stage in state["stages"]}
    state["status"] = (
        "failed" if "failed" in statuses else
        "escalated" if "escalated" in statuses else
        "completed" if statuses <= {"completed", "skipped"} else
        "running" if any(stage["attempts"] for stage in state["stages"]) else
        "pending"
    )
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_object(run_path, state)
    return recorded
