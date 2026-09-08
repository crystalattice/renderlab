"""Offline preparation of versioned, externally authored API workflow profiles."""

import base64
import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
POLICIES = {"preserve_visible_geometry", "allow_hidden_region_reconstruction"}


class ProfileError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def profile_hash(profile):
    return digest(canonical({k: v for k, v in profile.items() if k != "content_hash"}))


def read_json(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ProfileError(f"Cannot read JSON {path}: {exc}") from exc


def require(condition, message):
    if not condition:
        raise ProfileError(message)


def fields(value, required, optional, label):
    require(isinstance(value, dict), f"{label} must be an object")
    require(not required - value.keys(), f"{label}: missing fields {sorted(required - value.keys())}")
    require(not value.keys() - required - optional, f"{label}: unknown fields {sorted(value.keys() - required - optional)}")


def typed(value, binding, label):
    kind = binding["type"]
    valid = {"string": type(value) is str, "integer": type(value) is int, "number": type(value) in (int, float), "boolean": type(value) is bool}
    require(kind in valid and valid[kind], f"{label}: expected {kind}")
    if kind in {"integer", "number"}:
        require(type(value) is int or math.isfinite(value), f"{label}: non-finite number")
        require("minimum" not in binding or value >= binding["minimum"], f"{label}: below minimum")
        require("maximum" not in binding or value <= binding["maximum"], f"{label}: above maximum")


def profile_path(value):
    path = Path(value).expanduser()
    return (PROFILE_DIR / (str(value) + ".json") if len(path.parts) == 1 and path.suffix != ".json" else path).resolve()


def load_profile(value):
    path = profile_path(value)
    p = read_json(path)
    required = {"schema_version", "profile_id", "capability", "backend", "workflow", "resource_inventory_path", "bindings", "prompt_binding", "source_image_bindings", "seed_bindings", "numeric_bindings", "model_references", "asset_references", "output_nodes", "reconstruction_policy", "immutable_source_policy", "metadata_capture_policy", "content_hash"}
    fields(p, required, {"annotations"}, "profile")
    require(type(p["schema_version"]) is int and p["schema_version"] == 1, "Unsupported profile schema_version")
    for name in ("profile_id", "capability", "backend", "resource_inventory_path", "content_hash"):
        require(isinstance(p[name], str) and bool(p[name]), f"{name} must be a nonempty string")
    require(isinstance(p.get("annotations", {}), dict), "annotations must be an object")
    require(p["reconstruction_policy"] in POLICIES if isinstance(p["reconstruction_policy"], str) else False, "Invalid reconstruction_policy")
    require(p["immutable_source_policy"] == "read_only_verified", "immutable_source_policy must be read_only_verified")
    require(p["metadata_capture_policy"] == "complete_v1", "metadata_capture_policy must be complete_v1")
    fields(p["workflow"], {"path", "sha256"}, set(), "workflow")
    for name in ("path", "sha256"):
        require(isinstance(p["workflow"][name], str) and p["workflow"][name], f"workflow.{name} must be a string")
    require(isinstance(p["bindings"], dict) and p["bindings"], "bindings must be a nonempty object")
    targets = set()
    for name, b in p["bindings"].items():
        fields(b, {"node", "input", "type"}, {"default", "minimum", "maximum"}, f"binding {name}")
        require(all(isinstance(b[k], str) and b[k] for k in ("node", "input", "type")), f"binding {name}: invalid target/type")
        require(b["type"] in {"string", "integer", "number", "boolean"}, f"binding {name}: unknown type")
        for bound in ("minimum", "maximum"):
            if bound in b:
                require(b["type"] in {"integer", "number"} and type(b[bound]) in (int, float) and (type(b[bound]) is int or math.isfinite(b[bound])), f"binding {name}: invalid {bound}")
        require(not {"minimum", "maximum"} <= b.keys() or b["minimum"] <= b["maximum"], f"binding {name}: reversed bounds")
        target = (b["node"], b["input"])
        require(target not in targets, f"Duplicate binding target {target}")
        targets.add(target)
        if "default" in b:
            typed(b["default"], b, name)
    for role in ("source_image_bindings", "seed_bindings", "numeric_bindings", "output_nodes"):
        require(isinstance(p[role], list) and all(isinstance(v, str) for v in p[role]), f"{role} must be a string list")
        require(len(p[role]) == len(set(p[role])), f"Duplicate {role}")
    require(p["output_nodes"], "output_nodes must not be empty")
    require(isinstance(p["prompt_binding"], str), "prompt_binding must name a binding")
    roles = [p["prompt_binding"], *p["source_image_bindings"], *p["seed_bindings"], *p["numeric_bindings"]]
    require(len(roles) == len(set(roles)), "Binding roles must not overlap")
    for name in roles:
        require(name in p["bindings"], f"Missing binding {name}")
        expected = {"string"} if name == p["prompt_binding"] or name in p["source_image_bindings"] else ({"integer"} if name in p["seed_bindings"] else {"integer", "number"})
        require(p["bindings"][name]["type"] in expected, f"Wrong type for binding {name}")
    require(isinstance(p["model_references"], dict), "model_references must be an object")
    require(isinstance(p["asset_references"], dict), "asset_references must be an object")
    require(set(p["source_image_bindings"]) <= p["asset_references"].keys(), "Missing source asset reference")
    require(not p["model_references"].keys() & p["asset_references"].keys(), "Model and asset bindings overlap")
    for name, model in p["model_references"].items():
        require(isinstance(model, str) and model, f"Invalid model reference {name}")
    for name, asset in p["asset_references"].items():
        fields(asset, {"path", "sha256", "backend_value"}, set(), f"asset {name}")
        require(all(isinstance(v, str) and v for v in asset.values()), f"Invalid asset reference {name}")
    for name in [*p["model_references"], *p["asset_references"]]:
        require(name in p["bindings"] and p["bindings"][name]["type"] == "string", f"Missing string binding for reference {name}")
    require(p["content_hash"] == profile_hash(p), "Profile content hash mismatch")
    return path, p


def file_bytes(path, label):
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProfileError(f"Missing/unreadable {label} {path}: {exc}") from exc


def check_graph(graph):
    require(isinstance(graph, dict) and graph, "Workflow must be an API-format object")
    edges = {}
    for node, data in graph.items():
        require(isinstance(data, dict) and isinstance(data.get("class_type"), str) and isinstance(data.get("inputs"), dict), f"Invalid API node {node}")
        edges[node] = []
        for name, value in data["inputs"].items():
            if isinstance(value, list):
                require(len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int and value[1] >= 0, f"Invalid connection {node}.{name}")
                require(value[0] in graph, f"Missing connected node {value[0]}")
                edges[node].append(value[0])
    done, active = set(), set()
    def visit(node):
        require(node not in active, f"Workflow cycle at node {node}")
        if node in done:
            return
        active.add(node)
        for dependency in edges[node]:
            visit(dependency)
        active.remove(node)
        done.add(node)
    for node in graph:
        visit(node)


def prepare_job(profile, values=None, backend=None, parent_job=None):
    path, p = load_profile(profile)
    values = {} if values is None else values
    require(isinstance(values, dict), "Bindings must be a JSON object")
    require(not values.keys() - p["bindings"].keys(), f"Unknown bindings: {sorted(values.keys() - p['bindings'].keys())}")
    backend = p["backend"] if backend is None else backend
    require(isinstance(backend, str) and backend, "backend must be a nonempty identifier")
    require(parent_job is None or isinstance(parent_job, str) and parent_job, "parent_job must be an identifier")
    workflow_path = (path.parent / p["workflow"]["path"]).resolve()
    raw = file_bytes(workflow_path, "workflow")
    require(digest(raw) == p["workflow"]["sha256"], "Base workflow hash mismatch")
    try:
        graph = json.loads(raw)
    except ValueError as exc:
        raise ProfileError(f"Invalid workflow JSON {workflow_path}: {exc}") from exc
    check_graph(graph)
    resolved = copy.deepcopy(graph)
    inventory = read_json(path.parent / p["resource_inventory_path"])
    fields(inventory, {"models", "assets"}, {"annotations"}, "resource inventory")
    for kind in ("models", "assets"):
        require(isinstance(inventory[kind], list) and all(isinstance(v, str) for v in inventory[kind]), f"Invalid inventory {kind}")
    bound = {}
    for name, b in p["bindings"].items():
        require(b["node"] in graph, f"Missing node {b['node']} for binding {name}")
        require(b["input"] in graph[b["node"]]["inputs"], f"Missing input {b['node']}.{b['input']}")
        require(not isinstance(graph[b["node"]]["inputs"][b["input"]], list), f"Cannot replace connection with binding {name}")
        require(name in values or "default" in b, f"Missing binding value {name}")
        value = values[name] if name in values else b["default"]
        typed(value, b, name)
        if name in p["seed_bindings"]:
            require(0 <= value < 2**64, f"Seed {name} outside unsigned 64-bit range")
        bound[name] = value
        resolved[b["node"]]["inputs"][b["input"]] = value
    for name, model in p["model_references"].items():
        require(model in inventory["models"], f"Missing model in supplied inventory: {model}")
        require(bound[name] == model, f"Model binding {name} differs from declared reference")
    assets = {}
    for name, ref in p["asset_references"].items():
        local = (path.parent / ref["path"]).expanduser().resolve()
        sha = digest(file_bytes(local, "asset"))
        require(sha == ref["sha256"], f"Asset hash mismatch: {name}")
        require(ref["backend_value"] in inventory["assets"], f"Missing backend asset in supplied inventory: {name}")
        require(bound[name] == ref["backend_value"], f"Asset binding {name} differs from declared reference")
        assets[name] = {"path": str(local), "sha256": sha, "backend_value": bound[name]}
    for node in p["output_nodes"]:
        require(node in resolved, f"Missing output node {node}")
    encoded = canonical(resolved)
    job = {
        "schema_version": 1, "job_id": str(uuid.uuid4()), "status": "prepared", "submitted": False,
        "created_at": datetime.now(timezone.utc).isoformat(), "profile_id": p["profile_id"],
        "profile_hash": p["content_hash"], "profile": p, "profile_path": str(path),
        "capability": p["capability"], "backend": backend, "base_workflow_hash": digest(raw),
        "base_workflow_bytes_base64": base64.b64encode(raw).decode("ascii"),
        "resolved_workflow_hash": digest(encoded), "workflow_bytes_base64": base64.b64encode(encoded).decode("ascii"),
        "workflow": resolved, "resolved_bindings": bound,
        "prompt_bytes_base64": base64.b64encode(bound[p["prompt_binding"]].encode("utf-8")).decode("ascii"),
        "numeric_settings": {name: bound[name] for name in p["seed_bindings"] + p["numeric_bindings"]},
        "assets": assets, "source_hashes": {name: assets[name]["sha256"] for name in p["source_image_bindings"]},
        "lineage": {"parent_job_id": parent_job, "sources": {name: assets[name] for name in p["source_image_bindings"]}},
        "model_references": p["model_references"], "resource_inventory": inventory,
        "resource_verification": "Supplied offline inventory only; no backend availability check performed",
        "output_nodes": p["output_nodes"], "reconstruction_policy": p["reconstruction_policy"],
        "immutable_source_policy": p["immutable_source_policy"], "metadata_capture_policy": p["metadata_capture_policy"],
    }
    job["record_hash"] = digest(canonical(job))
    return job


def replay_job(path):
    job = read_json(path)
    require(isinstance(job, dict) and job.get("schema_version") == 1, "Unsupported job record")
    require(job.get("record_hash") == digest(canonical({k: v for k, v in job.items() if k != "record_hash"})), "Job record hash mismatch")
    require(job["profile_hash"] == profile_hash(job["profile"]), "Recorded profile hash mismatch")
    raw = base64.b64decode(job["workflow_bytes_base64"], validate=True)
    require(digest(raw) == job["resolved_workflow_hash"] and json.loads(raw) == job["workflow"], "Recorded workflow hash/content mismatch")
    require(digest(base64.b64decode(job["base_workflow_bytes_base64"], validate=True)) == job["base_workflow_hash"], "Recorded base workflow hash mismatch")
    for name, asset in job["assets"].items():
        require(digest(file_bytes(Path(asset["path"]), "replay asset")) == asset["sha256"], f"Replay asset changed: {name}")
    return job


def write_job(path, job):
    """Never overwrite any existing file, including a source or workflow."""
    try:
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(job, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except OSError as exc:
        raise ProfileError(f"Cannot create job record {path}: {exc}") from exc
