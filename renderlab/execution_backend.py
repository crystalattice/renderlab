"""Single-attempt execution of the existing resolved workflow job contract."""

import base64
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from PIL import Image

from .workflow_profiles import digest, validate_job, write_job


class ExecutionError(RuntimeError):
    """Messages are fixed local diagnostics, never raw remote response bodies."""


def require(condition, message):
    if not condition:
        raise ExecutionError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


class ExecutionBackend(Protocol):
    name: str
    identity: str

    def readiness(self, job: dict) -> dict: ...
    def assets_present(self, job: dict) -> bool: ...
    def validate(self, workflow: bytes) -> dict: ...
    def submit(self, workflow: bytes) -> str: ...
    def status(self, job_id: str) -> dict: ...
    def outputs(self, job_id: str, nodes: list[str]) -> list[dict]: ...
    def download(self, output: dict) -> bytes: ...
    def cancel(self, job_id: str) -> None: ...


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ExecutionError("Backend redirect rejected; no request repeated")


class ComfyLocalBackend:
    name = "comfy-local"

    def __init__(self, server="http://127.0.0.1:8188", request_timeout=30):
        parsed = urlsplit(server)
        require(parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and parsed.path in {"", "/"}, "Invalid server URL; embedded credentials are not supported")
        require(math.isfinite(request_timeout) and 0 < request_timeout <= 60, "request_timeout must be in (0, 60]")
        self.server = server.rstrip("/")
        self.identity = self.server
        self.request_timeout = request_timeout
        self.opener = build_opener(NoRedirect())
        self.schemas = None

    def request(self, path, body=None, binary=False):
        request = Request(self.server + path, data=body, method="POST" if body is not None else "GET")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self.opener.open(request, timeout=self.request_timeout) as response:
                content = response.read(64 * 1024 * 1024 + 1)
            require(len(content) <= 64 * 1024 * 1024, "Backend response exceeds 64 MiB limit")
            return content if binary else json.loads(content)
        except HTTPError as exc:
            error = ExecutionError("Backend HTTP failure; request not retried")
            error.http_status = exc.code
            raise error from None
        except (URLError, OSError, ValueError):
            raise ExecutionError("Backend transport or JSON failure; request not retried") from None

    def readiness(self, job):
        self.schemas = self.request("/object_info")
        require(isinstance(self.schemas, dict) and self.schemas, "Malformed node catalog")
        for node in job["workflow"].values():
            require(node["class_type"] in self.schemas, "Required node class is unavailable")
        return {"ready": True, "cancel_supported": True}

    def assets_present(self, job):
        for asset in job["assets"].values():
            # /view without preview/channel returns the original file bytes.
            data = self.request("/view?" + urlencode({"filename": asset["backend_value"], "type": "input"}), binary=True)
            require(digest(data) == asset["sha256"], "Backend input missing or hash differs; separate authorized upload required")
        return True

    def validate(self, workflow):
        require(self.schemas is not None, "Readiness check required")
        graph = json.loads(workflow)
        for node in graph.values():
            schema = self.schemas[node["class_type"]]
            require(isinstance(schema, dict) and isinstance(schema.get("input"), dict), "Malformed node schema")
            definitions = {**schema["input"].get("required", {}), **schema["input"].get("optional", {})}
            require(set(schema["input"].get("required", {})) <= node["inputs"].keys(), "Required backend input missing")
            for key, value in node["inputs"].items():
                require(key in definitions, "Unknown backend input")
                definition = definitions[key]
                require(isinstance(definition, list) and definition, "Malformed backend input schema")
                kind = definition[0]
                if isinstance(value, list):
                    upstream = graph[value[0]]
                    outputs = self.schemas[upstream["class_type"]].get("output", [])
                    require(value[1] < len(outputs) and outputs[value[1]] == kind, "Backend connection type or slot mismatch")
                elif isinstance(kind, list):
                    require(value in kind, "Model, asset or combo value missing from backend catalog")
                elif kind in {"STRING", "INT", "FLOAT", "BOOLEAN"}:
                    types = {"STRING": (str,), "INT": (int,), "FLOAT": (int, float), "BOOLEAN": (bool,)}
                    require(type(value) in types[kind], "Backend scalar type mismatch")
                    bounds = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
                    if kind in {"INT", "FLOAT"}:
                        require(("min" not in bounds or value >= bounds["min"]) and ("max" not in bounds or value <= bounds["max"]), "Backend numeric range mismatch")
                else:
                    raise ExecutionError("Unsupported backend scalar schema")
        return {"status": "validated", "submitted": False, "warnings": [], "scope": "live_object_info_schema_only"}

    def submit(self, workflow):
        # Embed the exact resolved document, without loading/re-serializing it.
        result = self.request("/prompt", b'{"prompt":' + workflow + b'}')
        require(isinstance(result, dict) and not result.get("error") and not result.get("node_errors"), "Backend rejected submission; not retried")
        job_id = result.get("prompt_id")
        require(isinstance(job_id, str) and job_id, "Malformed submission response; submission state unknown, do not resubmit")
        return job_id

    def history(self, job_id):
        result = self.request("/history/" + quote(job_id, safe=""))
        require(isinstance(result, dict), "Malformed history response")
        return result.get(job_id)

    def status(self, job_id):
        history = self.history(job_id)
        if history is None:
            return {"status": "pending"}
        require(isinstance(history, dict) and isinstance(history.get("status"), dict), "Malformed job status")
        state = history["status"]
        if state.get("status_str") == "error":
            return {"status": "failed", "error_code": "backend_execution_error"}
        require(state.get("status_str") == "success" and state.get("completed") is True, "Unrecognized terminal history status")
        result = {"status": "completed"}
        messages = history.get("messages", state.get("messages", []))
        if isinstance(messages, list):
            stamps = {}
            for message in messages:
                if isinstance(message, list) and len(message) == 2 and message[0] in {"execution_start", "execution_success"} and isinstance(message[1], dict):
                    stamp = message[1].get("timestamp")
                    if type(stamp) in (int, float) and math.isfinite(stamp):
                        stamps[message[0]] = stamp
            if stamps.keys() >= {"execution_start", "execution_success"} and stamps["execution_success"] >= stamps["execution_start"]:
                result["duration_seconds"] = (stamps["execution_success"] - stamps["execution_start"]) / 1000
        return result

    def outputs(self, job_id, nodes):
        history = self.history(job_id)
        require(isinstance(history, dict) and isinstance(history.get("outputs"), dict), "Malformed output history")
        outputs = []
        for node in nodes:
            entry = history["outputs"].get(node)
            require(isinstance(entry, dict) and isinstance(entry.get("images"), list) and entry["images"], "Declared image output missing")
            for image in entry["images"]:
                require(isinstance(image, dict) and image.get("type") == "output", "Malformed output descriptor")
                outputs.append({"node_id": node, "filename": image.get("filename"), "subfolder": image.get("subfolder", ""), "type": "output"})
        return outputs

    def download(self, output):
        return self.request("/view?" + urlencode({key: output[key] for key in ("filename", "subfolder", "type")}), binary=True)

    def cancel(self, job_id):
        self.request("/queue", json.dumps({"delete": [job_id]}).encode())
        self.request("/interrupt", json.dumps({"prompt_id": job_id}).encode())


def run_job(job, backend, directory, *, execute=False, max_polls=120, poll_interval=2, sleep=time.sleep):
    """Write evidence before any submission; never retry or implicitly cancel."""
    validate_job(job)
    credential_inputs = {"api_key", "authorization", "password", "token", "access_token", "secret"}
    require(not any(credential_inputs & {key.lower() for key in node["inputs"]} for node in job["workflow"].values()), "Credential-bearing workflow inputs cannot be recorded")
    require(type(execute) is bool, "Execution authorization must be boolean")
    require(type(max_polls) is int and 1 <= max_polls <= 10000, "Invalid polling bound")
    require(math.isfinite(poll_interval) and 0 <= poll_interval <= 60, "Invalid polling interval")
    require(job["backend"] == backend.name, "Prepared backend identity differs from adapter")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    evidence = {"schema_version": 1, "resolved_job_hash": job["record_hash"], "resolved_workflow_hash": job["resolved_workflow_hash"], "backend": backend.name, "backend_identity": backend.identity,
                "backend_job_id": None, "submitted_workflow_hash": None, "submission_timestamp": None,
                "submission_attempts": 0, "submitted": False, "execution_authorized": execute,
                "status": "checking", "status_history": [], "outputs": [], "output_nodes": job["output_nodes"],
                "lineage": job["lineage"], "max_polls": max_polls, "poll_interval": poll_interval, "created_at": now()}
    evidence_path = directory / "execution.json"
    def checkpoint():
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    started = time.monotonic()
    stage = "preflight"
    try:
        checkpoint()
        write_job(directory / "prepared_job.json", job)
        readiness = backend.readiness(job)
        require(isinstance(readiness, dict) and readiness.get("ready") is True, "Backend is not ready")
        stage = "asset_presence"
        require(backend.assets_present(job) is True, "Backend assets missing; upload remains separately gated")
        workflow = base64.b64decode(job["workflow_bytes_base64"], validate=True)
        stage = "validation"
        verdict = backend.validate(workflow)
        require(isinstance(verdict, dict) and verdict.get("status") == "validated" and verdict.get("submitted") is False and verdict.get("warnings") == [], "Backend dry-run validation failed or warned")
        evidence["validation"] = {"status": "validated", "submitted": False, "warnings": []}
        if not execute:
            evidence["status"] = "validated_pending_explicit_execution"
            checkpoint()
            return evidence
        # Recheck source hashes immediately before the one allowed submission.
        validate_job(job)
        stage = "submission"
        evidence.update(submission_attempts=1, submission_timestamp=now(), submitted_workflow_hash=digest(workflow), status="submission_outcome_unknown")
        checkpoint()
        job_id = backend.submit(workflow)
        require(isinstance(job_id, str) and 0 < len(job_id) <= 128 and all(c.isalnum() or c in "-_" for c in job_id), "Malformed backend job ID")
        evidence.update(backend_job_id=job_id, submitted=True, status="submitted")
        checkpoint()
        stage = "polling"
        for index in range(max_polls):
            status = backend.status(job_id)
            require(isinstance(status, dict) and status.get("status") in {"pending", "running", "completed", "failed", "cancelled"}, "Malformed backend status")
            entry = {"status": status["status"], "observed_at": now()}
            # Only typed numeric usage fields enter evidence, never raw responses.
            for key in ("gpu_seconds", "duration_seconds", "credits_used", "cost_usd"):
                if key in status:
                    require(type(status[key]) in (int, float) and math.isfinite(status[key]) and status[key] >= 0, "Malformed backend usage")
                    entry[key] = status[key]
            if status.get("error_code") == "backend_execution_error":
                entry["error_code"] = "backend_execution_error"
            evidence["status_history"].append(entry)
            evidence["status"] = entry["status"]
            checkpoint()
            require(entry["status"] not in {"failed", "cancelled"}, "Backend reached unsuccessful terminal state")
            if entry["status"] == "completed":
                break
            if index + 1 < max_polls:
                sleep(poll_interval)
        else:
            raise ExecutionError("Polling bound exhausted; job may still be running; no cancel or resubmission performed")
        stage = "download"
        outputs = backend.outputs(job_id, job["output_nodes"])
        require(isinstance(outputs, list) and 0 < len(outputs) <= 64 and all(isinstance(o, dict) for o in outputs), "Malformed output list")
        selected = [o for o in outputs if o.get("node_id") in job["output_nodes"]]
        require({o.get("node_id") for o in selected} == set(job["output_nodes"]), "Declared outputs missing")
        seen = set()
        for index, output in enumerate(selected):
            filename = output.get("filename")
            require(isinstance(filename, str) and filename and Path(filename).name == filename and not any(c in filename for c in ("\\", "\n", "\r")), "Malformed output filename")
            require(isinstance(output.get("subfolder"), str) and output.get("type") == "output", "Malformed output fields")
            key = (output["node_id"], filename, output["subfolder"])
            require(key not in seen, "Duplicate output descriptor")
            seen.add(key)
            data = backend.download(output)
            require(isinstance(data, bytes) and 0 < len(data) <= 64 * 1024 * 1024, "Invalid output bytes")
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                dimensions, mode = list(image.size), image.mode
                require(image.format in {"PNG", "JPEG", "WEBP"}, "Unsupported output image format")
                extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[image.format]
            path = directory / f"output_{index:03d}{extension}"
            with path.open("xb") as stream:
                stream.write(data)
            evidence["outputs"].append({"node_id": output["node_id"], "returned_filename": filename, "path": str(path.resolve()), "sha256": digest(data), "bytes": len(data), "dimensions": dimensions, "mode": mode})
            checkpoint()
        evidence["status"] = "completed"
        evidence["finished_at"] = now()
        evidence["elapsed_seconds"] = time.monotonic() - started
        checkpoint()
        return evidence
    except Exception as exc:
        # Never copy exception text, headers, signed links or arbitrary backend JSON.
        evidence["status"] = "failed"
        evidence["finished_at"] = now()
        evidence["elapsed_seconds"] = time.monotonic() - started
        evidence["error"] = {"stage": stage, "code": "execution_stage_failed", "submission_may_have_succeeded": stage == "submission", "detail": "No retry performed; inspect backend using recorded job ID if available"}
        if isinstance(exc, ExecutionError):
            http_status = getattr(exc, "http_status", None)
            if type(http_status) is int and 100 <= http_status <= 599:
                evidence["error"]["http_status"] = http_status
            # Fixed messages raised by this module only. Backend exceptions may contain secrets.
            evidence["error"]["code"] = "backend_contract_or_operation_failed"
        checkpoint()
        raise ExecutionError(f"{stage} failed; evidence saved in execution.json; no retry performed") from None
