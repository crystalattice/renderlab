from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_WORKFLOW = PACKAGE_DIR / "workflows" / "z_image_turbo_int8.json"
DEFAULT_OUTPUT_DIR = REPO_DIR / "output"
MAX_SEED = (1 << 64) - 1


class RenderError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="renderlab",
        description="Submit one local text-to-image render to ComfyUI.",
    )
    parser.add_argument("prompt", help="positive text prompt")
    parser.add_argument("--seed", type=int, help="fixed seed; random 64-bit seed by default")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=float, default=600.0, help="completion timeout in seconds")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="filesystem path matching the ComfyUI output directory",
    )
    args = parser.parse_args(argv)

    if args.seed is not None and not 0 <= args.seed <= MAX_SEED:
        parser.error(f"--seed must be between 0 and {MAX_SEED}")
    for name in ("width", "height", "steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.timeout <= 0 or args.poll_interval <= 0:
        parser.error("--timeout and --poll-interval must be greater than zero")

    parsed = urlparse(args.server)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("--server must be an http(s) URL")
    args.server = args.server.rstrip("/")
    return args


def load_workflow(path: Path) -> dict[str, Any]:
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot load workflow {path}: {exc}") from exc
    if not isinstance(workflow, dict):
        raise RenderError(f"workflow {path} is not a Comfy API object")
    return workflow


def inject_parameters(
    workflow: dict[str, Any], *, prompt: str, seed: int, width: int, height: int, steps: int
) -> None:
    try:
        workflow["4"]["inputs"]["text"] = prompt
        workflow["6"]["inputs"]["width"] = width
        workflow["6"]["inputs"]["height"] = height
        workflow["8"]["inputs"]["seed"] = seed
        workflow["8"]["inputs"]["steps"] = steps
    except (KeyError, TypeError) as exc:
        raise RenderError(f"workflow does not match the RenderLab v1 node contract: {exc}") from exc


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RenderError(f"ComfyUI returned HTTP {exc.code}: {detail}") from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"ComfyUI request failed: {exc}") from exc


def submit(server: str, workflow: dict[str, Any]) -> str:
    result = request_json(
        "POST", f"{server}/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())}
    )
    try:
        return str(result["prompt_id"])
    except (KeyError, TypeError) as exc:
        raise RenderError(f"ComfyUI did not return a prompt_id: {result!r}") from exc


def wait_for_history(server: str, prompt_id: str, timeout: float, poll_interval: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    url = f"{server}/history/{quote(prompt_id, safe='')}"
    while time.monotonic() < deadline:
        result = request_json("GET", url)
        if prompt_id in result:
            history = result[prompt_id]
            status = history.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RenderError(f"ComfyUI execution failed: {messages!r}")
            return history
        time.sleep(poll_interval)
    raise RenderError(f"render {prompt_id} did not finish within {timeout:g} seconds")


def find_saved_image(history: dict[str, Any]) -> dict[str, str]:
    for output in history.get("outputs", {}).values():
        for image in output.get("images", []):
            if image.get("type") == "output" and image.get("filename"):
                return {
                    "filename": str(image["filename"]),
                    "subfolder": str(image.get("subfolder", "")),
                    "type": "output",
                }
    raise RenderError("completed render has no SaveImage output")


def local_output_path(output_dir: Path, image: dict[str, str]) -> Path:
    root = output_dir.expanduser().resolve()
    path = (root / image["subfolder"] / image["filename"]).resolve()
    if path != root and root not in path.parents:
        raise RenderError("ComfyUI returned an output path outside --output-dir")
    return path


def write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    if not path.is_file():
        raise RenderError(
            f"ComfyUI completed, but output is not visible at {path}; "
            "set --output-dir to the server's output directory"
        )
    metadata_path = path.with_name(path.name + ".json")
    temporary_path = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(metadata_path)
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    started_at = datetime.now(timezone.utc)
    try:
        workflow = load_workflow(args.workflow)
        inject_parameters(
            workflow,
            prompt=args.prompt,
            seed=seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
        )
        prompt_id = submit(args.server, workflow)
        print(f"prompt_id: {prompt_id}", file=sys.stderr)
        print(f"seed: {seed}", file=sys.stderr)
        history = wait_for_history(args.server, prompt_id, args.timeout, args.poll_interval)
        image = find_saved_image(history)
        output_path = local_output_path(args.output_dir, image)
        finished_at = datetime.now(timezone.utc)
        metadata_path = write_metadata(
            output_path,
            {
                "schema_version": 1,
                "prompt_id": prompt_id,
                "prompt": args.prompt,
                "seed": seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "cfg": 1,
                "profile": "z_image_turbo_int8",
                "models": {
                    "diffusion_model": "z_image_turbo_int8_convrot.safetensors",
                    "text_encoder": "qwen_3_4b_fp8_mixed.safetensors",
                    "vae": "ae.safetensors",
                },
                "server": args.server,
                "workflow": str(args.workflow.resolve()),
                "output": str(output_path),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            },
        )
    except RenderError as exc:
        print(f"renderlab: error: {exc}", file=sys.stderr)
        return 1

    print(output_path)
    print(f"metadata: {metadata_path}", file=sys.stderr)
    return 0
