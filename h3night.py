#!/usr/bin/env python3
"""
h3night.py

Single-file overnight MiniMax H3 runner for local ComfyUI.

Baseline intentionally matches the known-good local workflow:
- 864x480
- nominal 5 seconds (snapped to H3's valid frame grid)
- 20 steps
- Turbo LoRA OFF
- local H3 UNet + Qwen3VL text encoder + video/audio VAEs

Usage:
    python3 h3night.py --check
    python3 h3night.py
    python3 h3night.py --from 3
    python3 h3night.py --force

Edit JOBS below before bedtime.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

COMFY_URL = "http://127.0.0.1:8188"

WIDTH = 608
HEIGHT = 352
DURATION_SECONDS = 5.0
STEPS = 20

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

OUTPUT_ROOT = "video/h3night"
MIN_FREE_VRAM_MIB = 4500
POLL_SECONDS = 10
JOB_TIMEOUT_SECONDS = 2 * 60 * 60

# Keep this fixed when comparing prompt variants.
DEFAULT_SEED = 1001

# One dict = one overnight render.
# Change/add/remove entries here. No other files are needed.
JOBS = [
    {
        "name": "s1_opening_a",
        "seed": DEFAULT_SEED,
        "prompt": """1960s/1970s Hollywood science-fiction computer room with physical consoles, oscilloscopes, blinking lamps and chunky control panels, combined with glossy black digital surfaces and cyan and amber illuminated channels. A synthetic male presenter stands centered, perfectly serious and restrained, facing camera. One background technician quietly works at a transparent control board. Subtle broadcast pathology: one short horizontal sync fault and a momentary duplicate of the presenter's face, then normal again. Cinematic live action, practical-set feeling, no copied franchise designs. Native audio: low machine hum, relay clicks, restrained electronic room tone, one brief synchronization beep. No subtitles, logos or watermarks."""
    },
    {
        "name": "s1_opening_b",
        "seed": DEFAULT_SEED,
        "prompt": """Dead-serious synthetic presenter in a retro-futurist television control room built from large 1970s-style consoles, CRT-like scopes, physical switches and indicator lamps. Glossy black surfaces carry narrow cyan and amber geometric light channels. Camera mostly locked, slight slow push-in. The presenter begins speaking calmly while a polling beep causes a brief horizontal video tear and a doubled face for a fraction of a second. One technician in the deep background keeps working without reacting. Native stereo audio: quiet fans, relays, soft scope tones, brief digital polling beep. Live-action physical-set realism, no text, no logos, no watermark."""
    },
    {
        "name": "s2_inputs_a",
        "seed": DEFAULT_SEED,
        "prompt": """Inside the same retro-futurist control room, the serious synthetic presenter demonstrates multimodal input. A physical console receives three clearly distinct signals at once: glowing typed prompt data, a moving audio waveform, and small illuminated image/video frames. The signals travel through separate cyan and amber channels into one central processing cabinet. Oscilloscope traces respond in sync with audible tones. The presenter remains composed while one tiny frame stutter affects him and the nearby displays simultaneously. Native audio: machine hum, keyed data clicks, short waveform tones, restrained electronic pulses. Live-action practical machinery, readable visual causality, no floating fantasy hologram clutter, no subtitles, logos or watermark."""
    },
    {
        "name": "s2_inputs_b",
        "seed": DEFAULT_SEED,
        "prompt": """Retro 1970s science-fiction computer room with modern digital logic embedded in physical machinery. The presenter points toward a large input console as text instructions, waveform activity, and moving picture frames enter through separate illuminated channels. Each incoming signal causes a matching physical response: meters jump, lamps sequence, scopes pulse. The three paths converge into one machine. One brief synchronization error duplicates part of the presenter's face and causes a short burst of static, then everything continues. Native stereo audio tightly synchronized to visible events: relay ticks, data tones, static burst, low machinery. Live action, restrained camera, no text overlays, logos or watermark."""
    },
    {
        "name": "s3_generation_a",
        "seed": DEFAULT_SEED,
        "prompt": """The central generation machine activates in the retro-futurist control room. Large physical relays engage in sequence, oscilloscopes pulse, cyan and amber channels illuminate step by step, then green indicators appear for validated stages. The serious synthetic presenter stands beside the machine explaining the process without broad gestures. Every audible processing tone produces a visible meter or light response. A brief frame freeze affects the presenter while his eyes still move naturally, then playback resumes. Native audio: deep relay clunks, stepped electronic processing tones, fan noise, short static artifact. Live-action practical-set realism, no subtitles, logos or watermark."""
    },
    {
        "name": "s3_generation_b",
        "seed": DEFAULT_SEED,
        "prompt": """A large physical AI generation cabinet dominates a 1970s-inspired science-fiction control room. The machine progresses through a visibly understandable sequence: first amber lamps, then cyan scope traces, then green completion lamps. Heavy internal mechanisms move slowly behind glass panels. The presenter remains dead serious and nearly motionless while the system visibly and audibly processes around him. One controlled broadcast glitch briefly shifts horizontal synchronization across the entire frame. Native stereo audio: low machinery, measured processing tones, relay clacks synchronized to lamps, one short static burst. Cinematic live action, no text, logos or watermark."""
    },
    {
        "name": "s4_output_a",
        "seed": DEFAULT_SEED,
        "prompt": """Final output sequence in the same retro-futurist control room. A large generation machine works absurdly slowly. It makes one heavy mechanical STOMP, then a long pause, a second STOMP, another long pause, then a third STOMP. The serious presenter waits beside it and briefly checks his wristwatch with restrained impatience. A rectangular physical output finally emerges from a slot. The presenter takes it and begins to unfold or shake it open. Native audio must emphasize the three widely separated heavy STOMP sounds, quiet room tone during the pauses, and small paper or flexible-material handling sounds. Live-action deadpan comedy, no subtitles, logos or watermark."""
    },
    {
        "name": "s4_output_b",
        "seed": DEFAULT_SEED,
        "prompt": """Deadpan ending in a retro-futurist generation room. The giant machine produces a rectangular finished output after three slow, widely spaced mechanical STOMPS. During the long wait the synthetic presenter glances at his wristwatch once, then immediately resumes professional composure. He removes the finished rectangular output and unfolds or shakes it open. The revealed image on the physical output is unmistakably the same presenter himself. He gives only a tiny realization beat, no exaggerated reaction. Native stereo audio: three heavy separated STOMPS, machine hum, quiet pause, material unfolding sound, tiny final electronic chirp. Live-action practical effect feeling, no subtitles, logos or watermark."""
    },
]


# ---------------------------------------------------------------------------
# COMFY API
# ---------------------------------------------------------------------------

REQUIRED_NODES = {
    "VAELoader",
    "VAEDecodeAudio",
    "VAEDecode",
    "KSamplerSelect",
    "BasicScheduler",
    "SamplerCustomAdvanced",
    "BasicGuider",
    "UNETLoader",
    "CLIPLoader",
    "RandomNoise",
    "CreateVideo",
    "MiniMaxH3ImageToVideo",
    "SaveVideo",
}


def http_json(path: str, method: str = "GET", data=None, timeout: int = 30):
    url = COMFY_URL.rstrip("/") + path
    body = None
    headers = {}

    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def h3_length(seconds: float) -> int:
    # Exact expression used by the GUI workflow:
    # max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
    base = max(5, round(seconds * 24))
    return base + (5 - (base % 17)) % 17


def build_prompt(job: dict, index: int) -> dict:
    """
    Flattened API graph. Turbo branch intentionally omitted because the
    known-good baseline has turbo_mode=false and therefore uses:
      raw H3 model + 20 scheduler steps.
    """
    seed = int(job.get("seed", DEFAULT_SEED))
    prompt_text = job["prompt"]
    length = h3_length(DURATION_SECONDS)
    prefix = f"{OUTPUT_ROOT}/{index:02d}_{job['name']}"

    return {
        "119": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "120": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "121": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["125", 0],
                "vae": ["120", 0],
            },
        },
        "122": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["125", 0],
                "vae": ["119", 0],
            },
        },
        "123": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "124": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["127", 0],
                "scheduler": "simple",
                "steps": STEPS,
                "denoise": 1.0,
            },
        },
        "125": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["129", 0],
                "guider": ["126", 0],
                "sampler": ["123", 0],
                "sigmas": ["124", 0],
                "latent_image": ["131", 1],
            },
        },
        "126": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["127", 0],
                "conditioning": ["131", 0],
            },
        },
        "127": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNET,
                "weight_dtype": "default",
            },
        },
        "128": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP,
                "type": "minimax",
                "device": "default",
            },
        },
        "129": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "130": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["122", 0],
                "audio": ["121", 0],
                "fps": 24,
                "bit_depth": 8,
            },
        },
        "131": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["128", 0],
                "vae": ["119", 0],
                "prompt": prompt_text,
                "width": WIDTH,
                "height": HEIGHT,
                "length": length,
            },
        },
        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["130", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------

def nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
    total, used, free = [int(x.strip()) for x in out.split(",")]
    return total, used, free


def find_llama_gpu_processes():
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError:
        return []

    hits = []
    for line in out.splitlines():
        if "llama" in line.lower():
            hits.append(line.strip())
    return hits


def preflight(force: bool = False):
    print("H3 overnight preflight")
    print("----------------------")

    # 1. Comfy API alive
    try:
        stats = http_json("/system_stats")
    except Exception as e:
        raise RuntimeError(f"ComfyUI API is not reachable at {COMFY_URL}: {e}") from e

    print(f"[OK] ComfyUI reachable: {COMFY_URL}")

    # 2. Required node classes installed
    info = http_json("/object_info")
    missing = sorted(REQUIRED_NODES - set(info))
    if missing:
        raise RuntimeError("Missing Comfy node classes: " + ", ".join(missing))

    print(f"[OK] Required Comfy nodes present ({len(REQUIRED_NODES)})")

    # 3. Model names visible to their loader nodes
    checks = [
        ("UNETLoader", "unet_name", UNET),
        ("CLIPLoader", "clip_name", CLIP),
        ("VAELoader", "vae_name", VIDEO_VAE),
        ("VAELoader", "vae_name", AUDIO_VAE),
    ]

    for node, field, model in checks:
        blob = json.dumps(info.get(node, {}))
        if model not in blob:
            raise RuntimeError(f"{model} is not visible to {node}")
        print(f"[OK] Model visible: {model}")

    # 4. Waldo/llama must be off GPU
    llama = find_llama_gpu_processes()
    if llama:
        msg = "llama is still using the GPU:\n  " + "\n  ".join(llama)
        if force:
            print("[WARN] " + msg)
        else:
            raise RuntimeError(msg)

    # 5. Enough free VRAM to make the known-good configuration plausible
    try:
        total, used, free = nvidia_smi()
        print(f"[INFO] VRAM: {used} MiB used / {total} MiB total; {free} MiB free")
        if free < MIN_FREE_VRAM_MIB:
            msg = (
                f"Only {free} MiB VRAM free; require at least "
                f"{MIN_FREE_VRAM_MIB} MiB for unattended run."
            )
            if force:
                print("[WARN] " + msg)
            else:
                raise RuntimeError(msg)
    except FileNotFoundError:
        print("[WARN] nvidia-smi not found; skipping VRAM check")

    # 6. Validate our graph against Comfy's currently installed node schemas
    first_graph = build_prompt(JOBS[0], 1)
    for node_id, node in first_graph.items():
        class_type = node["class_type"]
        if class_type not in info:
            raise RuntimeError(f"Graph node {node_id}: unknown class_type {class_type}")

    print(f"[OK] Embedded graph classes validate")
    print(f"[INFO] Resolution: {WIDTH}x{HEIGHT}")
    print(f"[INFO] Nominal duration: {DURATION_SECONDS}s -> {h3_length(DURATION_SECONDS)} frames")
    print(f"[INFO] Steps: {STEPS}")
    print(f"[INFO] Jobs queued: {len(JOBS)}")
    print("[PASS] Preflight complete")


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def submit(graph: dict) -> str:
    response = http_json(
        "/prompt",
        method="POST",
        data={"prompt": graph, "client_id": "h3night"},
        timeout=60,
    )

    if response.get("node_errors"):
        raise RuntimeError("Comfy rejected graph:\n" + json.dumps(response["node_errors"], indent=2))

    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("Comfy returned no prompt_id:\n" + json.dumps(response, indent=2))

    return prompt_id


def error_text(history_entry: dict) -> str:
    parts = []

    status = history_entry.get("status") or {}
    for message in status.get("messages") or []:
        try:
            parts.append(json.dumps(message))
        except Exception:
            parts.append(str(message))

    return "\n".join(parts)


def is_fatal_failure(text: str) -> bool:
    t = text.lower()
    fatal_markers = [
        "out of memory",
        "not enough gpu memory",
        "cuda error",
        "cuda out",
        "no space left on device",
        "disk full",
    ]
    return any(marker in t for marker in fatal_markers)


def wait_for_job(prompt_id: str):
    deadline = time.time() + JOB_TIMEOUT_SECONDS

    while time.time() < deadline:
        try:
            hist = http_json(f"/history/{prompt_id}", timeout=30)
        except Exception as e:
            print(f"  API poll failed: {e}; retrying...", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        entry = hist.get(prompt_id)
        if not entry:
            time.sleep(POLL_SECONDS)
            continue

        status = entry.get("status") or {}
        status_str = str(status.get("status_str", "")).lower()
        completed = bool(status.get("completed"))

        if completed or status_str == "success":
            return True, entry, ""

        if status_str in {"error", "failed"}:
            text = error_text(entry)
            return False, entry, text

        time.sleep(POLL_SECONDS)

    raise TimeoutError(f"Job {prompt_id} exceeded {JOB_TIMEOUT_SECONDS // 60} minute timeout")


def append_log(record: dict):
    log_path = Path(__file__).with_name("h3night.log.jsonl")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_jobs(start_at: int, force: bool):
    preflight(force=force)

    print()
    print("Starting overnight run")
    print("======================")

    total = len(JOBS)

    for index, job in enumerate(JOBS, start=1):
        if index < start_at:
            continue

        started = datetime.now().astimezone()
        name = job["name"]

        print()
        print(f"[{index}/{total}] {name}")
        print(f"  seed={job.get('seed', DEFAULT_SEED)}")
        print(f"  started={started.isoformat(timespec='seconds')}", flush=True)

        graph = build_prompt(job, index)

        try:
            prompt_id = submit(graph)
            print(f"  prompt_id={prompt_id}", flush=True)

            ok, history, err = wait_for_job(prompt_id)
            ended = datetime.now().astimezone()
            runtime = (ended - started).total_seconds()

            outputs = history.get("outputs") or {}

            record = {
                "index": index,
                "name": name,
                "seed": job.get("seed", DEFAULT_SEED),
                "prompt_id": prompt_id,
                "started": started.isoformat(),
                "ended": ended.isoformat(),
                "runtime_seconds": runtime,
                "status": "success" if ok else "failed",
                "outputs": outputs if ok else None,
                "error": err if not ok else None,
            }
            append_log(record)

            if ok:
                print(f"  SUCCESS in {runtime / 60:.1f} min", flush=True)
                continue

            print(f"  FAILED in {runtime / 60:.1f} min", flush=True)
            if err:
                print(err[-4000:], flush=True)

            if is_fatal_failure(err):
                print("  Fatal infrastructure/GPU failure detected; stopping overnight run.")
                return 2

            print("  Non-fatal job failure; continuing to next job.", flush=True)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as e:
            ended = datetime.now().astimezone()
            runtime = (ended - started).total_seconds()
            text = f"{type(e).__name__}: {e}"

            append_log(
                {
                    "index": index,
                    "name": name,
                    "started": started.isoformat(),
                    "ended": ended.isoformat(),
                    "runtime_seconds": runtime,
                    "status": "runner_error",
                    "error": text,
                }
            )

            print(f"  RUNNER ERROR: {text}", flush=True)

            # A broken API / graph / infrastructure issue is likely to poison
            # every remaining job, so stop rather than waste the night.
            return 3

    print()
    print("Overnight queue complete.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Overnight local MiniMax H3 runner")
    parser.add_argument(
        "--check",
        action="store_true",
        help="run preflight only; submit no render",
    )
    parser.add_argument(
        "--from",
        dest="start_at",
        type=int,
        default=1,
        help="start at 1-based job number (default: 1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore llama/free-VRAM preflight failures",
    )

    args = parser.parse_args()

    try:
        if args.check:
            preflight(force=args.force)
            return 0
        return run_jobs(start_at=max(1, args.start_at), force=args.force)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
