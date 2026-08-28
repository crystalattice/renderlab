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

OUTPUT_ROOT = "video/h3opening"
MIN_FREE_VRAM_MIB = 4500
POLL_SECONDS = 10
JOB_TIMEOUT_SECONDS = 2 * 60 * 60

# Keep this fixed when comparing prompt variants.
DEFAULT_SEED = 1001

# One dict = one overnight render.
# Change/add/remove entries here. No other files are needed.
JOBS = [
    {
        "name": "opening_vo_male_exact",
        "seed": 2001,
        "prompt": """Cinematic movie trailer opening, completely serious and polished.

Visual: begin on black for a brief beat, then reveal a vast retro-futuristic machine complex, dramatic practical lighting, large physical consoles and industrial computer machinery, cinematic widescreen composition.

Native audio is critical: a deep, clear, professional male movie-trailer narrator says exactly and distinctly, "In a world..." The English words must be intelligible, slow, deliberate, and easy to understand. No other spoken words.

Immediately after the narrator finishes the phrase, a deep cinematic impact sounds and a restrained orchestral-synth trailer swell begins. Near the end, introduce one brief digital audio hiccup and a matching horizontal video synchronization glitch, suggesting the trailer is beginning to malfunction.

Prioritize clear intelligible speech for the phrase "In a world..." over complex action. No subtitles, no logos, no watermark."""
    },
    {
        "name": "opening_vo_male_minimal",
        "seed": 2002,
        "prompt": """A polished dramatic movie trailer opening.

Visual: black screen, then a slow reveal of an enormous retro-futuristic computer facility with practical 1970s-style machinery, oscilloscopes, indicator lamps and dark cinematic lighting.

Native audio: one deep professional male trailer narrator speaks only these exact words, clearly and slowly: "In a world..." No other dialogue or vocalization. Make the three English words fully intelligible.

After the phrase, one powerful trailer boom, then a low cinematic synth swell. In the final second, a tiny digital glitch distorts both sound and picture.

Keep the scene simple so the voice remains clean. No subtitles, logos or watermark."""
    },
    {
        "name": "opening_vo_female_exact",
        "seed": 2003,
        "prompt": """Serious cinematic trailer opening with premium live-action photography.

Visual: a vast mysterious retro-futuristic laboratory emerges from darkness, filled with physical control panels, old oscilloscopes, glowing cyan and amber signal paths, and large industrial machinery.

Native audio is the priority: a clear, authoritative female movie-trailer narrator says exactly, "In a world..." with slow deliberate English diction. The words must be intelligible. No other spoken words.

After the line finishes, a deep cinematic impact and low orchestral-synth trailer music begin. A brief video sync tear and short audio corruption appear only after the spoken phrase has completed.

No subtitles, logos or watermark."""
    },
    {
        "name": "opening_vo_male_where",
        "seed": 2004,
        "prompt": """Traditional dramatic movie trailer opening.

Visual: black screen resolving into a huge retro-futuristic AI control room, physical consoles and industrial machinery, cinematic lighting, serious tone.

Native audio: a deep professional male trailer narrator clearly says, "In a world where..." The phrase must be intelligible English, spoken slowly and distinctly. No additional dialogue.

A deep trailer impact lands immediately afterward. Low dramatic synth-orchestral music rises. At the very end, the narrator audio briefly stutters and the image suffers one horizontal synchronization fault.

No subtitles, logos or watermark."""
    },
    {
        "name": "opening_visible_male",
        "seed": 2005,
        "prompt": """Cinematic movie trailer opening presented completely seriously.

Visual: a formal male presenter stands in a retro-futuristic 1970s control room surrounded by large consoles, oscilloscopes, indicator lamps and dark industrial machinery. Medium shot, restrained movement, professional posture.

Native audio is critical: the presenter looks toward camera and clearly says exactly, "In a world..." The English words must be intelligible and synchronized to his mouth. No additional spoken words.

After the phrase, a deep cinematic boom and restrained trailer music begin. Only after he finishes speaking, a brief audio glitch and horizontal sync distortion interrupt the polished image.

No subtitles, logos or watermark."""
    },
    {
        "name": "opening_visible_female",
        "seed": 2006,
        "prompt": """Polished dramatic movie trailer opening.

Visual: a poised female presenter in a retro-futuristic machine laboratory, physical 1960s and 1970s computer hardware, oscilloscopes, switches, indicator lamps, cinematic practical lighting. Medium shot, almost no unnecessary movement.

Native audio is the priority: she clearly says exactly, "In a world..." in intelligible English with deliberate movie-trailer delivery. Her mouth movement matches the phrase. No additional dialogue.

After the spoken phrase ends, a deep trailer impact sounds, then low cinematic synth music. In the last second, one brief digital audio stutter occurs with a matching horizontal video glitch.

No subtitles, logos or watermark."""
    },
    {
        "name": "opening_vo_clean_then_corrupt",
        "seed": 2007,
        "prompt": """A convincing serious movie trailer opening that begins perfectly normal and then starts to malfunction.

Visual: start on black, then reveal a grand retro-futuristic industrial computer complex, practical 1970s hardware mixed with subtle modern illuminated signal paths.

Native audio: first, a deep professional male trailer narrator clearly and intelligibly says, "In a world..." The phrase must be clean English. Keep the audio stable until the final word is complete.

Only after the phrase finishes, the narrator signal breaks into a short burst of unintelligible synthetic gibberish, accompanied by a digital stutter, static, and a horizontal synchronization tear. A dramatic trailer boom and low music continue underneath.

The contrast between the clean opening phrase and the corrupted sound must be obvious. No subtitles, logos or watermark."""
    },
    {
        "name": "opening_vo_two_stage",
        "seed": 2008,
        "prompt": """Classic cinematic trailer opening.

Visual: black screen, then a sweeping reveal of an enormous retro-futuristic computer room with physical consoles, relay cabinets, oscilloscopes and glowing signal indicators.

Native audio: a deep, clear, authoritative movie-trailer narrator says exactly, "In a world..." in fully intelligible English. Pause briefly after the phrase.

Then the same narrator begins another sentence, but the speech immediately degrades into rhythmic unintelligible synthetic gibberish as the image develops a brief horizontal sync fault. A deep trailer impact and dramatic synth swell support the transition.

The first phrase "In a world..." must remain clean and understandable. Everything after it may become corrupted. No subtitles, logos or watermark."""
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
        data={"prompt": graph, "client_id": "h3opening"},
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
    print("Starting H3 opening audio search")
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
    print("Opening search complete.")
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
