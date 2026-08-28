#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8188"

WIDTH = 608
HEIGHT = 352
DURATION_SECONDS = 5.0
STEPS = 20

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

OUTPUT_ROOT = "video/h3dinner"
MIN_FREE_VRAM_MIB = 4500
POLL_SECONDS = 10
JOB_TIMEOUT_SECONDS = 2 * 60 * 60

JOBS = [
    {
        "name": "input_atomic_1950s",
        "seed": 3101,
        "prompt": """A dead-serious 1950s atomic-age laboratory vignette showing the INPUT stage of an AI creation pipeline.

A poised female scientist in a white lab coat stands beside a huge analog computer made of vacuum-tube cabinets, oversized gauges, relays, blinking lamps and physical switches. Three visibly distinct inputs enter the machine: a waveform signal, a strip of moving image frames, and a text-data channel. Cyan and amber illuminated paths carry them into the same system.

She speaks confident, rhythmic, intentionally unintelligible synthetic gibberish, like fictional Animal-Crossing-style speech. Every chirp, beep or relay click corresponds to a visible gauge movement, lamp flash or signal pulse.

Live-action practical-set realism, period-film texture, absolutely serious institutional tone. No subtitles, logos or watermark."""
    },
    {
        "name": "routing_corporate_1990s",
        "seed": 3102,
        "prompt": """A completely serious 1990s corporate training-video vignette showing the ROUTING / CONDITIONING stage of an AI creation pipeline.

A bland office presenter stands in a low-budget computer training room with beige PCs, CRT monitors, fluorescent lighting, bad carpet, plastic speakers, dot-matrix-printer-era equipment and cheap office furniture. On the monitors and physical routing boxes, separate text, image and audio signals are visibly sorted and directed toward one central processing system.

The presenter speaks formal but completely unintelligible Animal-Crossing-style synthetic gibberish as if delivering crucial technical instruction. MIDI-like beeps, printer-like clicks and electronic chirps synchronize tightly with visible routing lights and monitor changes.

Dead-serious tone. Live-action 1990s institutional video aesthetic. No subtitles, logos or watermark."""
    },
    {
        "name": "generation_hybrid_future_retro",
        "seed": 3103,
        "prompt": """A dead-serious near-future industrial vignette showing the GENERATION stage of an AI creation pipeline.

The facility looks expensive and futuristic on one side, with glossy dark surfaces and precise illuminated channels, while the other side is obviously still built from bulky 1970s computer cabinets, analog meters, relays and old industrial machinery, as though the upgrade budget ran out halfway through.

A strange synthetic presenter calmly supervises the process and speaks authoritative unintelligible Animal-Crossing-style gibberish. The machine advances through a visible sequence: amber processing lamps, then cyan oscilloscope-like activity, then green validation indicators. Heavy physical relays and internal mechanisms move in sync with stepped electronic tones, clunks and one brief static burst.

Live-action practical machinery, dry institutional seriousness, no subtitles, logos or watermark."""
    },
    {
        "name": "output_stomp_watch_1980s",
        "seed": 3104,
        "prompt": """A deadpan 1980s retro-futurist vignette showing the OUTPUT stage of an AI creation pipeline.

A glamorous but completely serious technical presenter waits beside an absurdly large output machine in a neon-accented video-production facility filled with CRT displays, chunky controls, physical switches and industrial mechanisms.

The machine produces output painfully slowly. It makes one enormous mechanical STOMP with obvious synchronized physical movement, then a long pause. It makes a second enormous mechanical STOMP, then another pause. During the wait, the presenter briefly checks their wristwatch with restrained impatience, mutters a short burst of unintelligible Animal-Crossing-style synthetic gibberish, then resumes professional composure.

End with the output mechanism beginning to open, but do not reveal the final product yet. Native audio: low machine hum, two heavy synchronized STOMPS, quiet pauses, subtle electronic ambience, short gibberish mutter.

Live-action deadpan comedy. No subtitles, logos or watermark."""
    },
]

REQUIRED_NODES = {
    "VAELoader", "VAEDecodeAudio", "VAEDecode", "KSamplerSelect",
    "BasicScheduler", "SamplerCustomAdvanced", "BasicGuider", "UNETLoader",
    "CLIPLoader", "RandomNoise", "CreateVideo", "MiniMaxH3ImageToVideo",
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
    base = max(5, round(seconds * 24))
    return base + (5 - (base % 17)) % 17

def build_prompt(job: dict, index: int) -> dict:
    return {
        "119": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "120": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "121": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["125", 0], "vae": ["120", 0]}},
        "122": {"class_type": "VAEDecode", "inputs": {"samples": ["125", 0], "vae": ["119", 0]}},
        "123": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "124": {"class_type": "BasicScheduler", "inputs": {"model": ["127", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
        "125": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["129", 0], "guider": ["126", 0], "sampler": ["123", 0], "sigmas": ["124", 0], "latent_image": ["131", 1]}},
        "126": {"class_type": "BasicGuider", "inputs": {"model": ["127", 0], "conditioning": ["131", 0]}},
        "127": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "128": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(job["seed"])}},
        "130": {"class_type": "CreateVideo", "inputs": {"images": ["122", 0], "audio": ["121", 0], "fps": 24, "bit_depth": 8}},
        "131": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["128", 0], "vae": ["119", 0], "prompt": job["prompt"], "width": WIDTH, "height": HEIGHT, "length": h3_length(DURATION_SECONDS)}},
        "92": {"class_type": "SaveVideo", "inputs": {"video": ["130", 0], "filename_prefix": f"{OUTPUT_ROOT}/{index:02d}_{job['name']}", "format": "auto", "codec": "auto"}},
    }

def nvidia_smi():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines()[0]
    return tuple(int(x.strip()) for x in out.split(","))

def find_llama_gpu_processes():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in out.splitlines() if "llama" in line.lower()]

def preflight(force=False):
    print("H3 dinner-batch preflight")
    print("-------------------------")
    http_json("/system_stats")
    print(f"[OK] ComfyUI reachable: {COMFY_URL}")

    info = http_json("/object_info")
    missing = sorted(REQUIRED_NODES - set(info))
    if missing:
        raise RuntimeError("Missing Comfy node classes: " + ", ".join(missing))
    print(f"[OK] Required Comfy nodes present ({len(REQUIRED_NODES)})")

    for node, model in [
        ("UNETLoader", UNET), ("CLIPLoader", CLIP),
        ("VAELoader", VIDEO_VAE), ("VAELoader", AUDIO_VAE),
    ]:
        if model not in json.dumps(info.get(node, {})):
            raise RuntimeError(f"{model} is not visible to {node}")
        print(f"[OK] Model visible: {model}")

    llama = find_llama_gpu_processes()
    if llama and not force:
        raise RuntimeError("llama is still using the GPU:\n  " + "\n  ".join(llama))

    total, used, free = nvidia_smi()
    print(f"[INFO] VRAM: {used} MiB used / {total} MiB total; {free} MiB free")
    if free < MIN_FREE_VRAM_MIB and not force:
        raise RuntimeError(f"Only {free} MiB VRAM free; require at least {MIN_FREE_VRAM_MIB} MiB.")

    print(f"[INFO] Resolution: {WIDTH}x{HEIGHT}")
    print(f"[INFO] Duration: {DURATION_SECONDS}s -> {h3_length(DURATION_SECONDS)} frames")
    print(f"[INFO] Steps: {STEPS}")
    print(f"[INFO] Dinner vignettes: {len(JOBS)}")
    print("[PASS] Preflight complete")

def submit(graph):
    response = http_json("/prompt", method="POST",
                         data={"prompt": graph, "client_id": "h3dinner"}, timeout=60)
    if response.get("node_errors"):
        raise RuntimeError(json.dumps(response["node_errors"], indent=2))
    pid = response.get("prompt_id")
    if not pid:
        raise RuntimeError("No prompt_id returned")
    return pid

def error_text(entry):
    parts = []
    for msg in (entry.get("status") or {}).get("messages") or []:
        parts.append(json.dumps(msg))
    return "\n".join(parts)

def wait_for_job(prompt_id):
    deadline = time.time() + JOB_TIMEOUT_SECONDS
    while time.time() < deadline:
        hist = http_json(f"/history/{prompt_id}", timeout=30)
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            state = str(status.get("status_str", "")).lower()
            if status.get("completed") or state == "success":
                return True, entry, ""
            if state in {"error", "failed"}:
                return False, entry, error_text(entry)
        time.sleep(POLL_SECONDS)
    raise TimeoutError(prompt_id)

def append_log(record):
    p = Path(__file__).with_name("h3dinner.log.jsonl")
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def run_jobs(start_at=1, force=False):
    preflight(force=force)
    print("\nStarting H3 dinner vignette batch")
    print("================================")
    total = len(JOBS)

    for index, job in enumerate(JOBS, 1):
        if index < start_at:
            continue

        started = datetime.now().astimezone()
        print(f"\n[{index}/{total}] {job['name']}")
        print(f"  seed={job['seed']}")
        print(f"  started={started.isoformat(timespec='seconds')}", flush=True)

        try:
            pid = submit(build_prompt(job, index))
            print(f"  prompt_id={pid}", flush=True)
            ok, history, err = wait_for_job(pid)
            ended = datetime.now().astimezone()
            runtime = (ended - started).total_seconds()

            append_log({
                "index": index,
                "name": job["name"],
                "seed": job["seed"],
                "prompt_id": pid,
                "started": started.isoformat(),
                "ended": ended.isoformat(),
                "runtime_seconds": runtime,
                "status": "success" if ok else "failed",
                "outputs": history.get("outputs") if ok else None,
                "error": err if not ok else None,
            })

            if ok:
                print(f"  SUCCESS in {runtime/60:.1f} min", flush=True)
                continue

            print(f"  FAILED in {runtime/60:.1f} min", flush=True)
            if any(x in err.lower() for x in ("out of memory", "cuda", "disk full", "no space left")):
                print("  Fatal GPU/infrastructure failure; stopping.")
                return 2

        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as e:
            print(f"  RUNNER ERROR: {type(e).__name__}: {e}", flush=True)
            return 3

    print("\nDinner vignette batch complete.")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--from", dest="start_at", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        if args.check:
            preflight(force=args.force)
            return 0
        return run_jobs(max(1, args.start_at), args.force)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
