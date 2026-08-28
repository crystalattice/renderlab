#!/usr/bin/env python3
"""
h3overnight_vignettes.py

Overnight local MiniMax H3 candidate-harvest runner for ComfyUI.

Purpose:
Generate A/B/C coverage for a classified institutional "day in the life"
montage that visibly follows the H3 creation ritual:

    INTAKE -> ROUTING -> CONDITIONING -> GENERATION
    -> VALIDATION -> OUTPUT -> REVIEW

Creative invariants:
- dead-serious institutional tone
- different eras / presenters are allowed and encouraged
- recurring physical-machine ritual
- cyan = active/input, amber = processing, green = validated/complete
- native H3 beeps/chirps/relay clunks/static
- intentionally unintelligible synthetic "Animal Crossing-like" speech
- occasional brief Jacob's-Ladder-like face/signal smear near transitions
- no generated subtitles; final BITCHIN'. is added in edit

Known-good hardware baseline:
- RTX 2060 Super 8 GB
- 608x352
- 5 seconds nominal
- 20 steps
- Turbo LoRA OFF
- first cold render ~27 min
- warm renders ~12 min

39 jobs should land around ~8.3 hours on the observed machine, leaving margin
inside an 8+ hour unattended window.

Usage:
    python3 h3overnight_vignettes.py --check
    python3 h3overnight_vignettes.py
    python3 h3overnight_vignettes.py --from 13
    python3 h3overnight_vignettes.py --force

Outputs:
    ComfyUI/output/video/h3overnight/

Log:
    h3overnight.log.jsonl next to this script
"""

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

OUTPUT_ROOT = "video/h3overnight"
MIN_FREE_VRAM_MIB = 4500
POLL_SECONDS = 10
JOB_TIMEOUT_SECONDS = 2 * 60 * 60

COMMON = """
The presentation is completely sincere and institutional, never winking at the camera.
Native H3 audio is essential. Speech, when present, is confident rhythmic synthetic
gibberish resembling a fictional video-game language: expressive but intentionally
unintelligible. Sound effects must correspond to visible machine actions.
Use practical physical machinery rather than abstract magical holograms.
No subtitles, logos, watermarks, copyrighted characters, recognizable brands,
or unlicensed likenesses.
"""

JOBS = [
    # ------------------------------------------------------------------
    # INTAKE / INPUT   3 variants
    # ------------------------------------------------------------------
    {
        "name": "01_intake_atomic_lab_A",
        "stage": "INTAKE",
        "seed": 4001,
        "prompt": """1950s atomic-age classified intake laboratory. A poised female
scientist in a white lab coat receives three anomalous inputs at a huge analog machine:
a visible audio waveform, a moving strip of image frames, and a text-data stream.
Oversized gauges, vacuum tubes, relays, cyan input lamps and amber routing lamps.
Each chirp triggers a gauge or lamp. Near the final second her face briefly smears
side-to-side like a damaged archival signal, then snaps back.""" + COMMON
    },
    {
        "name": "02_intake_space_control_B",
        "stage": "INTAKE",
        "seed": 4002,
        "prompt": """1960s space-program-style classified intake room. A formal male
operator receives waveform, image-frame and text-data inputs through separate physical
channels into one central console. Rotary switches, oscilloscopes, patch cables and
status lamps. Cyan paths illuminate as signals arrive. Short gibberish briefing,
clean beeps matched to visible meter jumps. End on one brief horizontal sync tear.""" + COMMON
    },
    {
        "name": "03_intake_neon_broadcast_C",
        "stage": "INTAKE",
        "seed": 4003,
        "prompt": """1980s neon broadcast intake suite. A glamorous but dead-serious
technical host demonstrates three incoming channels: audio waveform, image/video frames,
and text-data, feeding a large physical processing rack. CRT walls, cyan and amber
signal paths, chunky switches and VU meters. Synthetic gibberish speech and synth chirps
sync to visible activity. Finish with a brief doubled-face broadcast glitch.""" + COMMON
    },

    # ------------------------------------------------------------------
    # ROUTING / CLASSIFICATION   3 variants
    # ------------------------------------------------------------------
    {
        "name": "04_routing_mainframe_1970s_A",
        "stage": "ROUTING",
        "seed": 4004,
        "prompt": """1970s classified mainframe routing department. Beige cabinets,
tape reels, patch panels and cable trays fill the room. An android-like technician
routes the three incoming signal types through different physical paths toward one
processing core. Amber lamps chase along the route. Relay clicks and chirps match every
visible route change. Brief controlled face vibration near the cut.""" + COMMON
    },
    {
        "name": "05_routing_corporate_1990s_B",
        "stage": "ROUTING",
        "seed": 4005,
        "prompt": """Cheap 1990s internal training-video routing department. Beige CRT
PCs, bad carpet, fluorescent lighting, plastic speakers, dot-matrix-era hardware.
A painfully ordinary office presenter solemnly routes text, image and audio inputs
through clunky network boxes. MIDI-like beeps, printer-ish clicks, blinking amber route
lights. Presenter speaks formal synthetic gibberish as though this is vital policy.""" + COMMON
    },
    {
        "name": "06_routing_alien_annex_C",
        "stage": "ROUTING",
        "seed": 4006,
        "prompt": """Unspecified-era classified routing annex staffed by an elegant
non-human operator. The room mixes 1960s oscilloscopes, 1980s rack gear and modern
signal displays. The operator calmly directs separate text, image and audio signals
through physical switches and illuminated cyan-to-amber channels. End with a short
signal-smear across the operator's face and nearby monitors at the same instant.""" + COMMON
    },

    # ------------------------------------------------------------------
    # CONDITIONING / PREPROCESSING   3 variants
    # ------------------------------------------------------------------
    {
        "name": "07_conditioning_reelroom_A",
        "stage": "CONDITIONING",
        "seed": 4007,
        "prompt": """Classified 1960s/1970s conditioning room. Large tape reels,
oscilloscopes and analog filters visibly transform incoming signals before generation.
A stern technician adjusts knobs as waveform shapes, picture strips and text pulses are
normalized into matching cyan channels. Beeps change pitch as meters settle. Sparse
gibberish explanation. The final conditioned signals align in perfect rhythm.""" + COMMON
    },
    {
        "name": "08_conditioning_1980s_video_B",
        "stage": "CONDITIONING",
        "seed": 4008,
        "prompt": """1980s video-processing conditioning bay. CRT waveform monitors,
time-base correctors, rack processors and illuminated patch panels. A female operator
tunes incoming audio, image and text signals until their meters align. Rhythmic synthetic
gibberish, calibration tones, clicks, and one VHS-like sync fault affecting the entire
frame for a moment before stabilizing.""" + COMMON
    },
    {
        "name": "09_conditioning_future_retro_C",
        "stage": "CONDITIONING",
        "seed": 4009,
        "prompt": """Near-future conditioning chamber where expensive glossy equipment
is awkwardly bolted to ancient 1970s cabinets. A synthetic presenter supervises input
normalization as cyan signals are measured, filtered and converted to steady amber
processing channels. Physical relays move in time with tones. The presenter briefly
develops a Jacob's-Ladder-like face smear exactly when a signal overload occurs.""" + COMMON
    },

    # ------------------------------------------------------------------
    # GENERATION   3 variants
    # ------------------------------------------------------------------
    {
        "name": "10_generation_heavy_industrial_A",
        "stage": "GENERATION",
        "seed": 4010,
        "prompt": """Heavy 1970s industrial generation room. The main synthesis machine
dominates the shot: huge cabinets, relay banks, glass windows, motors and meters.
Processing visibly advances from amber lamps to cyan scope activity to intermittent
green checks. Deep clunks, stepped tones and fan noise line up with mechanical actions.
A background technician is secondary to the machine.""" + COMMON
    },
    {
        "name": "11_generation_glasslab_2000s_B",
        "stage": "GENERATION",
        "seed": 4011,
        "prompt": """Pristine early-2000s glass-and-brushed-aluminum generation lab.
A calm synthetic host stands beside a physical processing chamber. Amber processing
indicators sequence around the chamber while cyan traces pulse on embedded displays.
Green validation lamps begin appearing near the end. Elegant electronic tones and
quiet mechanical clicks synchronize with each stage. Brief subtle signal corruption.""" + COMMON
    },
    {
        "name": "12_generation_budget_hybrid_C",
        "stage": "GENERATION",
        "seed": 4012,
        "prompt": """Near-future generation facility whose upgrade budget clearly ran
out halfway through: one side sleek and modern, the other side bulky 1970s machinery.
A strange presenter gives authoritative synthetic gibberish while both generations of
equipment somehow work together. Amber process lights, cyan waveform activity, then
green completion checks. Heavy old relays answer precise futuristic tones.""" + COMMON
    },

    # ------------------------------------------------------------------
    # VALIDATION / MONITORING   3 variants
    # ------------------------------------------------------------------
    {
        "name": "13_validation_missioncontrol_A",
        "stage": "VALIDATION",
        "seed": 4013,
        "prompt": """1960s mission-control-style validation room. Rows of serious
operators watch scopes and status boards as generated audiovisual output is checked.
Amber warning lights progressively turn green. Every green light produces a crisp
confirmation beep. One operator delivers short synthetic gibberish status reports.
One screen and one face briefly smear together during a failed check, then recover.""" + COMMON
    },
    {
        "name": "14_validation_1980s_wall_B",
        "stage": "VALIDATION",
        "seed": 4014,
        "prompt": """1980s validation center with a wall of CRT monitors showing
different views of the generated material. A poised presenter watches meters move from
amber to green while physical switches click into place. Confirmation chirps build into
a rhythmic pattern. The presenter speaks dead-serious gibberish. A single monitor
distorts and drags the presenter's face sideways for a split second.""" + COMMON
    },
    {
        "name": "15_validation_future_clean_C",
        "stage": "VALIDATION",
        "seed": 4015,
        "prompt": """Minimal near-future validation bay with clean dark surfaces
surrounding conspicuously old physical meters and toggle switches. A non-human analyst
reviews generated audiovisual output. Green validation lamps illuminate one by one,
each synchronized to a distinct beep. Clinical synthetic gibberish status report.
One brief corrupted-face event appears when the final check completes.""" + COMMON
    },

    # ------------------------------------------------------------------
    # TRANSITION / GLITCH MONTAGE MATERIAL   3 variants
    # ------------------------------------------------------------------
    {
        "name": "16_glitch_transition_presenter_A",
        "stage": "TRANSITION",
        "seed": 4016,
        "prompt": """Dead-serious classified presenter gives a short synthetic gibberish
briefing in front of retro computer equipment. During the final second, the speech
stutters, the face rapidly smears side-to-side in a disturbing damaged-signal effect,
nearby monitors duplicate the same distortion, static bursts, then a hard-looking visual
break suitable for cutting to a different era.""" + COMMON
    },
    {
        "name": "17_glitch_transition_console_B",
        "stage": "TRANSITION",
        "seed": 4017,
        "prompt": """Close institutional insert of a retro control console during an
H3 process. Cyan signal enters, amber processing lights cascade, then the whole panel
briefly loses synchronization: meters jump, CRT image tears horizontally, audio becomes
a short burst of rhythmic gibberish/static, then green lamps snap on. Designed as a
hard montage transition with no presenter required.""" + COMMON
    },
    {
        "name": "18_glitch_transition_group_C",
        "stage": "TRANSITION",
        "seed": 4018,
        "prompt": """Archival classified operations room with several serious staff
members working normally. Beeps and machinery maintain a steady rhythm. Suddenly every
visible face undergoes the same brief high-speed side-to-side signal smear while the
equipment displays horizontal sync faults. A static burst hits, then everything returns
to professional normality immediately.""" + COMMON
    },

    # ------------------------------------------------------------------
    # OUTPUT MACHINE / STOMP   3 variants
    # ------------------------------------------------------------------
    {
        "name": "19_output_stomp_industrial_A",
        "stage": "OUTPUT",
        "seed": 4019,
        "prompt": """Huge classified output machine in a cavernous industrial hall.
The machine is absurdly oversized for the task. It makes one enormous synchronized
mechanical STOMP with pistons and cabinets visibly moving, then settles into a long
quiet pause with only machine hum. A serious operator waits without reacting.""" + COMMON
    },
    {
        "name": "20_output_stomp_neon_B",
        "stage": "OUTPUT",
        "seed": 4020,
        "prompt": """1980s neon-accented output department. CRT monitors, chunky
switches, magenta/cyan practical lighting around a massive physical output machine.
One deep mechanical STOMP visibly shakes part of the machine, followed by a ridiculous
pause. A glamorous technical host remains solemn and murmurs a tiny burst of gibberish.""" + COMMON
    },
    {
        "name": "21_output_stomp_cheapoffice_C",
        "stage": "OUTPUT",
        "seed": 4021,
        "prompt": """Low-budget 1990s office annex containing an impossibly large,
ancient output machine squeezed between beige cubicles and CRT computers. The machine
performs one gigantic synchronized STOMP while office staff behave as though this is
routine. Long awkward pause afterward. Fluorescent office ambience and mechanical hum.""" + COMMON
    },

    # ------------------------------------------------------------------
    # WAIT / WATCH CHECK   3 variants
    # ------------------------------------------------------------------
    {
        "name": "22_wait_watch_male_A",
        "stage": "WAIT",
        "seed": 4022,
        "prompt": """Dead-serious male institutional presenter waits beside a painfully
slow output machine after a heavy processing cycle. Long quiet pause. He glances down
at his wristwatch once with restrained impatience, mutters a short synthetic gibberish
phrase, then immediately resumes perfect professional posture. Low machine-room hum
and one distant relay click.""" + COMMON
    },
    {
        "name": "23_wait_watch_female_B",
        "stage": "WAIT",
        "seed": 4023,
        "prompt": """Poised female classified technical presenter waits beside an
absurd output machine in a 1970s/1980s control room. Nothing happens for an awkwardly
long beat. She checks her wristwatch once, almost imperceptibly annoyed, gives a tiny
gibberish mutter, then faces the machine again. No broad comedy. Machinery hum continues.""" + COMMON
    },
    {
        "name": "24_wait_watch_nonhuman_C",
        "stage": "WAIT",
        "seed": 4024,
        "prompt": """Elegant non-human institutional operator waits beside an ancient
mechanical output device inside a futuristic facility. The contrast is absurd but
treated seriously. During a long machine pause the operator checks a wristwatch or
wrist-mounted timepiece once, speaks a tiny unintelligible synthetic phrase, then waits
again. Low hum, occasional electronic chirp.""" + COMMON
    },

    # ------------------------------------------------------------------
    # REVEAL / SELF OUTPUT   3 variants
    # ------------------------------------------------------------------
    {
        "name": "25_reveal_self_board_A",
        "stage": "REVEAL",
        "seed": 4025,
        "prompt": """Final review department. A serious male presenter receives a large
rectangular physical output from a machine and opens or unfolds it. The output clearly
contains an image of the same presenter. He pauses, recognizing himself, then turns the
board toward camera. Quiet machine ambience and material-handling sounds. Keep the
self-image readable and central. Do not generate any text.""" + COMMON
    },
    {
        "name": "26_reveal_self_female_B",
        "stage": "REVEAL",
        "seed": 4026,
        "prompt": """Final review department in a different era. A poised female
presenter receives a large physical printed/generated output and reveals that it
contains a recognizable image of herself. Tiny realization pause, then she calmly
displays it toward camera. Native audio: machine hum, paper or flexible-material handling,
small confirmation chirp. No text on screen.""" + COMMON
    },
    {
        "name": "27_reveal_self_synthetic_C",
        "stage": "REVEAL",
        "seed": 4027,
        "prompt": """Classified final review chamber. A slightly uncanny synthetic
presenter takes a large rectangular output from the machine and discovers that the
output depicts the same synthetic presenter. Hold the image clearly toward camera after
a brief recognition beat. Restrained machine ambience, material handling, one green
validation chirp. No generated text.""" + COMMON
    },

    # ------------------------------------------------------------------
    # FINAL GIBBERISH / THUMBS UP   3 variants
    # ------------------------------------------------------------------
    {
        "name": "28_final_speech_male_A",
        "stage": "FINAL",
        "seed": 4028,
        "prompt": """Final classified approval shot. A serious male presenter holds a
large board containing a clear image of himself. Facing camera, he launches into a long,
confident, enthusiastic speech entirely in rhythmic synthetic gibberish. The speech
sounds authoritative despite being meaningless. While continuing the speech he gives
a strong thumbs-up with his free hand. He finishes cleanly and holds the pose for a beat.
No generated subtitles or text.""" + COMMON
    },
    {
        "name": "29_final_speech_female_B",
        "stage": "FINAL",
        "seed": 4029,
        "prompt": """Final classified approval shot. A poised female presenter holds a
large physical output containing a recognizable image of herself. She delivers several
seconds of grand, formal, completely unintelligible synthetic gibberish directly to
camera, as though concluding an important technical presentation. During the speech she
raises a confident thumbs-up. End with a clean silence and held pose. No generated text.""" + COMMON
    },
    {
        "name": "30_final_speech_nonhuman_C",
        "stage": "FINAL",
        "seed": 4030,
        "prompt": """Final classified approval shot. An elegant non-human presenter
holds a large output depicting the same presenter. They deliver a long, triumphant,
authoritative synthetic gibberish monologue directly to camera while giving a confident
thumbs-up. Machine ambience and subtle confirmation tones underneath. End abruptly after
the speech with the pose held cleanly, leaving room for an externally added final subtitle.
No generated text.""" + COMMON
    },
    {
        "name": "31_intake_containment_crate_D",
        "stage": "INTAKE",
        "seed": 4031,
        "prompt": """Classified 1970s containment-intake bay. A sealed metal crate,
a waveform monitor, a strip of image frames and a text-data printout are all treated
as equivalent anomalous inputs and routed into one enormous computer system. Two
technicians in period uniforms work with dead-serious precision. Cyan intake lamps
trigger with every chirp and relay click. One technician's face briefly smears
sideways during a signal spike, then instantly resolves.""" + COMMON
    },
    {
        "name": "32_routing_switchboard_D",
        "stage": "ROUTING",
        "seed": 4032,
        "prompt": """1960s classified routing switchboard room packed with patch cords,
telephone-style plugs, oscilloscopes and blinking status lamps. A stern operator
manually patches text, image and audio channels into a central generation line.
Every plug insertion produces a crisp synchronized click and light pulse.
Short synthetic gibberish status phrases. The final route locks green with a
satisfying mechanical clack.""" + COMMON
    },
    {
        "name": "33_generation_core_closeup_D",
        "stage": "GENERATION",
        "seed": 4033,
        "prompt": """Extreme close institutional insert of the physical H3 generation
core. No presenter required. Huge relays, spinning mechanisms, glowing vacuum tubes,
oscilloscopes and modern cyan channels work together inside one machine. Amber
processing lights move rhythmically across the cabinet, mechanical clunks hit on
visible actions, then several green validation lamps snap on. Designed as dense
trailer-montage machine footage.""" + COMMON
    },
    {
        "name": "34_generation_face_smear_D",
        "stage": "GENERATION",
        "seed": 4034,
        "prompt": """Dead-serious synthetic presenter beside a generation machine in a
dark retro-futurist control room. The presenter delivers confident gibberish while
amber and cyan processing indicators sequence behind them. During one brief overload,
the presenter's face violently smears side-to-side in a damaged-signal effect while
the machinery keeps operating normally. Static burst, then the face resolves and the
briefing continues.""" + COMMON
    },
    {
        "name": "35_validation_stamp_D",
        "stage": "VALIDATION",
        "seed": 4035,
        "prompt": """Bureaucratic classified validation desk in a strange mix of 1950s
office furniture and futuristic electronics. A serious reviewer watches generated
audiovisual output on a small monitor, checks analog meters, then pulls a huge physical
approval lever. A green validation indicator lamp illuminates without readable text.
The lever action lands with a heavy clunk and confirmation chirp. Synthetic gibberish
commentary remains completely formal.""" + COMMON
    },
    {
        "name": "36_output_double_stomp_D",
        "stage": "OUTPUT",
        "seed": 4036,
        "prompt": """Massive classified output machine in a dim industrial chamber.
The machine performs TWO clearly separated heavy mechanical STOMPS. Each STOMP has
obvious synchronized piston and cabinet movement. Between them is a long awkward pause
with only low machinery hum. A lone operator waits motionless beside the machine and
does not react. Deadpan, physical, weighty, slow.""" + COMMON
    },
    {
        "name": "37_output_slot_open_D",
        "stage": "OUTPUT",
        "seed": 4037,
        "prompt": """Close-up of an ancient classified output slot built into a huge
retro machine. After a deep mechanical clunk, locks release one by one, amber lamps
turn green, and the slot slowly opens. A rectangular physical output begins to emerge
but remains mostly hidden. Native audio is entirely mechanical: locks, relays, hum,
one confirmation chirp. Designed as a trailer montage insert.""" + COMMON
    },
    {
        "name": "38_reveal_self_glitch_D",
        "stage": "REVEAL",
        "seed": 4038,
        "prompt": """Final review vignette. A dead-serious presenter receives and opens
a rectangular output that clearly depicts the same presenter. On recognition, both the
real face and the generated face briefly smear sideways in the same synchronized
damaged-signal glitch, accompanied by a short static burst. Both resolve immediately.
The presenter calmly holds the self-image toward camera. No generated text.""" + COMMON
    },
    {
        "name": "39_final_speech_board_D",
        "stage": "FINAL",
        "seed": 4039,
        "prompt": """Strong final approval candidate. A serious institutional presenter
stands centered, holding a large rectangular board showing a clear image of the same
presenter. They launch into a long, grand, authoritative burst of rhythmic synthetic
gibberish directly to camera, then raise a confident thumbs-up while continuing the
speech. The speech ends cleanly. Hold the thumbs-up and self-image for one silent beat
afterward, leaving clean visual space for an externally added final subtitle. No
generated text.""" + COMMON
    },

]


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
    base = max(5, round(seconds * 24))
    return base + (5 - (base % 17)) % 17


def build_prompt(job: dict, index: int) -> dict:
    return {
        "119": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "120": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "121": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["125", 0], "vae": ["120", 0]},
        },
        "122": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["125", 0], "vae": ["119", 0]},
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
            "inputs": {"noise_seed": int(job["seed"])},
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
                "prompt": job["prompt"],
                "width": WIDTH,
                "height": HEIGHT,
                "length": h3_length(DURATION_SECONDS),
            },
        },
        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["130", 0],
                "filename_prefix": f"{OUTPUT_ROOT}/{index:02d}_{job['name']}",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def nvidia_smi():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    return tuple(int(x.strip()) for x in out.split(","))


def find_llama_gpu_processes():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in out.splitlines() if "llama" in line.lower()]


def preflight(force=False):
    print("H3 overnight vignette preflight")
    print("------------------------------")
    http_json("/system_stats")
    print(f"[OK] ComfyUI reachable: {COMFY_URL}")

    info = http_json("/object_info")
    missing = sorted(REQUIRED_NODES - set(info))
    if missing:
        raise RuntimeError("Missing Comfy node classes: " + ", ".join(missing))
    print(f"[OK] Required Comfy nodes present ({len(REQUIRED_NODES)})")

    for node, model in [
        ("UNETLoader", UNET),
        ("CLIPLoader", CLIP),
        ("VAELoader", VIDEO_VAE),
        ("VAELoader", AUDIO_VAE),
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
        raise RuntimeError(
            f"Only {free} MiB VRAM free; require at least {MIN_FREE_VRAM_MIB} MiB."
        )

    warm_minutes = 12.2
    estimated = 26.8 + max(0, len(JOBS) - 1) * warm_minutes
    print(f"[INFO] Resolution: {WIDTH}x{HEIGHT}")
    print(f"[INFO] Duration: {DURATION_SECONDS}s -> {h3_length(DURATION_SECONDS)} frames")
    print(f"[INFO] Steps: {STEPS}")
    print(f"[INFO] Jobs: {len(JOBS)}")
    print(f"[INFO] Estimated runtime from observed timings: ~{estimated/60:.1f} hours")
    print("[PASS] Preflight complete")


def submit(graph):
    response = http_json(
        "/prompt",
        method="POST",
        data={"prompt": graph, "client_id": "h3overnight"},
        timeout=60,
    )
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
        try:
            hist = http_json(f"/history/{prompt_id}", timeout=30)
        except Exception as exc:
            print(f"  poll error: {exc}; retrying", flush=True)
            time.sleep(POLL_SECONDS)
            continue

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
    p = Path(__file__).with_name("h3overnight.log.jsonl")
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_fatal(err: str) -> bool:
    text = err.lower()
    return any(
        marker in text
        for marker in (
            "out of memory",
            "not enough gpu memory",
            "cuda error",
            "cuda out",
            "disk full",
            "no space left on device",
        )
    )


def run_jobs(start_at=1, force=False):
    preflight(force=force)

    print("\nStarting H3 overnight vignette harvest")
    print("======================================")
    total = len(JOBS)

    for index, job in enumerate(JOBS, 1):
        if index < start_at:
            continue

        started = datetime.now().astimezone()
        print(f"\n[{index}/{total}] {job['name']}  stage={job['stage']}")
        print(f"  seed={job['seed']}")
        print(f"  started={started.isoformat(timespec='seconds')}", flush=True)

        try:
            pid = submit(build_prompt(job, index))
            print(f"  prompt_id={pid}", flush=True)

            ok, history, err = wait_for_job(pid)
            ended = datetime.now().astimezone()
            runtime = (ended - started).total_seconds()

            append_log(
                {
                    "index": index,
                    "name": job["name"],
                    "stage": job["stage"],
                    "seed": job["seed"],
                    "prompt_id": pid,
                    "started": started.isoformat(),
                    "ended": ended.isoformat(),
                    "runtime_seconds": runtime,
                    "status": "success" if ok else "failed",
                    "outputs": history.get("outputs") if ok else None,
                    "error": err if not ok else None,
                }
            )

            if ok:
                print(f"  SUCCESS in {runtime/60:.1f} min", flush=True)
                continue

            print(f"  FAILED in {runtime/60:.1f} min", flush=True)
            if err:
                print(err[-4000:], flush=True)

            if is_fatal(err):
                print("  Fatal GPU/infrastructure failure; stopping overnight run.")
                return 2

            print("  Non-fatal failure; continuing to next candidate.", flush=True)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as exc:
            ended = datetime.now().astimezone()
            runtime = (ended - started).total_seconds()
            text = f"{type(exc).__name__}: {exc}"
            append_log(
                {
                    "index": index,
                    "name": job["name"],
                    "stage": job["stage"],
                    "seed": job["seed"],
                    "started": started.isoformat(),
                    "ended": ended.isoformat(),
                    "runtime_seconds": runtime,
                    "status": "runner_error",
                    "error": text,
                }
            )
            print(f"  RUNNER ERROR: {text}", flush=True)
            return 3

    print("\nOvernight vignette harvest complete.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Overnight local MiniMax H3 vignette harvest")
    ap.add_argument("--check", action="store_true", help="preflight only; submit no render")
    ap.add_argument("--from", dest="start_at", type=int, default=1, help="1-based job number")
    ap.add_argument("--force", action="store_true", help="ignore llama/free-VRAM preflight failures")
    args = ap.parse_args()

    try:
        if args.check:
            preflight(force=args.force)
            return 0
        return run_jobs(max(1, args.start_at), args.force)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
