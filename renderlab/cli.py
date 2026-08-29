from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import struct
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from . import __version__
from .video import (
    H3_AUDIO_VAE,
    H3_CLIP,
    H3_FPS,
    H3_REQUIRED_NODES,
    H3_UNET,
    H3_VIDEO_VAE,
    build_h3_t2v_workflow,
    h3_frame_count,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_WORKFLOW = PACKAGE_DIR / "workflows" / "z_image_turbo_int8.json"
DEFAULT_IMG2IMG_WORKFLOW = PACKAGE_DIR / "workflows" / "z_image_turbo_int8_img2img.json"
DEFAULT_INPAINT_WORKFLOW = PACKAGE_DIR / "workflows" / "z_image_turbo_int8_inpaint.json"
REALVISXL_WORKFLOW = PACKAGE_DIR / "workflows" / "realvisxl_v5.json"
REALVISXL_IMG2IMG_WORKFLOW = PACKAGE_DIR / "workflows" / "realvisxl_v5_img2img.json"
REALVISXL_INPAINT_WORKFLOW = PACKAGE_DIR / "workflows" / "realvisxl_v5_inpaint.json"
SAM3_MASK_WORKFLOW = PACKAGE_DIR / "workflows" / "sam3_mask.json"
DEFAULT_OUTPUT_DIR = REPO_DIR / "output"
MAX_SEED = (1 << 64) - 1
SUPPORTED_INPUT_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

PROFILES = {
    "z-image": {
        "workflow": DEFAULT_WORKFLOW,
        "img2img_workflow": DEFAULT_IMG2IMG_WORKFLOW,
        "inpaint_workflow": DEFAULT_INPAINT_WORKFLOW,
        "steps": 8,
        "cfg": 1,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "profile_name": "z_image_turbo_int8",
        "models": {
            "diffusion_model": "z_image_turbo_int8_convrot.safetensors",
            "text_encoder": "qwen_3_4b_fp8_mixed.safetensors",
            "vae": "ae.safetensors",
        },
    },
    "realvisxl": {
        "workflow": REALVISXL_WORKFLOW,
        "img2img_workflow": REALVISXL_IMG2IMG_WORKFLOW,
        "inpaint_workflow": REALVISXL_INPAINT_WORKFLOW,
        "steps": 30,
        "cfg": 7,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "profile_name": "realvisxl_v5_fp16",
        "models": {"checkpoint": "RealVisXL_V5.0_fp16.safetensors"},
    },
}

LORA_PRESETS = {
    "angelica": ("SDXL_Angelica.safetensors", 0.75, 0.75, "Angelica character archetype"),
    "artful-nsfw": ("ArtfulNSFWV2SDXL.safetensors", 0.3, 0.0, "erotic editorial style"),
    "cyber-goth": ("90s_Cyber-goth_Sci-Fi.safetensors", 0.5, 0.5, "90s cyber-goth style"),
    "dark-fantasy": ("Dark_Fantasy_Pulp_Pinup.safetensors", 0.5, 0.5, "dark fantasy pulp style"),
    "marie-rose": ("Marie_Rose_Cosplay_outfit.safetensors", 0.75, 0.75, "Marie Rose outfit and character"),
    "natural-body": ("natural breasts_epoch_5.safetensors", 0.3, 0.0, "natural body proportions"),
    "pov": ("NsfwPovAllInOneLoraSdxl-000009.safetensors", 0.75, 0.75, "first-person composition"),
    "realistic-eyes": ("Realistic_eyes.safetensors", 0.4, 0.4, "eye and face enhancement"),
    "samane": ("SDXL_Samane.safetensors", 0.75, 0.75, "Samane character archetype"),
    "vaporwave": ("Retro_80s_Vaporwave.safetensors", 0.75, 0.75, "retro vaporwave style"),
}


class RenderError(RuntimeError):
    pass


CONTROL_COMMANDS = {
    "jobs", "status", "cancel", "models", "loras", "lora-presets", "lora-sweep",
    "doctor", "mask", "replay", "video",
}

MODEL_NODE_INPUTS = (
    ("diffusion_models", "UNETLoader", "unet_name"),
    ("checkpoints", "CheckpointLoaderSimple", "ckpt_name"),
    ("text_encoders", "CLIPLoader", "clip_name"),
    ("vaes", "VAELoader", "vae_name"),
)

REQUIRED_WORKFLOW_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "EmptySD3LatentImage",
    "ModelSamplingAuraFlow",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "LoadImage",
    "VAEEncode",
    "LoadImageMask",
    "VAEEncodeForInpaint",
    "GrowMask",
    "MaskToImage",
    "ImageBlur",
    "ImageToMask",
    "ImageCompositeMasked",
    "ImagePadForOutpaint",
)

REQUIRED_MODEL_CHOICES = (
    ("UNETLoader", "unet_name", "z_image_turbo_int8_convrot.safetensors"),
    ("CLIPLoader", "clip_name", "qwen_3_4b_fp8_mixed.safetensors"),
    ("VAELoader", "vae_name", "ae.safetensors"),
)

REALVISXL_REQUIRED_NODES = (
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "LoadImage",
    "LoadImageMask",
    "VAEEncodeForInpaint",
    "GrowMask",
    "MaskToImage",
    "ImageBlur",
    "ImageToMask",
    "ImageCompositeMasked",
    "ImagePadForOutpaint",
)

REALVISXL_REQUIRED_MODEL_CHOICES = (
    ("CheckpointLoaderSimple", "ckpt_name", "RealVisXL_V5.0_fp16.safetensors"),
)


def add_server_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default="http://127.0.0.1:8188")


def validate_server(parser: argparse.ArgumentParser, server: str) -> str:
    parsed = urlparse(server)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("--server must be an http(s) URL")
    return server.rstrip("/")


def parse_control_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="renderlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    jobs_parser = subparsers.add_parser("jobs", help="list ComfyUI jobs")
    jobs_parser.add_argument("--limit", type=int, default=20)
    add_server_argument(jobs_parser)

    status_parser = subparsers.add_parser("status", help="show one ComfyUI job")
    status_parser.add_argument("prompt_id")
    add_server_argument(status_parser)

    cancel_parser = subparsers.add_parser("cancel", help="cancel a running or pending job")
    cancel_parser.add_argument("prompt_id")
    add_server_argument(cancel_parser)

    models_parser = subparsers.add_parser("models", help="list models visible to ComfyUI")
    add_server_argument(models_parser)

    loras_parser = subparsers.add_parser("loras", help="list LoRAs visible to ComfyUI")
    add_server_argument(loras_parser)

    presets_parser = subparsers.add_parser(
        "lora-presets", help="list RenderLab's tested LoRA presets"
    )
    add_server_argument(presets_parser)

    sweep_parser = subparsers.add_parser(
        "lora-sweep", help="render a fixed-seed baseline and LoRA strength sweep"
    )
    sweep_parser.add_argument("prompt", help="positive text prompt")
    sweep_lora = sweep_parser.add_mutually_exclusive_group(required=True)
    sweep_lora.add_argument("--lora", help="LoRA filename visible to ComfyUI")
    sweep_lora.add_argument("--lora-preset", choices=sorted(LORA_PRESETS))
    sweep_parser.add_argument(
        "--strengths", default="0.25,0.5,0.75,1.0",
        help="comma-separated model and CLIP strengths (default: 0.25,0.5,0.75,1.0)",
    )
    sweep_parser.add_argument("--seed", type=int, help="fixed seed; one is generated when omitted")
    sweep_parser.add_argument("--profile", choices=sorted(PROFILES), default="realvisxl")
    sweep_parser.add_argument("--width", type=int, default=1024)
    sweep_parser.add_argument("--height", type=int, default=1024)
    sweep_parser.add_argument("--steps", type=int)
    sweep_parser.add_argument("--cfg", type=float)
    sweep_parser.add_argument("--negative-prompt")
    sweep_parser.add_argument("--timeout", type=float, default=600.0)
    sweep_parser.add_argument("--poll-interval", type=float, default=1.0)
    sweep_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_server_argument(sweep_parser)

    doctor_parser = subparsers.add_parser("doctor", help="validate the RenderLab runtime")
    doctor_parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="z-image"
    )
    add_server_argument(doctor_parser)

    mask_parser = subparsers.add_parser(
        "mask", help="create a binary edit mask with ComfyUI's native SAM3"
    )
    mask_parser.add_argument("input_image", type=Path, help="image to segment")
    mask_parser.add_argument(
        "description", help="comma-separated objects or regions to make editable"
    )
    mask_parser.add_argument(
        "--within", help="keep only target pixels inside this enclosing object, e.g. person"
    )
    mask_parser.add_argument(
        "--allow-border", action="store_true",
        help="allow selected pixels to touch the image boundary",
    )
    mask_parser.add_argument("--threshold", type=float, default=0.5)
    mask_parser.add_argument("--refine-iterations", type=int, default=2)
    mask_parser.add_argument("--timeout", type=float, default=600.0)
    mask_parser.add_argument("--poll-interval", type=float, default=1.0)
    mask_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="filesystem path matching the ComfyUI output directory",
    )
    add_server_argument(mask_parser)

    replay_parser = subparsers.add_parser(
        "replay", help="rerun a render from its provenance sidecar"
    )
    replay_parser.add_argument("metadata", type=Path, help="RenderLab output .json sidecar")
    replay_seed = replay_parser.add_mutually_exclusive_group()
    replay_seed.add_argument(
        "--new-seed", action="store_true", help="restore all settings with a new random seed"
    )
    replay_seed.add_argument("--seed", type=int, help="restore all settings with this seed")
    replay_parser.add_argument("--timeout", type=float, default=600.0)
    replay_parser.add_argument("--poll-interval", type=float, default=1.0)
    replay_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="filesystem path matching the ComfyUI output directory",
    )
    add_server_argument(replay_parser)

    video_parser = subparsers.add_parser(
        "video", help="render text-to-video with the local MiniMax H3 profile"
    )
    video_parser.add_argument("prompt", nargs="?", help="text prompt for the video")
    video_parser.add_argument("--check", action="store_true", help="check nodes and models only")
    video_parser.add_argument("--seed", type=int, help="fixed seed; random 64-bit seed by default")
    video_parser.add_argument("--width", type=int, default=608)
    video_parser.add_argument("--height", type=int, default=352)
    video_parser.add_argument("--duration", type=float, default=5.0, help="nominal seconds")
    video_parser.add_argument("--steps", type=int, default=20)
    video_parser.add_argument("--timeout", type=float, default=7200.0)
    video_parser.add_argument("--poll-interval", type=float, default=10.0)
    video_parser.add_argument("--filename-prefix", default="RenderLabVideo")
    video_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="filesystem path matching the ComfyUI output directory",
    )
    add_server_argument(video_parser)

    args = parser.parse_args(argv)
    if args.command == "jobs" and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.command in {"mask", "replay", "video", "lora-sweep"}:
        if args.timeout <= 0 or args.poll_interval <= 0:
            parser.error("--timeout and --poll-interval must be greater than zero")
    if args.command == "replay" and args.seed is not None and not 0 <= args.seed <= MAX_SEED:
        parser.error(f"--seed must be between 0 and {MAX_SEED}")
    if args.command == "lora-sweep":
        if args.lora_preset is not None and args.profile != "realvisxl":
            parser.error("RenderLab LoRA presets are SDXL-only and require --profile realvisxl")
        if args.seed is not None and not 0 <= args.seed <= MAX_SEED:
            parser.error(f"--seed must be between 0 and {MAX_SEED}")
        for name in ("width", "height"):
            if getattr(args, name) <= 0:
                parser.error(f"--{name} must be greater than zero")
        try:
            args.strengths = [float(value.strip()) for value in args.strengths.split(",")]
        except ValueError:
            parser.error("--strengths must be comma-separated numbers")
        if not args.strengths or any(not -10.0 <= value <= 10.0 for value in args.strengths):
            parser.error("--strengths values must be between -10.0 and 10.0")
    if args.command == "mask":
        if not args.description.strip():
            parser.error("description must not be empty")
        if not 0.0 <= args.threshold <= 1.0:
            parser.error("--threshold must be between 0.0 and 1.0")
        if not 0 <= args.refine_iterations <= 5:
            parser.error("--refine-iterations must be between 0 and 5")
    if args.command == "video":
        if not args.check and (args.prompt is None or not args.prompt.strip()):
            parser.error("prompt is required unless --check is used")
        if args.seed is not None and not 0 <= args.seed <= MAX_SEED:
            parser.error(f"--seed must be between 0 and {MAX_SEED}")
        for name in ("width", "height", "steps"):
            if getattr(args, name) <= 0:
                parser.error(f"--{name} must be greater than zero")
        if args.width % 16 or args.height % 16:
            parser.error("--width and --height must be divisible by 16")
        if args.duration <= 0:
            parser.error("--duration must be greater than zero")
        allowed_prefix = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not args.filename_prefix or any(
            character not in allowed_prefix for character in args.filename_prefix
        ):
            parser.error(
                "--filename-prefix may contain only letters, digits, underscore, and hyphen"
            )
    args.server = validate_server(parser, args.server)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="renderlab",
        description="Submit local text-to-image renders to ComfyUI.",
    )
    parser.add_argument("--version", action="version", version=f"renderlab {__version__}")
    parser.add_argument("prompt", help="positive text prompt")
    parser.add_argument("--seed", type=int, help="fixed seed; random 64-bit seed by default")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="z-image",
        help="render profile (default: z-image)",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--cfg",
        type=float,
        help="classifier-free guidance strength (profile default when omitted)",
    )
    parser.add_argument(
        "--negative-prompt",
        help="replace the profile's negative prompt; pass an empty string to clear it",
    )
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--lora",
        help="LoRA filename visible to ComfyUI; use `renderlab loras` to list choices",
    )
    lora_group.add_argument(
        "--lora-preset",
        choices=sorted(LORA_PRESETS),
        action="append",
        help="tested LoRA and strength combination; use `renderlab lora-presets` to list",
    )
    parser.add_argument(
        "--lora-model-strength",
        type=float,
        default=None,
        help="LoRA strength applied to the diffusion model (default: 1.0)",
    )
    parser.add_argument(
        "--lora-clip-strength",
        type=float,
        default=None,
        help="LoRA strength applied to the text encoder (default: 1.0)",
    )
    parser.add_argument("--input-image", type=Path, help="source image for img2img editing")
    parser.add_argument(
        "--mask-image",
        type=Path,
        help="black/white edit mask; white pixels may change and black pixels are protected",
    )
    parser.add_argument(
        "--mask-grow",
        type=int,
        default=6,
        help="expand the editable mask by this many pixels for seam blending (default: 6)",
    )
    parser.add_argument(
        "--mask-feather",
        type=int,
        default=6,
        help="blur the final mask edge by this many pixels when compositing (default: 6)",
    )
    parser.add_argument("--outpaint-left", type=int, default=0)
    parser.add_argument("--outpaint-top", type=int, default=0)
    parser.add_argument("--outpaint-right", type=int, default=0)
    parser.add_argument("--outpaint-bottom", type=int, default=0)
    parser.add_argument(
        "--outpaint-feather", type=int, default=40,
        help="blend width between the source and expanded canvas (default: 40)",
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=0.45,
        help="img2img change strength from 0.0 to 1.0 (default: 0.45)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of sequential renders; an explicit seed increments for each render",
    )
    parser.add_argument(
        "--variations",
        type=int,
        help="expand the intent into this many materially different prompts before rendering",
    )
    parser.add_argument(
        "--prompt-server",
        default="http://127.0.0.1:8084",
        help="OpenAI-compatible local prompt-expander server (default: 127.0.0.1:8084)",
    )
    parser.add_argument("--prompt-model", default="local")
    parser.add_argument(
        "--prompt-timeout",
        type=float,
        default=180.0,
        help="prompt-expander request timeout in seconds",
    )
    add_server_argument(parser)
    parser.add_argument("--timeout", type=float, default=600.0, help="completion timeout in seconds")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--filename-prefix", default="RenderLab", help=argparse.SUPPRESS)
    parser.add_argument("--expected-pixel-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--replay-source", help=argparse.SUPPRESS)
    parser.add_argument("--replay-kind", choices=("exact", "new-seed"), help=argparse.SUPPRESS)
    parser.add_argument("--parent-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="filesystem path matching the ComfyUI output directory",
    )
    args = parser.parse_args(argv)

    if args.steps is None:
        args.steps = PROFILES[args.profile]["steps"]
    if args.cfg is None:
        args.cfg = PROFILES[args.profile]["cfg"]

    if args.seed is not None and not 0 <= args.seed <= MAX_SEED:
        parser.error(f"--seed must be between 0 and {MAX_SEED}")
    for name in ("width", "height", "steps", "count"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if not 0.0 <= args.cfg <= 100.0:
        parser.error("--cfg must be between 0.0 and 100.0")
    if args.negative_prompt and args.cfg <= 1.0:
        parser.error("a non-empty --negative-prompt requires --cfg greater than 1.0")
    if args.variations is not None:
        if args.variations <= 0:
            parser.error("--variations must be greater than zero")
        if args.count != 1:
            parser.error("--count and --variations cannot be used together")
    if args.timeout <= 0 or args.poll_interval <= 0:
        parser.error("--timeout and --poll-interval must be greater than zero")
    if args.prompt_timeout <= 0:
        parser.error("--prompt-timeout must be greater than zero")
    if args.lora_preset is not None and args.profile != "realvisxl":
        parser.error("RenderLab LoRA presets are SDXL-only and require --profile realvisxl")
    if args.lora_preset is not None and len(args.lora_preset) > 1 and (
        args.lora_model_strength is not None or args.lora_clip_strength is not None
    ):
        parser.error("explicit LoRA strengths cannot override a multi-preset stack")
    args.loras = []
    if args.lora_preset is not None:
        for preset in args.lora_preset:
            preset_name, preset_model, preset_clip, _ = LORA_PRESETS[preset]
            args.loras.append(
                {
                    "preset": preset,
                    "name": preset_name,
                    "model_strength": (
                        args.lora_model_strength
                        if args.lora_model_strength is not None else preset_model
                    ),
                    "clip_strength": (
                        args.lora_clip_strength
                        if args.lora_clip_strength is not None else preset_clip
                    ),
                }
            )
        if len(args.loras) == 1:
            args.lora = args.loras[0]["name"]
            args.lora_model_strength = args.loras[0]["model_strength"]
            args.lora_clip_strength = args.loras[0]["clip_strength"]
    if args.lora is not None:
        if args.lora_model_strength is None:
            args.lora_model_strength = 1.0
        if args.lora_clip_strength is None:
            args.lora_clip_strength = 1.0
        if not args.loras:
            args.loras.append(
                {
                    "preset": None,
                    "name": args.lora,
                    "model_strength": args.lora_model_strength,
                    "clip_strength": args.lora_clip_strength,
                }
            )
    for name in ("lora_model_strength", "lora_clip_strength"):
        value = getattr(args, name)
        if value is not None and not -10.0 <= value <= 10.0:
            parser.error(f"--{name.replace('_', '-')} must be between -10.0 and 10.0")
    if args.lora is None and (
        args.lora_model_strength is not None or args.lora_clip_strength is not None
    ):
        parser.error("--lora-model-strength and --lora-clip-strength require --lora")
    if not 0.0 <= args.denoise <= 1.0:
        parser.error("--denoise must be between 0.0 and 1.0")
    if args.input_image is None and args.denoise != 0.45:
        parser.error("--denoise requires --input-image")
    if args.mask_image is not None and args.input_image is None:
        parser.error("--mask-image requires --input-image")
    outpaint_values = (
        args.outpaint_left, args.outpaint_top, args.outpaint_right, args.outpaint_bottom
    )
    args.outpaint = any(outpaint_values)
    if any(value < 0 or value % 8 for value in outpaint_values):
        parser.error("outpaint expansion values must be non-negative multiples of 8")
    if args.outpaint and args.input_image is None:
        parser.error("outpainting requires --input-image")
    if args.outpaint and args.mask_image is not None:
        parser.error("outpainting and --mask-image cannot be used together")
    if not 0 <= args.outpaint_feather <= 256:
        parser.error("--outpaint-feather must be between 0 and 256")
    if not args.outpaint and args.outpaint_feather != 40:
        parser.error("--outpaint-feather requires an outpaint expansion value")
    if args.outpaint and args.denoise == 0.45:
        args.denoise = 1.0
    if args.mask_grow < 0 or args.mask_grow > 64:
        parser.error("--mask-grow must be between 0 and 64")
    if args.mask_image is None and args.mask_grow != 6:
        parser.error("--mask-grow requires --mask-image")
    if args.mask_feather < 1 or args.mask_feather > 31:
        parser.error("--mask-feather must be between 1 and 31")
    if args.mask_image is None and args.mask_feather != 6:
        parser.error("--mask-feather requires --mask-image")
    allowed_prefix = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not args.filename_prefix or any(
        character not in allowed_prefix for character in args.filename_prefix
    ):
        parser.error("--filename-prefix may contain only letters, digits, underscore, and hyphen")
    args.server = validate_server(parser, args.server)
    args.prompt_server = validate_server(parser, args.prompt_server)
    if args.workflow is None:
        if args.mask_image or args.outpaint:
            args.workflow = PROFILES[args.profile]["inpaint_workflow"]
        elif args.input_image:
            args.workflow = PROFILES[args.profile]["img2img_workflow"]
        else:
            args.workflow = PROFILES[args.profile]["workflow"]
    return args


def resolve_seeds(seed: int | None, count: int) -> list[int]:
    if seed is None:
        return [secrets.randbits(64) for _ in range(count)]
    if seed + count - 1 > MAX_SEED:
        raise RenderError(
            f"seed range exceeds {MAX_SEED}; use --seed no greater than {MAX_SEED - count + 1}"
        )
    return [seed + index for index in range(count)]


def load_workflow(path: Path) -> dict[str, Any]:
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot load workflow {path}: {exc}") from exc
    if not isinstance(workflow, dict):
        raise RenderError(f"workflow {path} is not a Comfy API object")
    return workflow


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RenderError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pixel_sha256_file(path: Path) -> str:
    """Hash normalized RGBA pixels, excluding PNG encoding and metadata."""
    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            digest = hashlib.sha256()
            digest.update(struct.pack(">II", rgba.width, rgba.height))
            digest.update(rgba.tobytes())
            return digest.hexdigest()
    except (OSError, UnidentifiedImageError) as exc:
        raise RenderError(f"cannot decode image pixels from {path}: {exc}") from exc


def expand_prompts(
    server: str, model: str, intent: str, count: int, timeout: float = 180.0
) -> list[str]:
    director_briefs = (
        "1. World-first: radically change the environment and use a wide or unusual establishing "
        "composition; let the setting materially affect the subject.\n"
        "2. Subject-first: redesign the subject's identity, shape, clothing or machinery, pose, and "
        "action; use a markedly different camera distance and angle.\n"
        "3. Treatment-first: choose a distinct visual medium or photographic treatment, palette, "
        "weather, lighting direction, and emotional tone.\n"
        "For additional prompts, combine unused camera, action, environment, design, lighting, and "
        "treatment choices without repeating an earlier composition."
    )
    instruction = (
        f"Create exactly {count} materially different image-generation prompts from the user's "
        "rough intent. Preserve only its essential subjects and explicitly requested concepts. "
        "Treat every unspecified detail as permission to change it. Every prompt must differ from "
        "every other prompt on at least four of these axes: subject design or identity, action or "
        "pose, camera distance or angle, environment, lighting or palette, and visual medium or "
        "treatment. Do not merely paraphrase, substitute synonyms, or retain the same centered "
        "standing composition. Do not add bracketed tags.\n\n"
        f"Director briefs:\n{director_briefs}\n\n"
        "Each prompt must stand alone and must not refer to previous images or variations. Return "
        "only JSON in the form {\"prompts\":[\"...\"]}."
    )
    schema = {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["prompts"],
        "additionalProperties": False,
    }
    result = request_json(
        "POST",
        f"{server}/v1/chat/completions",
        {
            "model": model,
            "temperature": 0.9,
            "max_tokens": 1536,
            # Keep Qwen and GPT-OSS reasoning models from spending the completion
            # budget on reasoning_content without ever emitting the requested JSON.
            "reasoning_effort": "low",
            "chat_template_kwargs": {
                "enable_thinking": False,
                "reasoning_effort": "low",
            },
            "json_schema": schema,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": intent},
            ],
        },
        timeout=timeout,
    )
    try:
        choice = result["choices"][0]
        content = choice["message"]["content"]
        if (
            not content
            and choice.get("finish_reason") == "length"
            and choice["message"].get("reasoning_content")
        ):
            raise RenderError(
                "prompt expander exhausted its completion budget on reasoning without "
                "returning JSON; update llama.cpp or use a model that honors low reasoning"
            )
        start = content.index("{")
        end = content.rindex("}") + 1
        prompts = json.loads(content[start:end])["prompts"]
    except RenderError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RenderError(f"prompt expander returned invalid JSON: {result!r}") from exc
    if (
        not isinstance(prompts, list)
        or len(prompts) != count
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        returned_count = len(prompts) if isinstance(prompts, list) else "non-list"
        raise RenderError(
            f"prompt expander returned {returned_count} prompts; expected exactly {count}: "
            f"{prompts!r}"
        )
    normalized = [prompt.strip() for prompt in prompts]
    if len(set(normalized)) != count:
        raise RenderError("prompt expander returned duplicate prompts")
    return normalized


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


def inject_generation_controls(
    workflow: dict[str, Any], *, negative_prompt: str | None, cfg: float
) -> str | None:
    """Apply shared guidance controls and return the effective negative prompt."""
    try:
        workflow["8"]["inputs"]["cfg"] = cfg
        negative_node = workflow["5"]
        if negative_prompt is not None:
            if negative_node["class_type"] == "ConditioningZeroOut":
                negative_node.clear()
                negative_node.update(
                    {
                        "class_type": "CLIPTextEncode",
                        "inputs": {
                            "text": negative_prompt,
                            "clip": list(workflow["4"]["inputs"]["clip"]),
                        },
                    }
                )
            elif negative_node["class_type"] == "CLIPTextEncode":
                negative_node["inputs"]["text"] = negative_prompt
            else:
                raise RenderError(
                    "workflow negative node is not CLIPTextEncode or ConditioningZeroOut"
                )
        if negative_node["class_type"] == "CLIPTextEncode":
            return str(negative_node["inputs"]["text"])
        if negative_node["class_type"] == "ConditioningZeroOut":
            return None
        raise RenderError("workflow has an unsupported negative conditioning node")
    except RenderError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RenderError(
            f"workflow does not match the RenderLab guidance-control contract: {exc}"
        ) from exc


def inject_filename_prefix(workflow: dict[str, Any], prefix: str) -> None:
    save_nodes = [
        node for node in workflow.values()
        if isinstance(node, dict) and node.get("class_type") == "SaveImage"
    ]
    if not save_nodes:
        raise RenderError("workflow has no SaveImage node")
    for node in save_nodes:
        try:
            node["inputs"]["filename_prefix"] = prefix
        except (KeyError, TypeError) as exc:
            raise RenderError(f"SaveImage node has no filename_prefix input: {exc}") from exc


def inject_lora(
    workflow: dict[str, Any], *, name: str, model_strength: float, clip_strength: float
) -> str:
    """Insert one LoraLoader and route the graph's model and CLIP through it."""
    model_source: list[Any] | None = None
    clip_source: list[Any] | None = None
    numeric_ids: list[int] = []
    existing_loras: list[tuple[int, str]] = []
    for node_id, node in workflow.items():
        try:
            numeric_ids.append(int(node_id))
            class_type = node["class_type"]
        except (TypeError, ValueError, KeyError):
            continue
        if class_type == "CheckpointLoaderSimple":
            model_source = [node_id, 0]
            clip_source = [node_id, 1]
        elif class_type == "UNETLoader":
            model_source = [node_id, 0]
        elif class_type == "CLIPLoader":
            clip_source = [node_id, 0]
        elif class_type == "LoraLoader":
            existing_loras.append((int(node_id), node_id))

    if existing_loras:
        _, latest_lora = max(existing_loras)
        model_source = [latest_lora, 0]
        clip_source = [latest_lora, 1]

    if model_source is None or clip_source is None:
        raise RenderError("workflow has no compatible model and CLIP loaders for --lora")

    lora_id = str(max(numeric_ids, default=0) + 1)
    for node_id, node in workflow.items():
        if node_id == lora_id or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, value in inputs.items():
            if value == model_source:
                inputs[input_name] = [lora_id, 0]
            elif value == clip_source:
                inputs[input_name] = [lora_id, 1]

    workflow[lora_id] = {
        "class_type": "LoraLoader",
        "inputs": {
            "model": model_source,
            "clip": clip_source,
            "lora_name": name,
            "strength_model": model_strength,
            "strength_clip": clip_strength,
        },
    }
    return lora_id


def inject_img2img_parameters(
    workflow: dict[str, Any], *, prompt: str, seed: int, steps: int, denoise: float, image: str
) -> None:
    try:
        workflow["4"]["inputs"]["text"] = prompt
        workflow["6"]["inputs"]["image"] = image
        workflow["8"]["inputs"]["seed"] = seed
        workflow["8"]["inputs"]["steps"] = steps
        workflow["8"]["inputs"]["denoise"] = denoise
    except (KeyError, TypeError) as exc:
        raise RenderError(f"workflow does not match the RenderLab img2img node contract: {exc}") from exc


def inject_inpaint_parameters(
    workflow: dict[str, Any], *, prompt: str, seed: int, steps: int, denoise: float,
    image: str, mask: str, mask_grow: int, mask_feather: int = 6
) -> None:
    try:
        workflow["4"]["inputs"]["text"] = prompt
        workflow["6"]["inputs"]["image"] = image
        workflow["8"]["inputs"]["seed"] = seed
        workflow["8"]["inputs"]["steps"] = steps
        workflow["8"]["inputs"]["denoise"] = denoise
        workflow["11"]["inputs"]["grow_mask_by"] = mask_grow
        workflow["12"]["inputs"]["image"] = mask
        workflow["13"]["inputs"]["expand"] = mask_grow
        workflow["15"]["inputs"]["blur_radius"] = mask_feather
    except (KeyError, TypeError) as exc:
        raise RenderError(f"workflow does not match the RenderLab inpaint node contract: {exc}") from exc


def inject_outpaint_parameters(
    workflow: dict[str, Any], *, prompt: str, seed: int, steps: int, denoise: float,
    image: str, left: int, top: int, right: int, bottom: int, feather: int
) -> None:
    """Convert the inpaint graph to native canvas expansion around one source image."""
    inject_inpaint_parameters(
        workflow,
        prompt=prompt,
        seed=seed,
        steps=steps,
        denoise=denoise,
        image=image,
        mask="unused",
        mask_grow=0,
        mask_feather=1,
    )
    workflow["18"] = {
        "class_type": "ImagePadForOutpaint",
        "inputs": {
            "image": ["6", 0], "left": left, "top": top,
            "right": right, "bottom": bottom, "feathering": feather,
        },
    }
    workflow["11"]["inputs"]["pixels"] = ["18", 0]
    workflow["11"]["inputs"]["mask"] = ["18", 1]
    workflow["17"]["inputs"]["destination"] = ["18", 0]
    workflow["17"]["inputs"]["mask"] = ["18", 1]
    workflow.pop("12", None)
    for node_id in ("13", "14", "15", "16"):
        workflow.pop(node_id, None)


def inject_mask_parameters(
    workflow: dict[str, Any], *, image: str, description: str,
    threshold: float, refine_iterations: int, within: str | None = None
) -> None:
    targets = [target.strip() for target in description.split(",") if target.strip()]
    if not targets:
        raise RenderError("mask description contains no targets")
    try:
        workflow["2"]["inputs"]["image"] = image
    except (KeyError, TypeError) as exc:
        raise RenderError(f"workflow does not match the RenderLab SAM3 mask contract: {exc}") from exc

    next_id = 3

    def detection(text: str) -> str:
        nonlocal next_id
        text_id = str(next_id)
        detect_id = str(next_id + 1)
        next_id += 2
        workflow[text_id] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": text, "clip": ["1", 1]},
        }
        workflow[detect_id] = {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["1", 0], "image": ["2", 0],
                "conditioning": [text_id, 0], "threshold": threshold,
                "refine_iterations": refine_iterations, "individual_masks": False,
            },
        }
        return detect_id

    enclosure_id = detection(within.strip()) if within and within.strip() else None
    target_masks = []
    for target in targets:
        target_id = detection(target)
        if enclosure_id is not None:
            intersect_id = str(next_id)
            next_id += 1
            workflow[intersect_id] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": [target_id, 0], "source": [enclosure_id, 0],
                    "x": 0, "y": 0, "operation": "and",
                },
            }
            target_id = intersect_id
        target_masks.append(target_id)

    combined_id = target_masks[0]
    for target_id in target_masks[1:]:
        union_id = str(next_id)
        next_id += 1
        workflow[union_id] = {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": [combined_id, 0], "source": [target_id, 0],
                "x": 0, "y": 0, "operation": "or",
            },
        }
        combined_id = union_id

    image_id = str(next_id)
    save_id = str(next_id + 1)
    workflow[image_id] = {
        "class_type": "MaskToImage", "inputs": {"mask": [combined_id, 0]}
    }
    workflow[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "RenderLabMask", "images": [image_id, 0]},
    }


def upload_image(server: str, path: Path) -> str:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise RenderError(f"input image does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_INPUT_IMAGE_SUFFIXES:
        raise RenderError(
            f"unsupported input image extension {suffix or '(none)'}; expected one of "
            f"{', '.join(sorted(SUPPORTED_INPUT_IMAGE_SUFFIXES))}"
        )
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise RenderError(f"cannot read input image {source}: {exc}") from exc

    boundary = f"RenderLab-{uuid.uuid4().hex}"
    remote_name = f"renderlab_{uuid.uuid4().hex}{suffix}"
    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{remote_name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode("utf-8"),
        content,
        (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            "true\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8"),
    ]
    body = b"".join(part if isinstance(part, bytes) else part.encode("utf-8") for part in parts)
    request = Request(f"{server}/upload/image", data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urlopen(request, timeout=30.0) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RenderError(f"image upload returned HTTP {exc.code}: {detail}") from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"image upload failed: {exc}") from exc
    try:
        name = str(result["name"])
        subfolder = str(result.get("subfolder", "")).strip("/")
    except (KeyError, TypeError) as exc:
        raise RenderError(f"ComfyUI returned an invalid image upload response: {result!r}") from exc
    if not name or name in {".", ".."} or "/" in name or "\\" in name or ".." in subfolder.split("/"):
        raise RenderError(f"ComfyUI returned an unsafe image upload path: {result!r}")
    return f"{subfolder}/{name}" if subfolder else name


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RenderError(f"API returned HTTP {exc.code}: {detail}") from exc
    except ConnectionResetError as exc:
        raise RenderError(
            "server reset the connection and may have crashed; check its terminal output. "
            "If this happened while ComfyUI was rendering, stop other GPU workloads "
            "(including GPU-offloaded Waldo), restart ComfyUI, and retry"
        ) from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"API request failed: {exc}") from exc


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


def find_saved_output(history: dict[str, Any]) -> dict[str, str]:
    """Find a SaveImage or SaveVideo artifact in ComfyUI history."""
    try:
        return find_saved_image(history)
    except RenderError as exc:
        raise RenderError("completed render has no saved output") from exc


def local_output_path(output_dir: Path, image: dict[str, str]) -> Path:
    root = output_dir.expanduser().resolve()
    path = (root / image["subfolder"] / image["filename"]).resolve()
    if path != root and root not in path.parents:
        raise RenderError("ComfyUI returned an output path outside --output-dir")
    return path


def validate_binary_mask_png(path: Path, *, allow_border: bool = False) -> dict[str, int]:
    """Validate the simple 8-bit PNG emitted by ComfyUI's MaskToImage node."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RenderError(f"cannot read generated mask {path}: {exc}") from exc
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RenderError(f"generated mask is not a PNG: {path}")

    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR" and len(chunk_data) == 13:
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if not width or not height or bit_depth != 8 or channels is None or interlace != 0:
        raise RenderError("generated mask PNG uses an unsupported pixel format")
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise RenderError(f"generated mask PNG is corrupt: {exc}") from exc

    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise RenderError("generated mask PNG has invalid scanline data")
    previous = bytearray(stride)
    white = black = 0
    selected = bytearray(width * height)
    for row_index in range(height):
        start = row_index * (stride + 1)
        filter_type = raw[start]
        encoded = raw[start + 1:start + 1 + stride]
        row = bytearray(stride)
        for index, value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                decoded = value
            elif filter_type == 1:
                decoded = value + left
            elif filter_type == 2:
                decoded = value + up
            elif filter_type == 3:
                decoded = value + ((left + up) // 2)
            elif filter_type == 4:
                predictor = left + up - upper_left
                distances = (
                    abs(predictor - left), abs(predictor - up), abs(predictor - upper_left)
                )
                decoded = value + (left if distances[0] <= min(distances[1:]) else
                                   up if distances[1] <= distances[2] else upper_left)
            else:
                raise RenderError(f"generated mask PNG uses invalid filter {filter_type}")
            row[index] = decoded & 0xFF
        previous = row
        for pixel_start in range(0, stride, channels):
            if color_type in {0, 4}:
                rgb = row[pixel_start:pixel_start + 1]
            else:
                rgb = row[pixel_start:pixel_start + 3]
            value = rgb[0]
            if any(channel != value for channel in rgb) or value not in {0, 255}:
                raise RenderError("generated SAM3 mask is not strictly black and white")
            if value == 255:
                white += 1
                selected[row_index * width + pixel_start // channels] = 1
            else:
                black += 1
    if white == 0:
        raise RenderError(
            "SAM3 found no matching region; try a simpler description or lower --threshold"
        )
    if black == 0:
        raise RenderError("SAM3 selected the entire image; use a more specific description")
    border_selected = any(selected[x] or selected[(height - 1) * width + x] for x in range(width))
    border_selected = border_selected or any(
        selected[y * width] or selected[y * width + width - 1] for y in range(height)
    )
    if border_selected and not allow_border:
        raise RenderError(
            "SAM3 mask touches the image boundary; likely background contamination. "
            "Use --allow-border only when the intended subject really reaches the frame edge"
        )

    visited = bytearray(width * height)
    component_sizes = []
    for start_index, is_selected in enumerate(selected):
        if not is_selected or visited[start_index]:
            continue
        visited[start_index] = 1
        stack = [start_index]
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            x = index % width
            y = index // width
            neighbors = []
            if x:
                neighbors.append(index - 1)
            if x + 1 < width:
                neighbors.append(index + 1)
            if y:
                neighbors.append(index - width)
            if y + 1 < height:
                neighbors.append(index + width)
            for neighbor in neighbors:
                if selected[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
        component_sizes.append(size)

    significant_floor = max(16, white // 500)
    significant_components = sum(size >= significant_floor for size in component_sizes)
    if significant_components > 8:
        raise RenderError(
            f"SAM3 mask contains {significant_components} scattered regions; "
            "use fewer, more concrete targets"
        )
    return {
        "width": width, "height": height, "white_pixels": white, "black_pixels": black,
        "components": len(component_sizes), "significant_components": significant_components,
        "touches_border": int(border_selected),
    }


def run_mask(args: argparse.Namespace) -> int:
    source = args.input_image.expanduser().resolve()
    source_sha256 = sha256_file(source)
    uploaded = upload_image(args.server, source)
    workflow = load_workflow(SAM3_MASK_WORKFLOW)
    inject_mask_parameters(
        workflow,
        image=uploaded,
        description=args.description.strip(),
        threshold=args.threshold,
        refine_iterations=args.refine_iterations,
        within=args.within,
    )
    prompt_id = submit(args.server, workflow)
    print(f"prompt_id: {prompt_id}", file=sys.stderr)
    history = wait_for_history(args.server, prompt_id, args.timeout, args.poll_interval)
    output_path = local_output_path(args.output_dir, find_saved_image(history))
    stats = validate_binary_mask_png(output_path, allow_border=args.allow_border)
    metadata_path = write_metadata(
        output_path,
        {
            "schema_version": 2,
            "renderlab_version": __version__,
            "prompt_id": prompt_id,
            "mode": "mask",
            "description": args.description.strip(),
            "targets": [target.strip() for target in args.description.split(",") if target.strip()],
            "within": args.within.strip() if args.within else None,
            "allow_border": args.allow_border,
            "threshold": args.threshold,
            "refine_iterations": args.refine_iterations,
            "model": "sam3.1_multiplex_fp16.safetensors",
            "source_image": {
                "path": str(source), "sha256": source_sha256, "comfy_input": uploaded
            },
            "mask": {**stats, "white_is_editable": True},
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "submitted_workflow": workflow,
            "submitted_workflow_sha256": sha256_json(workflow),
        },
    )
    print(output_path)
    print(f"metadata: {metadata_path}", file=sys.stderr)
    print(
        f"editable pixels: {stats['white_pixels']} / "
        f"{stats['white_pixels'] + stats['black_pixels']}",
        file=sys.stderr,
    )
    return 0


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


def load_render_metadata(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot load render metadata {source}: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("mode") not in {
        "txt2img", "img2img", "inpaint", "outpaint"
    }:
        raise RenderError(f"{source} is not RenderLab render provenance")
    return metadata


def replay_asset(record: dict[str, Any], label: str) -> str:
    try:
        path = Path(record["path"]).expanduser().resolve()
        expected_hash = record["sha256"]
    except (KeyError, TypeError) as exc:
        raise RenderError(f"{label} provenance is missing {exc}") from exc
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RenderError(
            f"{label} hash changed: expected {expected_hash}, found {actual_hash} at {path}"
        )
    return str(path)


def replay_arguments(args: argparse.Namespace) -> list[str]:
    """Translate one render sidecar back into normal render CLI arguments."""
    metadata = load_render_metadata(args.metadata)
    profile_names = {
        values["profile_name"]: name for name, values in PROFILES.items()
    }
    try:
        profile = profile_names[metadata["profile"]]
        prompt = metadata["effective_prompt"]
        parent_seed = metadata["seed"]
        steps = metadata["steps"]
        cfg = metadata["cfg"]
    except (KeyError, TypeError) as exc:
        raise RenderError(f"render provenance is missing replay field {exc}") from exc
    if not isinstance(prompt, str) or not prompt:
        raise RenderError("render provenance has an invalid effective prompt")

    replay = [
        prompt,
        "--profile", profile,
        "--seed", str(
            args.seed if args.seed is not None
            else secrets.randbits(64) if args.new_seed
            else parent_seed
        ),
        "--steps", str(steps),
        "--cfg", str(cfg),
        "--server", args.server,
        "--timeout", str(args.timeout),
        "--poll-interval", str(args.poll_interval),
        "--output-dir", str(args.output_dir),
        "--filename-prefix", (
            f"RenderLabVariant_{uuid.uuid4().hex[:12]}"
            if args.new_seed or args.seed is not None
            else f"RenderLabReplay_{uuid.uuid4().hex[:12]}"
        ),
        "--replay-source", str(args.metadata.expanduser().resolve()),
        "--replay-kind", "new-seed" if args.new_seed or args.seed is not None else "exact",
        "--parent-seed", str(parent_seed),
    ]
    recorded_output = Path(str(metadata.get("output", ""))).expanduser()
    adjacent_output = args.metadata.expanduser().resolve().with_suffix("")
    if not recorded_output.is_file() and adjacent_output.is_file():
        recorded_output = adjacent_output
    if not recorded_output.is_file():
        raise RenderError(
            "replay source output is missing; expected the recorded output path or a PNG "
            "adjacent to the sidecar"
        )
    if not args.new_seed and args.seed is None:
        replay.extend(["--expected-pixel-sha256", pixel_sha256_file(recorded_output.resolve())])
    if metadata.get("negative_prompt") is not None:
        replay.extend(["--negative-prompt", str(metadata["negative_prompt"])])

    mode = metadata["mode"]
    if mode == "txt2img":
        try:
            replay.extend([
                "--width", str(metadata["width"]),
                "--height", str(metadata["height"]),
            ])
        except KeyError as exc:
            raise RenderError(f"render provenance is missing replay field {exc}") from exc
    else:
        source = metadata.get("source_image")
        if not isinstance(source, dict):
            raise RenderError("image-edit provenance is missing source_image")
        replay.extend(["--input-image", replay_asset(source, "source image")])
        replay.extend(["--denoise", str(metadata["denoise"])])

    if mode == "inpaint":
        mask = metadata.get("mask_image")
        if not isinstance(mask, dict):
            raise RenderError("inpaint provenance is missing mask_image")
        replay.extend(["--mask-image", replay_asset(mask, "mask image")])
        replay.extend(["--mask-grow", str(mask["grow_pixels"])])
        replay.extend(["--mask-feather", str(mask["feather_pixels"])])

    if mode == "outpaint":
        outpaint = metadata.get("outpaint")
        if not isinstance(outpaint, dict):
            raise RenderError("outpaint provenance is missing expansion settings")
        for side in ("left", "top", "right", "bottom"):
            replay.extend([f"--outpaint-{side}", str(outpaint[side])])
        replay.extend(["--outpaint-feather", str(outpaint["feather"])])

    loras = metadata.get("loras")
    if loras is not None:
        if not isinstance(loras, list) or any(not isinstance(lora, dict) for lora in loras):
            raise RenderError("render provenance has invalid LoRA stack")
        presets = [lora.get("preset") for lora in loras]
        uses_preset_defaults = all(
            isinstance(preset, str)
            and lora.get("model_strength") == LORA_PRESETS[preset][1]
            and lora.get("clip_strength") == LORA_PRESETS[preset][2]
            for preset, lora in zip(presets, loras, strict=True)
        )
        if uses_preset_defaults:
            for preset in presets:
                replay.extend(["--lora-preset", preset])
        elif len(loras) == 1:
            lora = loras[0]
            replay.extend(["--lora", str(lora["name"])])
            replay.extend(["--lora-model-strength", str(lora["model_strength"])])
            replay.extend(["--lora-clip-strength", str(lora["clip_strength"])])
        else:
            raise RenderError("raw multi-LoRA stacks cannot yet be replayed")
    else:
        lora = metadata.get("lora")
        if lora is not None:
            if not isinstance(lora, dict):
                raise RenderError("render provenance has invalid LoRA settings")
            replay.extend(["--lora", str(lora["name"])])
            replay.extend(["--lora-model-strength", str(lora["model_strength"])])
            replay.extend(["--lora-clip-strength", str(lora["clip_strength"])])
    return replay


def discover_node_choices(server: str, node_name: str, input_name: str) -> list[str]:
    result = request_json("GET", f"{server}/object_info/{quote(node_name, safe='')}")
    try:
        definition = result[node_name]["input"]["required"][input_name]
        choices = definition[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RenderError(
            f"ComfyUI node {node_name} does not expose required input {input_name}"
        ) from exc
    if not isinstance(choices, list):
        raise RenderError(f"ComfyUI node {node_name} returned an invalid {input_name} list")
    return [str(choice) for choice in choices]


def print_discovered_group(label: str, choices: list[str]) -> None:
    print(f"{label}:")
    if choices:
        for choice in choices:
            print(f"  {choice}")
    else:
        print("  (none)")


def run_doctor(server: str, profile: str = "z-image") -> int:
    failures = 0
    request_json("GET", f"{server}/system_stats")
    print(f"[ok] ComfyUI: {server}")

    if profile == "realvisxl":
        required_nodes = REALVISXL_REQUIRED_NODES
        required_models = REALVISXL_REQUIRED_MODEL_CHOICES
    else:
        required_nodes = REQUIRED_WORKFLOW_NODES
        required_models = REQUIRED_MODEL_CHOICES

    for node_name in required_nodes:
        result = request_json("GET", f"{server}/object_info/{quote(node_name, safe='')}")
        if node_name in result:
            print(f"[ok] node: {node_name}")
        else:
            print(f"[missing] node: {node_name}")
            failures += 1

    for node_name, input_name, filename in required_models:
        choices = discover_node_choices(server, node_name, input_name)
        if filename in choices:
            print(f"[ok] model: {filename}")
        else:
            print(f"[missing] model: {filename}")
            failures += 1

    if failures:
        print(f"RenderLab doctor found {failures} problem(s).", file=sys.stderr)
        return 1
    print(f"RenderLab profile {profile} is ready.")
    return 0


def run_video_check(server: str) -> int:
    failures = 0
    request_json("GET", f"{server}/system_stats")
    print(f"[ok] ComfyUI: {server}")
    object_info = request_json("GET", f"{server}/object_info")
    for node_name in sorted(H3_REQUIRED_NODES):
        if node_name in object_info:
            print(f"[ok] node: {node_name}")
        else:
            print(f"[missing] node: {node_name}")
            failures += 1
    required_models = (
        ("UNETLoader", "unet_name", H3_UNET),
        ("CLIPLoader", "clip_name", H3_CLIP),
        ("VAELoader", "vae_name", H3_VIDEO_VAE),
        ("VAELoader", "vae_name", H3_AUDIO_VAE),
    )
    for node_name, input_name, filename in required_models:
        choices = discover_node_choices(server, node_name, input_name)
        if filename in choices:
            print(f"[ok] model: {filename}")
        else:
            print(f"[missing] model: {filename}")
            failures += 1
    if failures:
        print(f"RenderLab video check found {failures} problem(s).", file=sys.stderr)
        return 1
    print("RenderLab MiniMax H3 text-to-video is ready.")
    return 0


def run_video(args: argparse.Namespace) -> int:
    if args.check:
        return run_video_check(args.server)
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    frame_count = h3_frame_count(args.duration)
    workflow = build_h3_t2v_workflow(
        prompt=args.prompt.strip(), seed=seed, width=args.width, height=args.height,
        seconds=args.duration, steps=args.steps, filename_prefix=args.filename_prefix,
    )
    started_at = datetime.now(timezone.utc)
    prompt_id = submit(args.server, workflow)
    print(f"prompt_id: {prompt_id}", file=sys.stderr)
    print(f"seed: {seed}", file=sys.stderr)
    print(
        f"frames: {frame_count} at {H3_FPS} fps ({frame_count / H3_FPS:.3f}s)",
        file=sys.stderr,
    )
    history = wait_for_history(args.server, prompt_id, args.timeout, args.poll_interval)
    output_path = local_output_path(args.output_dir, find_saved_output(history))
    finished_at = datetime.now(timezone.utc)
    metadata_path = write_metadata(
        output_path,
        {
            "schema_version": 2,
            "renderlab_version": __version__,
            "prompt_id": prompt_id,
            "mode": "t2v",
            "profile": "minimax_h3_int8",
            "prompt": args.prompt.strip(),
            "effective_prompt": args.prompt.strip(),
            "seed": seed,
            "width": args.width,
            "height": args.height,
            "requested_duration_seconds": args.duration,
            "frame_count": frame_count,
            "fps": H3_FPS,
            "actual_duration_seconds": frame_count / H3_FPS,
            "steps": args.steps,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "models": {
                "diffusion_model": H3_UNET,
                "text_encoder": H3_CLIP,
                "video_vae": H3_VIDEO_VAE,
                "audio_vae": H3_AUDIO_VAE,
            },
            "native_audio": True,
            "server": args.server,
            "submitted_workflow": workflow,
            "submitted_workflow_sha256": sha256_json(workflow),
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "render_seconds": (finished_at - started_at).total_seconds(),
        },
    )
    print(output_path)
    print(f"metadata: {metadata_path}", file=sys.stderr)
    return 0


def lora_sweep_arguments(args: argparse.Namespace) -> list[list[str]]:
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    lora_name = args.lora
    if args.lora_preset is not None:
        lora_name = LORA_PRESETS[args.lora_preset][0]
    common = [
        args.prompt,
        "--profile", args.profile,
        "--seed", str(seed),
        "--width", str(args.width),
        "--height", str(args.height),
        "--timeout", str(args.timeout),
        "--poll-interval", str(args.poll_interval),
        "--output-dir", str(args.output_dir),
        "--server", args.server,
    ]
    if args.steps is not None:
        common.extend(("--steps", str(args.steps)))
    if args.cfg is not None:
        common.extend(("--cfg", str(args.cfg)))
    if args.negative_prompt is not None:
        common.extend(("--negative-prompt", args.negative_prompt))

    runs = [common + ["--filename-prefix", "LoRASweep_Base"]]
    for strength in args.strengths:
        label = str(strength).replace("-", "neg").replace(".", "_")
        runs.append(common + [
            "--lora", lora_name,
            "--lora-model-strength", str(strength),
            "--lora-clip-strength", str(strength),
            "--filename-prefix", f"LoRASweep_{label}",
        ])
    return runs


def run_control_command(args: argparse.Namespace) -> int:
    try:
        if args.command == "video":
            return run_video(args)

        if args.command == "replay":
            return main(replay_arguments(args))

        if args.command == "mask":
            return run_mask(args)

        if args.command == "doctor":
            return run_doctor(args.server, args.profile)

        if args.command == "models":
            for label, node_name, input_name in MODEL_NODE_INPUTS:
                print_discovered_group(
                    label, discover_node_choices(args.server, node_name, input_name)
                )
            return 0

        if args.command == "loras":
            print_discovered_group(
                "loras", discover_node_choices(args.server, "LoraLoader", "lora_name")
            )
            return 0

        if args.command == "lora-presets":
            for key, (name, model, clip, description) in LORA_PRESETS.items():
                print(f"{key}\t{name}\tmodel={model:g}\tclip={clip:g}\t{description}")
            return 0

        if args.command == "lora-sweep":
            runs = lora_sweep_arguments(args)
            for index, render_argv in enumerate(runs, start=1):
                print(f"sweep [{index}/{len(runs)}]", file=sys.stderr)
                result = main(render_argv)
                if result != 0:
                    return result
            return 0

        if args.command == "jobs":
            result = request_json("GET", f"{args.server}/api/jobs?limit={args.limit}")
            jobs = result.get("jobs", [])
            if not jobs:
                print("No jobs.")
                return 0
            for job in jobs:
                print(f"{job.get('id', '?')}\t{job.get('status', 'unknown')}")
            return 0

        prompt_id = quote(args.prompt_id, safe="")
        if args.command == "status":
            job = request_json("GET", f"{args.server}/api/jobs/{prompt_id}")
            print(json.dumps(job, indent=2))
            return 0

        result = request_json("POST", f"{args.server}/api/jobs/{prompt_id}/cancel", {})
        if result.get("cancelled"):
            print(f"cancelled: {args.prompt_id}")
        else:
            print(f"not cancelled: {args.prompt_id}")
        return 0
    except RenderError as exc:
        print(f"renderlab: error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    actual_argv = sys.argv[1:] if argv is None else argv
    if actual_argv and actual_argv[0] in CONTROL_COMMANDS:
        return run_control_command(parse_control_args(actual_argv))
    args = parse_args(actual_argv)
    try:
        if args.variations is not None:
            effective_prompts = expand_prompts(
                args.prompt_server,
                args.prompt_model,
                args.prompt,
                args.variations,
                args.prompt_timeout,
            )
        else:
            effective_prompts = [args.prompt] * args.count
        render_count = len(effective_prompts)
        seeds = resolve_seeds(args.seed, render_count)
        batch_id = str(uuid.uuid4())
        workflow_source = args.workflow.expanduser().resolve()
        workflow_source_sha256 = sha256_file(workflow_source)
        input_source = args.input_image.expanduser().resolve() if args.input_image else None
        input_source_sha256 = sha256_file(input_source) if input_source else None
        uploaded_image = upload_image(args.server, input_source) if input_source else None
        mask_source = args.mask_image.expanduser().resolve() if args.mask_image else None
        mask_source_sha256 = sha256_file(mask_source) if mask_source else None
        uploaded_mask = upload_image(args.server, mask_source) if mask_source else None
        for batch_index, (seed, effective_prompt) in enumerate(
            zip(seeds, effective_prompts, strict=True), start=1
        ):
            started_at = datetime.now(timezone.utc)
            workflow = load_workflow(args.workflow)
            if args.outpaint:
                inject_outpaint_parameters(
                    workflow,
                    prompt=effective_prompt,
                    seed=seed,
                    steps=args.steps,
                    denoise=args.denoise,
                    image=uploaded_image,
                    left=args.outpaint_left,
                    top=args.outpaint_top,
                    right=args.outpaint_right,
                    bottom=args.outpaint_bottom,
                    feather=args.outpaint_feather,
                )
            elif uploaded_mask is not None:
                inject_inpaint_parameters(
                    workflow,
                    prompt=effective_prompt,
                    seed=seed,
                    steps=args.steps,
                    denoise=args.denoise,
                    image=uploaded_image,
                    mask=uploaded_mask,
                    mask_grow=args.mask_grow,
                    mask_feather=args.mask_feather,
                )
            elif uploaded_image is not None:
                inject_img2img_parameters(
                    workflow,
                    prompt=effective_prompt,
                    seed=seed,
                    steps=args.steps,
                    denoise=args.denoise,
                    image=uploaded_image,
                )
            else:
                inject_parameters(
                    workflow,
                    prompt=effective_prompt,
                    seed=seed,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                )
            effective_negative_prompt = inject_generation_controls(
                workflow,
                negative_prompt=args.negative_prompt,
                cfg=args.cfg,
            )
            inject_filename_prefix(workflow, args.filename_prefix)
            for lora in args.loras:
                inject_lora(
                    workflow,
                    name=lora["name"],
                    model_strength=lora["model_strength"],
                    clip_strength=lora["clip_strength"],
                )
            submitted_workflow_sha256 = sha256_json(workflow)
            prompt_id = submit(args.server, workflow)
            print(f"[{batch_index}/{render_count}] prompt_id: {prompt_id}", file=sys.stderr)
            print(f"[{batch_index}/{render_count}] seed: {seed}", file=sys.stderr)
            history = wait_for_history(args.server, prompt_id, args.timeout, args.poll_interval)
            image = find_saved_image(history)
            output_path = local_output_path(args.output_dir, image)
            output_pixel_sha256 = pixel_sha256_file(output_path)
            if (
                args.expected_pixel_sha256 is not None
                and output_pixel_sha256 != args.expected_pixel_sha256
            ):
                raise RenderError(
                    "replay pixel mismatch: expected "
                    f"{args.expected_pixel_sha256}, found {output_pixel_sha256}"
                )
            if args.expected_pixel_sha256 is not None:
                print("replay pixels: identical", file=sys.stderr)
            finished_at = datetime.now(timezone.utc)
            metadata_path = write_metadata(
                output_path,
                {
                    "schema_version": 2,
                    "renderlab_version": __version__,
                    "prompt_id": prompt_id,
                    "batch_id": batch_id,
                    "intent": args.prompt,
                    "prompt": effective_prompt,
                    "effective_prompt": effective_prompt,
                    "negative_prompt": effective_negative_prompt,
                    "prompt_expander": (
                        {
                            "server": args.prompt_server,
                            "model": args.prompt_model,
                            "variation_policy": "directors_v1",
                            "variation_index": batch_index,
                            "variation_count": render_count,
                        }
                        if args.variations is not None
                        else None
                    ),
                    "replay": (
                        {
                            "kind": args.replay_kind,
                            "source_metadata": args.replay_source,
                            "parent_seed": args.parent_seed,
                        }
                        if args.replay_source is not None
                        else None
                    ),
                    "seed": seed,
                    "mode": (
                        "outpaint" if args.outpaint else
                        "inpaint" if mask_source else
                        "img2img" if input_source else "txt2img"
                    ),
                    "batch_index": batch_index,
                    "batch_count": render_count,
                    "width": None if input_source else args.width,
                    "height": None if input_source else args.height,
                    "steps": args.steps,
                    "denoise": args.denoise if input_source else 1,
                    "source_image": (
                        {
                            "path": str(input_source),
                            "sha256": input_source_sha256,
                            "comfy_input": uploaded_image,
                        }
                        if input_source
                        else None
                    ),
                    "mask_image": (
                        {
                            "path": str(mask_source),
                            "sha256": mask_source_sha256,
                            "comfy_input": uploaded_mask,
                            "white_is_editable": True,
                            "grow_pixels": args.mask_grow,
                            "feather_pixels": args.mask_feather,
                        }
                        if mask_source
                        else None
                    ),
                    "outpaint": (
                        {
                            "left": args.outpaint_left,
                            "top": args.outpaint_top,
                            "right": args.outpaint_right,
                            "bottom": args.outpaint_bottom,
                            "feather": args.outpaint_feather,
                        }
                        if args.outpaint
                        else None
                    ),
                    "cfg": args.cfg,
                    "sampler": PROFILES[args.profile]["sampler"],
                    "scheduler": PROFILES[args.profile]["scheduler"],
                    "profile": PROFILES[args.profile]["profile_name"],
                    "models": PROFILES[args.profile]["models"],
                    "lora": (
                        {
                            "name": args.lora,
                            "model_strength": args.lora_model_strength,
                            "clip_strength": args.lora_clip_strength,
                        }
                        if args.lora is not None
                        else None
                    ),
                    "loras": args.loras or None,
                    "server": args.server,
                    "filename_prefix": args.filename_prefix,
                    "workflow": str(workflow_source),
                    "workflow_source_sha256": workflow_source_sha256,
                    "submitted_workflow_sha256": submitted_workflow_sha256,
                    "submitted_workflow": workflow,
                    "output": str(output_path),
                    "output_sha256": sha256_file(output_path),
                    "output_pixel_sha256": output_pixel_sha256,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
                },
            )
            print(output_path)
            print(f"[{batch_index}/{render_count}] metadata: {metadata_path}", file=sys.stderr)
    except RenderError as exc:
        print(f"renderlab: error: {exc}", file=sys.stderr)
        return 1

    return 0
