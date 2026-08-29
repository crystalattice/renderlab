from __future__ import annotations

from typing import Any


H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_FPS = 24

H3_REQUIRED_NODES = {
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


def h3_frame_count(seconds: float) -> int:
    """Snap nominal duration to the frame grid required by MiniMax H3."""
    base = max(5, round(seconds * H3_FPS))
    return base + (5 - (base % 17)) % 17


def build_h3_t2v_workflow(
    *, prompt: str, seed: int, width: int, height: int, seconds: float,
    steps: int, filename_prefix: str,
) -> dict[str, Any]:
    return {
        "119": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": H3_VIDEO_VAE},
        },
        "120": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": H3_AUDIO_VAE},
        },
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
                "model": ["127", 0], "scheduler": "simple",
                "steps": steps, "denoise": 1.0,
            },
        },
        "125": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["129", 0], "guider": ["126", 0],
                "sampler": ["123", 0], "sigmas": ["124", 0],
                "latent_image": ["131", 1],
            },
        },
        "126": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["127", 0], "conditioning": ["131", 0]},
        },
        "127": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"},
        },
        "128": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"},
        },
        "129": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "130": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["122", 0], "audio": ["121", 0],
                "fps": H3_FPS, "bit_depth": 8,
            },
        },
        "131": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["128", 0], "vae": ["119", 0], "prompt": prompt,
                "width": width, "height": height, "length": h3_frame_count(seconds),
            },
        },
        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["130", 0], "filename_prefix": filename_prefix,
                "format": "auto", "codec": "auto",
            },
        },
    }
