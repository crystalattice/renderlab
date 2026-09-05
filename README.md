<div align="center">

# ComfyUI
**The most powerful and modular AI engine for content creation.**


[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://discord.com/invite/comfyorg
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

<img width="1590" height="795" alt="ComfyUI Screenshot" src="https://github.com/user-attachments/assets/36e065e0-bfae-4456-8c7f-8369d5ea48a2" />
<br>
</div>

ComfyUI is the AI creation engine for visual professionals who demand control over every model, every parameter, and every output. Its powerful and modular node graph interface empowers creatives to generate images, videos, 3D models, audio, and more...
- ComfyUI natively supports the latest open-source state of the art models.
- [Partner nodes](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) provide access to the best closed source models such as Nano Banana, Seedance, Hunyuan3D, etc.
- It is available on Windows, Linux, and macOS, locally with our [desktop application](https://www.comfy.org/download), our [portable install](#installing) or on our [cloud](https://www.comfy.org/cloud).
- The most sophisticated workflows can be exposed through a simple UI thanks to App Mode.
- It integrates seamlessly into production pipelines with our API endpoints.

## Get Started

### Local

#### [Desktop Application](https://www.comfy.org/download)
- The easiest way to get started.
- Available on Windows & macOS.

#### [Windows Portable Package](#installing)
- Get the latest commits and completely portable.
- Available on Windows.

#### [Manual Install](#manual-install-windows-linux)
Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

### Cloud

#### [Comfy Cloud](https://www.comfy.org/cloud)
- Our official paid cloud version for those who can't afford local hardware.

## Examples
See what ComfyUI can do with the [newer template workflows](https://comfy.org/workflows) or old [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Features
- A visual node graph for building and reusing image, video, audio, 3D, and text workflows without code.
- Reusable subgraphs, workflow templates, App Mode, and a local API for integrating workflows into applications.
- Efficient local execution with asynchronous queueing, partial graph re-execution, smart VRAM and RAM management, model offloading, and support for quantized models.
- Broad native model support. This is a representative list; browse the [workflow library](https://comfy.org/workflows/) for maintained, ready-to-run templates.
  - [Image generation](https://comfy.org/workflows/tag/text-to-image/): Stable Diffusion 1.5, SDXL, SD3.5, Flux.1, Flux.2, Qwen Image, Z-Image, Hunyuan Image 2.1, HiDream, Lumina Image 2.0, Chroma, Anima, LongCat Image, Ideogram 4, Krea 2, MageFlow, Microsoft Lens, PixelDiT, Kandinsky 5, and Ernie Image.
  - [Image editing](https://comfy.org/workflows/tag/image-edit/): Flux Kontext, Flux.2 Klein, Qwen Image Edit, HiDream E1.1 and O1, OmniGen2, Boogu, JoyImage Edit, MageFlow Edit, and LongCat Image Edit.
  - [Video generation](https://comfy.org/workflows/tag/video-generation/): Wan 2.1 and 2.2, LTX-Video 2 and 2.3, HunyuanVideo 1.5, Kandinsky 5 Video, CogVideoX, Cosmos Predict2, Bernini-R, SCAIL 2, and Mochi.
  - [Audio and video generation](https://comfy.org/workflows/): MiniMax H3 and LTX-AV.
  - [Audio generation](https://comfy.org/workflows/tag/text-to-audio/): ACE-Step 1.5, Stable Audio 3 and MiniMax Music 3
  - [3D and vision](https://comfy.org/workflows/): Hunyuan3D 2.1, TripoSplat, SeedVR2, SUPIR, Depth Anything 3, MoGe, SAM 3 and 3.1, RT-DETRv4, and BiRefNet.
  - [Text generation](https://comfy.org/workflows/tag/text-generation/): Gemma 3 and 4, Qwen3, Qwen3.5, and Qwen3-VL, including multimodal inputs.
- Load complete checkpoints or separate diffusion models, VAEs, text encoders, LoRAs, ControlNets, adapters, and upscalers from supported model formats.

## RenderLab local text-to-image CLI

This fork includes a focused local CLI using the bundled Z-Image Turbo INT8 API workflow.
[Corpus import, deduplication, experiment subsets, paired-edit validation, and
Base-vs-Distilled preparation](docs/CORPUS_EXPERIMENTS.md) use the same CLI.
Start ComfyUI, then validate the required nodes and models:

```bash
python -m renderlab --version
python -m renderlab doctor
```

Render one image or a sequential warm batch:

```bash
python -m renderlab "starry night, giant robot silhouetted against the sky" \
  --count 3
```

For the photorealistic RealVisXL V5 profile, install
`RealVisXL_V5.0_fp16.safetensors` under ComfyUI's checkpoints model path and run:

```bash
python -m renderlab "editorial portrait in a sunlit apartment" \
  --profile realvisxl
```

The RealVisXL profile defaults to 30 steps, CFG 7, and DPM++ 2M with the Karras
scheduler. Z-Image remains the default profile. `--input-image` and masked inpainting
use the selected profile, so the same v2 editing commands work with
`--profile realvisxl`.

Override classifier-free guidance and the profile's negative prompt explicitly:

```bash
python -m renderlab "studio photograph of a red convertible" \
  --profile realvisxl \
  --cfg 6.5 \
  --negative-prompt "people, text, watermark, distorted wheels"
```

RealVisXL has a built-in anatomy-quality negative prompt when the option is omitted;
`--negative-prompt ""` clears it. Z-Image normally uses zeroed negative conditioning at
CFG 1. Supplying a negative prompt converts that path to text conditioning, but a
non-empty negative prompt requires CFG greater than 1 because CFG 1 cannot use it. The
resolved negative prompt and CFG are recorded in every provenance sidecar.

Validate either installed profile explicitly:

```bash
python -m renderlab doctor --profile z-image
python -m renderlab doctor --profile realvisxl
```

Apply one installed LoRA to either profile with independent model and text-encoder
strengths. The LoRA must match the selected model architecture:

```bash
python -m renderlab loras
python -m renderlab "editorial portrait in a sunlit apartment" \
  --profile realvisxl \
  --lora "styles/example-sdxl.safetensors" \
  --lora-model-strength 0.8 \
  --lora-clip-strength 0.8
```

RenderLab also includes named settings from its local RealVisXL LoRA survey:

```bash
python -m renderlab lora-presets
python -m renderlab "close portrait" --profile realvisxl --lora-preset realistic-eyes
```

Repeat `--lora-preset` to stack tested LoRAs in the given order. Presets are SDXL-only,
so RenderLab rejects them unless `--profile realvisxl` is selected:

```bash
python -m renderlab "rainy street portrait" --profile realvisxl \
  --lora-preset angelica \
  --lora-preset realistic-eyes \
  --lora-preset cyber-goth
```

An explicit strength option overrides that part of a preset. To compare a new LoRA
without seed drift, render a no-LoRA baseline followed by a fixed-seed strength sweep:

```bash
python -m renderlab lora-sweep "close portrait" \
  --lora "Realistic_eyes.safetensors" --strengths 0.25,0.5,0.75,1.0
```

Each sweep image uses the ordinary render path and receives its own provenance sidecar.

For img2img, add a source image. RenderLab holds the seed and prompt fixed, then renders
a baseline plus every LoRA strength at denoise `0.25`, `0.45`, and `0.65`:

```bash
python -m renderlab lora-sweep "adult woman in a bedroom, photographic" \
  --input-image ./source.png \
  --lora "ArtfulNSFWV2SDXL.safetensors" \
  --strengths 0.25,0.5,0.75
```

Use `--denoises 0.3,0.5` to replace the default denoise axis. Output names encode both
values, such as `LoRAI2I_D0_45_L0_5`, so comparisons remain mechanically identifiable.
Preset sweeps vary model strength while retaining the preset's tested CLIP strength;
for an explicit LoRA, `--clip-strength` can hold CLIP influence fixed independently.

Both strengths default to `1.0` and accept values from `-10.0` through `10.0`. RenderLab
routes the active model and text encoder through ComfyUI's `LoraLoader`; the LoRA filename
and resolved strengths are recorded in the output provenance sidecar. A LoRA built for
SDXL will not work with Z-Image, and vice versa.

Validated RTX 2060 Super 8 GB performance for RealVisXL at 1024x1024 and 30 steps is
approximately 97 seconds for the first checkpoint load and 22.5 seconds warm. The
832x1216 portrait profile rendered warm in approximately 22.7 seconds.

Modify a provided image with img2img. Lower denoise preserves more of the source; higher
denoise permits larger changes:

```bash
python -m renderlab "change the daytime scene to a rainy neon night" \
  --input-image ./source.png \
  --denoise 0.45
```

RenderLab uploads the source to ComfyUI's input directory and records its local path,
SHA-256, uploaded name, and denoise strength in the output sidecar.

Outpainting is intentionally disabled. A normal RealVisXL checkpoint repeatedly rendered
the source boundary as a literal billboard or screen; RenderLab will not expose that path
again until it has a dedicated inpainting model and a visually validated workflow.

For a localized edit, provide a same-size black/white mask. White pixels are editable;
black pixels are protected:

```bash
python -m renderlab "an elegant fitted black gothic dress with black lace" \
  --input-image ./source.png \
  --mask-image ./dress-mask.png \
  --denoise 0.65
```

`--mask-grow` defaults to 6 pixels so the generated region overlaps its boundary rather
than leaving a hard seam. The decoded edit is composited back over the original through
that grown mask, with a 6-pixel edge blur controlled by `--mask-feather`. Pixels outside
the composite mask therefore remain byte-for-byte sourced from the input instead of the
model's re-decoded approximation. Source and mask hashes are both recorded in provenance.

To generate the mask automatically, install
`sam3.1_multiplex_fp16.safetensors` under ComfyUI's checkpoints model path, then describe
the existing object or body region to select:

```bash
python -m renderlab mask ./source.png "chest, abdomen, pelvis" --within "person"
```

The command uses ComfyUI's built-in `SAM3_Detect` node and writes a
`RenderLabMask_....png` file. Comma-separated targets are detected independently and
combined; `--within` intersects every target with an enclosing object so a body-region
request cannot select similar-looking furniture or background. It rejects empty,
full-frame, border-touching, excessively scattered, grayscale, and image-bearing masks;
accepted output is strictly black and white, with white pixels editable. Use the printed
path directly with `--mask-image`. If SAM3 finds nothing, use simpler nouns or lower
`--threshold` from its default of `0.5`. `--allow-border` overrides the border guard for
subjects that legitimately extend beyond the frame.

Compile a click-and-go appearance preset into a backend-neutral render plan:

```bash
python -m renderlab appearance presets
python -m renderlab appearance plan \
  renderlab/experiments/examples/bikini_appearance_request.json \
  --output ./bikini-plan.json
```

Presets describe intent rather than a canned prompt. The compiled plan records the source
semantics, target appearance, preservation constraints, selected backend, acceptance gates,
and a conditional identity-repair stage. `source_semantics: semantic_evidence` reserves the
same contract for future manga semantic-restaging workflows where the source is evidence,
not an editable canvas.

The bundled Qwen workflows also back the `Repair Region` and `Extend Canvas` presets.
The first outpainting capability check extends the existing gym reference downward by 512
pixels:

```bash
python -m renderlab appearance plan \
  renderlab/experiments/examples/outpaint_lower_body_request.json \
  --output /tmp/outpaint-plan.json
```

In ComfyUI, import `blueprints/Image Outpainting (Qwen-Image).json`, load the source,
and set left/top/right/bottom to `0/0/0/512`. Enter the request's `target.continuation`
text in the root text widget. The blueprint definition contains an internal feathering
input, but the imported Cloud root node does not expose it; RenderLab therefore does not
claim it as a user control. The visible expansion defaults are all zero, so the bottom
value must be changed for the test to outpaint.

The blueprint's bypassed `ImageCompositeMasked` path was evaluated as a strict source-
preservation variant. With pad feathering trapped at zero, it produced a conspicuous
horizontal seam across the subject and scene. RenderLab retains the validated generative
output path and does not expose strict preservation until an overlap/blend control is
available.

Operational commands:

```bash
python -m renderlab jobs
python -m renderlab status PROMPT_ID
python -m renderlab cancel PROMPT_ID
python -m renderlab models
python -m renderlab loras
python -m renderlab lora-presets
```

Every output receives an adjacent JSON provenance sidecar containing the resolved seed,
effective workflow, hashes, model filenames, timing, and batch identity.

Replay any completed render from that sidecar. RenderLab restores the effective prompt,
seed, profile, dimensions, steps, CFG, negative prompt, LoRA settings, and image-edit
controls instead of asking you to reconstruct the command by hand:

```bash
python -m renderlab replay ./output/RenderLab_00049_.png.json
```

Restore the same prompt, profile, dimensions, controls, inputs, and LoRA stack while
changing only the seed:

```bash
python -m renderlab replay ./output/RenderLab_00049_.png.json --new-seed
python -m renderlab replay ./output/RenderLab_00049_.png.json --seed 424242
```

Variant sidecars record the parent sidecar and parent seed. Pixel-identity verification
is intentionally limited to exact replay because a new seed is supposed to change pixels.

Replay gives the `SaveImage` node a unique `RenderLabReplay_...` prefix. This forces
ComfyUI to write a new PNG even when every generation node is already cached; the sampler
inputs remain identical. Because ComfyUI embeds the changed workflow in the PNG, raw file
hashes will differ. RenderLab therefore records a normalized RGBA pixel hash and reports
`replay pixels: identical` only when every decoded pixel matches the original.

For img2img and inpainting, replay verifies the recorded source and mask SHA-256 hashes
before submitting anything. A changed or missing input stops the replay rather than
quietly producing a different experiment. `--server`, `--output-dir`, `--timeout`, and
`--poll-interval` may be supplied after the sidecar path for a different runtime location.

### MiniMax H3 text-to-video

Check that ComfyUI exposes the H3 nodes and all four model files, then render a video:

```bash
python -m renderlab video --check
python -m renderlab video \
  "A red fox trots through a moonlit forest, cinematic tracking shot" \
  --seed 1001
```

The local H3 profile defaults to 608x352, a nominal five seconds, 20 steps, 24 fps,
and native generated audio. H3 requires a specific frame grid, so five requested seconds
becomes 124 frames (5.167 seconds); both durations are recorded in provenance. Override
these controls with `--width`, `--height`, `--duration`, and `--steps`.

On an RTX 2060 Super 8 GB, a five-second H3 render is expected to take minutes rather
than seconds and the default timeout is two hours. Stop other GPU-offloaded workloads
before rendering. ComfyUI should remain running so its loaded models can stay warm.

For semantic variation rather than seed-only variation, point RenderLab at a local
OpenAI-compatible prompt model (for example, llama.cpp server) and request variations:

```bash
python -m renderlab "starry night, giant robot, shooting stars" \
  --variations 3 \
  --prompt-server http://127.0.0.1:8084 \
  --prompt-model local
```

The expander runs once before rendering. The original intent and every effective prompt
are preserved separately in the provenance sidecars.

The default `8084` endpoint is the local prompt-expander service. GPT-OSS 20B Q4 is the
validated director model: run its llama.cpp server CPU-only with zero GPU layers. Smaller
models may satisfy the JSON contract while producing conservative variations that merely
swap backgrounds.

On the validated RTX 2060 Super 8 GB profile, stop Waldo before rendering. Waldo's
GPU-offloaded model consumes enough VRAM to make Z-Image's text-encoder load fail or crash
ComfyUI even when the prompt expander itself is CPU-only. A running CPU-only GPT-OSS prompt
server is compatible with rendering.
- Built-in tools for inpainting, outpainting, reference conditioning, masks and compositing, model merging, upscaling, frame interpolation, segmentation, depth estimation, and media processing.
- Save and load workflows as JSON, or recover complete workflows and seeds from supported generated media.
- Runs fully offline: core does not download anything unless you request it. Use `--disable-api-nodes` to disable the optional paid [Comfy API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) and force all built-in functionality to stay offline.
- Extend ComfyUI with custom nodes
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).


## Release Process

ComfyUI follows a weekly release cycle targeting Monday but this regularly changes because of model releases or large changes to the codebase. There are three interconnected repositories:

1. **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
   - Releases a new major stable version (e.g., v0.7.0) roughly every 2 weeks.
   - Starting from v0.4.0 patch versions will be used for fixes backported onto the current stable release.
   - Minor versions will be used for releases off the master branch.
   - Patch versions may still be used for releases on the master branch in cases where a backport would not make sense.
   - Commits outside of the stable release tags may be very unstable and break many custom nodes.
   - Serves as the foundation for the desktop release

2. **[Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop)**
   - Builds a new release using the latest stable core version

3. **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
   - Every 2+ weeks frontend updates are merged into the core repository
   - Features are frozen for the upcoming core release
   - Development continues for the next release cycle

## Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users

# Installing

## Windows and Mac

We highly recommend using the [desktop app](https://comfy.org/download):

### [Link to Download](https://comfy.org/download)

The desktop app is the easiest and best way to use ComfyUI for new users.

## Windows Portable

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only. It is not recommended for regular users. Regular users should use the desktop app above.

[Direct link to download (nvidia)](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

Simply download, extract with [7-Zip](https://7-zip.org) or with the windows explorer on recent windows versions and run. For smaller models you normally only need to put the checkpoints (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints but many of the larger models have multiple files. Make sure to follow the instructions to know which subfolder to put them in ComfyUI\models\

If you have trouble extracting it, right click the file -> properties -> unblock

The portable above currently comes with python 3.13 and pytorch cuda 13.0. Update your Nvidia drivers if it doesn't start.

#### All Official Portable Downloads:

[Portable for AMD GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z)

[Portable for Intel GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z)

[Portable for Nvidia GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z) (supports 20 series and above).

[Portable for Nvidia GPUs with pytorch cuda 12.6 and python 3.12](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z) (Supports Nvidia 10 series and older GPUs, DO NOT USE THIS ON NEWER 20 SERIES AND ABOVE GPUS).

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

You can install and start ComfyUI using comfy-cli:
```bash
pip install comfy-cli
comfy install
```

## Manual Install (Windows, Linux)

Python 3.14 works but some custom nodes may have issues. The free threaded variant works but some dependencies will enable the GIL so it's not fully supported.

Python 3.13 is very well supported. If you have trouble with some custom node dependencies on 3.13 you can try 3.12

torch 2.7 is minimally supported but using a newer version is extremely recommended. Using a cu130 or above version of pytorch is required on Nvidia 20 series and above. Some features and optimizations might only work on newer versions. We generally recommend using the latest major version of pytorch with the latest cuda version unless it is less than 2 weeks old. If your pytorch is more than 6 months old, please update it.

### Instructions:

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae


### AMD GPUs (Linux)

AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2```

This is the command to install the nightly with ROCm 7.2 which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2```


### AMD GPUs (Experimental: Windows and Linux), RDNA 3, 3.5 and 4 only.

These have less hardware support than the builds above but they work on windows. You also need to install the pytorch version specific to your hardware.

RDNA 3 (RX 7000 series):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/```

RDNA 3.5 (Strix halo/Ryzen AI Max+ 365):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1151/```

RDNA 4 (RX 9000 series):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/```

### Intel GPUs (Windows and Linux)

Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More information can be found [here](https://pytorch.org/docs/main/notes/get_start_xpu.html)

1. To install PyTorch xpu, use the following command:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

This is the command to install the Pytorch xpu nightly which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

### NVIDIA

Nvidia users should install stable pytorch using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130```

This is the command to install pytorch nightly instead which might have performance improvements.

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```pip install -r requirements.txt```

After this you should have everything installed and can proceed to running ComfyUI.

### Others:

#### Apple Mac silicon

You can install ComfyUI in Apple Mac silicon (M1, M2, M3 or M4) with any recent macOS version.

1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
1. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
1. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. Launch ComfyUI by running `python main.py`

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### Ascend NPUs

For models compatible with Ascend Extension for PyTorch (torch_npu). To get started, ensure your environment meets the prerequisites outlined on the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page. Here's a step-by-step guide tailored to your platform and installation method:

1. Begin by installing the recommended or newer kernel version for Linux as specified in the Installation page of torch-npu, if necessary.
2. Proceed with the installation of Ascend Basekit, which includes the driver, firmware, and CANN, following the instructions provided for your specific platform.
3. Next, install the necessary packages for torch-npu by adhering to the platform-specific instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
4. Finally, adhere to the [ComfyUI manual installation](#manual-install-windows-linux) guide for Linux. Once all components are installed, you can run ComfyUI as described earlier.

#### Cambricon MLUs

For models compatible with Cambricon Extension for PyTorch (torch_mlu). Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
2. Next, install the PyTorch(torch_mlu) following the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
3. Launch ComfyUI by running `python main.py`

#### Iluvatar Corex

For models compatible with Iluvatar Extension for PyTorch. Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
2. Launch ComfyUI by running `python main.py`


## [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** is an extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

### Setup

1. Install the manager dependencies:
   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:
   ```bash
   python main.py --enable-manager
   ```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (implies `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |


# Running

```python main.py```

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

For AMD 7600 and maybe other RDNA3 cards: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm Tips

You can try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed things up at the cost of a very slow initial run.

# Notes

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```


## How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.

## How to use TLS/SSL?
Generate a self-signed certificate (not appropriate for shared/production use) and key by running the command: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL, the app will now be accessible with `https://...` instead of `http://...`.

> Note: Windows users can use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) or one of the [3rd party binary distributions](https://wiki.openssl.org/index.php/Binaries) to run the command example above.
<br/><br/>If you use a container, note that the volume mount `-v` can be a relative path so `... -v ".\:/openssl-certs" ...` would create the key & cert files in the current directory of your command prompt or powershell terminal.

## Support and dev channel

[Discord](https://comfy.org/discord): Try the #help or #feedback channels.

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).

See also: [https://www.comfy.org/](https://www.comfy.org/)

> _psst — we're hiring!_ Help build ComfyUI: [comfy.org/careers](https://www.comfy.org/careers)

## Frontend Development

As of August 15, 2024, we have transitioned to a new frontend, which is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend). The compiled JS files (from TS/Vue) are published to [pypi](https://pypi.org/project/comfyui-frontend-package) and installed as a dependency in ComfyUI.

### Reporting Issues and Requesting Features

For any bugs, issues, or feature requests related to the frontend, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend). This will help us manage and address frontend-specific concerns more efficiently.

### Using the Latest Frontend

The new frontend is now the default for ComfyUI. However, please note:

1. The frontend in the main ComfyUI repository is updated fortnightly.
2. Daily releases are available in the separate frontend repository.

To use the most up-to-date frontend version:

1. For the latest daily release, launch ComfyUI with this command line argument:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. For a specific version, replace `latest` with the desired version number:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

This approach allows you to easily switch between the stable fortnightly release and the cutting-edge daily updates, or even specific versions for testing purposes.

# QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)
