# RenderLab v1

RenderLab submits one Z-Image Turbo INT8 text-to-image render to a local ComfyUI server.

Start ComfyUI normally, then run from the repository root:

```bash
python -m renderlab "a red fox sitting beside a brass telescope at sunrise"
```

Defaults are 1024x1024, 8 steps, CFG 1, and a random 64-bit seed resolved by RenderLab before submission. The final image path is written to stdout. Progress identifiers, the resolved seed, and the metadata sidecar path are written to stderr.

Use an exact seed:

```bash
python -m renderlab "a red fox sitting beside a brass telescope at sunrise" --seed 123456
```

The API defaults to `http://127.0.0.1:8188`. If ComfyUI writes somewhere other than this repository's `output/` directory, pass its filesystem path with `--output-dir`.

```bash
python -m renderlab "prompt" --server http://127.0.0.1:8188 --output-dir /path/to/ComfyUI/output
```

Each image receives an adjacent `<image>.json` metadata file containing the resolved seed, prompt, dimensions, steps, model profile, prompt ID, output path, timestamps, and duration.

Run the focused tests with:

```bash
python -m unittest discover -s tests/renderlab -v
```
