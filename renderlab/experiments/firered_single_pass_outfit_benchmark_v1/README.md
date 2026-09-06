# FireRed single-pass outfit benchmark v1

All three cases are prepared and passed Cloud dry-run preflight: `validated`, `submitted: false`, no warnings. They are **not execution-ready**: two unique originals still require separately authorized upload and authoritative filename binding. No upload, inference, batch or retry is authorized or performed.

This benchmark validates reproducible API behavior for clothing-editing capabilities already observed manually. It is not exploratory proof that FireRed can edit clothing.

| Case | Approved source | Native dimensions | Expected local output |
|---|---|---|---|
| shoes_to_stilettos | S0040 | 1440×2160 | shoes_to_stilettos/native_output.png |
| dress_to_top_and_miniskirt | S0534 | 680×1024 | dress_to_top_and_miniskirt/native_output.png |
| clothing_to_bikini | S0040 | 1440×2160 | clothing_to_bikini/native_output.png |

S0040 reuse is intentional: the footwear and bikini cases compare a localized edit against a large outfit edit while holding source bytes, dimensions, model settings, topology and seed constant. Only positive prompt and output prefix differ. Three cases are not three independent source samples; do not infer a general pass rate from them.

Source paths, SHA-256, anatomy/garment visibility, suitability and approval are recorded in the case manifests and source_review/selection.json. Both are clothed adult-appearing subjects on visual inspection, and the user approved these final sources; documentary ages were not independently verified. S0040 contains both complete shoes, torso, hips and limbs. S0534 contains the full dress, waist/hips, knees and calves; its feet meet the bottom edge, which must remain unchanged. No off-frame body reconstruction is required for the dress transformation.

## Frozen configuration

FireRed-Image-Edit-1.1-transformer.safetensors; qwen_2.5_vl_7b_fp8_scaled.safetensors; qwen_image_vae.safetensors. Non-turbo, no LoRA, 40 steps, CFG 4, Euler/simple, denoise 1, shift 3.1, CFGNorm 1, fixed seed 3407. Exact user prompts remain byte-for-byte in each prompt.txt. Negative prompt is empty.

Each 12-node graph loads one original, encodes it with the Qwen VAE, applies full-image KSampler denoising and saves the decoded image directly. No mask, latent noise mask, compositing, outpainting, reframing, face swap, pose control or repair nodes exist. No image resizing is introduced; both source dimensions are multiples of eight. Preserve native output download bytes unchanged. Do not promise pixel-exact preservation from native full-image editing.

## Artifacts and gates

Each case includes api.prepared.json (local source path), api.cloud.pending.json (explicit unresolved upload placeholder), workflow.json (editor form matching the pending API graph), manifest.json, evaluation_rubric.json, cloud_bindings.json, dry-run request/response, structural_validation.json and cloud_submit.gated.json. SaveImage prefixes are unique and end in s3407_native. Exact Cloud output filenames remain unknown until execution; expected filename patterns are documented per case.

Upload only these two originals after separate authorization:

- `/home/codyjackson/Datasets/renderlab-source/additional/Ziggy_Star/SCPE02977_001.jpg`: bind its one successful upload response filename to node 1 in shoes and bikini.
- `/home/codyjackson/Datasets/renderlab-source/round2/clothing3/250917858c660eb225f3.jpg`: bind its successful response filename to node 1 in dress.

Do not infer a Cloud name from these basenames and do not duplicate S0040's upload. cloud_bindings.json is the unique-asset registry. After binding only node 1, repeat dry-run validation and review every warning/error. The current clean preflight does not mean placeholder files exist at runtime. Obtain explicit authorization for each one-job call; do not batch, retry or reuse a consumed authorization. The three case-local cloud_submit.gated.json files hold the exact call templates. No executable resolved graph is claimed before upload.

## Evaluation and validation

Each case has twelve independent 1–5 scores, with artifact severity lower-is-better, and nine hard-fail conditions. One-piece substitution, identity/limb distortion, forced pose/framing changes and major unrelated changes fail regardless of averaged scores. All scores remain null: no benchmark inference has run.

Run `PYTHONDONTWRITEBYTECODE=1 openai_env/bin/python renderlab/experiments/firered_single_pass_outfit_benchmark_v1/validate_benchmark.py` from the repository root. It verifies sources, source approval, exact frozen prompts/settings, native decode-to-save routing, typed acyclic links, editor/API equivalence, same-source controls, unique upload registry, dry-run verdicts and independent execution gates. Run the complete RenderLab suite with `PYTHONDONTWRITEBYTECODE=1 openai_env/bin/python -m unittest discover -s tests/renderlab -v`.

Inventory metadata and the local contact-sheet script are in source_review/. Corpus originals and contact sheets are not copied into Git. Contact sheets are local review aids only.

## Existing evidence, kept separate

The two completed experiments remain unchanged at their recorded evidence. Controlled inpainting and deterministic containment are validated. Automatic garment masking, accessory preservation and feathered compositing remain unresolved. The tank-top case failed visual acceptance due to its white hem halo, missing necklace and bodysuit-like lower silhouette.

User-supplied informal observations live separately in informal_observations.json: FireRed generally performs existing-canvas outfit changes, reclothing and body morphing well; the approximate one-in-four informal failure rate concerns incomplete instruction adherence and is not a formal benchmark pass rate. Outpainting is unreliable and can compress or distort visible limbs to force additions into the existing canvas. FireRed is not the preferred outpainting backend.
