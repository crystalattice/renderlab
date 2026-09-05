# FireRed controlled inpainting v1 — prepared, not executed

One fixed-seed feasibility case: recolor a **128 × 128 rectangular patch** on the
existing opaque white T-shirt to royal blue. The clothed source is the previously
validated target from `qwen_face_swap_clothed_v1/manifest.json`, SHA-256
`da52612e6f04b4f030e4413d54f49246a2ea34e30a4f0ec93c9641cee4f84b75`.
No source image is copied into Git. This is a localized color-patch test, not a
whole-shirt replacement or an outfit-change test.

`mask.png` is an RGB binary mask at the source's 1160 × 896 resolution. White is
editable; black is protected. The exclusive rectangle is `[552, 424, 680, 552]`,
entirely inside the front shirt fabric, below the face/neck and away from arms,
hem, skin and background. There is no feathering or resizing. It remains a local reference only. The executed mask is generated natively: a zero
1160 × 896 SolidMask, a one 128 × 128 SolidMask, and MaskComposite(add) at (552, 424).
No uploaded mask filename is used.

`api.prepared.json` is a local API-format candidate graph, not a submission call.
The bundled FireRed Image Edit 1.1 blueprint has no mask input. This experiment
uses native `SolidMask`/`MaskComposite` and `SetLatentNoiseMask` before `KSampler`, using the blueprint's
non-turbo settings: seed 3407, 40 steps, CFG 4, Euler/simple, denoise 1, model shift
3.1 and CFGNorm 1. No Lightning LoRA is used. The blueprint's megapixel resize is
omitted to preserve mask coordinates. This adaptation is unvalidated at runtime;
Cloud catalog and dry-run checks pass; native FireRed inpainting quality remains unproven.

Only the raw decoded result is scored. There is no source recompositing or
postprocessing to conceal outside-mask drift. A VAE round-trip can alter pixels
outside a latent sampling mask; this is precisely a failure the protocol must
detect. A later containment/compositing experiment would be a separate variant.

| Acceptance criterion | Gate | Measurement |
|---|---:|---|
| Requested change inside mask | ≥4/5 | Royal-blue coverage, retained folds, shading and fabric; report coverage separately. |
| Pixels outside mask | 5/5 | Exactly unchanged decoded RGB pixels: changed-pixel fraction 0 and maximum channel error 0; report MAE too. |
| Semantics outside mask | 5/5 | No visible change to protected clothing, subject or scene, independently of numeric pixel error. |
| Edge integration | ≥4/5 | Inspect 8-pixel inner/outer boundary rings for halos, spill, tearing, doubled edges or broken folds. A deliberate color-patch edge is expected. |
| Identity preservation | 5/5 | Face, hairstyle and hair length unchanged. |
| Pose preservation | 5/5 | Head angle, torso lean, limbs, hands and grips unchanged. |
| Body preservation | 5/5 | Visible proportions and silhouette unchanged. |
| Background preservation | 5/5 | Machine, pads, cables, weights and gym background unchanged. |
| Instruction adherence | ≥4/5 | Only the specified shirt patch recolored; no text, logo, new garment or accessories. |
| Artifact severity | ≤2/5 | No more than minor new artifacts; severe anatomy errors automatically fail. |

The 1–5 anchors and measurement definitions are frozen in `experiment.json` before
execution. Evaluate at original resolution without registering/resizing the output.
Dimension mismatch fails. The entire black mask region remains protected, including
the outer edge ring. Every gate must pass; semantic success cannot override pixel
failure. Numeric blue coverage is descriptive, not a substitute for visual texture
and color assessment.

`evaluation.pending.json` deliberately contains null scores, measurements and
disposition. No execution is authorized by these files. The planned single attempt
must not be replaced by a favorable seed. Any future run must retain the exact
prompt/settings, model versions, source/mask hashes, raw output, output hash and
dimensions, runtime status, and all failed criteria. One case cannot establish a
backend-wide success rate.

`workflow.json` is the local editor graph; `api.cloud.resolved.json` retains the
verified Cloud source filename. `validate_native_mask.py` runs the native mask
operations on CPU and compares normalized mask tensors and decoded RGB bytes
against `mask.png`; no diffusion inference is performed.

The native-mask dry run is validated with zero warnings and `submitted: false`.
See `execution_readiness.json`. `cloud_submit.gated.json` contains the exact
one-job arguments for `mcp__comfy_cloud__submit_workflow`, including `dry_run: false`;
it is reference data only and requires explicit execution approval. No run was submitted.
