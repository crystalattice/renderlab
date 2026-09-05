# FireRed controlled inpainting v1 — evaluated raw + deterministic production

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
omitted to preserve mask coordinates. The raw graph completed one Cloud run.
Semantic editing passed this case; strict raw pixel containment failed. The revised
compositing branch has been verified locally and passed Cloud dry-run validation.

The frozen primary evaluation scores only the raw decoded result and remains FAIL.
The later production derivative uses explicit deterministic compositing, reported separately. A VAE round-trip can alter pixels
outside a latent sampling mask; this is precisely a failure the protocol must
detect. The production containment extension is assessed separately from that raw protocol.

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

`evaluation.json` records the completed raw evaluation. `EVALUATION.md` separates
semantic preservation, exact pixels, masked instruction success and boundary quality.
`raw_output_1bdff9f9.png` is the unchanged downloaded output; `production_composited.png`
is the deterministic production image. The latter is not substituted into the raw score.

The revised graph keeps raw SaveImage node 14 and adds ImageCompositeMasked node 17:
destination node 1, source node 13, mask node 2, x=0, y=0, resize_source=false.
SaveImage node 18 saves the production result. Both full-resolution inputs and the mask
share the 1160×896 canvas; the rectangle remains x=552, y=424, 128×128.

`api.executed.1bdff9f9.json` preserves the exact graph used by the completed job.
`workflow.json` and `api.cloud.resolved.json` describe the revised two-output graph.
`validate_native_mask.py` validates mask equivalence and editor/API structure.
`validate_composite.py` reproduces native CPU compositing using existing images only.
No further inference is authorized; any gated submission data is reference only.
