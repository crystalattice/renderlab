# Operation/backend evidence

These are current candidate recommendations, not formal performance rankings or
automatic changes to runtime dispatch. The machine-readable record is
[`operation_backend_evidence.json`](../renderlab/experiments/operation_backend_evidence.json).

| Operation | Current recommendation | Evidence and limits |
|---|---|---|
| Outfit changes | FireRed Image Edit 1.1 preferred candidate | Informal manual observations: mostly successful existing-image edits; incomplete instruction-following is the main failure mode. |
| Reclothing | FireRed Image Edit 1.1 preferred candidate | Same informal editing evidence; no separate validated reclothing rate. |
| Body morphing | FireRed Image Edit 1.1 preferred candidate | Mostly successful in manual use; preservation and requested morphology must still be checked per case. |
| General existing-image editing | FireRed Image Edit 1.1 preferred candidate | Informal evidence, not approval for strict protected-pixel edits. |
| Outpainting | Qwen preferred current candidate | Existing Qwen semantic extension succeeded; original-canvas preservation was 3/5. FireRed manual outpainting is unreliable and can severely distort proportions, including abnormally shortened limbs. |
| Strict face swap | No approved backend | Qwen identity transfer 2/5, instruction adherence 2/5, final **FAIL**. The existing evaluation is unchanged. |
| Controlled inpainting | Unvalidated | The existing Qwen localized color-edit case failed. The FireRed masked-edit experiment is prepared but has not run. |

The user reports **approximately one failure in four manual FireRed attempts**.
Outfit changes, body morphing and similar existing-image modifications are mostly
successful; failures primarily involve incomplete instruction-following. This is
informal recollection without a controlled sample, complete attempt log, fixed case
mix or predeclared rubric. The approximate three-in-four impression is **not a
formal benchmark or validated pass rate**. Outpainting failures are a separate
operation-specific warning, not a measured rate inferred from the editing estimate.

Existing recorded evidence retains its narrower conclusions:

- [Qwen face swap](../renderlab/experiments/qwen_face_swap_clothed_v1/EVALUATION.md): weak identity transfer and unauthorized hairstyle changes.
- [Qwen outpainting](../renderlab/experiments/qwen_outpaint_v0.json): useful semantic extension; strict source recompositing introduced a severe seam.
- [Qwen controlled inpainting](../renderlab/experiments/qwen_inpaint_v0.json): changed garment cut and damaged protected regions/boundaries.
- [Two-stage FireRed material editing](../renderlab/experiments/wet_fabric_two_stage_v0.json): useful material behavior but garment/identity drift; not evidence of strict containment.

The [prepared FireRed controlled-inpainting protocol](../renderlab/experiments/firered_controlled_inpaint_v1/README.md)
separates masked change, outside-mask pixels, outside-mask semantics, boundary
quality, identity, pose, body, background, instruction adherence and artifacts.
It contains no execution result. Current CLI defaults in `appearance.py` describe
available planning paths; they must not be read as approval or as this evidence matrix.
