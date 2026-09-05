# FireRed outfit change v1

Prepared from controlled-inpainting commit `1a50854c`. No outfit inference or uploads have occurred. The approved geometry is frozen. The composite correction changed 9,821 weights to make all core pixels fully generated; core and generation PNG hashes are unchanged.

`api.prepared.json` records local asset references. `workflow.json` is the editor graph matching `api.cloud.pending.json`. The source retains its previously successful Cloud binding. Mask nodes 2 and 19 deliberately contain `UPLOAD_REQUIRED_...` placeholders; neither graph is execution-ready until those bindings are resolved.

Seed 3407, 40 steps, CFG 4, Euler/simple, denoise 1, AuraFlow shift 3.1, CFGNorm strength 1, and 1160×896 are preserved. Node 2 loads the binary generation mask through its red channel into SetLatentNoiseMask. Node 19 loads grayscale composite weights through its red channel into ImageCompositeMasked, with original destination, decoded FireRed source, x=0, y=0, resize_source=false. Node 14 saves raw; node 18 saves production separately.

The composite is 255 throughout core and 0 outside generation. The previous inward distance ramp remains only in generation minus core. Production must preserve every original pixel outside generation and equal raw throughout core. Feather-zone pixels intentionally blend raw and original.

Cloud dry run: `validated`, `submitted: false`, with two warnings for the unuploaded masks. All 14 node classes and all three exact model filenames appear in the discovery schemas/catalog. Bundled preflight does not verify a live GPU filesystem. The warnings are unresolved dependencies, not declared false positives. The estimator reports 0 paid-API credits but excludes GPU, queue and storage; total cost is unknown.

Before any separately authorized execution, upload exactly these two files after hash/dimension verification and bind their successful response filenames:

- `cloud_assets/shirt_generation_mask.rgb.png` → node 2 image
- `cloud_assets/shirt_composite_mask.rgb.png` → node 19 image

Do not upload core, overlays, contact sheet, or another source copy. Repeat dry run after binding; investigate every warning/error. `cloud_submit.gated.json` is a one-job template with explicit blockers, not a runnable authorized request. The eventual call is `submit_workflow({workflow: <fully resolved graph>, dry_run: false})`; no such call was made.

Run `python validate_preparation.py` for frozen hash checks, mask relationships, editor/API roundtrip, model/settings/connection checks, actual native red-channel mask loading, and adversarial native CPU compositor containment. Run `python -m unittest discover -s . -p 'test_*.py' -v` for rejection tests. These require the RenderLab environment (NumPy, Pillow, Torch and native ComfyUI dependencies).

After a separately approved job, preserve the raw PNG unchanged and run `python validate_preparation.py --raw /path/raw.png --production /path/production.png` to enforce RGB dimensions, exact outside-generation preservation, exact raw/core equality and complete native-composite equivalence. Review every visual gate in `evaluation_rubric.json`; no garment or boundary-quality score exists before execution.

The upload assets are lossless equal-channel RGB copies. Native LoadImageMask decoded the L reference white as 0.99610895 locally; RGB copies load every weight exactly (white=1). The approved L geometry and corrected L composite remain the reference assets. See `mask_loader_evidence.json`.
