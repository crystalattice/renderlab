# FireRed single-pass outfit benchmark v1

Status: source selection blocked. The full tagged RenderLab corpus/archives have not been located. No source is selected and no runnable payload or submission object is claimed. Supply the authoritative corpus location to complete selection, workflows, dry runs and gating.

This benchmark validates reproducible API behavior for capabilities already observed manually. It is not exploratory proof that FireRed can edit clothing. User-supplied informal observations live separately in informal_observations.json; the roughly one-in-four informal failure rate is not a benchmark pass rate. FireRed is not the preferred outpainting backend.

The three separate cases are shoes_to_stilettos, dress_to_top_and_miniskirt and clothing_to_bikini. Each has an exact fixed prompt, source requirements and unique native-output prefix. settings.json freezes the requested non-turbo, no-LoRA configuration and seed 3407. evaluation_rubric.json defines independent scores and hard failures.

Only full-image unmasked editing is permitted. No canvas extension, reframing, missing-body reconstruction, invented off-frame anatomy, inpainting masks, outpainting, deterministic compositing, face swaps, pose controls or post-generation repair. Native output must be preserved unchanged. Sources must be clothed adults whose requested result fits entirely inside the existing canvas.

Read-only Cloud discovery found all 11 required node types and all three exact models. Dry runs await concrete source selection. Uploads and inference remain unauthorized. Cases must have separate one-job gates; no batch and no retries.

Completed evidence at commit 2e37059c is preserved without modifying either prior experiment. Controlled inpainting and deterministic containment are validated. Automatic garment masking, accessory preservation and feathered compositing remain unresolved. The tank-top experiment failed visual acceptance due to the white hem halo, missing necklace and bodysuit-like lower silhouette.
