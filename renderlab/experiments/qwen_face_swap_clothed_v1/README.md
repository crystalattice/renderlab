Completed Qwen Edit 2509 clothed two-image validation: **FAIL**.
See `evaluation.json` and `EVALUATION.md` for the visual assessment, and
`cloud_execution.json` for the single completed run. The preparation notes below
record the original pre-execution baseline and are historical, not current run instructions.

The authoritative graph is RenderLab's local `blueprints/Image Edit (Qwen 2509).json`.
The broken saved workflow `dddf4297-3e13-4aa2-a573-e7a9eb3188dd` is not used.

- `workflow.json`: editor graph, original subgraph plus two LoadImage nodes and SaveImage.
- `api.json`: local API graph; turbo switch/primitive controls resolved from the blueprint.
- `manifest.json`: complete input paths, SHA-256 hashes, prompt, settings, and provenance.
- `request.json`: corrected request; same prompt and acceptance criteria as the authoritative request.
- `experiment.json`: updated experiment record; body/morphology and pose thresholds are 5.
- `validation.json`: static validation report; no inference or model loading.
- `local_node_schemas.json`: schemas extracted from the dev checkout, not a live Cloud inventory.
- `prepare.py` / `validate.py`: local preparation and static validation, no network operations.
- `after_approval.js`: reference upload/execution calls; intentionally not invoked.

Image 1 is the opaque white-T-shirt img_00562 derivative. Its PNG metadata records
the original img_00562 input hash. Image 2 is a clothed synthetic adult portrait
already in RenderLab's input assets, with a clearly visible face. Adulthood is
assessed visually; it is not independently documented. This reference replaces
img_00317, so results are not a paired comparison with the old identity fixture.

Fixed seed: 3407. Qwen Edit 2509 FP8, Lightning LoRA strength 1, turbo enabled,
4 steps, CFG 1, Euler/simple, denoise 1; negative prompt empty; image 3 and mask unused.
The blueprint's existing image scale node controls output dimensions and may
resize/crop the 1160 x 896 target to 1184 x 880. There is no explicit mask restricting editing to the face.

Output prefix: `renderlab_qwen_face_swap_clothed_v1_s3407`.
Expected first filename: `renderlab_qwen_face_swap_clothed_v1_s3407_00001_.png`.
SaveImage adds an available numeric counter; an exact suffix cannot be guaranteed
before checking the output namespace at execution.

Pricing: the previous read-only template estimate returned 0 recognized paid-API
credits. This corrected graph uses native nodes only, so its static paid-API
component is also 0. GPU/queue time and storage are excluded and unpriced; total
execution cost is unknown. The fresh estimate and schema requests failed with
`Auth required`. Restore the connector and re-estimate before seeking spend approval.

After approval, upload exactly the two manifest files using the upload calls in
`after_approval.js`; execute each tool's returned single-use PUT command. Bind the
actual returned Cloud `name` values to nodes 470 and 471. Local absolute paths in
api.json are not Cloud input names. Then the single execution call is
`submit_workflow({workflow: <rebound API object>, confirm: true})` as implemented
in that reference file. No Cloud save is needed. Upload response names and PUT URLs
cannot be known before uploads; they must never be invented. Revalidate hashes and
the rebound payload before submission. Do not send the original nude fixtures.

Recheck locally with:
`python renderlab/experiments/qwen_face_swap_clothed_v1/validate.py`

Validation covers every editor link, every API input, required inputs, connection
types/output slots, ranges/enums, cycles, output reachability, image ordering,
conditioning, sampler settings, and six deliberately invalid mutations. It does
not establish Cloud model availability or runtime generation quality.
