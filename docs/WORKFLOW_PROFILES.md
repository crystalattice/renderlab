# Versioned workflow profiles

Application capabilities can be supplied by versioned external workflow profiles. `renderlab/workflow_profiles.py` treats prompts, capability identifiers, backend identifiers and model names as opaque data. It performs no semantic prompt compilation, model-specific dispatch, upload, submission, retry, workflow creation or network access. Existing render commands and their built-in presets remain unchanged.

The first integration profile is `renderlab/profiles/firered_shoes_v1.json`. It references the successful shoes-to-stilettos benchmark API file in place, without copying or changing that workflow, prompt or source image. Its resource inventory records historical availability; validation does not assert current Cloud account access.

## Profile schema version 1

Unknown fields are rejected except inside the optional `annotations` object. Paths are absolute or relative to the profile file, independent of the working directory.

| Field | Contract |
|---|---|
| `schema_version` | Integer `1` |
| `profile_id` | Nonempty versioned identifier chosen by the author |
| `capability` | Public, nonempty application capability name |
| `backend` | Nonempty opaque backend identifier |
| `workflow` | `{ "path": "external-api.json", "sha256": "..." }`; hash of exact file bytes |
| `resource_inventory_path` | Offline JSON object with `models` and `assets` arrays of exact backend names; optional annotations |
| `bindings` | Map of names to `{node, input, type, default?, minimum?, maximum?}`; scalar types: string, integer, number, boolean |
| `prompt_binding` | Name of one string binding; literal UTF-8, no substitution or normalization |
| `source_image_bindings` | Names of string bindings with corresponding asset references |
| `seed_bindings` | Integer binding names, restricted to unsigned 64-bit values |
| `numeric_bindings` | Integer or number binding names |
| `model_references` | Binding name to opaque model name; must match resolved value and supplied inventory |
| `asset_references` | Binding name to `{path, sha256, backend_value}`; local original required and verified, backend name must occur in inventory |
| `output_nodes` | Nonempty list of existing API node IDs selected by the author |
| `reconstruction_policy` | `preserve_visible_geometry` or `allow_hidden_region_reconstruction` |
| `immutable_source_policy` | `read_only_verified` |
| `metadata_capture_policy` | `complete_v1` |
| `annotations` | Optional profile-specific JSON object |
| `aliases` | Optional nonempty string aliases for registered profile lookup |
| `content_hash` | SHA-256 of canonical profile JSON excluding this field |

Canonical JSON is UTF-8 from Python `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`. Whitespace changes in the profile file do not change its content hash. Any byte change in the external workflow does change its file hash. Authors can compute the profile hash with `renderlab.workflow_profiles.profile_hash(profile)`; loading never silently repairs hashes.

Bindings may only replace declared existing scalar inputs. Unknown bindings, duplicate target assignments, invalid types/ranges, connection replacements, missing nodes/inputs, invalid references and graph cycles fail explicitly. A binding without a default must be supplied during preparation. Numeric bounds are inclusive. Boolean values are not integers. References cannot be changed with a value override; author a new profile and matching inventory instead.

The author declares output nodes and input types. This module checks graph structure, not backend node-class catalogs, output-slot signatures, model compatibility or prompt semantics. An offline inventory is an explicit dependency snapshot, not proof that a backend currently has those resources. Local and Cloud adapters consume the same record shape. Changing `--backend` changes only the routing identifier: it never translates asset names, transfers files or grants access. Supply a profile/inventory whose exact names are valid on the intended backend.

Reconstruction policy records caller intent; it does not add controls to the external graph or guarantee geometry preservation. Originals are opened read-only and SHA-256 checked. A new job record is created exclusively: it cannot overwrite an original, workflow, existing job or symlink. This is application-level immutability, not a filesystem lock against another process changing files later.

## Preparation, lineage and exact replay

`prepare_job(profile, values=None, backend=None, parent_job=None, source=None)` returns a copied, resolved graph plus a version-1 job record. It never submits. Each record includes a unique job ID, UTC creation timestamp, parent job identifier, source paths/hashes, all resolved bindings, prompt UTF-8 bytes, numeric settings, output declarations, policies, profile snapshot/hash, original workflow bytes/hash, resolved workflow bytes/hash and the resource inventory used. Full graph bytes capture fixed settings beyond configurable bindings. `record_hash` covers the complete record except itself.

`workflow_bytes_base64` is the exact serialized UTF-8 API workflow document for a future backend adapter. An adapter must use those decoded bytes as the workflow document; backend-specific transport envelopes and server-assigned job/output metadata are not fabricated at preparation time. The execution adapter in `renderlab/execution_backend.py` consumes this document directly. Comfy Local embeds it unchanged in the `/prompt` envelope and records its hash, job ID, status history and downloaded outputs.

`replay_job(record_path)` verifies the record, embedded hashes and current original asset hashes, then returns the unchanged record. It does not reread the current profile or base workflow, regenerate a seed, recompile a prompt, create a child job or submit. Thus recorded workflow-input bytes are reproducible even if the original profile/workflow files were moved or changed. Exact input replay does not promise identical generated pixels. `parent_job_id` plus hashed sources records lineage for a new preparation; it is caller-supplied provenance, not an inferred relationship.

Hashes detect accidental corruption, not malicious edits by an author who can recompute them. Profiles and recorded API graphs are executable backend specifications and must be reviewed before any future authorized dispatch.

## CLI

```bash
python -m renderlab profiles list
python -m renderlab profiles list --directory /path/to/profiles
python -m renderlab profiles inspect firered_shoes_v1
python -m renderlab profiles validate /path/to/profile.json
python -m renderlab jobs prepare /path/to/profile.json --bindings values.json --parent-job previous-job-id --output job.json
python -m renderlab jobs replay job.json
python -m renderlab generate --profile firered_shoes_v1 --dry-run --output shoes-job.json
```

`profiles inspect` loads and validates the profile schema/hash. `profiles validate` additionally checks the workflow, default bindings and local dependency inventory using preparation, without saving or submitting a job. For profiles with required values and no defaults, use `jobs prepare --bindings` to validate a complete request. `profiles list` scans only immediate JSON profile files; put inventories in a subdirectory. A bare profile ID resolves in `renderlab/profiles`; external JSON paths are supported.

`jobs` without a subcommand still lists existing ComfyUI jobs. `jobs prepare`, `jobs replay` and profile-based `generate` are offline and print JSON; `--output` saves a new record only. The `generate` command requires exactly one of `--dry-run` and `--execute`. Preparation-only dry run remains offline; `--dry-run --check-backend` additionally performs read-only readiness, source-presence and live-schema checks. Existing positional-prompt generation and existing top-level `replay` retain their prior behavior, including submission; they are separate from these profile preparation commands.

To load a user-authored external profile, provide the schema-v1 profile with its computed content hash, the exact API JSON and hash, the offline resource inventory, verified local asset files and any required binding values. No model-specific Python code is needed. Actual execution requires the separate explicit `--execute` gate and a ready backend. `profiles`, `jobs prepare` and `jobs replay` never submit.

## One-job execution adapter

`ExecutionBackend` declares bounded readiness, asset-presence, validation, single submission, status, output selection/download and explicit cancellation operations. `run_job` accepts the existing prepared record, verifies it with `validate_job`, and passes its exact decoded workflow bytes to the adapter. It does not bind inputs again. It verifies originals again immediately before submission.

The first concrete adapter is **Comfy Local**, using the same HTTP routes already used by RenderLab and defined by this repository's `server.py`: GET `/object_info`, `/view`, `/history/{id}`; POST `/prompt`; explicit targeted cancellation via `/queue` and `/interrupt`. HTTP polling is used instead of introducing a WebSocket client. Only image outputs (PNG, JPEG or WebP) are currently supported. No API endpoint is invented and no core ComfyUI module is changed.

A live schema check is not a server execution-validator call: Comfy Local's POST `/prompt` validates and queues together, so it must never be used as a dry run. Preflight checks required nodes, scalar types/ranges, model/combo membership and connection slots/types against `/object_info`, and compares `/view?type=input` bytes against every declared asset SHA-256. Backend-specific custom validation still occurs during the single authorized POST. Missing models, inputs or remote originals block submission. No upload operation is implemented here.

The sibling `comfy_cloud_mcp` repository was inspected read-only. Its modern MCP client uses `mcp`, `httpx`, `X-API-Key` and the supported `https://cloud.comfy.org/mcp` transport; it also defaults to connection retries. RenderLab's active Python environment lacks `mcp`. Its older render adapter additionally rebinds workflows, polls without a bound and can create placeholder output images. Neither adapter was imported or reused. Cloud requires deliberate integration of its optional SDK/authenticated transport and authoritative private-asset presence checks; this implementation does not borrow Codex-session credentials or infer undocumented HTTP endpoints. No credential contents were read and no Cloud calls were made.

`generate --execute` authorizes at most one submission per invocation. No retry, batch, upload, workflow save, automatic cancellation or output repair exists. HTTP redirects are rejected, avoiding repeated POSTs. Requests have a 30-second socket timeout and a 64-MiB response limit. Status polling is bounded by `--max-polls` (default 120), with `--poll-interval` default 2 seconds. A polling timeout leaves the remote job potentially running; it fails locally and never resubmits. Cancellation is exposed on the adapter for explicit callers only.

Execution creates a new directory exclusively and writes `prepared_job.json` and `execution.json` before submitting. Submission intent is written before POST. An ambiguous submission response remains a failure with `submission_may_have_succeeded: true`; it is never retried. Evidence records the prepared-record hash, exact submitted document hash, credential-free backend origin, backend job ID, submission timestamp, status observations, declared outputs, timing/usage numbers when returned, error stage/code, and original lineage. Downloads use new index-based filenames and retain raw bytes, dimensions, mode and SHA-256. Existing directories and output files are not overwritten.

Evidence excludes response headers, arbitrary backend response fields, raw exception text, signed URLs and authentication material. Terminal errors retain normalized codes and HTTP status when available; full remote tracebacks are deliberately excluded. Credential-bearing workflow input fields are rejected before recording. Prepared metadata and output names must themselves be suitable for persistent evidence.

The fixture remains `firered_shoes_v1`, now with the generic alias `firered_shoes_to_stilettos` and selected `comfy-local` backend. Its graph, prompts, seed and input filename are unchanged. `--source` verifies an alternate local path to the same declared source SHA-256; it never changes the remote binding or uploads bytes.

The exact separately authorized command is:

```bash
python -m renderlab generate --profile firered_shoes_to_stilettos \
  --source /home/codyjackson/Datasets/renderlab-source/additional/Ziggy_Star/SCPE02977_001.jpg \
  --execute --server http://127.0.0.1:8188 \
  --execution-dir output/firered_shoes_authorized_001
```

Do not run this command until separately authorized. The prepared record is `renderlab/profiles/resources/shoes_execution.prepared.json`. The observed readiness result and remaining requirements are in `resources/shoes_execution_readiness.json`. At implementation time the local server was unavailable, so no live model or asset check could complete and no backend dry-run pass is claimed. Start Comfy Local with the three recorded model files and the exact original under its authoritative input filename before requesting execution. Any future upload must be a separate explicitly authorized operation.
