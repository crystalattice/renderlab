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
| `content_hash` | SHA-256 of canonical profile JSON excluding this field |

Canonical JSON is UTF-8 from Python `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`. Whitespace changes in the profile file do not change its content hash. Any byte change in the external workflow does change its file hash. Authors can compute the profile hash with `renderlab.workflow_profiles.profile_hash(profile)`; loading never silently repairs hashes.

Bindings may only replace declared existing scalar inputs. Unknown bindings, duplicate target assignments, invalid types/ranges, connection replacements, missing nodes/inputs, invalid references and graph cycles fail explicitly. A binding without a default must be supplied during preparation. Numeric bounds are inclusive. Boolean values are not integers. References cannot be changed with a value override; author a new profile and matching inventory instead.

The author declares output nodes and input types. This module checks graph structure, not backend node-class catalogs, output-slot signatures, model compatibility or prompt semantics. An offline inventory is an explicit dependency snapshot, not proof that a backend currently has those resources. Local and Cloud adapters consume the same record shape. Changing `--backend` changes only the routing identifier: it never translates asset names, transfers files or grants access. Supply a profile/inventory whose exact names are valid on the intended backend.

Reconstruction policy records caller intent; it does not add controls to the external graph or guarantee geometry preservation. Originals are opened read-only and SHA-256 checked. A new job record is created exclusively: it cannot overwrite an original, workflow, existing job or symlink. This is application-level immutability, not a filesystem lock against another process changing files later.

## Preparation, lineage and exact replay

`prepare_job(profile, values=None, backend=None, parent_job=None)` returns a copied, resolved graph plus a version-1 job record. It never submits. Each record includes a unique job ID, UTC creation timestamp, parent job identifier, source paths/hashes, all resolved bindings, prompt UTF-8 bytes, numeric settings, output declarations, policies, profile snapshot/hash, original workflow bytes/hash, resolved workflow bytes/hash and the resource inventory used. Full graph bytes capture fixed settings beyond configurable bindings. `record_hash` covers the complete record except itself.

`workflow_bytes_base64` is the exact serialized UTF-8 API workflow document for a future backend adapter. An adapter must use those decoded bytes as the workflow document; backend-specific transport envelopes and server-assigned job/output metadata are not fabricated at preparation time. This change intentionally provides no executing adapter for profiles. A later explicitly authorized adapter must capture its request envelope, remote job ID, terminal status and outputs as execution evidence.

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

`jobs` without a subcommand still lists existing ComfyUI jobs. `jobs prepare`, `jobs replay` and profile-based `generate` are offline and print JSON; `--output` saves a new record only. The new `generate` command requires `--dry-run` and rejects its omission. Existing positional-prompt generation and existing top-level `replay` retain their prior behavior, including submission; they are separate from these profile preparation commands.

To load a user-authored external profile, provide the schema-v1 profile with its computed content hash, the exact API JSON and hash, the offline resource inventory, verified local asset files and any required binding values. No model-specific Python code is needed. Actual execution requires a separately authorized backend adapter; none is invoked by these commands.
