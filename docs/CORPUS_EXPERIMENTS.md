# Corpus and experiments

RenderLab keeps two different data contracts:

- Reference manifests describe unaligned identity, morphology, body/fabric, environment, and quality references.
- Paired-edit manifests describe tightly aligned source/target examples suitable for edit LoRA training.

Reference images are never promoted to training pairs merely because two images show the same person. A training pair must preserve identity, pose, camera, lighting, background, and anatomy while changing clothing state. One accepted pair may be represented in both directions with distinct `pair_id` values.

## Corpus v0

Validate the authoritative tagged manifest:

```bash
python -m renderlab corpus validate /path/to/renderlab_corpus_v0_tagged_manifest_v2.jsonl
```

Import original images from directories, individual images, or ZIP archives into a content-addressed asset directory. Existing SHA-256 values are skipped:

```bash
python -m renderlab corpus import /path/to/round2.zip \
  --manifest ./corpus/local/imported.jsonl \
  --asset-dir ./corpus/local/assets
```

Generate deterministic experiment subsets:

```bash
python -m renderlab corpus subset MANIFEST \
  renderlab/experiments/subsets/morphology_canon.json \
  output/subsets/morphology_canon.jsonl
```

The included subset specs keep morphology canon, body/fabric interaction, and face-swap identity references separate from aligned-pair training.

## Klein paired LoRA

Populate `corpus/pairs/klein_v0.jsonl` with reviewed aligned pairs, then prepare the run:

```bash
python -m renderlab experiment prepare \
  renderlab/experiments/klein_base_4b_paired_lora.json \
  --output-dir output/experiments/klein-v0
```

Compile training and holdout records before training:

```bash
python -m renderlab experiment dataset \
  renderlab/experiments/klein_base_4b_paired_lora.json \
  --output-dir output/datasets/klein-v0
```

The split is deterministic and performed by `identity_id` before direction and
caption expansion. With the default configuration, each accepted physical pair
produces eight forward and eight reverse records. No identity can occur in both
`train.jsonl` and `holdout.jsonl`. The generated `dataset.json` records input and
output hashes, the split seed, and the exact identity assignment.

Preparation validates the pair contract and writes `run.json` plus a machine-readable four-case `results.jsonl` matrix: Base and Distilled, each with and without the same LoRA. Empty or invalid pair manifests stop preparation; there is no fallback to the unaligned reference corpus.

Record each evaluation output and its numeric metrics. Completed cases require an
existing output file, whose SHA-256 is stored with the result:

```bash
python -m renderlab experiment record output/experiments/klein-v0 \
  klein-base-4b__baseline --status completed \
  --output output/eval/base-baseline.png \
  --metrics output/eval/base-baseline.metrics.json
```

Compare the Base and Distilled LoRA cases against their respective baselines:

```bash
python -m renderlab experiment compare output/experiments/klein-v0
```

The comparison is JSON. For every numeric metric it reports `LoRA - baseline`;
missing measurements remain `null` rather than being treated as zero.

Each paired record has this shape:

```json
{"pair_id":"pair_0001","identity_id":"subject_001","source":{"path":"assets/clothed.png","sha256":"..."},"target":{"path":"assets/unclothed.png","sha256":"..."},"source_state":"clothed","target_state":"unclothed","garment_description":"a fitted black dress","alignment_checks":{"identity":true,"pose":true,"camera":true,"lighting":true,"background":true,"anatomy":true},"review_status":"accepted"}
```
