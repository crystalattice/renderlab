from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_REFERENCE_FIELDS = {
    "id", "archive", "path", "filename", "sha256", "width", "height", "format",
    "content_role", "subject_group", "reference_use", "alignment_type", "training_value",
    "review_status",
}
PAIR_STATES = {"clothed", "unclothed"}
INSTRUCTION_TEMPLATES = {
    "unclothed": (
        "Remove the clothing while preserving the person and scene.",
        "Change the subject from clothed to unclothed without changing identity or pose.",
        "Remove the subject's clothing; keep anatomy, camera, lighting, and background fixed.",
        "Make the subject unclothed while preserving all non-clothing details.",
        "Replace the clothed state with an unclothed state and keep the composition unchanged.",
        "Edit only the clothing state to unclothed.",
        "Render the same person in the same scene without clothing.",
        "Preserve identity and body morphology while removing the clothing.",
    ),
    "clothed": (
        "Dress the subject in {garment} while preserving the person and scene.",
        "Change the subject from unclothed to wearing {garment} without changing identity or pose.",
        "Add {garment}; keep anatomy, camera, lighting, and background fixed.",
        "Make the subject clothed in {garment} while preserving all other details.",
        "Replace the unclothed state with {garment} and keep the composition unchanged.",
        "Edit only the clothing state by adding {garment}.",
        "Render the same person in the same scene wearing {garment}.",
        "Preserve identity and body morphology while dressing the subject in {garment}.",
    ),
}


class CorpusError(RuntimeError):
    pass


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise CorpusError(f"{path}:{line_number}: record must be a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def validate_reference_manifest(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    errors = []
    ids = Counter(row.get("id") for row in rows)
    hashes = Counter(row.get("sha256") for row in rows)
    for index, row in enumerate(rows, 1):
        missing = sorted(REQUIRED_REFERENCE_FIELDS - row.keys())
        if missing:
            errors.append(f"record {index}: missing {', '.join(missing)}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append(f"record {index}: sha256 must be 64 lowercase hex characters")
        if not isinstance(row.get("width"), int) or row.get("width", 0) <= 0:
            errors.append(f"record {index}: width must be a positive integer")
        if not isinstance(row.get("height"), int) or row.get("height", 0) <= 0:
            errors.append(f"record {index}: height must be a positive integer")
    duplicate_ids = sorted(value for value, count in ids.items() if count > 1)
    duplicate_hashes = sorted(value for value, count in hashes.items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate ids: {', '.join(duplicate_ids[:10])}")
    if duplicate_hashes:
        errors.append(f"duplicate sha256 values: {', '.join(duplicate_hashes[:10])}")
    if errors:
        raise CorpusError("manifest validation failed:\n" + "\n".join(errors))
    return {
        "schema": "renderlab.reference-manifest.v1",
        "records": len(rows),
        "unique_sha256": len(hashes),
        "content_roles": dict(sorted(Counter(row["content_role"] for row in rows).items())),
        "reference_uses": dict(sorted(Counter(row["reference_use"] for row in rows).items())),
        "subject_groups": dict(sorted(Counter(row["subject_group"] for row in rows).items())),
    }


def validate_pair_manifest(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise CorpusError("pair manifest contains no records")
    errors = []
    pair_ids = Counter(row.get("pair_id") for row in rows)
    for index, row in enumerate(rows, 1):
        missing = {"pair_id", "identity_id", "source", "target", "source_state", "target_state", "garment_description"} - row.keys()
        if missing:
            errors.append(f"record {index}: missing {', '.join(sorted(missing))}")
            continue
        if row["source_state"] not in PAIR_STATES or row["target_state"] not in PAIR_STATES:
            errors.append(f"record {index}: source_state and target_state must be clothed or unclothed")
        if row["source_state"] == row["target_state"]:
            errors.append(f"record {index}: source_state and target_state must differ")
        for field in ("source", "target"):
            value = row[field]
            if not isinstance(value, dict) or not isinstance(value.get("sha256"), str) or not value.get("path"):
                errors.append(f"record {index}: {field} must contain path and sha256")
            elif len(value["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in value["sha256"]):
                errors.append(f"record {index}: {field} sha256 must be 64 lowercase hex characters")
        checks = row.get("alignment_checks", {})
        failed = [name for name in ("identity", "pose", "camera", "lighting", "background", "anatomy") if checks.get(name) is not True]
        if failed:
            errors.append(f"record {index}: missing or failed alignment checks: {', '.join(failed)}")
        if row.get("review_status") != "accepted":
            errors.append(f"record {index}: review_status must be accepted")
        if not isinstance(row.get("garment_description"), str) or not row.get("garment_description", "").strip():
            errors.append(f"record {index}: garment_description must be non-empty")
    duplicates = sorted(value for value, count in pair_ids.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate pair_ids: {', '.join(duplicates[:10])}")
    if errors:
        raise CorpusError("pair manifest validation failed:\n" + "\n".join(errors))
    return {"schema": "renderlab.paired-edit-manifest.v1", "records": len(rows)}


def _matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, accepted in filters.items():
        values = accepted if isinstance(accepted, list) else [accepted]
        if row.get(field) not in values:
            return False
    return True


def generate_subset(manifest: Path, spec: Path, output: Path) -> dict[str, Any]:
    validate_reference_manifest(manifest)
    settings = json.loads(spec.read_text(encoding="utf-8"))
    filters = settings.get("filters", {})
    exclude = settings.get("exclude", {})
    rows = [
        row for row in read_jsonl(manifest)
        if _matches(row, filters) and not (exclude and _matches(row, exclude))
    ]
    sort_fields = settings.get("sort", ["archive", "path"])
    rows.sort(key=lambda row: tuple(str(row.get(field, "")) for field in sort_fields))
    limit = settings.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise CorpusError("subset limit must be a positive integer")
        rows = rows[:limit]
    write_jsonl(output, rows)
    return {
        "schema": "renderlab.subset-result.v1", "name": settings.get("name", spec.stem),
        "source_manifest": str(manifest.resolve()), "source_sha256": file_sha256(manifest),
        "spec": str(spec.resolve()), "spec_sha256": file_sha256(spec),
        "output": str(output.resolve()), "records": len(rows),
    }


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return sha256_stream(source)


def _image_info(stream) -> tuple[int, int, str]:
    try:
        with Image.open(stream) as image:
            return image.width, image.height, image.format or "UNKNOWN"
    except UnidentifiedImageError as exc:
        raise CorpusError("input has an image suffix but Pillow cannot identify it") from exc


def import_images(sources: list[Path], manifest: Path, asset_dir: Path) -> dict[str, Any]:
    existing = read_jsonl(manifest) if manifest.exists() else []
    hashes = {row["sha256"] for row in existing}
    next_id = max((int(row["id"].split("_")[-1]) for row in existing), default=0) + 1
    added = []
    duplicates = 0

    def accept(display_path: str, archive: str, suffix: str, open_stream) -> None:
        nonlocal next_id, duplicates
        with open_stream() as stream:
            digest = sha256_stream(stream)
        if digest in hashes:
            duplicates += 1
            return
        with open_stream() as stream:
            width, height, image_format = _image_info(stream)
        target = asset_dir / digest[:2] / f"{digest}{suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open_stream() as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        added.append({
            "id": f"img_{next_id:05d}", "archive": archive, "path": display_path,
            "folder": str(Path(display_path).parent), "subject_label": "unreviewed",
            "filename": Path(display_path).name, "sha256": digest, "width": width,
            "height": height, "orientation": "square" if width == height else "landscape" if width > height else "portrait",
            "format": image_format, "content_role": "unreviewed", "subject_group": "unreviewed",
            "reference_use": "unreviewed", "alignment_type": "unreviewed",
            "collection_axis": "unreviewed", "visual_state": "unreviewed", "viewpoint": "unreviewed",
            "framing": "unreviewed", "morphology_bucket": "unreviewed", "training_value": "unreviewed",
            "review_status": "imported", "notes": "",
        })
        hashes.add(digest)
        next_id += 1

    for source in sources:
        if source.is_dir():
            for path in sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
                accept(str(path.relative_to(source)), "", path.suffix, lambda path=path: path.open("rb"))
        elif source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive_file:
                for member in sorted(archive_file.infolist(), key=lambda item: item.filename):
                    suffix = Path(member.filename).suffix.lower()
                    if not member.is_dir() and suffix in IMAGE_SUFFIXES:
                        accept(member.filename, source.name, suffix, lambda member=member: archive_file.open(member))
        elif source.suffix.lower() in IMAGE_SUFFIXES:
            accept(source.name, "", source.suffix, lambda source=source: source.open("rb"))
        else:
            raise CorpusError(f"unsupported import source: {source}")
    write_jsonl(manifest, [*existing, *added])
    return {"schema": "renderlab.import-result.v1", "added": len(added), "duplicates": duplicates, "records": len(existing) + len(added)}


def prepare_experiment(config: Path, output_dir: Path) -> dict[str, Any]:
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings.get("schema") != "renderlab.experiment.v1":
        raise CorpusError("experiment config schema must be renderlab.experiment.v1")
    pair_manifest = (config.parent / settings["training"]["pair_manifest"]).resolve()
    pair_summary = validate_pair_manifest(pair_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = []
    metric_names = settings["evaluation"].get("metrics", [])
    for model in settings["evaluation"]["models"]:
        for lora in (False, True):
            matrix.append({
                "case_id": f"{model['id']}__{'lora' if lora else 'baseline'}",
                "model": model, "lora_enabled": lora,
                "pair_manifest": str(pair_manifest), "pair_manifest_sha256": file_sha256(pair_manifest),
                "status": "pending", "output": None,
                "metrics": {name: None for name in metric_names},
            })
    resolved = {
        "schema": "renderlab.experiment-run.v1", "experiment": settings["name"],
        "config": str(config.resolve()), "config_sha256": file_sha256(config),
        "training": settings["training"], "pair_summary": pair_summary, "evaluation_cases": matrix,
    }
    (output_dir / "run.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "results.jsonl", matrix)
    return resolved


def _identity_split(identity_id: str, seed: int, holdout_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{identity_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / (1 << 64)
    return "holdout" if value < holdout_fraction else "train"


def build_training_dataset(config: Path, output_dir: Path) -> dict[str, Any]:
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings.get("schema") != "renderlab.experiment.v1":
        raise CorpusError("experiment config schema must be renderlab.experiment.v1")
    training = settings["training"]
    pair_manifest = (config.parent / training["pair_manifest"]).resolve()
    summary = validate_pair_manifest(pair_manifest)
    rows = read_jsonl(pair_manifest)
    minimum = training.get("minimum_pair_count", 1)
    maximum = training.get("maximum_pair_count")
    if len(rows) < minimum:
        raise CorpusError(f"pair manifest has {len(rows)} records; minimum is {minimum}")
    if maximum is not None and len(rows) > maximum:
        raise CorpusError(f"pair manifest has {len(rows)} records; maximum is {maximum}")
    holdout_fraction = training.get("holdout_fraction", 0.2)
    split_seed = training.get("split_seed", 0)
    if not 0.0 <= holdout_fraction < 1.0:
        raise CorpusError("holdout_fraction must be at least 0.0 and less than 1.0")
    identities = sorted({row["identity_id"] for row in rows})
    identity_splits = {
        identity: _identity_split(identity, split_seed, holdout_fraction)
        for identity in identities
    }
    if len(identities) > 1 and holdout_fraction > 0 and "holdout" not in identity_splits.values():
        identity_splits[identities[-1]] = "holdout"
    if len(identities) > 1 and "train" not in identity_splits.values():
        identity_splits[identities[0]] = "train"
    compiled = {"train": [], "holdout": []}
    caption_count = training.get("caption_variants_per_pair", 8)
    if not 1 <= caption_count <= len(INSTRUCTION_TEMPLATES["clothed"]):
        raise CorpusError(f"caption_variants_per_pair must be between 1 and {len(INSTRUCTION_TEMPLATES['clothed'])}")
    direction_policy = training.get("direction_policy", "forward")
    if direction_policy not in {"forward", "bidirectional"}:
        raise CorpusError("direction_policy must be forward or bidirectional")
    for pair in sorted(rows, key=lambda row: row["pair_id"]):
        split = identity_splits[pair["identity_id"]]
        directions = [(pair["source"], pair["target"], pair["source_state"], pair["target_state"], "forward")]
        if direction_policy == "bidirectional":
            directions.append((pair["target"], pair["source"], pair["target_state"], pair["source_state"], "reverse"))
        for source, target, source_state, target_state, direction in directions:
            for caption_index, template in enumerate(INSTRUCTION_TEMPLATES[target_state][:caption_count], 1):
                compiled[split].append({
                    "sample_id": f"{pair['pair_id']}__{direction}__caption_{caption_index:02d}",
                    "pair_id": pair["pair_id"], "identity_id": pair["identity_id"],
                    "direction": direction, "source": source, "target": target,
                    "source_state": source_state,
                    "target_state": target_state,
                    "instruction": template.format(garment=pair["garment_description"].strip()),
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    holdout_path = output_dir / "holdout.jsonl"
    write_jsonl(train_path, compiled["train"])
    write_jsonl(holdout_path, compiled["holdout"])
    result = {
        "schema": "renderlab.training-dataset.v1", "experiment": settings["name"],
        "pair_manifest": str(pair_manifest), "pair_manifest_sha256": file_sha256(pair_manifest),
        "pair_records": summary["records"], "identity_splits": identity_splits,
        "split_seed": split_seed, "holdout_fraction": holdout_fraction,
        "train_records": len(compiled["train"]), "holdout_records": len(compiled["holdout"]),
        "train_sha256": file_sha256(train_path), "holdout_sha256": file_sha256(holdout_path),
    }
    (output_dir / "dataset.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def record_experiment_result(
    run_dir: Path,
    case_id: str,
    status: str,
    output: Path | None,
    metrics_path: Path | None,
) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    results_path = run_dir / "results.jsonl"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rows = read_jsonl(results_path)
    matches = [row for row in rows if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise CorpusError(f"experiment case not found: {case_id}")
    row = matches[0]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path else {}
    if not isinstance(metrics, dict):
        raise CorpusError("metrics file must contain a JSON object")
    expected_metrics = set(row.get("metrics", {}))
    unknown = sorted(set(metrics) - expected_metrics)
    if unknown:
        raise CorpusError(f"unknown metrics: {', '.join(unknown)}")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise CorpusError(f"metric {name} must be a finite number")
    if status == "completed" and (output is None or not output.is_file()):
        raise CorpusError("completed results require an existing --output file")
    row["status"] = status
    row["metrics"].update(metrics)
    row["output"] = (
        {"path": str(output.resolve()), "sha256": file_sha256(output)}
        if output is not None and output.is_file() else None
    )
    row["experiment"] = run["experiment"]
    write_jsonl(results_path, rows)
    return row


def compare_experiment_results(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "results.jsonl")
    models: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_id = row["model"]["id"]
        variant = "lora" if row["lora_enabled"] else "baseline"
        models.setdefault(model_id, {})[variant] = row
    comparisons = []
    for model_id, variants in sorted(models.items()):
        baseline = variants.get("baseline")
        lora = variants.get("lora")
        deltas = {}
        if baseline and lora:
            for name in sorted(set(baseline.get("metrics", {})) | set(lora.get("metrics", {}))):
                before = baseline.get("metrics", {}).get(name)
                after = lora.get("metrics", {}).get(name)
                deltas[name] = round(after - before, 8) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
        comparisons.append({
            "model": model_id,
            "baseline_status": baseline.get("status") if baseline else "missing",
            "lora_status": lora.get("status") if lora else "missing",
            "metric_delta_lora_minus_baseline": deltas,
        })
    return {
        "schema": "renderlab.experiment-comparison.v1",
        "run_dir": str(run_dir.resolve()),
        "cases": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "comparisons": comparisons,
    }
