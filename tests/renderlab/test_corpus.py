import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from renderlab.corpus import (
    CorpusError,
    build_training_dataset,
    compare_experiment_results,
    generate_subset,
    import_images,
    prepare_experiment,
    record_experiment_result,
    read_jsonl,
    validate_pair_manifest,
    validate_reference_manifest,
)


def reference_row(identifier: str, digest: str, reference_use: str = "morphology_canon") -> dict:
    return {
        "id": identifier, "archive": "set.zip", "path": f"set/{identifier}.png",
        "filename": f"{identifier}.png", "sha256": digest, "width": 32, "height": 64,
        "format": "PNG", "content_role": "identity_reference", "subject_group": "female",
        "reference_use": reference_use, "alignment_type": "unaligned_multi_view",
        "training_value": "reference", "review_status": "provenance_tagged",
    }


class CorpusTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_reference_validation_reports_counts_and_rejects_duplicate_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            self.write_jsonl(path, [reference_row("img_1", "a" * 64)])
            summary = validate_reference_manifest(path)
            self.assertEqual(summary["records"], 1)
            self.assertEqual(summary["reference_uses"], {"morphology_canon": 1})
            self.write_jsonl(path, [reference_row("img_1", "a" * 64), reference_row("img_2", "a" * 64)])
            with self.assertRaisesRegex(CorpusError, "duplicate sha256"):
                validate_reference_manifest(path)

    def test_subset_is_filtered_and_deterministically_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            self.write_jsonl(manifest, [
                reference_row("img_2", "b" * 64, "body_fabric_interaction"),
                reference_row("img_1", "a" * 64),
            ])
            spec = root / "spec.json"
            spec.write_text(json.dumps({"name": "morph", "filters": {"reference_use": "morphology_canon"}}), encoding="utf-8")
            output = root / "subset.jsonl"
            result = generate_subset(manifest, spec, output)
            self.assertEqual(result["records"], 1)
            self.assertEqual(read_jsonl(output)[0]["id"], "img_1")
            self.assertEqual(len(result["source_sha256"]), 64)

    def test_import_deduplicates_images_across_zip_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.png"
            Image.new("RGB", (8, 12), "red").save(image)
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.write(image, "one.png")
                output.write(image, "two.png")
            manifest = root / "manifest.jsonl"
            result = import_images([archive], manifest, root / "assets")
            self.assertEqual(result, {"schema": "renderlab.import-result.v1", "added": 1, "duplicates": 1, "records": 1})
            row = read_jsonl(manifest)[0]
            self.assertEqual((row["width"], row["height"]), (8, 12))
            self.assertEqual(len(list((root / "assets").rglob("*.png"))), 1)

    def test_pair_validation_rejects_alignment_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            self.write_jsonl(path, [{
                "pair_id": "pair_1", "identity_id": "person_1",
                "source": {"path": "on.png", "sha256": "a" * 64},
                "target": {"path": "off.png", "sha256": "b" * 64},
                "source_state": "clothed", "target_state": "unclothed",
                "alignment_checks": {"identity": True, "pose": False},
                "garment_description": "a black dress",
                "review_status": "accepted",
            }])
            with self.assertRaisesRegex(CorpusError, "missing or failed alignment checks: pose"):
                validate_pair_manifest(path)

    def test_pair_validation_rejects_empty_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.touch()
            with self.assertRaisesRegex(CorpusError, "contains no records"):
                validate_pair_manifest(path)

    def test_experiment_preparation_builds_four_case_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.jsonl"
            self.write_jsonl(pairs, [{
                "pair_id": "pair_1", "identity_id": "person_1",
                "source": {"path": "on.png", "sha256": "a" * 64},
                "target": {"path": "off.png", "sha256": "b" * 64},
                "source_state": "clothed", "target_state": "unclothed",
                "alignment_checks": {name: True for name in ("identity", "pose", "camera", "lighting", "background", "anatomy")},
                "garment_description": "a black dress",
                "review_status": "accepted",
            }])
            config = root / "experiment.json"
            config.write_text(json.dumps({
                "schema": "renderlab.experiment.v1", "name": "test",
                "training": {"pair_manifest": "pairs.jsonl"},
                "evaluation": {"models": [{"id": "base"}, {"id": "distilled"}]},
            }), encoding="utf-8")
            result = prepare_experiment(config, root / "run")
            self.assertEqual(len(result["evaluation_cases"]), 4)
            self.assertEqual(len(read_jsonl(root / "run" / "results.jsonl")), 4)

    def test_training_dataset_splits_by_identity_before_bidirectional_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.jsonl"
            rows = []
            for number, identity in enumerate(("person_a", "person_b"), 1):
                rows.append({
                    "pair_id": f"pair_{number}", "identity_id": identity,
                    "source": {"path": f"{identity}_on.png", "sha256": "a" * 64},
                    "target": {"path": f"{identity}_off.png", "sha256": "b" * 64},
                    "source_state": "clothed", "target_state": "unclothed",
                    "garment_description": "a fitted black dress",
                    "alignment_checks": {name: True for name in ("identity", "pose", "camera", "lighting", "background", "anatomy")},
                    "review_status": "accepted",
                })
            self.write_jsonl(pairs, rows)
            config = root / "experiment.json"
            config.write_text(json.dumps({
                "schema": "renderlab.experiment.v1", "name": "test",
                "training": {
                    "pair_manifest": "pairs.jsonl", "direction_policy": "bidirectional",
                    "caption_variants_per_pair": 2, "holdout_fraction": 0.2, "split_seed": 42,
                },
            }), encoding="utf-8")
            result = build_training_dataset(config, root / "dataset")
            train = read_jsonl(root / "dataset" / "train.jsonl")
            holdout = read_jsonl(root / "dataset" / "holdout.jsonl")
            self.assertEqual((result["train_records"], result["holdout_records"]), (4, 4))
            self.assertTrue({row["identity_id"] for row in train}.isdisjoint({row["identity_id"] for row in holdout}))
            self.assertEqual({row["direction"] for row in train + holdout}, {"forward", "reverse"})
            reverse = next(row for row in train + holdout if row["direction"] == "reverse")
            self.assertEqual(reverse["target_state"], "clothed")
            self.assertIn("fitted black dress", reverse["instruction"])

    def test_record_and_compare_experiment_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(json.dumps({"experiment": "test"}), encoding="utf-8")
            rows = [
                {"case_id": "base__baseline", "model": {"id": "base"}, "lora_enabled": False, "status": "pending", "output": None, "metrics": {"identity": None}},
                {"case_id": "base__lora", "model": {"id": "base"}, "lora_enabled": True, "status": "pending", "output": None, "metrics": {"identity": None}},
            ]
            self.write_jsonl(run_dir / "results.jsonl", rows)
            output = root / "result.png"
            output.write_bytes(b"result")
            baseline_metrics = root / "baseline.json"
            baseline_metrics.write_text(json.dumps({"identity": 0.7}), encoding="utf-8")
            lora_metrics = root / "lora.json"
            lora_metrics.write_text(json.dumps({"identity": 0.85}), encoding="utf-8")
            recorded = record_experiment_result(run_dir, "base__baseline", "completed", output, baseline_metrics)
            record_experiment_result(run_dir, "base__lora", "completed", output, lora_metrics)
            self.assertEqual(recorded["output"]["sha256"], "f6a214f7a5fcda0c2cee9660b7fc29f5649e3c68aad48e20e950137c98913a68")
            comparison = compare_experiment_results(run_dir)
            self.assertEqual(comparison["completed"], 2)
            self.assertEqual(comparison["comparisons"][0]["metric_delta_lora_minus_baseline"]["identity"], 0.15)

    def test_record_rejects_unknown_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({"experiment": "test"}), encoding="utf-8")
            self.write_jsonl(root / "results.jsonl", [{
                "case_id": "base__baseline", "model": {"id": "base"},
                "lora_enabled": False, "status": "pending", "output": None,
                "metrics": {"identity": None},
            }])
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps({"made_up": 1}), encoding="utf-8")
            with self.assertRaisesRegex(CorpusError, "unknown metrics: made_up"):
                record_experiment_result(root, "base__baseline", "failed", None, metrics)


if __name__ == "__main__":
    unittest.main()
