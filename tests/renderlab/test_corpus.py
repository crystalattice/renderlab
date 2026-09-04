import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from renderlab.corpus import (
    CorpusError,
    generate_subset,
    import_images,
    prepare_experiment,
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


if __name__ == "__main__":
    unittest.main()
