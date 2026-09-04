import json
import tempfile
import unittest
from pathlib import Path

from renderlab.render_run import RenderRunError, initialize_render_run, record_stage_result, validate_render_spec


def spec() -> dict:
    return {
        "schema": "renderlab.render-spec.v1",
        "run_id": "char-1__studio",
        "scene": {"prompt": "neutral studio"},
        "canon": {"character_id": "char-1"},
        "stages": [
            {"id": "scene", "type": "environment", "backend": {"id": "klein-base-4b"}, "owns": ["environment", "composition"], "preserves": [], "depends_on": []},
            {"id": "identity", "type": "identity", "backend": {"id": "face-swap"}, "owns": ["face_identity"], "preserves": ["environment", "body"], "depends_on": ["scene"]},
        ],
    }


class RenderRunTests(unittest.TestCase):
    def test_rejects_forward_dependency(self):
        value = spec()
        value["stages"][0]["depends_on"] = ["identity"]
        with self.assertRaisesRegex(RenderRunError, "earlier stages"):
            validate_render_spec(value)

    def test_initialize_and_record_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "input.json"
            spec_path.write_text(json.dumps(spec()), encoding="utf-8")
            run_dir = root / "run"
            state = initialize_render_run(spec_path, run_dir)
            self.assertEqual(state["status"], "pending")
            output = root / "scene.png"
            output.write_bytes(b"scene")
            result = root / "result.json"
            result.write_text(json.dumps({
                "schema": "renderlab.stage-result.v1", "stage_id": "scene",
                "status": "completed", "backend": {"id": "klein-base-4b"},
                "inputs": [], "output": {"path": str(output)},
                "validation": {"decision": "accept", "metrics": {"composition": 0.9}},
                "provenance": {"seed": 3407},
            }), encoding="utf-8")
            recorded = record_stage_result(run_dir, "scene", result)
            self.assertEqual(recorded["attempt"], 1)
            updated = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(updated["status"], "running")
            self.assertEqual(updated["stages"][0]["status"], "completed")
            self.assertEqual(len(updated["stages"][0]["attempts"]), 1)

    def test_retry_leaves_stage_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "input.json"
            spec_path.write_text(json.dumps(spec()), encoding="utf-8")
            run_dir = root / "run"
            initialize_render_run(spec_path, run_dir)
            result = root / "result.json"
            result.write_text(json.dumps({
                "schema": "renderlab.stage-result.v1", "stage_id": "scene",
                "status": "failed", "validation": {"decision": "retry"},
            }), encoding="utf-8")
            record_stage_result(run_dir, "scene", result)
            state = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(state["stages"][0]["status"], "pending")

    def test_blocks_stage_until_dependency_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "input.json"
            spec_path.write_text(json.dumps(spec()), encoding="utf-8")
            run_dir = root / "run"
            initialize_render_run(spec_path, run_dir)
            result = root / "result.json"
            result.write_text(json.dumps({
                "schema": "renderlab.stage-result.v1", "stage_id": "identity",
                "status": "failed", "validation": {"decision": "escalate"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(RenderRunError, "dependency scene is not complete"):
                record_stage_result(run_dir, "identity", result)


if __name__ == "__main__":
    unittest.main()
