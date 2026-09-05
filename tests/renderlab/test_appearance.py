import json
import tempfile
import unittest
from pathlib import Path

from renderlab.appearance import BACKEND_BLUEPRINTS, AppearanceError, list_presets, plan_appearance
from renderlab.cli import main


class AppearanceTests(unittest.TestCase):
    def write_request(self, root: Path, **overrides) -> Path:
        request = {
            "schema": "renderlab.appearance-request.v1",
            "preset": "bikini",
            "source": {"path": "input.png", "sha256": "a" * 64},
            **overrides,
        }
        path = root / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return path

    def test_built_in_presets_expose_click_and_go_operations(self):
        presets = {preset["id"]: preset for preset in list_presets()}
        self.assertEqual(presets["bikini"]["operation"], "outfit_change")
        self.assertEqual(presets["auto_unclothe"]["operation"], "unclothe")
        self.assertEqual(presets["face_swap"]["operation"], "face_swap")
        self.assertEqual(presets["extend_canvas"]["operation"], "outpaint")
        self.assertEqual(presets["repair_region"]["operation"], "inpaint")

    def test_available_backends_reference_bundled_blueprints(self):
        repository = Path(__file__).resolve().parents[2]
        for blueprint in BACKEND_BLUEPRINTS.values():
            self.assertTrue((repository / blueprint).is_file(), blueprint)

    def test_plan_merges_preset_and_request_target(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.write_request(
                Path(directory), target={"color": "black", "material": "matte"}
            )
            plan = plan_appearance(request)
            self.assertEqual(plan["schema"], "renderlab.render-plan.v1")
            self.assertEqual(plan["intent"]["target"]["clothing_state"], "bikini")
            self.assertEqual(plan["intent"]["target"]["color"], "black")
            self.assertEqual(plan["stages"][0]["backend"]["id"], "qwen-image-edit")
            self.assertEqual(
                plan["stages"][0]["backend"]["workflow_blueprint"],
                "blueprints/Image Edit (Qwen 2509).json",
            )
            self.assertTrue(plan["stages"][0]["backend"]["available"])
            self.assertFalse(plan["stages"][1]["backend"]["available"])
            self.assertEqual(plan["stages"][1]["runs_when"], "identity_preservation < acceptance.identity_preservation")
            self.assertEqual(len(plan["request_sha256"]), 64)

    def test_outpaint_defaults_are_executable_not_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.write_request(Path(directory), preset="extend_canvas")
            stage = plan_appearance(request)["stages"][0]
            self.assertEqual(stage["backend"]["id"], "qwen-image-outpaint")
            self.assertEqual(stage["controls"], {
                "left": 0, "top": 0, "right": 0, "bottom": 512,
            })

    def test_controls_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.write_request(Path(directory), controls=[])
            with self.assertRaisesRegex(AppearanceError, "controls must be an object"):
                plan_appearance(request)

    def test_semantic_evidence_is_preserved_for_future_manga_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.write_request(Path(directory), source_semantics="semantic_evidence")
            self.assertEqual(plan_appearance(request)["source_semantics"], "semantic_evidence")

    def test_unknown_preset_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.write_request(Path(directory), preset="missing")
            with self.assertRaisesRegex(AppearanceError, "unknown appearance preset"):
                plan_appearance(request)

    def test_acceptance_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.write_request(Path(directory), acceptance=[])
            with self.assertRaisesRegex(AppearanceError, "acceptance must be an object"):
                plan_appearance(request)

    def test_cli_writes_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.write_request(root)
            output = root / "plan.json"
            self.assertEqual(main(["appearance", "plan", str(request), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["intent"]["preset"], "bikini")


if __name__ == "__main__":
    unittest.main()
