import json
import tempfile
import unittest
from pathlib import Path

from renderlab.appearance import AppearanceError, list_presets, plan_appearance
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
            self.assertEqual(plan["stages"][1]["runs_when"], "identity_preservation < acceptance.identity_preservation")
            self.assertEqual(len(plan["request_sha256"]), 64)

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
