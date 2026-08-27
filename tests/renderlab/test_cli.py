import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from renderlab import cli


class RenderLabCliTests(unittest.TestCase):
    def test_inject_parameters(self):
        workflow = cli.load_workflow(cli.DEFAULT_WORKFLOW)
        cli.inject_parameters(
            workflow, prompt="a fox", seed=123, width=768, height=1024, steps=7
        )
        self.assertEqual(workflow["4"]["inputs"]["text"], "a fox")
        self.assertEqual(workflow["6"]["inputs"]["width"], 768)
        self.assertEqual(workflow["6"]["inputs"]["height"], 1024)
        self.assertEqual(workflow["8"]["inputs"]["seed"], 123)
        self.assertEqual(workflow["8"]["inputs"]["steps"], 7)

    def test_find_saved_image(self):
        history = {
            "outputs": {
                "10": {
                    "images": [
                        {"filename": "RenderLab_00001_.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
        self.assertEqual(cli.find_saved_image(history)["filename"], "RenderLab_00001_.png")

    def test_single_render_writes_resolved_random_seed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            image_path = output_dir / "RenderLab_00001_.png"
            image_path.write_bytes(b"png")
            history = {
                "outputs": {
                    "10": {
                        "images": [
                            {"filename": image_path.name, "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
            with (
                patch.object(cli.secrets, "randbits", return_value=987654321),
                patch.object(cli, "submit", return_value="prompt-123") as submit,
                patch.object(cli, "wait_for_history", return_value=history),
            ):
                result = cli.main(["a test fox", "--output-dir", str(output_dir)])

            self.assertEqual(result, 0)
            submitted_workflow = submit.call_args.args[1]
            self.assertEqual(submitted_workflow["8"]["inputs"]["seed"], 987654321)
            metadata = json.loads(Path(str(image_path) + ".json").read_text())
            self.assertEqual(metadata["seed"], 987654321)
            self.assertEqual(metadata["prompt_id"], "prompt-123")
            self.assertEqual(metadata["batch_index"], 1)
            self.assertEqual(metadata["batch_count"], 1)

    def test_batch_resolves_independent_random_seeds(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            histories = []
            for index in range(1, 4):
                image_path = output_dir / f"RenderLab_{index:05d}_.png"
                image_path.write_bytes(b"png")
                histories.append(
                    {
                        "outputs": {
                            "10": {
                                "images": [
                                    {"filename": image_path.name, "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    }
                )
            with (
                patch.object(cli.secrets, "randbits", side_effect=[101, 202, 303]),
                patch.object(cli, "submit", side_effect=["prompt-1", "prompt-2", "prompt-3"]) as submit,
                patch.object(cli, "wait_for_history", side_effect=histories),
            ):
                result = cli.main(
                    ["three foxes", "--count", "3", "--output-dir", str(output_dir)]
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args[1]["8"]["inputs"]["seed"] for call in submit.call_args_list],
                [101, 202, 303],
            )
            metadata = json.loads(
                (output_dir / "RenderLab_00003_.png.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["batch_index"], 3)
            self.assertEqual(metadata["batch_count"], 3)

    def test_explicit_batch_seed_increments(self):
        self.assertEqual(cli.resolve_seeds(500, 3), [500, 501, 502])

    def test_explicit_batch_seed_rejects_overflow(self):
        with self.assertRaises(cli.RenderError):
            cli.resolve_seeds(cli.MAX_SEED, 2)

    def test_output_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(cli.RenderError):
                cli.local_output_path(
                    Path(temporary_dir),
                    {"filename": "escape.png", "subfolder": "../outside", "type": "output"},
                )


if __name__ == "__main__":
    unittest.main()
