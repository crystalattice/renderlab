import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from renderlab import cli
from renderlab.video import (
    H3_AUDIO_VAE,
    H3_CLIP,
    H3_UNET,
    H3_VIDEO_VAE,
    build_h3_t2v_workflow,
    h3_frame_count,
)


class RenderLabVideoTests(unittest.TestCase):
    def test_frame_count_snaps_to_h3_grid(self):
        self.assertEqual(h3_frame_count(5.0), 124)
        self.assertEqual(h3_frame_count(1.0) % 17, 5)

    def test_workflow_uses_proven_h3_contract(self):
        workflow = build_h3_t2v_workflow(
            prompt="fox at night", seed=1001, width=608, height=352,
            seconds=5.0, steps=20, filename_prefix="RenderLabVideo",
        )
        self.assertEqual(workflow["127"]["inputs"]["unet_name"], H3_UNET)
        self.assertEqual(workflow["128"]["inputs"]["clip_name"], H3_CLIP)
        self.assertEqual(workflow["119"]["inputs"]["vae_name"], H3_VIDEO_VAE)
        self.assertEqual(workflow["120"]["inputs"]["vae_name"], H3_AUDIO_VAE)
        self.assertEqual(workflow["129"]["inputs"]["noise_seed"], 1001)
        self.assertEqual(workflow["131"]["inputs"]["prompt"], "fox at night")
        self.assertEqual(workflow["131"]["inputs"]["length"], 124)
        self.assertEqual(workflow["124"]["inputs"]["steps"], 20)
        self.assertEqual(workflow["92"]["class_type"], "SaveVideo")

    def test_video_argument_defaults_and_validation(self):
        args = cli.parse_control_args(["video", "fox at night"])
        self.assertEqual((args.width, args.height), (608, 352))
        self.assertEqual(args.duration, 5.0)
        self.assertEqual(args.steps, 20)
        self.assertEqual(args.timeout, 7200.0)
        with self.assertRaises(SystemExit):
            cli.parse_control_args(["video"])
        with self.assertRaises(SystemExit):
            cli.parse_control_args(["video", "fox", "--width", "601"])

    def test_video_check_does_not_require_prompt(self):
        args = cli.parse_control_args(["video", "--check"])
        self.assertTrue(args.check)
        self.assertIsNone(args.prompt)

    def test_video_render_writes_mp4_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            video_path = output_dir / "RenderLabVideo_00001_.mp4"
            video_path.write_bytes(b"fake mp4")
            history = {
                "outputs": {
                    "92": {
                        "images": [
                            {"filename": video_path.name, "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
            with (
                patch.object(cli, "submit", return_value="video-prompt") as submit,
                patch.object(cli, "wait_for_history", return_value=history),
            ):
                result = cli.main([
                    "video", "fox at night", "--seed", "1001",
                    "--output-dir", str(output_dir),
                ])

            self.assertEqual(result, 0)
            workflow = submit.call_args.args[1]
            self.assertEqual(workflow["129"]["inputs"]["noise_seed"], 1001)
            metadata = json.loads(Path(str(video_path) + ".json").read_text())
            self.assertEqual(metadata["renderlab_version"], "0.8.0")
            self.assertEqual(metadata["mode"], "t2v")
            self.assertEqual(metadata["frame_count"], 124)
            self.assertEqual(metadata["actual_duration_seconds"], 124 / 24)
            self.assertTrue(metadata["native_audio"])
            self.assertEqual(metadata["output_sha256"], cli.sha256_file(video_path))


if __name__ == "__main__":
    unittest.main()
