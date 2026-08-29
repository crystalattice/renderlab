import json
import io
import tempfile
import unittest
import struct
import zlib
from pathlib import Path
from unittest.mock import patch

from renderlab import cli


class RenderLabCliTests(unittest.TestCase):
    @staticmethod
    def write_binary_png(path: Path, rows: list[list[int]]) -> None:
        height = len(rows)
        width = len(rows[0])
        raw = b"".join(
            b"\x00" + b"".join(bytes((value, value, value)) for value in row)
            for row in rows
        )

        def chunk(kind: bytes, value: bytes) -> bytes:
            return (
                struct.pack(">I", len(value)) + kind + value
                + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
            )

        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )

    def test_version_prints_renderlab_version(self):
        with patch("sys.stdout") as stdout:
            with self.assertRaises(SystemExit) as raised:
                cli.parse_args(["--version"])

        self.assertEqual(raised.exception.code, 0)
        stdout.write.assert_called_once_with("renderlab 0.8.0\n")

    def test_replay_txt2img_restores_effective_render_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "image.png.json"
            output_path = Path(directory) / "image.png"
            self.write_binary_png(output_path, [[0]])
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "txt2img",
                        "profile": "realvisxl_v5_fp16",
                        "effective_prompt": "a red bowl",
                        "negative_prompt": "bananas, yellow fruit",
                        "seed": 271828,
                        "steps": 30,
                        "cfg": 7,
                        "width": 1024,
                        "height": 1024,
                        "output": str(output_path),
                        "lora": {
                            "name": "style.safetensors",
                            "model_strength": 0.75,
                            "clip_strength": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            control = cli.parse_control_args(
                ["replay", str(metadata_path), "--server", "http://comfy:8188"]
            )
            replay = cli.replay_arguments(control)

        parsed = cli.parse_args(replay)
        self.assertEqual(parsed.prompt, "a red bowl")
        self.assertEqual(parsed.profile, "realvisxl")
        self.assertEqual(parsed.seed, 271828)
        self.assertEqual(parsed.negative_prompt, "bananas, yellow fruit")
        self.assertEqual((parsed.width, parsed.height), (1024, 1024))
        self.assertEqual(parsed.lora, "style.safetensors")
        self.assertEqual(parsed.lora_model_strength, 0.75)
        self.assertEqual(parsed.lora_clip_strength, 0.5)
        self.assertEqual(parsed.server, "http://comfy:8188")
        self.assertRegex(parsed.filename_prefix, r"^RenderLabReplay_[0-9a-f]{12}$")
        self.assertEqual(parsed.replay_kind, "exact")
        self.assertEqual(parsed.parent_seed, 271828)

    def test_replay_new_seed_changes_only_seed_and_disables_pixel_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "image.png"
            self.write_binary_png(output_path, [[0]])
            metadata_path = root / "image.png.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "mode": "txt2img", "profile": "realvisxl_v5_fp16",
                        "effective_prompt": "rainy portrait", "seed": 100,
                        "steps": 30, "cfg": 7, "width": 1024, "height": 1024,
                        "output": str(output_path), "negative_prompt": "watermark",
                    }
                ),
                encoding="utf-8",
            )
            control = cli.parse_control_args([
                "replay", str(metadata_path), "--new-seed"
            ])
            with patch.object(cli.secrets, "randbits", return_value=200):
                replay = cli.replay_arguments(control)
            parsed = cli.parse_args(replay)

        self.assertEqual(parsed.seed, 200)
        self.assertEqual(parsed.parent_seed, 100)
        self.assertEqual(parsed.replay_kind, "new-seed")
        self.assertIsNone(parsed.expected_pixel_sha256)
        self.assertRegex(parsed.filename_prefix, r"^RenderLabVariant_[0-9a-f]{12}$")

    def test_filename_prefix_changes_only_save_nodes(self):
        workflow = cli.load_workflow(cli.DEFAULT_WORKFLOW)
        original_sampler = json.loads(json.dumps(workflow["8"]))
        cli.inject_filename_prefix(workflow, "RenderLabReplay_deadbeef1234")
        self.assertEqual(workflow["10"]["inputs"]["filename_prefix"], "RenderLabReplay_deadbeef1234")
        self.assertEqual(workflow["8"], original_sampler)

    def test_replay_inpaint_restores_verified_assets_and_mask_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            source.write_bytes(b"source")
            mask.write_bytes(b"mask")
            output_path = root / "edit.png"
            self.write_binary_png(output_path, [[0]])
            metadata_path = root / "edit.png.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "mode": "inpaint",
                        "profile": "realvisxl_v5_fp16",
                        "effective_prompt": "black lace",
                        "negative_prompt": "",
                        "seed": 42,
                        "steps": 30,
                        "cfg": 7,
                        "denoise": 0.75,
                        "output": str(output_path),
                        "source_image": {
                            "path": str(source), "sha256": cli.sha256_file(source)
                        },
                        "mask_image": {
                            "path": str(mask), "sha256": cli.sha256_file(mask),
                            "grow_pixels": 0, "feather_pixels": 1,
                        },
                        "lora": None,
                    }
                ),
                encoding="utf-8",
            )
            replay = cli.replay_arguments(cli.parse_control_args(["replay", str(metadata_path)]))
            parsed = cli.parse_args(replay)

        self.assertEqual(parsed.input_image, source.resolve())
        self.assertEqual(parsed.mask_image, mask.resolve())
        self.assertEqual(parsed.denoise, 0.75)
        self.assertEqual(parsed.mask_grow, 0)
        self.assertEqual(parsed.mask_feather, 1)

    def test_replay_rejects_changed_source_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            source.write_bytes(b"changed")
            output_path = root / "edit.png"
            self.write_binary_png(output_path, [[0]])
            metadata_path = root / "edit.png.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "mode": "img2img", "profile": "realvisxl_v5_fp16",
                        "effective_prompt": "edit", "seed": 1, "steps": 30, "cfg": 7,
                        "denoise": 0.5,
                        "output": str(output_path),
                        "source_image": {"path": str(source), "sha256": "0" * 64},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(cli.RenderError, "source image hash changed"):
                cli.replay_arguments(cli.parse_control_args(["replay", str(metadata_path)]))

    def test_replay_rejects_mask_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "mask.png.json"
            metadata_path.write_text(json.dumps({"mode": "mask"}), encoding="utf-8")
            with self.assertRaisesRegex(cli.RenderError, "not RenderLab render provenance"):
                cli.replay_arguments(cli.parse_control_args(["replay", str(metadata_path)]))

    def test_profile_cfg_defaults_and_validation(self):
        self.assertEqual(cli.parse_args(["fox"]).cfg, 1)
        self.assertEqual(cli.parse_args(["fox", "--profile", "realvisxl"]).cfg, 7)
        self.assertEqual(cli.parse_args(["fox", "--cfg", "2.5"]).cfg, 2.5)
        with self.assertRaises(SystemExit):
            cli.parse_args(["fox", "--cfg", "101"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["fox", "--negative-prompt", "dogs"])

    def test_z_image_negative_prompt_replaces_zero_conditioning(self):
        workflow = cli.load_workflow(cli.DEFAULT_WORKFLOW)
        effective = cli.inject_generation_controls(
            workflow, negative_prompt="dogs, watermark", cfg=2.0
        )
        self.assertEqual(effective, "dogs, watermark")
        self.assertEqual(workflow["5"]["class_type"], "CLIPTextEncode")
        self.assertEqual(workflow["5"]["inputs"]["text"], "dogs, watermark")
        self.assertEqual(workflow["5"]["inputs"]["clip"], ["2", 0])
        self.assertEqual(workflow["8"]["inputs"]["cfg"], 2.0)
        lora_id = cli.inject_lora(
            workflow,
            name="style.safetensors",
            model_strength=0.8,
            clip_strength=0.8,
        )
        self.assertEqual(workflow["4"]["inputs"]["clip"], [lora_id, 1])
        self.assertEqual(workflow["5"]["inputs"]["clip"], [lora_id, 1])

    def test_realvis_negative_prompt_can_be_replaced_or_cleared(self):
        workflow = cli.load_workflow(cli.REALVISXL_WORKFLOW)
        effective = cli.inject_generation_controls(
            workflow, negative_prompt="watermark", cfg=6.5
        )
        self.assertEqual(effective, "watermark")
        self.assertEqual(workflow["5"]["inputs"]["text"], "watermark")
        self.assertEqual(workflow["8"]["inputs"]["cfg"], 6.5)
        self.assertEqual(
            cli.inject_generation_controls(workflow, negative_prompt="", cfg=7), ""
        )

    def test_lora_arguments_require_lora_and_validate_strengths(self):
        args = cli.parse_args(
            ["a fox", "--lora", "fox.safetensors", "--lora-model-strength", "0.7"]
        )
        self.assertEqual(args.lora, "fox.safetensors")
        self.assertEqual(args.lora_model_strength, 0.7)
        self.assertEqual(args.lora_clip_strength, 1.0)

        with self.assertRaises(SystemExit):
            cli.parse_args(["a fox", "--lora-model-strength", "0.7"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["a fox", "--lora", "fox.safetensors", "--lora-clip-strength", "11"])

    def test_lora_preset_resolves_filename_and_tested_strengths(self):
        args = cli.parse_args(["portrait", "--profile", "realvisxl", "--lora-preset", "realistic-eyes"])
        self.assertEqual(args.lora, "Realistic_eyes.safetensors")
        self.assertEqual(args.lora_model_strength, 0.4)
        self.assertEqual(args.lora_clip_strength, 0.4)

        overridden = cli.parse_args([
            "portrait", "--profile", "realvisxl", "--lora-preset", "natural-body",
            "--lora-model-strength", "0.2"
        ])
        self.assertEqual(overridden.lora_model_strength, 0.2)
        self.assertEqual(overridden.lora_clip_strength, 0.0)

        with self.assertRaises(SystemExit):
            cli.parse_args(["portrait", "--lora", "x.safetensors", "--lora-preset", "samane"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["portrait", "--lora-preset", "samane"])

    def test_lora_presets_stack_in_order_and_replay(self):
        args = cli.parse_args([
            "portrait", "--profile", "realvisxl",
            "--lora-preset", "angelica", "--lora-preset", "realistic-eyes",
        ])
        self.assertEqual(
            [lora["name"] for lora in args.loras],
            ["SDXL_Angelica.safetensors", "Realistic_eyes.safetensors"],
        )
        workflow = cli.load_workflow(cli.REALVISXL_WORKFLOW)
        node_ids = [
            cli.inject_lora(
                workflow,
                name=lora["name"],
                model_strength=lora["model_strength"],
                clip_strength=lora["clip_strength"],
            )
            for lora in args.loras
        ]
        self.assertEqual(workflow[node_ids[1]]["inputs"]["model"], [node_ids[0], 0])
        self.assertEqual(workflow[node_ids[1]]["inputs"]["clip"], [node_ids[0], 1])
        self.assertEqual(workflow["8"]["inputs"]["model"], [node_ids[1], 0])
        self.assertEqual(workflow["4"]["inputs"]["clip"], [node_ids[1], 1])

    def test_lora_presets_command_lists_tested_settings(self):
        args = cli.parse_control_args(["lora-presets"])
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = cli.run_control_command(args)
        self.assertEqual(result, 0)
        self.assertIn("vaporwave\tRetro_80s_Vaporwave.safetensors\tmodel=0.75\tclip=0.75", stdout.getvalue())

    def test_lora_sweep_builds_baseline_and_fixed_seed_strength_runs(self):
        args = cli.parse_control_args([
            "lora-sweep", "a portrait", "--lora-preset", "realistic-eyes",
            "--strengths", "0.2,0.4", "--seed", "123",
        ])
        runs = cli.lora_sweep_arguments(args)
        self.assertEqual(len(runs), 3)
        self.assertNotIn("--lora", runs[0])
        self.assertIn("LoRASweep_Base", runs[0])
        for run in runs:
            self.assertEqual(run[run.index("--seed") + 1], "123")
        self.assertEqual(runs[1][runs[1].index("--lora") + 1], "Realistic_eyes.safetensors")
        self.assertEqual(runs[1][runs[1].index("--lora-model-strength") + 1], "0.2")
        self.assertEqual(runs[2][runs[2].index("--lora-clip-strength") + 1], "0.4")

    def test_lora_sweep_builds_img2img_denoise_strength_matrix(self):
        args = cli.parse_control_args([
            "lora-sweep", "change the outfit", "--lora", "nsfw.safetensors",
            "--input-image", "source.png", "--denoises", "0.3,0.5",
            "--strengths", "0.25,0.75", "--seed", "456",
        ])
        runs = cli.lora_sweep_arguments(args)
        self.assertEqual(len(runs), 6)
        self.assertEqual(
            [run[run.index("--denoise") + 1] for run in runs],
            ["0.3", "0.3", "0.3", "0.5", "0.5", "0.5"],
        )
        for run in runs:
            self.assertEqual(run[run.index("--input-image") + 1], "source.png")
            self.assertEqual(run[run.index("--seed") + 1], "456")
        self.assertIn("LoRAI2I_D0_3_Base", runs[0])
        self.assertIn("LoRAI2I_D0_5_L0_75", runs[-1])

        defaults = cli.parse_control_args([
            "lora-sweep", "change the outfit", "--lora", "nsfw.safetensors",
            "--input-image", "source.png",
        ])
        self.assertEqual(defaults.denoises, [0.25, 0.45, 0.65])

    def test_lora_injection_routes_z_image_model_and_clip(self):
        workflow = cli.load_workflow(cli.DEFAULT_WORKFLOW)
        node_id = cli.inject_lora(
            workflow,
            name="style.safetensors",
            model_strength=0.75,
            clip_strength=0.5,
        )
        self.assertEqual(workflow[node_id]["class_type"], "LoraLoader")
        self.assertEqual(workflow[node_id]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow[node_id]["inputs"]["clip"], ["2", 0])
        self.assertEqual(workflow["7"]["inputs"]["model"], [node_id, 0])
        self.assertEqual(workflow["4"]["inputs"]["clip"], [node_id, 1])

    def test_lora_injection_routes_realvis_model_and_clip(self):
        workflow = cli.load_workflow(cli.REALVISXL_WORKFLOW)
        node_id = cli.inject_lora(
            workflow,
            name="style-xl.safetensors",
            model_strength=0.8,
            clip_strength=0.6,
        )
        self.assertEqual(workflow[node_id]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow[node_id]["inputs"]["clip"], ["1", 1])
        self.assertEqual(workflow["8"]["inputs"]["model"], [node_id, 0])
        self.assertEqual(workflow["4"]["inputs"]["clip"], [node_id, 1])
        self.assertEqual(workflow["5"]["inputs"]["clip"], [node_id, 1])

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

    def test_realvisxl_profile_defaults_and_parameter_injection(self):
        args = cli.parse_args(["a studio portrait", "--profile", "realvisxl"])
        self.assertEqual(args.workflow, cli.REALVISXL_WORKFLOW)
        self.assertEqual(args.steps, 30)

        workflow = cli.load_workflow(args.workflow)
        cli.inject_parameters(
            workflow, prompt="a studio portrait", seed=321, width=832, height=1216, steps=24
        )
        self.assertEqual(workflow["1"]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(
            workflow["1"]["inputs"]["ckpt_name"], "RealVisXL_V5.0_fp16.safetensors"
        )
        self.assertEqual(workflow["4"]["inputs"]["text"], "a studio portrait")
        self.assertEqual(workflow["6"]["inputs"]["width"], 832)
        self.assertEqual(workflow["6"]["inputs"]["height"], 1216)
        self.assertEqual(workflow["8"]["inputs"]["seed"], 321)
        self.assertEqual(workflow["8"]["inputs"]["steps"], 24)
        self.assertEqual(workflow["8"]["inputs"]["cfg"], 7)
        self.assertEqual(workflow["8"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(workflow["8"]["inputs"]["scheduler"], "karras")

    def test_realvisxl_selects_img2img_and_inpaint_workflows(self):
        img2img = cli.parse_args(
            ["change the dress", "--profile", "realvisxl", "--input-image", "source.png"]
        )
        self.assertEqual(img2img.workflow, cli.REALVISXL_IMG2IMG_WORKFLOW)
        self.assertEqual(img2img.steps, 30)

        inpaint = cli.parse_args(
            [
                "change the dress",
                "--profile",
                "realvisxl",
                "--input-image",
                "source.png",
                "--mask-image",
                "mask.png",
            ]
        )
        self.assertEqual(inpaint.workflow, cli.REALVISXL_INPAINT_WORKFLOW)

        workflow = cli.load_workflow(inpaint.workflow)
        cli.inject_inpaint_parameters(
            workflow,
            prompt="change the dress",
            seed=12,
            steps=30,
            denoise=0.7,
            image="source.png",
            mask="mask.png",
            mask_grow=8,
            mask_feather=5,
        )
        self.assertEqual(workflow["1"]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(workflow["6"]["inputs"]["image"], "source.png")
        self.assertEqual(workflow["12"]["inputs"]["image"], "mask.png")
        self.assertEqual(workflow["11"]["inputs"]["grow_mask_by"], 8)
        self.assertEqual(workflow["13"]["inputs"]["expand"], 8)
        self.assertEqual(workflow["15"]["inputs"]["blur_radius"], 5)
        self.assertEqual(workflow["17"]["class_type"], "ImageCompositeMasked")
        self.assertEqual(workflow["17"]["inputs"]["destination"], ["6", 0])
        self.assertEqual(workflow["10"]["inputs"]["images"], ["17", 0])
        self.assertEqual(workflow["8"]["inputs"]["denoise"], 0.7)

    def test_realvisxl_render_records_profile_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            image_path = output_dir / "RenderLab_00001_.png"
            self.write_binary_png(image_path, [[0]])
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
                patch.object(cli.secrets, "randbits", return_value=456),
                patch.object(cli, "submit", return_value="prompt-realvis"),
                patch.object(cli, "wait_for_history", return_value=history),
            ):
                result = cli.main(
                    [
                        "an adult studio portrait",
                        "--profile",
                        "realvisxl",
                        "--lora",
                        "portrait-xl.safetensors",
                        "--lora-model-strength",
                        "0.8",
                        "--lora-clip-strength",
                        "0.6",
                        "--cfg",
                        "6.5",
                        "--negative-prompt",
                        "watermark, text",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(result, 0)
            metadata = json.loads(Path(str(image_path) + ".json").read_text())
            self.assertEqual(metadata["renderlab_version"], "0.8.0")
            self.assertEqual(metadata["profile"], "realvisxl_v5_fp16")
            self.assertEqual(
                metadata["models"],
                {"checkpoint": "RealVisXL_V5.0_fp16.safetensors"},
            )
            self.assertEqual(metadata["steps"], 30)
            self.assertEqual(metadata["cfg"], 6.5)
            self.assertEqual(metadata["negative_prompt"], "watermark, text")
            self.assertEqual(metadata["sampler"], "dpmpp_2m")
            self.assertEqual(metadata["scheduler"], "karras")
            self.assertEqual(
                metadata["lora"],
                {
                    "name": "portrait-xl.safetensors",
                    "model_strength": 0.8,
                    "clip_strength": 0.6,
                },
            )
            lora_nodes = [
                node
                for node in metadata["submitted_workflow"].values()
                if node["class_type"] == "LoraLoader"
            ]
            self.assertEqual(len(lora_nodes), 1)

    def test_img2img_defaults_and_parameter_injection(self):
        args = cli.parse_args(["make it rainy", "--input-image", "source.png"])
        self.assertEqual(args.workflow, cli.DEFAULT_IMG2IMG_WORKFLOW)
        self.assertEqual(args.denoise, 0.45)

        workflow = cli.load_workflow(args.workflow)
        cli.inject_img2img_parameters(
            workflow,
            prompt="make it rainy",
            seed=456,
            steps=8,
            denoise=0.35,
            image="renderlab/source.png",
        )
        self.assertEqual(workflow["4"]["inputs"]["text"], "make it rainy")
        self.assertEqual(workflow["6"]["inputs"]["image"], "renderlab/source.png")
        self.assertEqual(workflow["8"]["inputs"]["latent_image"], ["11", 0])
        self.assertEqual(workflow["8"]["inputs"]["denoise"], 0.35)

    def test_outpaint_is_rejected_without_a_dedicated_model(self):
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "extend", "--input-image", "source.png", "--outpaint-left", "256",
            ])

    def test_inpaint_defaults_and_parameter_injection(self):
        args = cli.parse_args(
            [
                "black gothic dress",
                "--input-image",
                "source.png",
                "--mask-image",
                "mask.png",
            ]
        )
        self.assertEqual(args.workflow, cli.DEFAULT_INPAINT_WORKFLOW)
        self.assertEqual(args.mask_grow, 6)

        workflow = cli.load_workflow(args.workflow)
        cli.inject_inpaint_parameters(
            workflow,
            prompt="black gothic dress",
            seed=789,
            steps=8,
            denoise=0.65,
            image="source.png",
            mask="mask.png",
            mask_grow=10,
            mask_feather=7,
        )
        self.assertEqual(workflow["6"]["inputs"]["image"], "source.png")
        self.assertEqual(workflow["12"]["inputs"]["image"], "mask.png")
        self.assertEqual(workflow["12"]["inputs"]["channel"], "red")
        self.assertEqual(workflow["11"]["class_type"], "VAEEncodeForInpaint")
        self.assertEqual(workflow["11"]["inputs"]["grow_mask_by"], 10)
        self.assertEqual(workflow["13"]["inputs"]["expand"], 10)
        self.assertEqual(workflow["15"]["inputs"]["blur_radius"], 7)
        self.assertEqual(workflow["16"]["class_type"], "ImageToMask")
        self.assertEqual(workflow["17"]["inputs"]["source"], ["9", 0])
        self.assertEqual(workflow["10"]["inputs"]["images"], ["17", 0])
        self.assertEqual(workflow["8"]["inputs"]["denoise"], 0.65)

    def test_mask_command_defaults_and_parameter_injection(self):
        args = cli.parse_control_args(
            ["mask", "source.png", "chest, abdomen, pelvis", "--within", "woman"]
        )
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.refine_iterations, 2)
        self.assertEqual(args.within, "woman")

        workflow = cli.load_workflow(cli.SAM3_MASK_WORKFLOW)
        cli.inject_mask_parameters(
            workflow,
            image="renderlab/source.png",
            description="chest, abdomen, pelvis",
            threshold=0.4,
            refine_iterations=3,
            within="woman",
        )
        self.assertEqual(workflow["2"]["inputs"]["image"], "renderlab/source.png")
        text_nodes = [
            node["inputs"]["text"] for node in workflow.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        self.assertEqual(text_nodes, ["woman", "chest", "abdomen", "pelvis"])
        self.assertEqual(workflow["4"]["class_type"], "SAM3_Detect")
        self.assertEqual(workflow["4"]["inputs"]["threshold"], 0.4)
        self.assertEqual(workflow["4"]["inputs"]["refine_iterations"], 3)
        operations = [
            node["inputs"]["operation"] for node in workflow.values()
            if node["class_type"] == "MaskComposite"
        ]
        self.assertEqual(operations, ["and", "and", "and", "or", "or"])
        self.assertEqual(sum(node["class_type"] == "MaskToImage" for node in workflow.values()), 1)
        self.assertEqual(sum(node["class_type"] == "SaveImage" for node in workflow.values()), 1)

    def test_mask_command_writes_validated_binary_mask_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.png"
            source.write_bytes(b"source")
            mask = root / "RenderLabMask_00001_.png"
            self.write_binary_png(
                mask,
                [[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]],
            )
            history = {
                "outputs": {
                    "6": {
                        "images": [
                            {"filename": mask.name, "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
            with (
                patch.object(cli, "upload_image", return_value="renderlab/source.png"),
                patch.object(cli, "submit", return_value="prompt-mask") as submit,
                patch.object(cli, "wait_for_history", return_value=history),
            ):
                result = cli.main(
                    [
                        "mask", str(source), "torso and hips",
                        "--threshold", "0.4", "--output-dir", str(root),
                    ]
                )

            self.assertEqual(result, 0)
            workflow = submit.call_args.args[1]
            self.assertEqual(workflow["3"]["inputs"]["text"], "torso and hips")
            metadata = json.loads(Path(str(mask) + ".json").read_text())
            self.assertEqual(metadata["mode"], "mask")
            self.assertEqual(metadata["model"], "sam3.1_multiplex_fp16.safetensors")
            self.assertEqual(metadata["mask"]["white_pixels"], 4)
            self.assertEqual(metadata["mask"]["black_pixels"], 12)
            self.assertFalse(metadata["mask"]["touches_border"])
            self.assertTrue(metadata["mask"]["white_is_editable"])

    def test_binary_mask_validation_rejects_empty_mask(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            mask = Path(temporary_dir) / "empty.png"
            self.write_binary_png(mask, [[0, 0], [0, 0]])
            with self.assertRaisesRegex(cli.RenderError, "found no matching region"):
                cli.validate_binary_mask_png(mask)

    def test_binary_mask_validation_rejects_border_contamination(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            mask = Path(temporary_dir) / "border.png"
            self.write_binary_png(mask, [[255, 0], [0, 0]])
            with self.assertRaisesRegex(cli.RenderError, "touches the image boundary"):
                cli.validate_binary_mask_png(mask)

    def test_upload_image_posts_multipart_and_returns_comfy_name(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.png"
            source.write_bytes(b"test-png")
            response = io.BytesIO(
                json.dumps(
                    {"name": "uploaded.png", "subfolder": "renderlab", "type": "input"}
                ).encode("utf-8")
            )
            with patch.object(cli, "urlopen", return_value=response) as urlopen:
                uploaded = cli.upload_image("http://127.0.0.1:8188", source)

        self.assertEqual(uploaded, "renderlab/uploaded.png")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8188/upload/image")
        self.assertIn(b"test-png", request.data)
        self.assertIn(b'name="overwrite"', request.data)

    def test_img2img_render_records_source_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.png"
            source.write_bytes(b"source-png")
            output = root / "RenderLab_00001_.png"
            self.write_binary_png(output, [[0]])
            history = {
                "outputs": {
                    "10": {
                        "images": [
                            {"filename": output.name, "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
            with (
                patch.object(cli, "upload_image", return_value="renderlab/uploaded.png") as upload,
                patch.object(cli, "submit", return_value="prompt-edit") as submit,
                patch.object(cli, "wait_for_history", return_value=history),
                patch.object(cli.secrets, "randbits", return_value=123),
            ):
                result = cli.main(
                    [
                        "make it rainy",
                        "--input-image",
                        str(source),
                        "--denoise",
                        "0.35",
                        "--output-dir",
                        str(root),
                    ]
                )

            self.assertEqual(result, 0)
            upload.assert_called_once_with("http://127.0.0.1:8188", source.resolve())
            workflow = submit.call_args.args[1]
            self.assertEqual(workflow["6"]["inputs"]["image"], "renderlab/uploaded.png")
            self.assertEqual(workflow["8"]["inputs"]["denoise"], 0.35)
            metadata = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["mode"], "img2img")
            self.assertEqual(metadata["denoise"], 0.35)
            self.assertIsNone(metadata["width"])
            self.assertEqual(metadata["source_image"]["path"], str(source.resolve()))
            self.assertEqual(metadata["source_image"]["sha256"], cli.sha256_file(source))
            self.assertEqual(metadata["source_image"]["comfy_input"], "renderlab/uploaded.png")

    def test_inpaint_render_records_mask_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.png"
            source.write_bytes(b"source-png")
            mask = root / "mask.png"
            mask.write_bytes(b"mask-png")
            output = root / "RenderLab_00001_.png"
            self.write_binary_png(output, [[0]])
            history = {
                "outputs": {
                    "10": {
                        "images": [
                            {"filename": output.name, "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
            with (
                patch.object(
                    cli,
                    "upload_image",
                    side_effect=["renderlab/source.png", "renderlab/mask.png"],
                ) as upload,
                patch.object(cli, "submit", return_value="prompt-inpaint") as submit,
                patch.object(cli, "wait_for_history", return_value=history),
                patch.object(cli.secrets, "randbits", return_value=321),
            ):
                result = cli.main(
                    [
                        "black gothic dress",
                        "--input-image",
                        str(source),
                        "--mask-image",
                        str(mask),
                        "--mask-grow",
                        "10",
                        "--mask-feather",
                        "8",
                        "--denoise",
                        "0.65",
                        "--output-dir",
                        str(root),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(upload.call_count, 2)
            workflow = submit.call_args.args[1]
            self.assertEqual(workflow["6"]["inputs"]["image"], "renderlab/source.png")
            self.assertEqual(workflow["12"]["inputs"]["image"], "renderlab/mask.png")
            metadata = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["mode"], "inpaint")
            self.assertEqual(metadata["mask_image"]["path"], str(mask.resolve()))
            self.assertEqual(metadata["mask_image"]["sha256"], cli.sha256_file(mask))
            self.assertTrue(metadata["mask_image"]["white_is_editable"])
            self.assertEqual(metadata["mask_image"]["grow_pixels"], 10)
            self.assertEqual(metadata["mask_image"]["feather_pixels"], 8)

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
            self.write_binary_png(image_path, [[0]])
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
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["renderlab_version"], "0.8.0")
            self.assertEqual(metadata["intent"], "a test fox")
            self.assertEqual(metadata["effective_prompt"], "a test fox")
            self.assertEqual(metadata["output_sha256"], cli.sha256_file(image_path))
            self.assertEqual(metadata["output_pixel_sha256"], cli.pixel_sha256_file(image_path))
            self.assertEqual(
                metadata["submitted_workflow_sha256"],
                cli.sha256_json(metadata["submitted_workflow"]),
            )
            self.assertEqual(
                metadata["submitted_workflow"]["8"]["inputs"]["seed"], 987654321
            )

    def test_batch_resolves_independent_random_seeds(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            histories = []
            for index in range(1, 4):
                image_path = output_dir / f"RenderLab_{index:05d}_.png"
                self.write_binary_png(image_path, [[index]])
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
            first_metadata = json.loads(
                (output_dir / "RenderLab_00001_.png.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_metadata["batch_id"], metadata["batch_id"])

    def test_explicit_batch_seed_increments(self):
        self.assertEqual(cli.resolve_seeds(500, 3), [500, 501, 502])

    def test_explicit_batch_seed_rejects_overflow(self):
        with self.assertRaises(cli.RenderError):
            cli.resolve_seeds(cli.MAX_SEED, 2)

    def test_expand_prompts_calls_local_openai_compatible_server(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"prompts":["desert mech","alpine mech"]}\n```'
                    }
                }
            ]
        }
        with patch.object(cli, "request_json", return_value=response) as request:
            prompts = cli.expand_prompts(
                "http://127.0.0.1:8080", "tiny-model", "starry mech", 2
            )

        self.assertEqual(prompts, ["desert mech", "alpine mech"])
        payload = request.call_args.args[2]
        self.assertEqual(request.call_args.kwargs["timeout"], 180.0)
        self.assertEqual(payload["model"], "tiny-model")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False, "reasoning_effort": "low"},
        )
        self.assertEqual(payload["messages"][1]["content"], "starry mech")
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("at least four", system_prompt)
        self.assertIn("Do not merely paraphrase", system_prompt)
        self.assertIn("Director briefs", system_prompt)
        self.assertEqual(payload["json_schema"]["properties"]["prompts"]["minItems"], 2)
        self.assertEqual(payload["json_schema"]["properties"]["prompts"]["maxItems"], 2)

    def test_expand_prompts_reports_reasoning_budget_exhaustion(self):
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "still thinking"},
                }
            ]
        }
        with patch.object(cli, "request_json", return_value=response):
            with self.assertRaisesRegex(cli.RenderError, "exhausted its completion budget"):
                cli.expand_prompts("http://127.0.0.1:8084", "local", "starry mech", 3)

    def test_variations_default_to_whiskers_cpu_summarizer(self):
        args = cli.parse_args(["starry mech", "--variations", "2"])
        self.assertEqual(args.prompt_server, "http://127.0.0.1:8084")
        self.assertEqual(args.prompt_timeout, 180.0)

    def test_connection_reset_reports_probable_server_crash(self):
        with patch.object(cli, "urlopen", side_effect=ConnectionResetError(104, "reset")):
            with self.assertRaisesRegex(
                cli.RenderError, "server reset the connection and may have crashed"
            ) as raised:
                cli.request_json("GET", "http://127.0.0.1:8188/history/prompt-1")

        self.assertIn("GPU-offloaded Waldo", str(raised.exception))

    def test_variations_render_expanded_prompts_and_record_intent(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            histories = []
            for index in range(1, 3):
                image_path = output_dir / f"RenderLab_{index:05d}_.png"
                self.write_binary_png(image_path, [[index]])
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
                patch.object(cli, "expand_prompts", return_value=["desert mech", "lake mech"]),
                patch.object(cli.secrets, "randbits", side_effect=[11, 22]),
                patch.object(cli, "submit", side_effect=["prompt-1", "prompt-2"]) as submit,
                patch.object(cli, "wait_for_history", side_effect=histories),
            ):
                result = cli.main(
                    ["starry mech", "--variations", "2", "--output-dir", str(output_dir)]
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args[1]["4"]["inputs"]["text"] for call in submit.call_args_list],
                ["desert mech", "lake mech"],
            )
            metadata = json.loads(
                (output_dir / "RenderLab_00002_.png.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["intent"], "starry mech")
            self.assertEqual(metadata["effective_prompt"], "lake mech")
            self.assertEqual(metadata["prompt_expander"]["variation_count"], 2)
            self.assertEqual(
                metadata["prompt_expander"]["variation_policy"], "directors_v1"
            )

    def test_jobs_lists_prompt_ids_and_statuses(self):
        with patch.object(
            cli,
            "request_json",
            return_value={
                "jobs": [
                    {"id": "prompt-running", "status": "in_progress"},
                    {"id": "prompt-waiting", "status": "pending"},
                ]
            },
        ) as request:
            result = cli.main(["jobs", "--limit", "2"])

        self.assertEqual(result, 0)
        request.assert_called_once_with("GET", "http://127.0.0.1:8188/api/jobs?limit=2")

    def test_status_fetches_one_job(self):
        with patch.object(
            cli, "request_json", return_value={"id": "prompt-123", "status": "completed"}
        ) as request:
            result = cli.main(["status", "prompt-123"])

        self.assertEqual(result, 0)
        request.assert_called_once_with("GET", "http://127.0.0.1:8188/api/jobs/prompt-123")

    def test_cancel_uses_targeted_job_endpoint(self):
        with patch.object(cli, "request_json", return_value={"cancelled": True}) as request:
            result = cli.main(["cancel", "prompt-123"])

        self.assertEqual(result, 0)
        request.assert_called_once_with(
            "POST", "http://127.0.0.1:8188/api/jobs/prompt-123/cancel", {}
        )

    def test_discover_node_choices_reads_comfy_input_options(self):
        with patch.object(
            cli,
            "request_json",
            return_value={
                "LoraLoader": {
                    "input": {"required": {"lora_name": [["style/a.safetensors"], {}]}}
                }
            },
        ):
            choices = cli.discover_node_choices(
                "http://127.0.0.1:8188", "LoraLoader", "lora_name"
            )

        self.assertEqual(choices, ["style/a.safetensors"])

    def test_models_queries_each_supported_loader(self):
        def object_info(_method, url, _payload=None):
            node_name = url.rsplit("/", 1)[-1]
            input_name = next(
                input_name
                for _, configured_node, input_name in cli.MODEL_NODE_INPUTS
                if configured_node == node_name
            )
            return {
                node_name: {"input": {"required": {input_name: [[f"{node_name}.model"], {}]}}}
            }

        with patch.object(cli, "request_json", side_effect=object_info) as request:
            result = cli.main(["models"])

        self.assertEqual(result, 0)
        self.assertEqual(request.call_count, len(cli.MODEL_NODE_INPUTS))

    def test_loras_queries_lora_loader(self):
        response = {
            "LoraLoader": {
                "input": {"required": {"lora_name": [["character.safetensors"], {}]}}
            }
        }
        with patch.object(cli, "request_json", return_value=response) as request:
            result = cli.main(["loras"])

        self.assertEqual(result, 0)
        request.assert_called_once_with(
            "GET", "http://127.0.0.1:8188/object_info/LoraLoader"
        )

    def test_doctor_accepts_complete_runtime(self):
        choice_by_node = {
            node_name: (input_name, filename)
            for node_name, input_name, filename in cli.REQUIRED_MODEL_CHOICES
        }

        def doctor_response(_method, url, _payload=None):
            if url.endswith("/system_stats"):
                return {"system": {}}
            node_name = url.rsplit("/", 1)[-1]
            response = {node_name: {"input": {"required": {}}}}
            if node_name in choice_by_node:
                input_name, filename = choice_by_node[node_name]
                response[node_name]["input"]["required"][input_name] = [[filename], {}]
            return response

        with patch.object(cli, "request_json", side_effect=doctor_response):
            result = cli.main(["doctor"])

        self.assertEqual(result, 0)

    def test_doctor_reports_missing_model(self):
        def doctor_response(_method, url, _payload=None):
            if url.endswith("/system_stats"):
                return {"system": {}}
            node_name = url.rsplit("/", 1)[-1]
            response = {node_name: {"input": {"required": {}}}}
            for configured_node, input_name, filename in cli.REQUIRED_MODEL_CHOICES:
                if node_name == configured_node:
                    choices = [] if node_name == "UNETLoader" else [filename]
                    response[node_name]["input"]["required"][input_name] = [choices, {}]
            return response

        with patch.object(cli, "request_json", side_effect=doctor_response):
            result = cli.main(["doctor"])

        self.assertEqual(result, 1)

    def test_doctor_validates_realvisxl_profile(self):
        def doctor_response(_method, url, _payload=None):
            if url.endswith("/system_stats"):
                return {"system": {}}
            node_name = url.rsplit("/", 1)[-1]
            response = {node_name: {"input": {"required": {}}}}
            if node_name == "CheckpointLoaderSimple":
                response[node_name]["input"]["required"]["ckpt_name"] = [
                    ["RealVisXL_V5.0_fp16.safetensors"],
                    {},
                ]
            return response

        with patch.object(cli, "request_json", side_effect=doctor_response):
            result = cli.main(["doctor", "--profile", "realvisxl"])

        self.assertEqual(result, 0)

    def test_output_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(cli.RenderError):
                cli.local_output_path(
                    Path(temporary_dir),
                    {"filename": "escape.png", "subfolder": "../outside", "type": "output"},
                )


if __name__ == "__main__":
    unittest.main()
