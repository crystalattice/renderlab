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
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["renderlab_version"], "0.1.0")
            self.assertEqual(metadata["intent"], "a test fox")
            self.assertEqual(metadata["effective_prompt"], "a test fox")
            self.assertEqual(metadata["output_sha256"], cli.hashlib.sha256(b"png").hexdigest())
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
        self.assertEqual(payload["model"], "tiny-model")
        self.assertEqual(payload["messages"][1]["content"], "starry mech")

    def test_variations_default_to_whiskers_cpu_summarizer(self):
        args = cli.parse_args(["starry mech", "--variations", "2"])
        self.assertEqual(args.prompt_server, "http://127.0.0.1:8084")

    def test_variations_render_expanded_prompts_and_record_intent(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            histories = []
            for index in range(1, 3):
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

    def test_output_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(cli.RenderError):
                cli.local_output_path(
                    Path(temporary_dir),
                    {"filename": "escape.png", "subfolder": "../outside", "type": "output"},
                )


if __name__ == "__main__":
    unittest.main()
