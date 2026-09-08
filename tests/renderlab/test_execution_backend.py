import base64
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from renderlab import cli
from renderlab.execution_backend import ComfyLocalBackend, ExecutionError, run_job
from renderlab.workflow_profiles import digest, load_profile, prepare_job, profile_hash, read_json, replay_job, write_job


class FakeBackend:
    name = "comfy-local"
    identity = "fake://offline"

    def __init__(self, data):
        self.data = data
        self.calls = []
        self.present = True
        self.states = [{"status": "completed", "gpu_seconds": 1.5}]
        self.submit_error = None
        self.items = [{"node_id": "14", "filename": "result.png", "subfolder": "", "type": "output"}]

    def readiness(self, job):
        self.calls.append(("readiness",))
        return {"ready": True}

    def assets_present(self, job):
        return self.present

    def validate(self, workflow):
        self.calls.append(("validate", workflow))
        return {"status": "validated", "submitted": False, "warnings": []}

    def submit(self, workflow):
        self.calls.append(("submit", workflow))
        if self.submit_error:
            raise self.submit_error
        return "job-1"

    def status(self, job_id):
        self.calls.append(("status", job_id))
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def outputs(self, job_id, nodes):
        return self.items

    def download(self, output):
        self.calls.append(("download", output["node_id"]))
        return self.data

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))


class ExecutionBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = io.BytesIO()
        Image.new("RGB", (4, 3), "blue").save(image, format="PNG")
        self.data = image.getvalue()
        self.source = self.root / "source.png"
        self.source.write_bytes(self.data)
        path, profile = load_profile("firered_shoes_v1")
        profile = copy.deepcopy(profile)
        profile["workflow"]["path"] = str((path.parent / profile["workflow"]["path"]).resolve())
        profile["resource_inventory_path"] = str((path.parent / profile["resource_inventory_path"]).resolve())
        profile["asset_references"]["source"].update(path=str(self.source), sha256=digest(self.data))
        profile["content_hash"] = profile_hash(profile)
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(profile))
        self.job = prepare_job(self.profile_path)
        self.backend = FakeBackend(self.data)
        self.directory = self.root / "execution"

    def run_execution(self, **kwargs):
        return run_job(self.job, self.backend, self.directory, poll_interval=0, **kwargs)

    def submits(self):
        return [call for call in self.backend.calls if call[0] == "submit"]

    def test_dry_run_never_submits(self):
        result = self.run_execution()
        self.assertFalse(result["submitted"])
        self.assertEqual(self.submits(), [])
        self.assertEqual(result["status"], "validated_pending_explicit_execution")

    def test_exact_bytes_single_submit_metadata_and_raw_download(self):
        result = self.run_execution(execute=True)
        expected = base64.b64decode(self.job["workflow_bytes_base64"])
        self.assertEqual(self.submits(), [("submit", expected)])
        self.assertEqual(result["resolved_job_hash"], self.job["record_hash"])
        self.assertEqual(result["submitted_workflow_hash"], digest(expected))
        self.assertEqual(result["outputs"][0]["dimensions"], [4, 3])
        self.assertEqual(result["outputs"][0]["sha256"], digest(self.data))
        self.assertEqual(Path(result["outputs"][0]["path"]).read_bytes(), self.data)
        self.assertEqual(result["lineage"], self.job["lineage"])
        self.assertEqual(result["status_history"][0]["gpu_seconds"], 1.5)
        self.assertTrue(result["submission_timestamp"])
        self.assertEqual(read_json(self.directory / "prepared_job.json"), self.job)
        self.assertEqual(self.source.read_bytes(), self.data)

    def test_submit_exception_is_not_retried_and_cannot_leak_secret(self):
        self.backend.submit_error = RuntimeError("Authorization: Bearer TOP_SECRET_CREDENTIAL")
        with self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        self.assertEqual(len(self.submits()), 1)
        evidence = (self.directory / "execution.json").read_text()
        self.assertNotIn("TOP_SECRET", evidence)
        self.assertTrue(json.loads(evidence)["error"]["submission_may_have_succeeded"])

    def test_missing_asset_blocks_submission(self):
        self.backend.present = False
        with self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        self.assertEqual(self.submits(), [])

    def test_local_source_change_blocks_every_backend_call(self):
        self.source.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.run_execution(execute=True)
        self.assertEqual(self.backend.calls, [])

    def test_failed_status_does_not_retry_or_download(self):
        self.backend.states = [{"status": "failed", "error_code": "backend_execution_error", "message": "secret"}]
        with self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        self.assertEqual(len(self.submits()), 1)
        self.assertFalse(any(c[0] == "download" for c in self.backend.calls))
        self.assertNotIn('"message"', (self.directory / "execution.json").read_text())

    def test_polling_is_bounded_without_cancel(self):
        self.backend.states = [{"status": "running"}]
        with self.assertRaises(ExecutionError):
            self.run_execution(execute=True, max_polls=3)
        self.assertEqual(sum(c[0] == "status" for c in self.backend.calls), 3)
        self.assertEqual(len(self.submits()), 1)
        self.assertFalse(any(c[0] == "cancel" for c in self.backend.calls))

    def test_declared_output_selection(self):
        self.backend.items.append({"node_id": "999", "filename": "unselected.png"})
        self.run_execution(execute=True)
        self.assertEqual([c for c in self.backend.calls if c[0] == "download"], [("download", "14")])

    def test_malformed_status_fails_closed(self):
        self.backend.states = [{"status": "probably_done", "token": "secret"}]
        with self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        self.assertEqual(len(self.submits()), 1)
        self.assertNotIn("secret", (self.directory / "execution.json").read_text())

    def test_malformed_output_fails_closed(self):
        self.backend.items[0]["filename"] = "../../source.png"
        with self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        self.assertFalse(any(c[0] == "download" for c in self.backend.calls))

    def test_download_failure_nonzero_without_retry(self):
        self.backend.data = b"not an image"
        with self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        self.assertEqual(sum(c[0] == "download" for c in self.backend.calls), 1)
        self.assertEqual(read_json(self.directory / "execution.json")["error"]["stage"], "download")

    def test_replayed_job_uses_identical_bytes(self):
        record = self.root / "job.json"
        write_job(record, self.job)
        self.job = replay_job(record)
        self.run_execution(execute=True)
        self.assertEqual(self.submits()[0][1], base64.b64decode(self.job["workflow_bytes_base64"]))

    def test_existing_evidence_directory_blocks_another_submission(self):
        self.run_execution(execute=True)
        with self.assertRaises(FileExistsError):
            self.run_execution(execute=True)
        self.assertEqual(len(self.submits()), 1)

    def test_local_adapter_preserves_wire_document(self):
        backend = ComfyLocalBackend()
        raw = base64.b64decode(self.job["workflow_bytes_base64"])
        with patch.object(backend, "request", return_value={"prompt_id": "job-1"}) as request:
            self.assertEqual(backend.submit(raw), "job-1")
        request.assert_called_once_with("/prompt", b'{"prompt":' + raw + b'}')
        with patch.object(backend, "request", return_value={"prompt_id": None}), self.assertRaises(ExecutionError):
            backend.submit(raw)

    def test_local_status_and_declared_history_outputs(self):
        backend = ComfyLocalBackend()
        history = {"status": {"status_str": "success", "completed": True}, "outputs": {"14": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}
        with patch.object(backend, "history", return_value=history):
            self.assertEqual(backend.status("job-1")["status"], "completed")
            self.assertEqual(backend.outputs("job-1", ["14"])[0]["node_id"], "14")

    def test_cli_execution_requires_explicit_flag_and_failures_are_nonzero(self):
        with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            with patch.object(cli, "ComfyLocalBackend", return_value=self.backend):
                result = cli.main(["generate", "--profile", str(self.profile_path), "--source", str(self.source), "--execute", "--execution-dir", str(self.directory)])
            self.assertEqual(result, 0)
            self.assertEqual(len(self.submits()), 1)
            with self.assertRaises(SystemExit):
                cli.main(["generate", "--profile", str(self.profile_path), "--execute", "--dry-run"])
            self.backend.present = False
            with patch.object(cli, "ComfyLocalBackend", return_value=self.backend):
                self.assertEqual(cli.main(["generate", "--profile", str(self.profile_path), "--execute", "--execution-dir", str(self.root / "failed")]), 1)
            self.assertEqual(len(self.submits()), 1)

    def test_local_validation_rejects_missing_model_and_wrong_slot(self):
        backend = ComfyLocalBackend()
        backend.schemas = {"Loader": {"input": {"required": {"model": [["available"]]}}, "output": ["IMAGE"]}, "Saver": {"input": {"required": {"image": ["IMAGE"]}}, "output": []}}
        graph = {"1": {"class_type": "Loader", "inputs": {"model": "available"}}, "2": {"class_type": "Saver", "inputs": {"image": ["1", 0]}}}
        self.assertEqual(backend.validate(json.dumps(graph).encode())["submitted"], False)
        graph["1"]["inputs"]["model"] = "missing"
        with self.assertRaises(ExecutionError):
            backend.validate(json.dumps(graph).encode())
        graph["1"]["inputs"]["model"] = "available"
        graph["2"]["inputs"]["image"] = ["1", 9]
        with self.assertRaises(ExecutionError):
            backend.validate(json.dumps(graph).encode())

    def test_local_remote_source_hash_is_checked(self):
        backend = ComfyLocalBackend()
        with patch.object(backend, "request", return_value=b"different source"), self.assertRaises(ExecutionError):
            backend.assets_present(self.job)
        with patch.object(backend, "request", return_value=self.data):
            self.assertTrue(backend.assets_present(self.job))

    def test_unknown_submit_response_fails_without_poll(self):
        with patch.object(self.backend, "submit", return_value=None) as submit, self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        submit.assert_called_once()
        self.assertFalse(any(c[0] == "status" for c in self.backend.calls))
        self.assertTrue(read_json(self.directory / "execution.json")["error"]["submission_may_have_succeeded"])

    def test_backend_warning_blocks_submission(self):
        verdict = {"status": "validated", "submitted": False, "warnings": ["warning"]}
        with patch.object(self.backend, "validate", return_value=verdict), self.assertRaises(ExecutionError):
            self.run_execution(execute=True)
        self.assertEqual(self.submits(), [])

    def test_requested_profile_alias(self):
        self.assertEqual(load_profile("firered_shoes_to_stilettos")[1]["profile_id"], "firered_shoes_v1")


if __name__ == "__main__":
    unittest.main()
