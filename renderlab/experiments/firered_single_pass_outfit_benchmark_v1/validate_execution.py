"""Read-only verification of the completed, single-attempt benchmark."""

import hashlib
import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image


P = Path(__file__).resolve().parent
REPO = P.parents[2]
BASELINE = "35bc10c97aea8219fc17bdf1c05decc43b587d42"
CASES = ["shoes_to_stilettos", "dress_to_top_and_miniskirt", "clothing_to_bikini"]


def validate():
    previous_terminal = None
    jobs = []
    for case in CASES:
        folder = P / case
        execution = json.loads((folder / "execution.json").read_text())
        for name in [
            "cloud_submit.gated.json",
            "api.cloud.resolved.json",
            "api.prepared.json",
            "workflow.json",
            "prompt.txt",
            "cloud_bindings.json",
            "evaluation_rubric.json",
        ]:
            path = folder / name
            assert path.read_bytes() == subprocess.check_output(
                ["git", "show", f"{BASELINE}:{path.relative_to(REPO)}"], cwd=REPO
            ), path
        gate = json.loads((folder / "cloud_submit.gated.json").read_text())
        assert execution["request"] == gate["call_template"]["arguments"]
        assert execution["request"]["dry_run"] is False
        assert execution["submission_attempts"] == 1 and execution["retries"] == 0
        assert execution["authorization_consumed"] is True
        assert (
            execution["terminal_response"]["structuredContent"]["result"]["job_status"]
            == "completed"
        )
        assert (
            execution["job_id"]
            == execution["submission"]["structuredContent"]["result"]["prompt_id"]
        )
        start = datetime.strptime(
            execution["started_at"]["current_time"], "%Y-%m-%d %H:%M:%S UTC"
        )
        terminal = datetime.strptime(
            execution["terminal_observed_at"]["current_time"], "%Y-%m-%d %H:%M:%S UTC"
        )
        assert start <= terminal
        if previous_terminal is not None:
            assert previous_terminal <= start
        previous_terminal = terminal
        source = json.loads((folder / "manifest.json").read_text())["source"]
        assert (
            hashlib.sha256(Path(source["original_path"]).read_bytes()).hexdigest()
            == source["sha256"]
        )
        raw = folder / "native_output.png"
        assert (
            hashlib.sha256(raw.read_bytes()).hexdigest()
            == execution["raw_file"]["sha256"]
        )
        assert raw.stat().st_size == execution["raw_file"]["bytes"]
        with Image.open(raw) as image:
            image.load()
            assert (
                list(image.size)
                == source["dimensions"]
                == execution["raw_file"]["dimensions"]
            )
            assert image.mode == "RGB" == execution["raw_file"]["mode"]
        assert execution["download_exit_code"] == 0
        assert execution["billing_event"]["params"]["job_id"] == execution["job_id"]
        evaluation = json.loads((folder / "evaluation.json").read_text())
        assert evaluation["raw_sha256"] == execution["raw_file"]["sha256"]
        assert len(evaluation["independent_scores"]) == 12
        assert all(
            1 <= score["score"] <= 5 and score["evidence"]
            for score in evaluation["independent_scores"].values()
        )
        assert len(evaluation["hard_fail_assessment"]) == 9
        failures = any(row["triggered"] for row in evaluation["hard_fail_assessment"])
        assert evaluation["result"] == ("FAIL" if failures else "PASS")
        jobs.append(execution["job_id"])
    assert len(set(jobs)) == 3
    for name in ["settings.json", "informal_observations.json", "cloud_bindings.json"]:
        assert (P / name).read_bytes() == subprocess.check_output(
            ["git", "show", f"{BASELINE}:{(P / name).relative_to(REPO)}"], cwd=REPO
        )
    return {
        "status": "PASS",
        "committed_arguments_and_workflows_unchanged": True,
        "three_unique_sequential_single_attempt_jobs": jobs,
        "raw_hashes_dimensions_rgb_verified": True,
        "source_hashes_unchanged": True,
        "scores_complete": True,
        "informal_observations_unchanged": True,
    }


if __name__ == "__main__":
    sys.stdout.write(json.dumps(validate(), indent=2) + "\n")
