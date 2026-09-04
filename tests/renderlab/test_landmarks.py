import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from renderlab.landmarks import LandmarkError, render_landmark_maps, validate_landmark_spec


def spec() -> dict:
    point = {"sigma_x": 0.08, "sigma_y": 0.06, "confidence": 0.8, "provenance": "canon_inferred", "visibility": "covered"}
    return {
        "schema": "renderlab.landmark-map.v1", "width": 32, "height": 48,
        "channels": {
            "left_nipple": {**point, "center_x": 0.4, "center_y": 0.35},
            "right_nipple": {**point, "center_x": 0.6, "center_y": 0.35},
            "navel": {**point, "center_x": 0.5, "center_y": 0.6},
        },
    }


class LandmarkTests(unittest.TestCase):
    def test_render_writes_deterministic_channels_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "landmarks.json"
            source.write_text(json.dumps(spec()), encoding="utf-8")
            result = render_landmark_maps(source, root / "rendered")
            self.assertEqual([value["channel"] for value in result["channels"]], ["left_nipple", "right_nipple", "navel"])
            with Image.open(result["preview"]["path"]) as preview:
                self.assertEqual(preview.size, (32, 48))
                self.assertEqual(preview.mode, "RGB")
            self.assertTrue((root / "rendered" / "manifest.json").is_file())

    def test_rejects_out_of_range_coordinate(self):
        value = spec()
        value["channels"]["navel"]["center_y"] = 1.2
        with self.assertRaisesRegex(LandmarkError, "center_y must be between 0 and 1"):
            validate_landmark_spec(value)


if __name__ == "__main__":
    unittest.main()
