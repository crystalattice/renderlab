import copy
import unittest

from renderlab.experiments.firered_single_pass_outfit_benchmark_v1.validate_benchmark import (
    CASES,
    read,
    validate_editor,
    validate_graph,
    validate_same_source,
)


class SinglePassOutfitBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.specs = read("cloud_availability.json")["node_schemas"]
        self.case = "shoes_to_stilettos"
        self.graph = read(self.case + "/api.cloud.resolved.json")

    def test_all_graphs_and_editors(self):
        for case in CASES:
            with self.subTest(case=case):
                graph = read(case + "/api.cloud.resolved.json")
                validate_graph(graph, case, self.specs)
                validate_editor(graph, read(case + "/workflow.json"), self.specs)

    def test_sampler_settings_must_remain_fixed(self):
        self.graph["12"]["inputs"]["seed"] = 3408
        with self.assertRaises(AssertionError):
            validate_graph(self.graph, self.case, self.specs)

    def test_mask_node_is_rejected(self):
        self.graph["20"] = {
            "class_type": "SolidMask",
            "inputs": {"value": 1, "width": 1440, "height": 2160},
        }
        with self.assertRaises(AssertionError):
            validate_graph(self.graph, self.case, self.specs)

    def test_save_must_receive_native_decode(self):
        self.graph["14"]["inputs"]["images"] = ["1", 0]
        with self.assertRaises(AssertionError):
            validate_graph(self.graph, self.case, self.specs)

    def test_same_source_pair_cannot_change_other_inputs(self):
        bikini = read("clothing_to_bikini/api.cloud.resolved.json")
        validate_same_source(self.graph, bikini)
        changed = copy.deepcopy(bikini)
        changed["1"]["inputs"]["image"] = "different.jpg"
        with self.assertRaises(AssertionError):
            validate_same_source(self.graph, changed)

    def test_editor_cannot_randomize_seed(self):
        editor = read(self.case + "/workflow.json")
        sampler = next(node for node in editor["nodes"] if node["type"] == "KSampler")
        index = sampler["widgets_values"].index("fixed")
        sampler["widgets_values"][index] = "randomize"
        with self.assertRaises(AssertionError):
            validate_editor(self.graph, editor, self.specs)

    def test_invalid_decode_slot_is_rejected(self):
        self.graph["13"]["inputs"]["samples"] = ["12", 1]
        with self.assertRaises(AssertionError):
            validate_graph(self.graph, self.case, self.specs)


if __name__ == "__main__":
    unittest.main()
