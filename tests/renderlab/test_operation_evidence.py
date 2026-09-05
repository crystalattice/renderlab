import hashlib
import json
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / 'renderlab' / 'experiments'


class OperationEvidenceTests(unittest.TestCase):
    def test_informal_evidence_does_not_promote_failed_or_unrun_backends(self):
        matrix = json.loads((EXPERIMENTS / 'operation_backend_evidence.json').read_text())
        informal = matrix['informal_firered']
        self.assertFalse(informal['formal_benchmark'])
        self.assertIsNone(informal['validated_pass_rate'])
        rows = {r['operation']: r for r in matrix['operations']}
        self.assertEqual(rows['strict_face_swap']['approval'], 'no_approved_backend')
        self.assertEqual(rows['controlled_inpaint']['approval'], 'unvalidated')
        prior = ROOT / matrix['qwen_face_swap']['evaluation']
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(), matrix['qwen_face_swap']['sha256'])
        evaluation = json.loads(prior.read_text())
        self.assertEqual(evaluation['criteria']['source_identity_transfer']['score'], 2)
        self.assertEqual(evaluation['criteria']['instruction_adherence']['score'], 2)
        self.assertEqual(evaluation['disposition'], 'FAIL')

    def test_mask_and_graph_preserve_the_predeclared_coordinate_contract(self):
        p = EXPERIMENTS / 'firered_controlled_inpaint_v1'
        case = json.loads((p / 'experiment.json').read_text())
        graph = json.loads((p / 'api.prepared.json').read_text())
        self.assertEqual(hashlib.sha256((p / 'mask.png').read_bytes()).hexdigest(), case['mask']['sha256'])
        with Image.open(p / 'mask.png') as image:
            self.assertEqual(image.size, tuple(case['settings']['output_dimensions']))
            self.assertEqual(image.mode, 'RGB')
            colors = dict((color, count) for count, color in image.getcolors(maxcolors=3))
            self.assertEqual(set(colors), {(0, 0, 0), (255, 255, 255)})
            self.assertEqual(image.getbbox(), tuple(case['mask']['bounds_xyxy_exclusive']))
            self.assertEqual(colors[(255, 255, 255)], case['mask']['editable_pixels'])
        sampler = next(n for n in graph.values() if n['class_type'] == 'KSampler')
        latent_id, slot = sampler['inputs']['latent_image']
        self.assertEqual(slot, 0)
        masked = graph[latent_id]
        self.assertEqual(masked['class_type'], 'SetLatentNoiseMask')
        mask_id, slot = masked['inputs']['mask']
        self.assertEqual(graph[mask_id]['class_type'], 'LoadImageMask')
        self.assertEqual(graph[mask_id]['inputs']['channel'], 'red')
        self.assertFalse(any('Resize' in n['class_type'] or 'Composite' in n['class_type'] for n in graph.values()))
        save = next(n for n in graph.values() if n['class_type'] == 'SaveImage')
        self.assertEqual(graph[save['inputs']['images'][0]]['class_type'], 'VAEDecode')
        for node in graph.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)

    def test_inpainting_remains_unexecuted_with_independent_preservation_gates(self):
        p = EXPERIMENTS / 'firered_controlled_inpaint_v1'
        case = json.loads((p / 'experiment.json').read_text())
        pending = json.loads((p / 'evaluation.pending.json').read_text())
        self.assertEqual(case['status'], 'prepared_not_executed')
        self.assertFalse(case['run_design']['execution_authorized'])
        self.assertIsNone(case['run_design']['job_id'])
        self.assertIsNone(case['run_design']['results'])
        self.assertIsNone(pending['disposition'])
        self.assertTrue(all(v is None for v in pending['scores'].values()))
        self.assertEqual(set(pending['scores']), set(case['metrics']))
        for metric in ['outside_mask_pixel_preservation', 'outside_mask_semantic_preservation',
                       'identity_preservation', 'pose_preservation', 'body_preservation', 'background_preservation']:
            self.assertEqual(case['thresholds'][metric], 5)
