import base64
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from renderlab import cli
from renderlab.workflow_profiles import (
    ProfileError, canonical, digest, load_profile, prepare_job, profile_hash,
    read_json, replay_job, write_job,
)


class WorkflowProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'
        self.source.write_bytes(b'immutable source bytes')
        self.workflow = self.root / 'workflow.json'
        self.workflow.write_text(json.dumps({'1': {'class_type': 'OpaqueInput', 'inputs': {'image': 'input.png', 'text': 'literal\ntext', 'seed': 43, 'scale': 0.7, 'model': 'opaque-model'}}, '2': {'class_type': 'OpaqueOutput', 'inputs': {'image': ['1', 0]}}}))
        (self.root / 'inventory.json').write_text(json.dumps({'models': ['opaque-model'], 'assets': ['input.png']}))
        self.profile = {
            'schema_version': 1, 'profile_id': 'test_v1', 'capability': 'external_operation', 'backend': 'comfy-local',
            'workflow': {'path': 'workflow.json', 'sha256': digest(self.workflow.read_bytes())},
            'resource_inventory_path': 'inventory.json',
            'bindings': {key: {'node': '1', 'input': inp, 'type': typ, 'default': val} for key, inp, typ, val in [('source', 'image', 'string', 'input.png'), ('prompt', 'text', 'string', 'literal\ntext'), ('seed', 'seed', 'integer', 43), ('scale', 'scale', 'number', 0.7), ('model', 'model', 'string', 'opaque-model')]},
            'prompt_binding': 'prompt', 'source_image_bindings': ['source'], 'seed_bindings': ['seed'], 'numeric_bindings': ['scale'],
            'model_references': {'model': 'opaque-model'},
            'asset_references': {'source': {'path': 'source.png', 'sha256': digest(self.source.read_bytes()), 'backend_value': 'input.png'}},
            'output_nodes': ['2'], 'reconstruction_policy': 'preserve_visible_geometry',
            'immutable_source_policy': 'read_only_verified', 'metadata_capture_policy': 'complete_v1',
        }
        self.path = self.root / 'profile.json'
        self.save_profile()

    def save_profile(self):
        self.profile['content_hash'] = profile_hash(self.profile)
        self.path.write_text(json.dumps(self.profile))

    def test_valid_loading_and_relative_workflow_path(self):
        path, profile = load_profile(self.path)
        self.assertEqual(path, self.path)
        self.assertEqual(profile['profile_id'], 'test_v1')
        self.assertFalse(prepare_job(self.path)['submitted'])

    def test_malformed_and_unknown_schema_fields(self):
        for field, value in [('schema_version', 2), ('bindings', []), ('output_nodes', '2'), ('immutable_source_policy', 'overwrite'), ('unexpected', True)]:
            with self.subTest(field=field):
                original = copy.deepcopy(self.profile)
                self.profile[field] = value
                self.save_profile()
                with self.assertRaises(ProfileError):
                    load_profile(self.path)
                self.profile = original
        self.path.write_text('{')
        with self.assertRaises(ProfileError):
            load_profile(self.path)

    def test_missing_node_input_output_and_workflow(self):
        for key, value in [('node', 'missing'), ('input', 'missing')]:
            original = copy.deepcopy(self.profile)
            self.profile['bindings']['prompt'][key] = value
            self.save_profile()
            with self.assertRaisesRegex(ProfileError, 'Missing'):
                prepare_job(self.path)
            self.profile = original
        self.profile['output_nodes'] = ['missing']
        self.save_profile()
        with self.assertRaisesRegex(ProfileError, 'Missing output'):
            prepare_job(self.path)
        self.workflow.unlink()
        with self.assertRaisesRegex(ProfileError, 'workflow'):
            prepare_job(self.path)

    def test_deterministic_binding_and_immutable_inputs(self):
        workflow_before = self.workflow.read_bytes()
        source_before = self.source.read_bytes()
        values = {'prompt': 'opaque λ\nverbatim ', 'seed': 3407}
        first = prepare_job(self.path, values)
        second = prepare_job(self.path, dict(reversed(list(values.items()))))
        self.assertEqual(first['workflow_bytes_base64'], second['workflow_bytes_base64'])
        self.assertEqual(first['resolved_workflow_hash'], second['resolved_workflow_hash'])
        self.assertEqual(base64.b64decode(first['prompt_bytes_base64']), values['prompt'].encode())
        self.assertEqual(self.workflow.read_bytes(), workflow_before)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(values, {'prompt': 'opaque λ\nverbatim ', 'seed': 3407})
        with self.assertRaises(ProfileError):
            write_job(self.source, first)
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_hashes_detect_changes(self):
        job = prepare_job(self.path)
        self.assertEqual(job['base_workflow_hash'], digest(self.workflow.read_bytes()))
        self.assertEqual(job['resolved_workflow_hash'], digest(canonical(job['workflow'])))
        self.assertEqual(job['profile_hash'], profile_hash(self.profile))
        self.profile['capability'] = 'changed'
        self.path.write_text(json.dumps(self.profile))
        with self.assertRaisesRegex(ProfileError, 'hash mismatch'):
            load_profile(self.path)
        self.save_profile()
        self.workflow.write_bytes(self.workflow.read_bytes() + b' ')
        with self.assertRaisesRegex(ProfileError, 'workflow hash'):
            prepare_job(self.path)

    def test_reconstruction_policies(self):
        for policy in ['preserve_visible_geometry', 'allow_hidden_region_reconstruction']:
            self.profile['reconstruction_policy'] = policy
            self.save_profile()
            self.assertEqual(prepare_job(self.path)['reconstruction_policy'], policy)
        self.profile['reconstruction_policy'] = 'invented'
        self.save_profile()
        with self.assertRaisesRegex(ProfileError, 'reconstruction_policy'):
            load_profile(self.path)

    def test_missing_models_assets_and_bindings(self):
        inventory = self.root / 'inventory.json'
        inventory.write_text(json.dumps({'models': [], 'assets': ['input.png']}))
        with self.assertRaisesRegex(ProfileError, 'Missing model'):
            prepare_job(self.path)
        inventory.write_text(json.dumps({'models': ['opaque-model'], 'assets': []}))
        with self.assertRaisesRegex(ProfileError, 'Missing backend asset'):
            prepare_job(self.path)
        inventory.write_text(json.dumps({'models': ['opaque-model'], 'assets': ['input.png']}))
        del self.profile['bindings']['prompt']['default']
        self.save_profile()
        with self.assertRaisesRegex(ProfileError, 'Missing binding value'):
            prepare_job(self.path)
        self.source.unlink()
        with self.assertRaisesRegex(ProfileError, 'asset'):
            prepare_job(self.path, {'prompt': 'text'})

    def test_binding_types_ranges_and_undeclared_values(self):
        for values in [{'seed': True}, {'seed': -1}, {'seed': 2**64}, {'scale': float('inf')}, {'unknown': 1}, {'source': 'other.png'}, {'model': 'other-model'}]:
            with self.subTest(values=values), self.assertRaises(ProfileError):
                prepare_job(self.path, values)
        self.profile['bindings']['scale']['maximum'] = 1
        self.save_profile()
        with self.assertRaisesRegex(ProfileError, 'above maximum'):
            prepare_job(self.path, {'scale': 2})

    def test_metadata_lineage_and_backend_neutrality(self):
        local = prepare_job(self.path, parent_job='parent-1')
        cloud = prepare_job(self.path, backend='comfy-cloud', parent_job='parent-1')
        self.assertEqual(local['workflow_bytes_base64'], cloud['workflow_bytes_base64'])
        self.assertEqual(local.keys(), cloud.keys())
        self.assertEqual(local['lineage']['parent_job_id'], 'parent-1')
        self.assertEqual(local['source_hashes']['source'], digest(self.source.read_bytes()))
        for key in ['profile_id', 'profile_hash', 'base_workflow_hash', 'resolved_workflow_hash', 'resolved_bindings', 'prompt_bytes_base64', 'numeric_settings', 'backend', 'created_at', 'output_nodes', 'record_hash']:
            self.assertIn(key, local)

    def test_exact_replay_does_not_require_profile_or_workflow(self):
        original = prepare_job(self.path)
        record = self.root / 'job.json'
        write_job(record, original)
        self.path.unlink()
        self.workflow.unlink()
        self.assertEqual(replay_job(record), original)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ProfileError, 'Replay asset changed'):
            replay_job(record)

    def test_replay_detects_tampering(self):
        job = prepare_job(self.path)
        job['resolved_bindings']['seed'] = 2
        record = self.root / 'job.json'
        write_job(record, job)
        with self.assertRaisesRegex(ProfileError, 'record hash'):
            replay_job(record)

    def test_cli_prepare_replay_and_dry_run_never_submit(self):
        record = self.root / 'job.json'
        with patch.object(cli, 'request_json', side_effect=AssertionError('network forbidden')), patch.object(cli, 'upload_image', side_effect=AssertionError('upload forbidden')), patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(cli.main(['jobs', 'prepare', str(self.path), '--output', str(record)]), 0)
            self.assertEqual(cli.main(['jobs', 'replay', str(record)]), 0)
            self.assertEqual(cli.main(['generate', '--profile', str(self.path), '--dry-run']), 0)
            self.assertEqual(cli.main(['profiles', 'inspect', str(self.path)]), 0)
            self.assertEqual(cli.main(['profiles', 'validate', str(self.path)]), 0)
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
            cli.main(['generate', '--profile', str(self.path)])
        self.assertFalse(read_json(record)['submitted'])

    def test_existing_shoes_profile_references_frozen_workflow(self):
        path, p = load_profile('firered_shoes_v1')
        raw = (path.parent / p['workflow']['path']).read_bytes()
        self.assertEqual(digest(raw), p['workflow']['sha256'])
        self.assertEqual(p['capability'], 'footwear_change')
        self.assertEqual(p['bindings']['seed']['default'], 3407)
        graph = json.loads(raw)
        for binding in p['bindings'].values():
            self.assertEqual(graph[binding['node']]['inputs'][binding['input']], binding['default'])


if __name__ == '__main__':
    unittest.main()
