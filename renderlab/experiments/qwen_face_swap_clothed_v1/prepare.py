"""Prepare and statically validate the clothed baseline; never runs inference or HTTP."""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path('/home/codyjackson/PycharmProjects/renderlab')
OUT = Path(__file__).resolve().parent
SEED = 3407
PREFIX = 'renderlab_qwen_face_swap_clothed_v1_s3407'
TARGET = Path('/home/codyjackson/Downloads/img_00562__firered-image-edit-1.1__single-pass__621146938678618_00001_.png')
IDENTITY = ROOT / 'input/renderlab_afccbc5cc07c4f88b8f397e2104664f5.png'

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

request = json.loads((ROOT / 'renderlab/experiments/examples/qwen_face_swap_request.json').read_text())
prompt = request['target']['instruction']
assets = []
for path, role in [(TARGET, 'target_image'), (IDENTITY, 'face_identity')]:
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        assets.append(dict(path=str(path), role=role, sha256=sha(path), width=im.width, height=im.height, mode=im.mode))
request['source'] = {k: assets[0][k] for k in ('path', 'role', 'sha256')}
request['references'] = [{k: assets[1][k] for k in ('path', 'role', 'sha256')}]
write('request.json', request)

blueprint = ROOT / 'blueprints/Image Edit (Qwen 2509).json'
workflow = json.loads(blueprint.read_text())
group = workflow['nodes'][0]
sg = workflow['definitions']['subgraphs'][0]
nodes = {n['id']: n for n in sg['nodes']}
links = {l['id']: l for l in sg['links']}
external = [['470', 0], ['471', 0], None, prompt, SEED, True,
            nodes[37]['widgets_values'][0], nodes[38]['widgets_values'][0], nodes[39]['widgets_values'][0]]
group['widgets_values'] = [prompt, SEED, True, external[6], external[7], external[8], 'fixed']
group['inputs'][0]['link'] = 1001
group['inputs'][1]['link'] = 1002
group['outputs'][0]['links'] = [1003]
nodes[111]['widgets_values'] = [prompt]
nodes[3]['widgets_values'][0:2] = [SEED, 'fixed']

def image_loader(node_id, path, title, link_id, pos):
    return dict(id=node_id, type='LoadImage', title=title, pos=pos, size=[320, 300], flags={}, order=0, mode=0,
                inputs=[], outputs=[dict(name='IMAGE', type='IMAGE', links=[link_id]), dict(name='MASK', type='MASK', links=[])],
                properties={'Node name for S&R': 'LoadImage'}, widgets_values=[str(path), 'image'])

workflow['nodes'] += [image_loader(470, TARGET, 'image_1: target canvas', 1001, [-350, -150]),
                      image_loader(471, IDENTITY, 'image_2: face identity', 1002, [-350, 250]),
                      dict(id=472, type='SaveImage', pos=[600, -150], size=[320, 300], flags={}, order=4, mode=0,
                           inputs=[dict(name='images', type='IMAGE', link=1003)], outputs=[],
                           properties={'Node name for S&R': 'SaveImage'}, widgets_values=[PREFIX])]
workflow['links'] = [[1001, 470, 0, 433, 0, 'IMAGE'], [1002, 471, 0, 433, 1, 'IMAGE'], [1003, 433, 0, 472, 0, 'IMAGE']]
workflow['last_node_id'] = 472
workflow['last_link_id'] = 1003

# Resolve only the blueprint's primitive controls and its chosen turbo branches.
def source(link_id):
    edge = links[link_id]
    nid = edge['origin_id']
    if nid == -10:
        return external[edge['origin_slot']]
    node = nodes[nid]
    if node['type'].startswith('Primitive'):
        incoming = node['inputs'][0]['link']
        return source(incoming) if incoming is not None else node['widgets_values'][0]
    if node['type'] == 'ComfySwitchNode':
        enabled = source(node['inputs'][2]['link'])
        assert type(enabled) is bool
        return source(node['inputs'][int(enabled)]['link'])
    return [f'433:{nid}', edge['origin_slot']]

api = {}
for nid, node in nodes.items():
    if node['type'].startswith('Primitive') or node['type'] in ('ComfySwitchNode', 'MarkdownNote'):
        continue
    widget_names = [i['name'] for i in node['inputs'] if 'widget' in i]
    values = list(node.get('widgets_values', []))
    if node['type'] == 'KSampler':
        del values[1]  # frontend control_after_generate is not an API input
    assert len(widget_names) == len(values), (nid, widget_names, values)
    inputs = dict(zip(widget_names, values))
    for field in node['inputs']:
        if field['link'] is not None:
            value = source(field['link'])
            if value is not None:
                inputs[field['name']] = value
            else:
                inputs.pop(field['name'], None)
    api[f'433:{nid}'] = dict(class_type=node['type'], inputs=inputs)
api['470'] = dict(class_type='LoadImage', inputs=dict(image=str(TARGET)))
api['471'] = dict(class_type='LoadImage', inputs=dict(image=str(IDENTITY)))
api['472'] = dict(class_type='SaveImage', inputs=dict(images=source(110), filename_prefix=PREFIX))

write('workflow.json', workflow)
write('api.json', api)
experiment = json.loads((ROOT / 'renderlab/experiments/qwen_face_swap_v0.json').read_text())
experiment.update(workflow='renderlab/experiments/qwen_face_swap_clothed_v1/workflow.json',
                  authoritative_blueprint='blueprints/Image Edit (Qwen 2509).json',
                  reference_ids=['img_00562_clothed_firered_621146938678618', 'renderlab_afccbc5cc07c4f88b8f397e2104664f5'],
                  input_protocol=dict(image_1='img_00562 clothed derivative: target composition/body/scene',
                                      image_2='clothed synthetic adult portrait: face identity', image_3='unused'),
                  acceptance=request['acceptance'],
                  thresholds=dict(source_identity_transfer=4, target_body_preservation=5, target_pose_preservation=5,
                                  target_scene_preservation=4, face_boundary_quality=4, instruction_adherence=4, artifact_severity_max=2),
                  decision_rule='Pass when source_identity_transfer, target_scene_preservation, face_boundary_quality, and instruction_adherence are at least 4; target_body_preservation and target_pose_preservation are 5; and artifact_severity is at most 2.',
                  status='prepared_not_executed', validation_seed=SEED,
                  input_manifest='renderlab/experiments/qwen_face_swap_clothed_v1/manifest.json')
experiment['metric_mapping'] = dict(identity_preservation='source_identity_transfer', morphology_preservation='target_body_preservation', pose_preservation='target_pose_preservation')
if 'instruction_adherence' not in experiment['metrics']:
    experiment['metrics'].append('instruction_adherence')
write('experiment.json', experiment)
write('manifest.json', dict(schema='renderlab.clothed-face-swap-validation.v1',
      project_commit='a27dc7db32b07b391d5624a4a4784ecfa0122996', branch='dev',
      blueprint=dict(path=str(blueprint), sha256=sha(blueprint)), inputs=assets,
      source_semantics='editable_canvas', image_3=None, mask=None, prompt=prompt, negative_prompt='',
      seed=SEED, control_after_generate='fixed', output_prefix=PREFIX,
      expected_first_output=PREFIX+'_00001_.png', planned_output_dimensions=[1184, 880], output_counter_note='SaveImage chooses an available counter; the exact suffix is assigned at execution.',
      settings=dict(unet=external[6], clip=external[7], vae=external[8],
                    lora=nodes[89]['widgets_values'][0], lora_strength=1, turbo=True,
                    steps=4, cfg=1, sampler='euler', scheduler='simple', denoise=1, model_shift=3, cfg_norm_strength=1),
      acceptance=request['acceptance'], cloud_input_bindings=None,
      cloud_status='Not uploaded. Replace only LoadImage image values with returned upload names after approval.',
      visual_review='Target: opaque white T-shirt with lower body covered, clear adult face. Reference: clothed synthetic adult portrait, clear face.',
      provenance='Target PNG input is_changed matches original img_00562 SHA-256 8b22ad2084cda890622612e5cf02aa5e97c18a7244b703cb8c906240f78e8c68. Reference PNG contains RealVisXL generation metadata; adulthood is a visual assessment, not an age document.'))
print(json.dumps(assets, indent=2))
print('Prepared', len(api), 'API nodes')
