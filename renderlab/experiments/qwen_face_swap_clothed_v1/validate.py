"""Static validation only: reads files and node schema ASTs, never imports inference code."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/codyjackson/PycharmProjects/renderlab')
HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / 'manifest.json').read_text())
workflow = json.loads((HERE / 'workflow.json').read_text())
api = json.loads((HERE / 'api.json').read_text())

def literal(node):
    return ast.literal_eval(node)

files = ['nodes.py', 'comfy_extras/nodes_cfg.py', 'comfy_extras/nodes_qwen.py',
         'comfy_extras/nodes_flux.py', 'comfy_extras/nodes_model_advanced.py']
classes = {}
for filename in files:
    for c in ast.parse((ROOT / filename).read_text()).body:
        if isinstance(c, ast.ClassDef):
            classes[c.name] = c

def assignment(c, name):
    for a in classes[c].body:
        if isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in a.targets):
            return literal(a.value)
    for base in classes[c].bases:
        if isinstance(base, ast.Name) and base.id in classes:
            result = assignment(base.id, name)
            if result is not None:
                return result

sampler_ast = ast.parse((ROOT / 'comfy/samplers.py').read_text())
sampler_names = []
schedulers = []
for stmt in sampler_ast.body:
    if isinstance(stmt, ast.Assign):
        names = [n.id for n in stmt.targets if isinstance(n, ast.Name)]
        if 'KSAMPLER_NAMES' in names:
            sampler_names = literal(stmt.value)
        if 'SCHEDULER_HANDLERS' in names:
            schedulers = [literal(k) for k in stmt.value.keys]
assert 'euler' in sampler_names and 'simple' in schedulers

model_names = {k: [manifest['settings'][v]] for k, v in
               [('diffusion_models', 'unet'), ('text_encoders', 'clip'), ('vae', 'vae'), ('loras', 'lora')]}
env = {'folder_paths': SimpleNamespace(get_filename_list=lambda kind: model_names[kind]),
       'comfy': SimpleNamespace(samplers=SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=sampler_names, SCHEDULERS=schedulers))),
       's': SimpleNamespace(vae_list=lambda _: model_names['vae'])}
schemas = {}
type_names = dict(Model='MODEL', Clip='CLIP', Vae='VAE', Image='IMAGE', Conditioning='CONDITIONING',
                  Float='FLOAT', Boolean='BOOLEAN', String='STRING')
for name in sorted({n['class_type'] for n in api.values()}):
    c = classes[name]
    methods = {m.name: m for m in c.body if isinstance(m, ast.FunctionDef)}
    if 'define_schema' in methods:
        schema_call = next(s.value for s in methods['define_schema'].body if isinstance(s, ast.Return))
        kw = {k.arg: k.value for k in schema_call.keywords}
        inputs = {'required': {}, 'optional': {}}
        for call in kw['inputs'].elts:
            kind = type_names[call.func.value.attr]
            opts = {k.arg: literal(k.value) for k in call.keywords}
            inputs['optional' if opts.get('optional') else 'required'][literal(call.args[0])] = [kind, opts]
        outputs = [type_names[call.func.value.attr] for call in kw['outputs'].elts]
    elif name == 'LoadImage':
        # Dynamic inventory is the actual verified local files, not a remote input catalog.
        inputs = {'required': {'image': [[a['path'] for a in manifest['inputs']], {'image_upload': True}]}}
        outputs = assignment(name, 'RETURN_TYPES')
    else:
        expression = next(s.value for s in methods['INPUT_TYPES'].body if isinstance(s, ast.Return))
        inputs = eval(compile(ast.Expression(expression), '<schema-only>', 'eval'), env)
        outputs = assignment(name, 'RETURN_TYPES')
    schemas[name] = dict(inputs=inputs, outputs=outputs)

def validate_api(payload):
    count = 0
    for nid, n in payload.items():
        schema = schemas[n['class_type']]
        required = schema['inputs'].get('required', {})
        fields = {**required, **schema['inputs'].get('optional', {})}
        assert set(required) <= set(n['inputs']), (nid, 'missing required inputs')
        assert set(n['inputs']) <= set(fields), (nid, 'unknown input')
        for key, value in n['inputs'].items():
            field = fields[key]
            kind = field[0]
            opts = field[1] if len(field) > 1 else {}
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
                assert value[0] in payload, (nid, key, 'dangling link')
                upstream = schemas[payload[value[0]]['class_type']]['outputs']
                assert 0 <= value[1] < len(upstream), (nid, key, 'output slot')
                assert upstream[value[1]] == kind, (nid, key, 'type mismatch')
                count += 1
            elif isinstance(kind, (list, tuple)):
                assert value in kind, (nid, key, 'invalid combo', value)
            elif kind == 'STRING':
                assert isinstance(value, str), (nid, key)
            elif kind in ('INT', 'FLOAT', 'BOOLEAN'):
                assert (type(value) is int if kind == 'INT' else type(value) in (int, float) if kind == 'FLOAT' else type(value) is bool), (nid, key)
                if 'min' in opts:
                    assert value >= opts['min'], (nid, key)
                if 'max' in opts:
                    assert value <= opts['max'], (nid, key)
            else:
                raise AssertionError((nid, key, 'required connection is a literal'))
    visiting, done = set(), set()
    def visit(nid):
        assert nid not in visiting, 'cycle'
        if nid in done:
            return
        visiting.add(nid)
        for value in payload[nid]['inputs'].values():
            if isinstance(value, list):
                visit(value[0])
        visiting.remove(nid)
        done.add(nid)
    visit('472')
    assert done == set(payload), 'disconnected API nodes'
    assert payload['470']['inputs']['image'] == manifest['inputs'][0]['path'], 'target role'
    assert payload['471']['inputs']['image'] == manifest['inputs'][1]['path'], 'identity role'
    for nid in ['433:110', '433:111']:
        assert payload[nid]['inputs']['image1'] == ['433:117', 0], 'target conditioning'
        assert payload[nid]['inputs']['image2'] == ['471', 0], 'identity conditioning'
        assert 'image3' not in payload[nid]['inputs'], 'unused image3'
    assert payload['433:111']['inputs']['prompt'] == manifest['prompt'], 'positive prompt'
    assert payload['433:110']['inputs']['prompt'] == '', 'negative prompt'
    assert payload['433:117']['inputs']['image'] == ['470', 0]
    assert payload['433:88']['inputs']['pixels'] == ['433:117', 0]
    assert payload['433:3']['inputs']['positive'] == ['433:111', 0]
    assert payload['433:3']['inputs']['negative'] == ['433:110', 0]
    assert payload['433:3']['inputs']['latent_image'] == ['433:88', 0]
    for key, value in dict(seed=3407, steps=4, cfg=1, sampler_name='euler', scheduler='simple', denoise=1).items():
        assert payload['433:3']['inputs'][key] == value, key
    assert payload['433:66']['inputs']['model'] == ['433:89', 0], 'Lightning branch'
    assert payload['433:89']['inputs']['strength_model'] == 1
    assert payload['472']['inputs'] == dict(images=['433:8', 0], filename_prefix=manifest['output_prefix'])
    return count

def check_graph(nodes, edges, ins=(), outs=()):
    by_id = {n['id']: n for n in nodes}
    assert len(by_id) == len(nodes)
    by_id[-10] = dict(outputs=ins)
    by_id[-20] = dict(inputs=outs)
    ids = set()
    for edge in edges:
        if isinstance(edge, dict):
            lid, origin, oslot, target, tslot, kind = [edge[k] for k in ['id', 'origin_id', 'origin_slot', 'target_id', 'target_slot', 'type']]
        else:
            lid, origin, oslot, target, tslot, kind = edge
        assert lid not in ids
        ids.add(lid)
        src, dst = by_id[origin]['outputs'][oslot], by_id[target]['inputs'][tslot]
        assert src['type'] in (kind, '*') and dst['type'] in (kind, '*'), lid
        assert lid in src.get('links', src.get('linkIds', [])), ('source link index', lid)
        assert (lid in dst['linkIds'] if 'linkIds' in dst else dst['link'] == lid), ('target link index', lid)
    for node in nodes:
        for field in node.get('inputs', []):
            assert field.get('link') is None or field['link'] in ids
        for field in node.get('outputs', []):
            assert set(field.get('links') or []) <= ids
    return len(ids)

for asset in manifest['inputs']:
    assert hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest() == asset['sha256']
assert hashlib.sha256(Path(manifest['blueprint']['path']).read_bytes()).hexdigest() == manifest['blueprint']['sha256']
graph_links = check_graph(workflow['nodes'], workflow['links'])
for sg in workflow['definitions']['subgraphs']:
    graph_links += check_graph(sg['nodes'], sg['links'], sg['inputs'], sg['outputs'])
api_links = validate_api(api)

# These mutations must fail: the mistakes found in the old Cloud graph and role reversals.
mutations = [('433:111', 'image2', ['470', 0]), ('433:111', 'prompt', ''),
             ('472', 'images', ['470', 0]), ('433:88', 'pixels', ['999', 0]),
             ('433:3', 'scheduler', 'invalid'), ('433:3', 'steps', 20)]
for nid, key, bad in mutations:
    changed = copy.deepcopy(api)
    changed[nid]['inputs'][key] = bad
    try:
        validate_api(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError(('validator accepted mutation', nid, key))

report = dict(status='PASS_STATIC_LOCAL', graph_links=graph_links, api_nodes=len(api), api_links=api_links,
              api_inputs=sum(len(n['inputs']) for n in api.values()), negative_checks_rejected=len(mutations),
              checks=['image hashes and blueprint hash', 'every editor edge, port type and reciprocal index',
                      'every API input against local source schemas', 'required inputs, types, scalar ranges, enums',
                      'acyclic graph and all API nodes reachable from SaveImage', 'image roles and positive/negative prompt paths',
                      'target latent, turbo model branch and fixed sampler settings'],
              limitations=['No model weights loaded or inference performed.',
                           'Model filenames match blueprint declarations; Cloud installation is unverified.',
                           'LoadImage inventory validated against local paths; Cloud names require upload response binding.',
                           'Cloud discovery and revised credit estimate returned Auth required.',
                           'Static validation is not a runtime execution guarantee.'],
              schema_sources=[dict(path=f, sha256=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()) for f in files])
(HERE / 'local_node_schemas.json').write_text(json.dumps(schemas, indent=2)+'\n')
(HERE / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
