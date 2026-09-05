"""Build a local review sheet; never rewrite inputs or the raw result."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / 'manifest.json').read_text())
execution = json.loads((HERE / 'cloud_execution.json').read_text())
output = execution['outputs'][0]
assets = [(Path(a['path']), a['sha256']) for a in manifest['inputs']]
assets.append((HERE / output['local_filename'], output['sha256']))
labels = ['Original target', 'Identity reference', 'Raw generated output']
crops = [(520, 50, 770, 340), (280, 110, 760, 740), (540, 35, 795, 335)]
sheet = Image.new('RGB', (1500, 1030), '#eeeeee')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('DejaVuSans.ttf', 22)
for column, ((path, expected), label, crop) in enumerate(zip(assets, labels, crops)):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    with Image.open(path) as source:
        full = ImageOps.contain(source.convert('RGB'), (480, 460))
        detail = ImageOps.contain(source.crop(crop).convert('RGB'), (480, 450))
    x = column * 500
    draw.text((x + 10, 12), label, fill='black', font=font)
    sheet.paste(full, (x + (500-full.width)//2, 50+(460-full.height)//2))
    draw.text((x + 10, 530), 'Face / hair detail (resized crop)', fill='black', font=font)
    sheet.paste(detail, (x + (500-detail.width)//2, 570+(450-detail.height)//2))
sheet.save(HERE / 'contact_sheet.local.png')
print(HERE / 'contact_sheet.local.png')
