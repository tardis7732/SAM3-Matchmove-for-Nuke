"""Native Transform regression: motion menu, stabilized crop and inverse export.

Run with configured Nuke -ti. No media, model inference or saved Nuke script.
"""
import base64
import json
import math
from pathlib import Path
import sys
import nuke

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nuke'))
import sam3_unified as unified
assert Path(unified.__file__).resolve().parent == ROOT / 'nuke'

plate = nuke.nodes.Constant()
plate['format'].setValue(nuke.addFormat('1280 720 1 SAM3_Crop_Test'))
group = unified.create()
group.setInput(0, plate)
assert list(group['tracking_mode'].values()) == unified.MOTION_CHOICES
assert group['tracking_mode'].value() == 'Translation + Scale'

def point(matrix, x, y):
    v = (x, y, 0., 1.)
    return [sum(matrix[c*4+r] * v[c] for c in range(4)) for r in (0, 1)]

def near(a, b):
    assert all(abs(x-y) < 2e-3 for x, y in zip(a, b)), (a, b)

for label, scale, angle in (('Translation', 1., 0.), ('Translation + Scale', .418, 0.),
                            ('Translation + Rotation', 1., 37.)):
    group['tracking_mode'].setValue(label)
    for width, height in ((720, 720), (1280, 720), (720, 1280)):
        group['aspect_lock'].setValue(False)
        group['output_width'].setValue(width)
        group['output_height'].setValue(height)
        rows = [dict(frame=f, center=center, translate=[center[0]-500, center[1]-300],
                     scale=s, rotate=r, confidence=1., crop_box=[center[0]-102, center[1]-102, center[0]+102, center[1]+102])
                for f, center, s, r in ((1, [500., 300.], 1., 0.), (2, [620., 430.], scale, angle))]
        data = dict(schema_version=1, width=1280, height=720, pixel_aspect=1.,
                    reference_frame=1, reference_center=[500., 300.], reference_box=[450.,250.,550.,350.], frames=rows)
        group['reference_frame'].setValue(1)
        group['analysisData'].setValue('b64:' + base64.b64encode(json.dumps(data).encode()).decode())
        group['analysisReady'].setValue(True)
        crop = unified.export_output(group, 'Plate Stabilize Crop')
        uncrop = unified.export_output(group, 'Generated Crop Matchmove')
        for f, center, s, r in ((1, [500.,300.], 1.,0.), (2, [620.,430.], scale,angle)):
            nuke.frame(f)
            near(crop[0]['scale'].value(), [204/min(width,height)*s]*2)
            assert abs(crop[0]['rotate'].value()-r) < 1e-6
            cm, um = crop[0]['matrix'].value(), uncrop[0]['matrix'].value()
            near(point(cm, *center), [width/2, height/2])
            near(point(um, width/2, height/2), center)
            near(point(um, *point(cm, 430.,280.)), [430.,280.])
        assert crop[1]['box_width'].value() == width
        assert crop[1]['box_height'].value() == height
        for node in crop + uncrop:
            nuke.delete(node)
print('NUKE_CROP_OK: three modes, three aspects, native inverse matrices and fixed output dimensions')
