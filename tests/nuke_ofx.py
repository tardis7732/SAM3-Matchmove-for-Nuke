"""Configured Nuke smoke check with synthetic RAM data, no media/model inference.

Run with Nuke -ti after configure/install. Does not save a Nuke script.
"""
from pathlib import Path
import sys
import zlib
import nuke

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nuke'))
import sam3_unified as unified
import sam3_memory as memory

assert Path(unified.__file__).resolve().parent == ROOT / 'nuke'
nuke.root()['first_frame'].setValue(1)
nuke.root()['last_frame'].setValue(3)
plate = nuke.nodes.Constant()
plate['format'].setValue(nuke.addFormat('64 64 1 SAM3_Test'))
plate['selected'].setValue(True)
group = unified.create()
group.setInput(0, plate)
unified.reset_range(group)
for name, value in (('first_frame', 1), ('last_frame', 3), ('reference_frame', 1)):
    group[name].setValue(value)
group['smoothing_window'].setValue(1)
assert group['outputMode'].value() == 'Mask'
assert not group.knob('adjustments')
assert group.knobs()['smoothing_window'].getFlag(nuke.STARTLINE)
assert group.knobs()['crop_margin'].getFlag(nuke.STARTLINE)
for name in ('resetRange', 'analyzeRange', 'solveMotion', 'exportTracker'):
    assert not group[name].getFlag(0x2000000)
engine = group.node('INFERENCE')
engine['autoStart'].setValue(False)
assert group.sample('rgba.alpha', 32, 32) == 0

# Build bit-packed masks and grayscale frames in the same RAM format as Analyze.
frames = {}
for frame in (1, 2, 3):
    bits = bytearray(64 * 64 // 8)
    gray = bytearray(64 * 64)
    for y in range(12, 28):
        for x in range(8 + frame, 24 + frame):
            i = y * 64 + x
            bits[i // 8] |= 1 << (7 - i % 8)
            gray[i] = (x * 11 + y * 7) % 256
    frames[frame] = zlib.compress(bytes(bits + gray))
token = 'synthetic-smoke-batch'
memory.BATCHES[token] = {'width': 64, 'height': 64, 'frames': frames,
                         'bytes': sum(map(len, frames.values()))}
engine['playbackToken'].setValue(token)
result = unified.solve(group)
assert len(result['frames']) == 3
assert result['frames'][-1]['translate'] == [2.0, 0.0], result['frames'][-1]['translate']
assert group['analysisReady'].value()
for choice in unified.EXPORT_CHOICES:
    made = unified.export_output(group, choice)
    assert made, choice
assert not any(n.knob('sam3Derived') for n in group.nodes())
print('SAM3_OFX_SMOKE_OK: OFX load, Properties flags, RAM Solve and all five exports')
