"""Run with Nuke -ti and SAM3_TEST_PORT pointing to a test video service."""
import os
import sys
from pathlib import Path
import nuke
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nuke'))
import sam3_unified as unified
import sam3_memory as memory
assert Path(unified.__file__).resolve().parent == ROOT / 'nuke'
assert Path(memory.__file__).resolve().parent == ROOT / 'nuke'

plate = nuke.nodes.Constant()
plate['format'].setValue(nuke.addFormat('64 64 1 SAM3_Video_Test'))
group = unified.create()
group.setInput(0, plate)
engine = group.node('INFERENCE')
engine['port'].setValue(int(os.environ['SAM3_TEST_PORT']))
engine['autoStart'].setValue(False)
for name, value in (('first_frame', 100), ('last_frame', 102), ('reference_frame', 101)):
    group[name].setValue(value)
group['smoothing_window'].setValue(1)
token = memory.analyze(group)
assert token in memory.BATCHES
assert memory.BATCHES[token]['temporal_tracking']
assert memory.BATCHES[token]['object_id'] == 10
for frame in range(100, 103):
    nuke.frame(frame)
    # Test service places a mask at x=8+i..23+i, y=12..27 (top-first).
    assert group.sample('rgba.alpha', 12, 44) == 1, (frame, group.sample('rgba.alpha', 12, 44))
    assert group.sample('rgba.alpha', 50, 44) == 0
try:
    memory.analyze(group)
except Exception as exc:
    assert 'Test tracking failure' in str(exc), exc
else:
    raise AssertionError('Failed Analyze must report its error')
assert engine['playbackToken'].value() == token
assert group.sample('rgba.alpha', 12, 44) == 1, 'Failed Analyze replaced the previous mask batch'
result = memory.solve(group)
assert result['frames'][-1]['translate'] == [1.0, 0.0], result['frames'][-1]
for choice in unified.EXPORT_CHOICES:
    assert unified.export_output(group, choice)
group['reference_frame'].setValue(100)
assert engine['playbackToken'].value() == ''
assert not group['analysisReady'].value()
print('NUKE_VIDEO_OK: Analyze, native RAM playback, Solve, all exports, reference invalidation')
