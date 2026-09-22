from pathlib import Path
import sys
import unittest
import base64
import io
import json
import struct
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'daemon'))
from ram_cache import RamCache
from sam3_daemon import Engine
from test_engine import FakeBackend


class RamTests(unittest.TestCase):
    def test_snapshot_can_solve_without_engine(self):
        from solve_worker import run
        engine = Engine(FakeBackend())
        engine.control({'cmd': 'capture_begin', 'token': 'snapshot'})
        blobs = []
        for frame in (1, 2, 3):
            pixels = np.zeros((16, 20, 3), dtype='<f4')
            pixels[3:8, 2+frame:7+frame, 0] = 1
            engine.infer({'width': 20, 'height': 16, 'prompt': 'face', 'capture_token': 'snapshot',
                          'frame': frame, 'input_colorspace': 'srgb'}, pixels.tobytes())
            snapshot = engine.control({'cmd': 'capture_frame', 'token': 'snapshot', 'frame': frame})
            blobs.append(base64.b64decode(snapshot['data']))
        self.assertEqual(engine.inferences, 3)
        with self.assertRaises(ValueError):
            engine.control({'cmd': 'capture_frame', 'token': 'snapshot', 'frame': 4})
        del engine
        job = {'width': 20, 'height': 16, 'frames': [1, 2, 3], 'reference_frame': 1, 'tracking_mode': 'bbox'}
        packet = json.dumps(job).encode() + b'\n' + b''.join(struct.pack('<I', len(b)) + b for b in blobs)
        result = run(io.BytesIO(packet))
        self.assertEqual(result['frames'][2]['translate'], [2.0, 0.0])
        self.assertTrue(all(row['mask_path'] is None for row in result['frames']))
        with self.assertRaises(ValueError):
            run(io.BytesIO(packet[:-1]))

    def test_preview_reused_and_controls_invalidate(self):
        engine = Engine(FakeBackend())
        pixels = np.zeros((4, 6, 3), dtype='<f4')
        pixels[1:3, 2:5, 0] = 1
        header = {'cmd': 'infer', 'width': 6, 'height': 4, 'prompt': 'face', 'input_colorspace': 'srgb'}
        meta, initial = engine.infer(header, pixels.tobytes())
        self.assertFalse(meta['cache_hit'])
        engine.control({'cmd': 'capture_begin', 'token': 'abc'})
        meta, again = engine.infer(dict(header, capture_token='abc', frame=1), pixels.tobytes())
        self.assertTrue(meta['cache_hit'])
        self.assertEqual(initial, again)
        self.assertEqual(engine.inferences, 1)
        for update in ({'prompt': 'person'}, {'confidence': 0.9}, {'object_index': 1}, {'cache_revision': 1}, {'input_colorspace': 'linear_srgb'}):
            meta, _ = engine.infer(dict(header, **update), pixels.tobytes())
            self.assertFalse(meta['cache_hit'])
        pixels[0, 0, 0] = 1
        meta, _ = engine.infer(header, pixels.tobytes())
        self.assertFalse(meta['cache_hit'])
        engine.control({'cmd': 'capture_abort', 'token': 'abc'})

    def test_memory_bbox_tracking_without_files(self):
        engine = Engine(FakeBackend())
        engine.control({'cmd': 'capture_begin', 'token': 'track'})
        for frame in (1, 2, 3):
            pixels = np.zeros((16, 20, 3), dtype='<f4')
            pixels[3:8, 2+frame:7+frame, 0] = 1
            engine.infer({'width': 20, 'height': 16, 'prompt': 'face', 'capture_token': 'track',
                          'frame': frame, 'input_colorspace': 'srgb'}, pixels.tobytes())
        reply = engine.control({'cmd': 'capture_finish', 'token': 'track', 'first': 1, 'last': 3,
                                'settings': {'reference_frame': 1, 'tracking_mode': 'bbox'}})
        self.assertEqual(reply['result']['frames'][2]['translate'], [2.0, 0.0])
        self.assertTrue(all(row['mask_path'] is None for row in reply['result']['frames']))
        self.assertFalse(engine.cache.captures)

    def test_pinned_cache_never_silently_spills_to_disk(self):
        cache = RamCache(max_bytes=550)
        mask = np.zeros((4, 4), dtype=bool)
        gray = np.zeros((4, 4), dtype=np.uint8)
        cache.begin('one')
        cache.put('a', mask, gray, {})
        cache.pin('one', 1, 'a')
        with self.assertRaises(MemoryError):
            cache.put('b', mask, gray, {})
        cache.end('one')
        cache.put('b', mask, gray, {})
        self.assertIsNone(cache.get('a'))
        self.assertLessEqual(cache.bytes, cache.max_bytes)


if __name__ == '__main__':
    unittest.main()
