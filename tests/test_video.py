"""Identity continuity, missing targets, bidirectional propagation and RAM transport."""
from pathlib import Path
import sys
import threading
import time
import unittest
import zlib
import base64
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'daemon'))
from video_backend import propagate
from sam3_daemon import Engine
from server_transport import Server
from protocol import request


def outputs(ids, areas):
    masks = np.zeros((len(ids), 4, 6), bool)
    for mask, size in zip(masks, areas):
        mask.flat[:size] = True
    return dict(out_obj_ids=np.array(ids), out_binary_masks=masks, out_probs=np.full(len(ids), .9))


class Predictor:
    def __init__(self):
        self.closed = False
        self.stream_request = None
    def handle_request(self, r):
        if r['type'] == 'start_session':
            assert all(isinstance(im, Image.Image) for im in r['resource_path'])
            return {'session_id': 'session'}
        if r['type'] == 'close_session':
            self.closed = True
            return {}
        assert r['frame_index'] == 1
        return dict(frame_index=1, outputs=outputs([10, 20], [10, 4]))
    def handle_stream_request(self, r):
        self.stream_request = r
        yield dict(frame_index=1, outputs=outputs([10, 20], [1, 20]))
        # Rank and output order both change, but selected ID 10 must remain selected.
        yield dict(frame_index=2, outputs=outputs([20, 10], [20, 2]))
        # Earlier frame contains only the other person. Must return black.
        yield dict(frame_index=0, outputs=outputs([20], [24]))


class FakeVideoBackend:
    def release_image(self):
        pass
    def predict(self, *args):
        raise AssertionError('Analyze must never use independent image detection')
    def track_video(self, images, reference, prompt, confidence, index, cancelled):
        yield from propagate(Predictor(), images, reference, prompt, confidence, index, cancelled)


class VideoTests(unittest.TestCase):
    def test_reference_identity_survives_rank_change_and_absence(self):
        predictor = Predictor()
        frames = [Image.new('RGB', (6, 4)) for _ in range(3)]
        result = {i: (m, meta) for i, m, meta in propagate(predictor, frames, 1, 'person', .45, 0, lambda: False)}
        self.assertEqual([int(result[i][0].sum()) for i in range(3)], [0, 10, 2])
        self.assertEqual({result[i][1]['object_id'] for i in result}, {10})
        self.assertEqual(predictor.stream_request['propagation_direction'], 'both')
        self.assertTrue(predictor.closed)

    def test_reference_selection_error_closes_session(self):
        predictor = Predictor()
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            list(propagate(predictor, [Image.new('RGB', (6, 4))]*3, 1, 'person', .45, 2, lambda: False))
        self.assertTrue(predictor.closed)
        self.assertIsNone(predictor.stream_request)

    def test_cancel_closes_session(self):
        predictor = Predictor()
        cancel = threading.Event()
        stream = propagate(predictor, [Image.new('RGB', (6, 4))]*3, 1, 'person', .45, 0, cancel.is_set)
        next(stream)
        cancel.set()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            next(stream)
        self.assertTrue(predictor.closed)

    def test_video_wire_upload_track_download_and_abort(self):
        engine = Engine(FakeVideoBackend())
        server = Server(engine, port=0, idle_timeout=30)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        deadline = time.monotonic() + 3
        while not server.port and time.monotonic() < deadline:
            time.sleep(.01)
        def control(cmd, **kwargs):
            return request(dict(cmd=cmd, token='test', **kwargs), port=server.port)[0]
        rgb = np.zeros((4, 6, 3), dtype='<f4').tobytes()
        def infer(frame):
            return request(dict(cmd='infer', capture_token='test', frame=frame, width=6, height=4,
                                prompt='person', input_colorspace='srgb'), rgb, port=server.port)
        try:
            self.assertEqual(request(dict(cmd='info'), port=server.port)[0]['video_tracking'], 1)
            control('video_begin', first=100, last=102, reference=101)
            for f in range(100, 103):
                self.assertFalse(np.frombuffer(infer(f)[1], '<f4').any())
            control('video_start')
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = control('video_status')
                if status['state'] != 'running':
                    break
                time.sleep(.01)
            self.assertEqual(status['state'], 'ready', status)
            for f, area in zip(range(100, 103), (0, 10, 2)):
                meta, payload = infer(f)
                self.assertTrue(meta['temporal_tracking'])
                self.assertEqual(meta['object_id'], 10)
                self.assertEqual(np.frombuffer(payload, '<f4').reshape(4, 4, 6)[0].sum(), area)
                snapshot = control('video_frame', frame=f)
                raw = zlib.decompress(base64.b64decode(snapshot['data']))
                self.assertEqual(np.unpackbits(np.frombuffer(raw[:3], np.uint8)).sum(), area)
            control('video_abort')
            self.assertFalse(engine.video.jobs)
            with self.assertLogs('sam3', level='ERROR'), self.assertRaisesRegex(RuntimeError, 'expired'):
                infer(100)
        finally:
            server.stop.set()
            thread.join(3)


if __name__ == '__main__':
    unittest.main()
