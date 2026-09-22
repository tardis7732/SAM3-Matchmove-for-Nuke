"""Deterministic tests of selection, color conversion, and wire framing."""
from pathlib import Path
import json
import socket
import struct
import sys
import threading
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'daemon'))
from backend import prepare_rgb, select_mask
from protocol import ProtocolError, read_request, request
from sam3_daemon import Engine
from server_transport import Server


class FakeBackend:
    def predict(self, rgb, prompt, confidence, object_index):
        return np.repeat((rgb[:, :, 0] > 0.5)[None], 4, axis=0).astype('<f4'), {'detections': 1}


class EngineTests(unittest.TestCase):
    def test_area_order_and_missing(self):
        masks = np.zeros((2, 3, 5), bool)
        masks[0, 0, 0] = True
        masks[1, 1:, 2:] = True
        self.assertEqual(select_mask(masks, [0.9, 0.7], 0, 3, 5)[3].sum(), 6)
        self.assertEqual(select_mask(masks, [0.9, 0.7], 1, 3, 5)[3].sum(), 1)
        with self.assertRaises(ValueError):
            select_mask(masks, [0.9, 0.7], -1, 3, 5)
        self.assertEqual(select_mask(masks, [0.9, 0.7], 2, 3, 5).sum(), 0)
        self.assertEqual(select_mask(np.empty((0, 3, 5)), [], 0, 3, 5).sum(), 0)

    def test_transfer_and_nonfinite_values(self):
        rgb = np.array([[[0, 0.0031308, 1], [np.nan, np.inf, -np.inf]]], dtype=np.float32)
        result = prepare_rgb(rgb, 'linear_srgb')
        np.testing.assert_allclose(result[0, 0], [0, 0.0404499, 1], atol=1e-6)
        np.testing.assert_allclose(result[0, 1], [0, 1, 0], atol=1e-6)
        with self.assertRaises(ValueError):
            prepare_rgb(rgb, 'ACEScg')

    def test_invalid_controls_rejected_before_inference(self):
        engine = Engine(FakeBackend())
        header = {'width': 1, 'height': 1, 'prompt': 'face'}
        for update in ({'prompt': ''}, {'prompt': 3}, {'confidence': float('nan')},
                       {'confidence': -1}, {'object_index': -2}, {'object_index': -1}, {'object_index': True}):
            with self.subTest(update=update), self.assertRaises(ProtocolError):
                engine.infer(dict(header, **update), b'\0' * 12)
        self.assertEqual(engine.count, 0)

    def test_incomplete_or_inconsistent_payload(self):
        for width, data in [(1, b'123'), (20000, b'')]:
            a, b = socket.socketpair()
            try:
                body = json.dumps({'cmd': 'infer', 'width': width, 'height': 1}).encode()
                a.sendall(b'S3Q1' + struct.pack('<I', len(body)) + body + struct.pack('<I', len(data)) + data)
                with self.assertRaises(ProtocolError):
                    read_request(b)
            finally:
                a.close()
                b.close()

    def test_loopback_preserves_non_square_pixel_order_and_survives_error(self):
        engine = Engine(FakeBackend())
        server = Server(engine, port=0, idle_timeout=10)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        import time
        deadline = time.monotonic() + 5
        while server.port == 0 and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            rgb = np.zeros((3, 5, 3), dtype='<f4')
            rgb[0, 1, 0] = rgb[2, 4, 0] = 1
            header = {'cmd': 'infer', 'width': 5, 'height': 3, 'prompt': 'face', 'input_colorspace': 'srgb'}
            with self.assertLogs('sam3', level='ERROR'), self.assertRaises(RuntimeError):
                request(dict(header, prompt=''), rgb.tobytes(), port=server.port)
            meta, payload = request(header, rgb.tobytes(), port=server.port)
            result = np.frombuffer(payload, dtype='<f4').reshape(4, 3, 5)
            np.testing.assert_array_equal(result[3], rgb[:, :, 0])
            self.assertEqual(meta['requests'], 1)
            info, _ = request({'cmd': 'info'}, port=server.port)
            self.assertEqual(info['engine'], 'SAM3 Mask')
        finally:
            server.stop.set()
            thread.join(3)


if __name__ == '__main__':
    unittest.main()
