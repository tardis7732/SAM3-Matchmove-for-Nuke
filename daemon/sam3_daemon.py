"""Loopback SAM3 image segmentation service. Requests are serialized on the GPU."""
import argparse
import base64
import zlib
import logging
import math
import hashlib
import os
import threading
import time
if os.name == 'nt':
    import ctypes
    # Nuke's inherited DLL search directory must not override PyTorch's DLLs.
    if not ctypes.windll.kernel32.SetDllDirectoryW(None):
        raise ctypes.WinError()
import numpy as np
from backend import Sam3Backend, prepare_rgb
from protocol import PORT, ProtocolError, dimensions
from server_transport import Server
from ram_cache import RamCache


class Engine:
    def __init__(self, backend, cache_bytes=512 * 1024 * 1024):
        self.backend = backend
        self.lock = threading.Lock()
        self.count = 0
        self.state = 'ready; model loads on first inference'
        self.last_error = None
        self.cache = RamCache(cache_bytes)
        self.inferences = 0
        self.cache_hits = 0

    def info(self):
        return {'engine': 'SAM3 Mask', 'protocol': 1, 'pid': os.getpid(), 'state': self.state,
                'requests': self.count, 'last_error': self.last_error,
                'gpu_metrics': getattr(self.backend, 'metrics', {}), 'ram_analysis': True,
                'ram_snapshot': True, 'inferences': self.inferences, 'cache_hits': self.cache_hits,
                'cache_frames': len(self.cache.entries), 'cache_bytes': self.cache.bytes,
                'cache_limit_bytes': self.cache.max_bytes}

    def control(self, header):
        token = header.get('token', '')
        with self.lock:
            if header['cmd'] == 'capture_begin':
                self.cache.begin(token)
                return {'ok': True}
            if header['cmd'] == 'capture_abort':
                self.cache.end(token)
                return {'ok': True}
            if header['cmd'] == 'capture_frame':
                frame = header.get('frame')
                if type(frame) is not int:
                    raise ProtocolError('Invalid frame')
                capture = self.cache.captures.get(token)
                item = self.cache.entries.get(capture['frames'].get(frame)) if capture else None
                if item is None:
                    raise ValueError('Frame %s is missing from RAM; Analyze again' % frame)
                h, w = item['shape']
                # One bounded frame per reply; no images or tracking files on disk.
                data = zlib.compress(item['packed'] + item['gray'], 1)
                return {'frame': frame, 'width': w, 'height': h,
                        'data': base64.b64encode(data).decode('ascii')}
            if header['cmd'] != 'capture_finish':
                raise ProtocolError('Unsupported cache command')
            try:
                first, last = header.get('first'), header.get('last')
                if type(first) is not int or type(last) is not int or not 0 <= last-first < 10000:
                    raise ProtocolError('Invalid frame range (maximum 10000 frames)')
                frames = self.cache.frames(token, first, last)
                h, w = frames[0]['mask'].shape
                from memory_tracking import track_memory
                job = dict(header.get('settings', {}), frames=frames, width=w, height=h)
                result = track_memory(job)
                return {'result': result, 'cache': self.info()}
            finally:
                self.cache.end(token)

    def infer(self, header, data):
        w, h = dimensions(header)
        prompt = header.get('prompt', '')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 4096:
            raise ProtocolError('Enter target text (maximum 4096 UTF-8 bytes)')
        threshold = header.get('confidence', 0.45)
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ProtocolError('Confidence must be between 0 and 1')
        index = header.get('object_index', 0)
        if type(index) is not int or not 0 <= index <= 1000:
            raise ProtocolError('Object index must be 0..1000')
        if len(data) != w * h * 12:
            raise ProtocolError('RGB payload length mismatch')
        encoding = header.get('input_colorspace', 'linear_srgb')
        revision = header.get('cache_revision', 0)
        if type(revision) is not int or revision < 0:
            raise ProtocolError('Invalid cache revision')
        key = (w, h, hashlib.sha256(data).digest(), prompt.strip(), float(threshold), index, encoding, revision)
        rgb = np.frombuffer(data, dtype='<f4').reshape(h, w, 3)
        rgb = prepare_rgb(rgb, encoding)
        with self.lock:
            started = time.monotonic()
            self.state = 'loading / inferring'
            try:
                item = self.cache.get(key)
                hit = item is not None
                if hit:
                    self.cache_hits += 1
                    planes = np.repeat(self.cache.mask(item)[None], 4, axis=0).astype('<f4')
                    metadata = item['metadata']
                else:
                    planes, metadata = self.backend.predict(rgb, prompt.strip(), threshold, index)
                    if planes.shape != (4, h, w) or not np.isfinite(planes).all():
                        raise ValueError('Invalid mask output shape or values')
                    self.inferences += 1
                    import cv2
                    gray = cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    item = self.cache.put(key, planes[0] > 0.5, gray, metadata)
                self.cache.pin(header.get('capture_token', ''), header.get('frame'), key)
                self.count += 1
                self.state = 'model resident'
                self.last_error = None
                return dict(metadata, elapsed_seconds=time.monotonic() - started, requests=self.count,
                            cache_hit=hit, inferences=self.inferences,
                            representation='binary_mask_rgba', temporal_tracking=False), np.asarray(planes, dtype='<f4').tobytes()
            except Exception as exc:
                self.state = 'error; see last_error'
                self.last_error = str(exc)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--idle-timeout', type=int, default=300)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.idle_timeout < 0:
        parser.error('Invalid port or idle timeout')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    Server(Engine(Sam3Backend(args.checkpoint)), args.port, args.idle_timeout).serve()


if __name__ == '__main__':
    main()
