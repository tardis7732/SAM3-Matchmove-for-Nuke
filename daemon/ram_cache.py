"""Bounded mask/gray RAM cache. Pinned capture frames are never silently evicted."""
from collections import OrderedDict
import time
import numpy as np


class RamCache:
    def __init__(self, max_bytes=512 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.captures = {}
        self.bytes = 0

    def expire(self):
        now = time.monotonic()
        for token, capture in list(self.captures.items()):
            if now - capture['touched'] > 600:
                del self.captures[token]

    def begin(self, token):
        self.expire()
        if not isinstance(token, str) or not 1 <= len(token) <= 64:
            raise ValueError('Invalid capture token')
        if token in self.captures or len(self.captures) >= 4:
            raise ValueError('A capture with this token already exists, or too many captures are active')
        self.captures[token] = {'frames': {}, 'touched': time.monotonic()}

    def get(self, key):
        item = self.entries.get(key)
        if item is not None:
            self.entries.move_to_end(key)
        return item

    def put(self, key, mask, gray, metadata):
        self.expire()
        packed = np.packbits(mask.reshape(-1)).tobytes()
        gray = np.asarray(gray, dtype=np.uint8).tobytes()
        size = len(packed) + len(gray) + 512
        pinned = {key for capture in self.captures.values() for key in capture['frames'].values()}
        if size > self.max_bytes:
            raise MemoryError('One frame exceeds the RAM cache limit. Resize the input.')
        for victim in list(self.entries):
            if self.bytes + size <= self.max_bytes:
                break
            if victim not in pinned:
                self.bytes -= self.entries.pop(victim)['size']
        if self.bytes + size > self.max_bytes:
            raise MemoryError('Analysis exceeds the RAM cache limit. Use a shorter range or smaller input; no files were written.')
        item = {'packed': packed, 'gray': gray, 'shape': mask.shape, 'metadata': dict(metadata), 'size': size}
        self.entries[key] = item
        self.bytes += size
        return item

    def pin(self, token, frame, key):
        if not token:
            return
        if token not in self.captures:
            raise ValueError('RAM capture expired. Start Analyze again.')
        if type(frame) is not int:
            raise ValueError('Analysis frame must be an integer')
        capture = self.captures[token]
        capture['frames'][frame] = key
        capture['touched'] = time.monotonic()

    def frames(self, token, first, last):
        if token not in self.captures:
            raise ValueError('RAM capture does not exist')
        result = []
        for frame in range(first, last + 1):
            key = self.captures[token]['frames'].get(frame)
            item = self.entries.get(key)
            if item is None:
                raise ValueError('Frame %d was not evaluated into RAM. Restart Nuke after installing the updated OFX.' % frame)
            result.append({'frame': frame, 'mask': self.mask(item),
                           'gray': np.frombuffer(item['gray'], dtype=np.uint8).reshape(item['shape'])})
        return result

    @staticmethod
    def mask(item):
        h, w = item['shape']
        return np.unpackbits(np.frombuffer(item['packed'], dtype=np.uint8), count=h*w).reshape(h, w).astype(bool)

    def end(self, token):
        self.captures.pop(token, None)
