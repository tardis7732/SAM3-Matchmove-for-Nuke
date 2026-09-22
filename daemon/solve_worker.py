"""CPU-only motion solve from a RAM stream. Does not load SAM3 or read images."""
import json
import os
import struct
import sys
import zlib

if os.name == 'nt':
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(None)


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Incomplete RAM frame stream')
    return data


def run(stream):
    import numpy as np
    from memory_tracking import track_memory
    job = json.loads(stream.readline())
    w, h = job['width'], job['height']
    if not 0 < w * h <= 16777216 or not 0 < len(job['frames']) <= 10000:
        raise ValueError('Invalid RAM batch size')
    records = []
    for frame in job['frames']:
        size, = struct.unpack('<I', read_exact(stream, 4))
        if size > 32 * 1024 * 1024:
            raise ValueError('RAM frame exceeds limit')
        pixels = zlib.decompress(read_exact(stream, size))
        packed_size = (w * h + 7) // 8
        if len(pixels) != packed_size + w * h:
            raise ValueError('Invalid RAM frame dimensions')
        mask = np.unpackbits(np.frombuffer(pixels[:packed_size], dtype=np.uint8), count=w*h).reshape(h, w).astype(bool)
        gray = np.frombuffer(pixels[packed_size:], dtype=np.uint8).reshape(h, w)
        records.append({'frame': frame, 'mask': mask, 'gray': gray})
    job['frames'] = records
    return track_memory(job)


if __name__ == '__main__':
    try:
        reply = {'result': run(sys.stdin.buffer)}
    except Exception as exc:
        reply = {'error': str(exc)}
    sys.stdout.write(json.dumps(reply, allow_nan=False))
