"""Run the real Nuke Analyze/RAM path against deterministic video masks (no GPU)."""
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'daemon'))
from sam3_daemon import Engine
from server_transport import Server

class Backend:
    calls = 0
    def release_image(self):
        pass
    def predict(self, *args):
        raise AssertionError('Independent detection must not be called')
    def track_video(self, images, reference, prompt, confidence, index, cancelled):
        self.calls += 1
        if self.calls == 2:
            raise ValueError('Test tracking failure')
        assert reference == 1
        for i in (1, 2, 0):
            mask = np.zeros((64, 64), bool)
            mask[12:28, 8+i:24+i] = True
            yield i, mask, dict(object_id=10, temporal_tracking=True)

server = Server(Engine(Backend()), port=0, idle_timeout=60)
thread = threading.Thread(target=server.serve, daemon=True)
thread.start()
while not server.port:
    time.sleep(.01)
try:
    env = dict(os.environ, SAM3_TEST_PORT=str(server.port), NUKE_PATH=str(ROOT / 'nuke'))
    result = subprocess.run([sys.argv[1], '-ti', str(ROOT / 'tests/nuke_video.py')], env=env, timeout=180)
    raise SystemExit(result.returncode)
finally:
    server.stop.set()
    thread.join(3)
