"""Bounded, cancellable RAM video jobs. No source or mask image files are created."""
import base64
import threading
import time
import zlib
import numpy as np


class VideoJobs:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.RLock()
        self.jobs = {}

    def control(self, header):
        token = header.get('token')
        if not isinstance(token, str) or not 1 <= len(token) <= 64:
            raise ValueError('Invalid video token')
        cmd = header['cmd']
        with self.lock:
            # Idle uploads/results can be abandoned when Nuke closes or disconnects.
            for old, job in list(self.jobs.items()):
                if job['state'] != 'running' and time.monotonic() - job['touched'] > 600:
                    del self.jobs[old]
            if cmd == 'video_begin':
                if self.jobs:
                    raise ValueError('Another RAM video job is active; finish or cancel it first')
                first, last, reference = [header.get(k) for k in ('first', 'last', 'reference')]
                if any(type(v) is not int for v in (first, last, reference)) or not first <= reference <= last or last-first >= 10000:
                    raise ValueError('Invalid video frame range/reference')
                self.jobs[token] = dict(first=first, last=last, reference=reference, state='uploading',
                    images={}, results={}, bytes=0, result_bytes=0, touched=time.monotonic(),
                    cancel=threading.Event(), message='Reading source frames', progress=0)
                return {'ok': True}
            job = self.jobs.get(token)
            if job is None:
                raise ValueError('RAM video job expired; Analyze again')
            job['touched'] = time.monotonic()
            if cmd == 'video_abort':
                job['cancel'].set()
                if job['state'] != 'running':
                    del self.jobs[token]
                return {'ok': True}
            if cmd == 'video_status':
                return {k: job.get(k) for k in ('state', 'progress', 'message', 'error', 'object_id')}
            if cmd == 'video_start':
                if job['state'] != 'uploading' or len(job['images']) != job['last']-job['first']+1:
                    raise ValueError('Complete all source frames before video tracking')
                job['state'] = 'running'
                job['message'] = 'Loading SAM3 video model'
                threading.Thread(target=self.run, args=(token, job), daemon=True).start()
                return {'ok': True}
            if cmd == 'video_frame':
                if job['state'] != 'ready':
                    raise ValueError('Video masks are not ready')
                frame = header.get('frame')
                data = job['results'].get(frame)
                if data is None:
                    raise ValueError('Missing tracked frame')
                return dict(frame=frame, width=job['width'], height=job['height'],
                            object_id=job['object_id'], data=base64.b64encode(data).decode('ascii'))
            raise ValueError('Unsupported video command')

    def infer(self, header, rgb):
        token, frame = header.get('capture_token'), header.get('frame')
        with self.lock:
            job = self.jobs.get(token)
            if job is None:
                return None
            if type(frame) is not int or not job['first'] <= frame <= job['last']:
                raise ValueError('Frame outside video job')
            h, w = rgb.shape[:2]
            settings = (header['prompt'].strip(), float(header.get('confidence', .45)), header.get('object_index', 0),
                        header.get('input_colorspace', 'linear_srgb'))
            if 'settings' not in job:
                count = job['last'] - job['first'] + 1
                # SAM3 also allocates normalized 1008-square CPU tensors for its video loader.
                if count * w * h * 3 > 2 * 1024**3 or count * 1008**2 * 12 > 4 * 1024**3:
                    raise MemoryError('Video input exceeds RAM budget. Use a shorter range or smaller plate.')
                job.update(settings=settings, width=w, height=h)
            if (w, h, settings) != (job['width'], job['height'], job['settings']):
                raise ValueError('Input dimensions or detection settings changed during Analyze')
            job['touched'] = time.monotonic()
            if job['state'] == 'uploading':
                job['images'][frame] = np.rint(rgb * 255).astype(np.uint8)
                mask = np.zeros((h, w), bool)
            elif job['state'] == 'ready':
                raw = zlib.decompress(job['results'][frame])
                mask = np.unpackbits(np.frombuffer(raw[:(w*h+7)//8], np.uint8), count=w*h).reshape(h, w)
            else:
                raise ValueError('Video analysis is not ready: ' + job['state'])
            return dict(temporal_tracking=True, object_id=job.get('object_id'),
                        elapsed_seconds=0, video_state=job['state']), np.repeat(mask[None], 4, axis=0).astype('<f4').tobytes()

    def run(self, token, job):
        images = []
        try:
            from PIL import Image
            import cv2
            with self.engine.lock:
                self.engine.state = 'tracking video'
                self.engine.backend.release_image()
                images = [Image.fromarray(job['images'][f]) for f in range(job['first'], job['last']+1)]
                job['images'].clear()
                prompt, confidence, index, _ = job['settings']
                stream = self.engine.backend.track_video(images, job['reference']-job['first'],
                                                         prompt, confidence, index, job['cancel'].is_set)
                try:
                    for i, mask, metadata in stream:
                        if job['cancel'].is_set():
                            raise RuntimeError('Video analysis cancelled')
                        gray = cv2.cvtColor(np.asarray(images[i]), cv2.COLOR_RGB2GRAY)
                        data = zlib.compress(np.packbits(mask.reshape(-1)).tobytes() + gray.tobytes(), 1)
                        frame = job['first'] + i
                        with self.lock:
                            size = job['result_bytes'] - len(job['results'].get(frame, b'')) + len(data)
                            if size > 512 * 1024**2:
                                raise MemoryError('RAM masks exceed budget; use a shorter range or smaller plate')
                            job['results'][frame] = data
                            job.update(result_bytes=size, object_id=metadata['object_id'],
                                progress=len(job['results']) / len(images),
                                message='Tracking frame %d / object %d' % (frame, metadata['object_id']))
                finally:
                    stream.close()
                if len(job['results']) != len(images):
                    raise RuntimeError('Incomplete video result')
                with self.lock:
                    job.update(state='ready', message='Video masks ready')
                self.engine.state = 'ready'
        except Exception as exc:
            with self.lock:
                job.update(state='error', error=str(exc), message=str(exc))
                job['results'].clear()
            self.engine.last_error = str(exc)
            self.engine.state = 'error; see last_error'
        finally:
            for im in images:
                im.close()
            job['images'].clear()
            with self.lock:
                if job['cancel'].is_set():
                    self.jobs.pop(token, None)

    def running(self):
        with self.lock:
            return any(j['state'] == 'running' for j in self.jobs.values())
