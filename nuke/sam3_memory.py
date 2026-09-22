"""Analyze OFX pixels in RAM; never render temporary images or write result JSON."""
import base64
import json
import uuid
import os
from pathlib import Path
import struct
import subprocess
import nuke

CANCELLED = set()
BATCHES = {}
MAX_BATCH_BYTES = 512 * 1024 * 1024


def prune_batches():
    active = set()
    for group in nuke.allNodes('Group', recurseGroups=True):
        if group.knob('sam3Unified'):
            engine = group.node('INFERENCE')
            if engine and engine.knob('playbackToken'):
                active.add(engine['playbackToken'].value())
    for token in list(BATCHES):
        if token not in active:
            del BATCHES[token]


def ensure(group):
    if not group.knob('analysisData'):
        knob = nuke.String_Knob('analysisData', '')
        group.addKnob(knob)
        knob.setVisible(False)
    if not group.knob('memoryMode'):
        knob = nuke.Boolean_Knob('memoryMode', '')
        group.addKnob(knob)
        knob.setValue(group.node('CACHED_MASK').Class() != 'Read')
        knob.setVisible(False)


def analyze(group):
    import sam3_unified as unified
    import sam3_ofx_nuke as adapter
    ensure(group)
    engine = group.node('INFERENCE')
    if not engine.knob('playbackOnly'):
        return unified.fail(group, ValueError('Restart Nuke to load the memory-analysis OFX update'))
    controller = group.node('ANALYSIS')
    first, last, reference = [int(controller[k].value()) for k in ('first_frame', 'last_frame', 'reference_frame')]
    if group.input(0) is None or not first <= reference <= last or last-first >= 10000:
        return unified.fail(group, ValueError('Connect a plate and set a valid range (maximum 10000 frames)'))
    if abs(group.input(0).format().pixelAspect() - 1.0) > 1e-6:
        return unified.fail(group, ValueError('Reformat to square pixels before analyzing'))
    token = uuid.uuid4().hex
    prune_batches()
    frames = {}
    batch_bytes = 0
    port = int(engine['port'].value())
    key = group.fullName()
    CANCELLED.discard(key)
    previous_frame = nuke.frame()
    progress = nuke.ProgressTask('SAM3 RAM analysis') if nuke.GUI else None
    unified.busy(group, True)
    unified.ACTIVE[token] = {'group': group, 'memory': True}
    started = False
    try:
        engine['playbackOnly'].setValue(False)
        group['status'].setValue('Preparing RAM cache...')
        nuke.frame(first)
        # Sampling a pixel evaluates the whole-frame OFX without a Write node.
        fmt = group.input(0).format()
        engine.sample('rgba.alpha', fmt.width() / 2, fmt.height() / 2)
        if engine.error():
            raise RuntimeError('Cannot read the input frame. Check the connected plate.')
        before, _ = adapter._wire.request({'cmd': 'info'}, port=port, timeout=3)
        if not before.get('ram_snapshot'):
            raise RuntimeError('Stop the old engine using Stop engine / free GPU, then Analyze again')
        adapter._wire.request({'cmd': 'capture_begin', 'token': token}, port=port, timeout=3)
        started = True
        engine['captureToken'].setValue(token)
        for frame in range(first, last+1):
            if key in CANCELLED or (progress and progress.isCancelled()):
                raise RuntimeError('Cancelled — no mask files were written')
            if progress:
                progress.setMessage('Frame %d / %d — reuse preview cache or infer' % (frame, last))
                progress.setProgress(int(90 * (frame-first) / (last-first+1)))
            nuke.frame(frame)
            engine.sample('rgba.alpha', fmt.width()/2, fmt.height()/2)
            if engine.error():
                raise RuntimeError('Failed to store frame %d in RAM' % frame)
            snapshot, _ = adapter._wire.request({'cmd': 'capture_frame', 'token': token, 'frame': frame}, port=port)
            if (snapshot['width'], snapshot['height']) != (fmt.width(), fmt.height()):
                raise ValueError('Analyze at full resolution with a fixed input format')
            data = base64.b64decode(snapshot['data'])
            batch_bytes += len(data)
            if batch_bytes + sum(b['bytes'] for b in BATCHES.values()) > MAX_BATCH_BYTES:
                raise MemoryError('RAM solve cache is full. Use a shorter range or smaller plate.')
            frames[frame] = data
        if progress:
            progress.setMessage('Storing RAM masks...')
            progress.setProgress(95)
        adapter._wire.request({'cmd': 'capture_abort', 'token': token}, port=port, timeout=3)
        started = False
        with group:
            cached = group.node('CACHED_MASK')
            if cached.Class() == 'Read':
                cached.setName('OLD_FILE_MASK')
                ram = nuke.nodes.Dot(name='CACHED_MASK', inputs=[engine])
                group.node('MASK').setInput(1, ram)
                controller.setInput(1, ram)
                nuke.delete(cached)
            for node in nuke.allNodes('Write'):
                if node.knob('sam3_owner') and node['sam3_owner'].value() == controller['sam3_id'].value():
                    nuke.delete(node)
        BATCHES[token] = {'frames': frames, 'width': fmt.width(), 'height': fmt.height(), 'bytes': batch_bytes}
        group['analysisData'].setValue('')
        group['memoryMode'].setValue(True)
        controller['result_file'].setValue('')
        group['analysisReady'].setValue(False)
        engine['playbackToken'].setValue(token)
        group['maskSource'].setValue('Analyzed')
        group['maskSource'].setTooltip('Plays the analyzed RAM batch. Run Analyze again to replace it.')
        group['status'].setValue('Masks ready: %d-%d (%d frames). Solve to calculate motion.' % (first, last, last-first+1))
        prune_batches()
        unified.layout_internal(group)
        if progress:
            progress.setProgress(100)
        return token
    except Exception as exc:
        return unified.fail(group, exc)
    finally:
        engine['captureToken'].setValue('')
        engine['playbackOnly'].setValue(True)
        if started:
            try:
                adapter._wire.request({'cmd': 'capture_abort', 'token': token}, port=port, timeout=3)
            except Exception:
                pass
        nuke.frame(previous_frame)
        unified.ACTIVE.pop(token, None)
        CANCELLED.discard(key)
        unified.busy(group, False)
        del progress


def solve(group):
    import sam3_unified as unified
    engine = group.node('INFERENCE')
    batch = BATCHES.get(engine['playbackToken'].value())
    if batch is None:
        return unified.fail(group, ValueError('Analyze the frame range to load RAM masks before Solve'))
    controller = group.node('ANALYSIS')
    first, last, reference = [int(controller[k].value()) for k in ('first_frame', 'last_frame', 'reference_frame')]
    if not first <= reference <= last or last - first >= 10000:
        return unified.fail(group, ValueError('Set a valid frame range and reference frame'))
    if any(frame not in batch['frames'] for frame in range(first, last + 1)):
        return unified.fail(group, ValueError('This range includes frames without RAM masks. Analyze the new range first.'))
    job = {'width': batch['width'], 'height': batch['height'], 'frames': list(range(first, last + 1)),
           'reference_frame': reference,
           'tracking_mode': 'features' if controller['tracking_mode'].value().startswith('Features') else 'bbox',
           'smoothing_window': int(controller['smoothing_window'].value()),
           'crop_margin': float(controller['crop_margin'].value()),
           'fixed_crop': bool(controller['fixed_crop'].value()), 'pixel_aspect': 1.0}
    token = uuid.uuid4().hex
    unified.ACTIVE[token] = {'group': group, 'memory': True}
    unified.busy(group, True)
    child = None
    try:
        group['status'].setValue('Solving motion from stored RAM masks...')
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'config/frontend.json').read_text(encoding='utf-8-sig'))
        env = dict(os.environ)
        for name in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONEXECUTABLE', 'PYTHONSTARTUP', 'NUKE_PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.pop(name, None)
        env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep)
                                   if not any(part.lower().startswith('nuke') for part in Path(p).parts))
        env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONNOUSERSITE='1')
        child = subprocess.Popen([config['python'], str(root / 'daemon/solve_worker.py')],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 env=env, cwd=str(root), creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        child.stdin.write(json.dumps(job).encode('utf-8') + b'\n')
        for frame in job['frames']:
            data = batch['frames'][frame]
            child.stdin.write(struct.pack('<I', len(data)))
            child.stdin.write(data)
        child.stdin.close()
        reply = json.loads(child.stdout.read())
        if child.wait() or 'error' in reply:
            raise RuntimeError(reply.get('error', 'Motion solve failed'))
        result = reply['result']
        unified.matchmove().validate_result(result)
        payload = json.dumps(result, separators=(',', ':')).encode('utf-8')
        group['analysisData'].setValue('b64:' + base64.b64encode(payload).decode('ascii'))
        group['analysisReady'].setValue(True)
        group['status'].setValue('Solved: %d-%d (%d frames). Ready to Export.' % (first, last, last-first+1))
        return result
    except Exception as exc:
        return unified.fail(group, exc)
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                child.wait()
            for pipe in (child.stdin, child.stdout):
                if pipe:
                    pipe.close()
        unified.ACTIVE.pop(token, None)
        unified.busy(group, False)
