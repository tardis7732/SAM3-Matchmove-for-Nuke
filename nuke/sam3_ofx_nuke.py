"""SAM3 OFX controls and optional bridge to the existing Matchmove tool."""
import datetime
import importlib.util
import json
from pathlib import Path
import sys
import uuid
import nuke

ROOT = Path(__file__).resolve().parents[1]
CLASS = 'OFXorg.sam3.mask_v1'
_spec = importlib.util.spec_from_file_location('sam3_ofx_wire', ROOT / 'daemon/protocol.py')
_wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wire)


def create():
    from sam3_unified import create as unified_create
    return unified_create()


def create_legacy():
    node = nuke.createNode(CLASS, inpanel=False)
    node.setName('SAM3Mask_OFX')
    node.addKnob(nuke.Tab_Knob('sam3Tools', 'Mask / Tracking'))
    node.addKnob(nuke.Text_Knob('sam3Help', '',
        'Output: binary mask in R, G, B and alpha.\n'
        'Frames are segmented independently. Object index does not lock identity.\n'
        'Use the original SAM3 Matchmove text mode for video identity tracking.'))
    for name, label, value in [('bakeFirst', 'first', int(nuke.root()['first_frame'].value())),
                               ('bakeLast', 'last', int(nuke.root()['last_frame'].value())),
                               ('bakeReference', 'reference', int(nuke.frame()))]:
        knob = nuke.Int_Knob(name, label)
        node.addKnob(knob)
        knob.setValue(value)
    knob = nuke.PyScript_Knob('bakeTrack', 'Bake masks + Analyze motion')
    knob.setCommand('import sam3_ofx_nuke; sam3_ofx_nuke.bake_and_analyze(nuke.thisNode())')
    knob.setTooltip('Render the mask range, create a SAM3 Matchmove controller using those masks, and analyze motion. Export Tracker from the controller when complete.')
    node.addKnob(knob)
    for name, label, func in [('engineStatus', 'Engine status', 'status'), ('stopEngine', 'Stop engine / free GPU', 'stop'), ('clearCache', 'Refresh mask', 'clear_cache')]:
        knob = nuke.PyScript_Knob(name, label)
        knob.setCommand('import sam3_ofx_nuke; sam3_ofx_nuke.' + func + '(nuke.thisNode())')
        node.addKnob(knob)
    if nuke.GUI:
        node.showControlPanel()
    return node


def clear_cache(node):
    if node.knob('sam3Unified'):
        from sam3_unified import live
        live(node)
        node = node.node('INFERENCE')
    node['cacheRevision'].setValue(int(node['cacheRevision'].value()) + 1)


def status(node):
    if node.knob('sam3Unified'):
        node = node.node('INFERENCE')
    try:
        info, _ = _wire.request({'cmd': 'info'}, port=int(node['port'].value()), timeout=3)
        message = json.dumps(info, indent=2)
    except Exception as exc:
        message = 'Engine is not running. Viewing the mask starts it automatically.\n' + str(exc)
    if nuke.GUI:
        nuke.message(message)
    else:
        print(message)


def stop(node):
    if node.knob('sam3Unified'):
        node = node.node('INFERENCE')
    try:
        _wire.request({'cmd': 'shutdown'}, port=int(node['port'].value()), timeout=3)
    except OSError:
        pass
    # Retain the mask cache so stopping does not immediately trigger a reload.


def bake_and_analyze(node):
    if node.knob('sam3Unified'):
        from sam3_unified import analyze
        return analyze(node)
    try:
        return _bake_and_analyze(node)
    except Exception as exc:
        if not nuke.GUI:
            raise
        nuke.message('SAM3 mask analysis failed:\n' + str(exc))


def _bake_and_analyze(node):
    plate = node.input(0)
    if plate is None:
        raise ValueError('Connect an image to the SAM3 OFX node')
    first, last, reference = [int(node[k].value()) for k in ('bakeFirst', 'bakeLast', 'bakeReference')]
    if not first <= reference <= last:
        raise ValueError('First <= Reference <= Last is required')
    if abs(plate.format().pixelAspect() - 1) > 1e-6:
        raise ValueError('Reformat to square pixels before motion analysis')
    settings = json.loads((ROOT / 'config/frontend.json').read_text(encoding='utf-8'))
    matchmove = Path(settings['matchmove_root'])
    if str(matchmove / 'nuke') not in sys.path:
        sys.path.insert(0, str(matchmove / 'nuke'))
    import sam3_matchmove_nuke as sm
    folder = ROOT / 'output' / ('bake_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    pattern = (folder / 'mask.%06d.png').as_posix()
    writer = nuke.nodes.Write(inputs=[node], file=pattern, file_type='png', channels='rgba', raw=True)
    writer['datatype'].setValue('8 bit')
    previous_frame = nuke.frame()
    try:
        nuke.execute(writer, first, last)
    finally:
        nuke.delete(writer)
        nuke.frame(previous_frame)
    reader = nuke.nodes.Read(file=pattern, first=first, last=last, raw=True)
    reader.setName('SAM3_OFX_BakedMask')
    reader.setXYpos(node.xpos() + 160, node.ypos() + 80)
    controller = sm.create_node()
    controller.setInput(0, plate)
    controller.setInput(1, reader)
    controller.setXYpos(node.xpos(), node.ypos() + 170)
    controller['backend'].setValue('Input mask')
    controller['mask_channel'].setValue('alpha')
    controller['target_text'].setValue(node['targetText'].value())
    for name, value in [('first_frame', first), ('last_frame', last), ('reference_frame', reference)]:
        controller[name].setValue(value)
    controller['label'].setValue('OFX masks: independent detections\nExport Tracker after analysis')
    # Automatically render any plate conversion required by intermediate nodes/EXR.
    if sm.export_inputs(controller, folder) is None:
        owner = controller['sam3_id'].value()
        for item in nuke.allNodes('Write', recurseGroups=True):
            if item.knob('sam3_owner') and item['sam3_owner'].value() == owner:
                nuke.execute(item, first, last)
    token = sm._start(controller)
    if token is None:
        raise RuntimeError('Could not prepare the tracking input')
    return controller, token
