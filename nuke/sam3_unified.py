"""Single user node: native OFX, cached mask, and motion analysis inside a Group."""
import base64
import json
from pathlib import Path
import sys
import nuke

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {}
EXPORT_CHOICES = ['Tracker', 'Matchmove', 'Stabilize', 'Plate Stabilize Crop', 'Generated Crop Matchmove']
VIEW_CHOICES = ['Plate', 'Mask', 'Plate + mask alpha', 'Mask overlay']
MOTION_CHOICES = ['Translation', 'Translation + Scale', 'Translation + Scale + Rotation']
MOTION_MODES = dict(zip(MOTION_CHOICES, ('translation', 'translation_scale', 'translation_scale_rotation')))
MIGRATING_MOTION = set()


def matchmove():
    settings = json.loads((ROOT / 'config/frontend.json').read_text(encoding='utf-8'))
    path = str(Path(settings['matchmove_root']) / 'nuke')
    if path not in sys.path:
        sys.path.insert(0, path)
    import sam3_matchmove_nuke
    return sam3_matchmove_nuke


def button(node, name, label, function):
    knob = nuke.PyScript_Knob(name, label)
    knob.setCommand('import sam3_unified; sam3_unified.' + function + '(nuke.thisNode())')
    node.addKnob(knob)


def link(node, name, label, child, target=None):
    knob = nuke.Link_Knob(name, label)
    knob.makeLink(child, target or name)
    node.addKnob(knob)


def create():
    import sam3_ofx_nuke
    sm = matchmove()
    upstream = nuke.selectedNode() if nuke.selectedNodes() else None
    group = nuke.nodes.Group(name='SAM3_Mask')
    try:
        with group:
            source = nuke.nodes.Input(name='SOURCE')
            engine = nuke.createNode(sam3_ofx_nuke.CLASS, inpanel=False)
            engine.setName('INFERENCE')
            engine.setInput(0, source)
            cached = nuke.nodes.Dot(name='CACHED_MASK', inputs=[engine])
            switch = nuke.nodes.Switch(name='MASK', inputs=[engine, cached])
            alpha = nuke.nodes.Copy(name='PLATE_ALPHA', inputs=[source, switch])
            alpha['from0'].setValue('rgba.alpha')
            alpha['to0'].setValue('rgba.alpha')
            output = nuke.nodes.Switch(name='RESULT', inputs=[switch, alpha, source, source, source])
            nuke.nodes.Output(inputs=[output])
            controller = sm.create_node()
            controller.setName('ANALYSIS')
            controller['aspect_preset'].setValue('1:1')
            controller['aspect_lock'].setValue(True)
            controller['output_height'].setValue(720)
            controller.setInput(0, source)
            controller.setInput(1, cached)
            controller['backend'].setValue('Input mask')
            if nuke.GUI:
                controller.hideControlPanel()
        if upstream is not None:
            group.setInput(0, upstream)
        group.addKnob(nuke.Tab_Knob('sam3', 'SAM3'))
        marker = nuke.Boolean_Knob('sam3Unified', '')
        group.addKnob(marker)
        marker.setValue(True)
        marker.setVisible(False)
        for name, label in [('targetText', 'Target'), ('confidence', 'Confidence'), ('objectIndex', 'Object index')]:
            link(group, name, label, 'INFERENCE')
        for name, label, values in [('outputMode', 'View', VIEW_CHOICES),
                                    ('maskSource', 'Mask', ['Live', 'Analyzed'])]:
            group.addKnob(nuke.Enumeration_Knob(name, label, values))
        group['outputMode'].setValue('Mask')
        group['maskSource'].setTooltip('Analyze tracks the selected reference object through the video and stores masks in RAM.')
        switch['which'].setExpression('parent.maskSource')
        output['which'].setExpression('parent.outputMode')
        first, last = input_frame_range(upstream)
        for name, value in [('first_frame', first), ('last_frame', last), ('reference_frame', first)]:
            controller[name].setValue(value)
        for name, label in [('first_frame', 'First'), ('last_frame', 'Last'), ('reference_frame', 'Reference'), ('tracking_mode', 'Motion')]:
            link(group, name, label, 'ANALYSIS')
        button(group, 'analyzeRange', 'Analyze', 'analyze')
        button(group, 'exportTracker', 'Export Tracker', 'export_tracker')
        group.addKnob(nuke.Text_Knob('status', 'Status', 'Set Frame range, then Analyze to load RAM masks'))
        group.addKnob(nuke.Tab_Knob('adjustments', 'Adjustments'))
        link(group, 'inputColorspace', 'Input encoding', 'INFERENCE')
        for name, label in [('smoothing_window', 'Smoothing'), ('crop_margin', 'Crop margin'),
                            ('aspect_preset', 'Aspect ratio'), ('aspect_lock', 'Lock ratio'),
                            ('output_width', 'Crop width'), ('output_height', 'Crop height')]:
            link(group, name, label, 'ANALYSIS')
        button(group, 'refreshLive', 'Refresh mask', 'live')
        for name, label, func in [('engineStatus', 'Engine status', 'engine_status'), ('stopEngine', 'Stop engine / free GPU', 'stop_engine')]:
            button(group, name, label, func)
        ready = nuke.Boolean_Knob('analysisReady', '')
        group.addKnob(ready)
        ready.setVisible(False)
        group['knobChanged'].setValue('import sam3_unified; sam3_unified.changed(nuke.thisNode(), nuke.thisKnob())')
        engine['knobChanged'].setValue('import sam3_unified; sam3_unified.engine_changed(nuke.thisNode(), nuke.thisKnob())')
        controller['knobChanged'].setValue('import sam3_unified; sam3_unified.controller_changed(nuke.thisNode(), nuke.thisKnob())')
        group['tile_color'].setValue(0x358A91FF)
        group['label'].setValue('[value targetText]\n[value outputMode]')
        ensure_export_controls(group)
        if nuke.GUI:
            group.showControlPanel()
        return group
    except Exception:
        nuke.delete(group)
        raise


def ensure_view_controls(group):
    """Keep four preview choices and only their active render branches."""
    group['outputMode'].setLabel('View')
    if list(group['outputMode'].values()) != VIEW_CHOICES:
        selected = group['outputMode'].value()
        group['outputMode'].setValues(VIEW_CHOICES)
        group['outputMode'].setValue(selected if selected in VIEW_CHOICES else 'Plate')
    with group:
        overlay = group.node('MASK_OVERLAY')
        # Upgrade the old Copy-based overlay, which required source alpha.
        if overlay is not None and overlay.Class() == 'Copy':
            nuke.delete(overlay)
            overlay = None
            old_tint = group.node('MASK_OVERLAY_RGB')
            if old_tint is not None:
                nuke.delete(old_tint)
        if overlay is None:
            # Premultiplied red, masked at 50%. Merge only RGB, so the source
            # need not have alpha; existing source alpha passes through untouched.
            tint = nuke.nodes.Expression(name='MASK_OVERLAY_RGB', inputs=[group.node('MASK')])
            weight = 'clamp(a,0,1)*0.5'
            for index, expression in enumerate((weight, '0', '0', weight)):
                tint['expr%d' % index].setValue(expression)
            overlay = nuke.nodes.Merge2(name='MASK_OVERLAY', inputs=[group.node('SOURCE'), tint],
                                       operation='over', output='rgb')
        result = group.node('RESULT')
        for index in range(result.inputs()):
            result.setInput(index, None)
        for index, node in enumerate((group.node('SOURCE'), group.node('MASK'), group.node('PLATE_ALPHA'), overlay)):
            result.setInput(index, node)
        result['which'].setExpression('parent.outputMode')
    group['outputMode'].setTooltip('Mask overlay blends red at 50% inside the mask over the plate. Export creates separate nodes.')
    if group.knob('maskSource'):
        group['maskSource'].setVisible(False)
    if group.knob('refreshLive'):
        group['refreshLive'].setLabel('Refresh mask')
    configure_index(group.node('INFERENCE'))
    group.knobs()['objectIndex'].setLabel('Object index')
    group.knobs()['objectIndex'].clearFlag(0x2)
    if group['label'].value() == '[value targetText]\n[value maskSource] / [value outputMode]':
        group['label'].setValue('[value targetText]\n[value outputMode]')


def configure_index(engine):
    index = engine['objectIndex']
    index.setLabel('Object index')
    index.setTooltip('Target rank by mask area on Reference frame (0 = largest). Analyze follows that object ID through the video.')
    index.setRange(0, 1000)
    # DDImage/Knob.h: Knob::SLIDER = 0x2 (not exported by Nuke's Python module).
    index.clearFlag(0x2)
    if index.value() < 0:
        index.setValue(0)


def ensure_export_controls(group):
    """Restore the six export choices while preserving the v0.2 control layout."""
    if not group.knob('sam3Unified'):
        return
    # Knob migration removes and reattaches Link_Knobs. Close any existing
    # Properties widgets before changing their owning knob list.
    if nuke.GUI:
        group.hideControlPanel()
    from sam3_memory import ensure
    ensure(group)
    ensure_view_controls(group)
    ensure_frame_controls(group)
    ensure_solve_controls(group)
    layout_internal(group)
    engine = group.node('INFERENCE')
    if engine.knob('playbackOnly'):
        engine['playbackOnly'].setValue(True)
    ensure_aspect_controls(group)
    ensure_compact_controls(group)
    if group.knob('export_type'):
        group['export_type'].setTooltip('Create motion or crop nodes below SAM3 from the analyzed curves. Use View > Mask for the mask output.')
        if list(group['export_type'].values()) != EXPORT_CHOICES:
            selected = group['export_type'].value()
            group['export_type'].setValues(EXPORT_CHOICES)
            group['export_type'].setValue(selected if selected in EXPORT_CHOICES else 'Tracker')
        return
    # Link knobs must be moved as their links, not as resolved child knobs.
    knobs = group.knobs()
    names = list(knobs)
    if 'exportTracker' not in names:
        return
    tail = [knobs[name] for name in names[names.index('exportTracker'):]]
    for knob in tail:
        group.removeKnob(knob)
    group.addKnob(nuke.Text_Knob('export_section', 'Export', ''))
    group.addKnob(nuke.Enumeration_Knob('export_type', 'Output', EXPORT_CHOICES))
    group['export_type'].setTooltip('Create motion or crop nodes below SAM3 from the analyzed curves. Use View > Mask for the mask output.')
    for knob in tail:
        group.addKnob(knob)
    group['exportTracker'].setLabel('Export')
    group['exportTracker'].setCommand('import sam3_unified; sam3_unified.export_selected(nuke.thisNode())')
    group['exportTracker'].clearFlag(nuke.STARTLINE)
    group['onCreate'].setValue('import sam3_unified; sam3_unified.ensure_export_controls(nuke.thisNode())')


def ensure_compact_controls(group):
    """Keep motion and crop settings together on the main SAM3 tab."""
    knobs = group.knobs()
    if 'adjustments' in knobs:
        group.removeKnob(knobs['adjustments'])
    knobs = group.knobs()
    names = list(knobs)
    row = ['smoothing_window', 'crop_margin', 'output_width', 'output_height', 'aspect_preset', 'aspect_lock']
    start = names.index('solveMotion') + 1
    if names[start:start + len(row)] != row:
        tail = [knobs[name] for name in names[start:]]
        for knob in tail:
            group.removeKnob(knob)
        for name in row:
            group.addKnob(knobs[name])
        for knob in tail:
            if knob.name() not in row:
                group.addKnob(knob)
    knobs['smoothing_window'].setFlag(nuke.STARTLINE)
    knobs['crop_margin'].setFlag(nuke.STARTLINE)
    # Keep numeric fields on the same label/input grid as Target and View.
    # Clear both the Link_Knob and its source so old scenes migrate consistently.
    for name in ('confidence', 'smoothing_window', 'crop_margin'):
        knobs[name].clearFlag(0x2)  # DDImage::Knob::SLIDER
        group[name].clearFlag(0x2)
    for name in ('resetRange', 'analyzeRange', 'solveMotion', 'exportTracker'):
        # Restore the normal Properties button widget. SMALL_UI is intended for
        # toolbar widgets and is not exported by Nuke's Python knob API.
        knobs[name].clearFlag(0x2000000)
    # Preserve legacy scripts and the selected encoding without exposing controls.
    for name in ('inputColorspace', 'refreshLive', 'engineStatus', 'stopEngine'):
        if name in knobs:
            knobs[name].setVisible(False)


def ensure_solve_controls(group):
    controller = group.node('ANALYSIS')
    motion = controller['tracking_mode']
    if list(motion.values()) != MOTION_CHOICES:
        old = motion.value()
        selected = old if old in MOTION_CHOICES else (
            'Translation + Scale + Rotation' if old.startswith('Features') or old == 'Translation + Rotation' else 'Translation + Scale')
        MIGRATING_MOTION.add(group.fullName())
        try:
            motion.setValues(MOTION_CHOICES)
            motion.setValue(selected)
        finally:
            MIGRATING_MOTION.discard(group.fullName())
        if old == 'Translation + Rotation' and group.knob('analysisReady'):
            group['analysisReady'].setValue(False)
            group['status'].setValue('Motion mode updated - Solve again before Export')
    motion.setTooltip('Translation: position only. Translation + Scale: position and uniform scale. '
                      'Translation + Scale + Rotation: position, uniform scale and rotation. Run Solve after changes.')
    if not group.knob('solveMotion'):
        button(group, 'solveMotion', 'Solve', 'solve')
    knobs = group.knobs()
    names = list(knobs)
    row = ['tracking_mode', 'reference_frame', 'solveMotion']
    start = min(names.index(name) for name in row)
    if names[start:start + len(row)] != row:
        tail = [knobs[name] for name in names[start:]]
        for knob in tail:
            group.removeKnob(knob)
        for name in row:
            group.addKnob(knobs[name])
        for knob in tail:
            if knob.name() not in row:
                group.addKnob(knob)
    knobs['tracking_mode'].setFlag(nuke.STARTLINE)
    knobs['reference_frame'].setLabel('  Reference frame')
    knobs['reference_frame'].clearFlag(nuke.STARTLINE)
    knobs['solveMotion'].clearFlag(nuke.STARTLINE)
    knobs['solveMotion'].setTooltip('Calculate motion using masks already generated by Analyze. No SAM3 inference; the GPU engine can be stopped.')
    knobs['analyzeRange'].setTooltip('Select the object on Reference frame, track its ID in both directions, and store masks in RAM. Then Solve motion.')


def layout_internal(group):
    """Arrange owned nodes; remove only obsolete, tagged preview outputs."""
    with group:
        owned = {node for node in nuke.allNodes() if node.knob('sam3Derived')}
        protected = {node for node in owned if any(other not in owned and other.name() != 'RESULT'
                     for other in node.dependent(nuke.INPUTS | nuke.HIDDEN_INPUTS | nuke.EXPRESSIONS))}
        pending = list(protected)
        while pending:
            for source in pending.pop().dependencies(nuke.INPUTS | nuke.HIDDEN_INPUTS | nuke.EXPRESSIONS):
                if source in owned and source not in protected:
                    protected.add(source)
                    pending.append(source)
        for node in owned - protected:
            nuke.delete(node)
        positions = {'SOURCE': (0, 0), 'INFERENCE': (-240, 110),
                     'CACHED_MASK': (-206, 220), 'MASK': (-240, 290),
                     'MASK_OVERLAY_RGB': (-240, 400), 'MASK_OVERLAY': (0, 510),
                     'PLATE_ALPHA': (240, 400), 'RESULT': (0, 650),
                     'ANALYSIS': (480, 110)}
        for name, (x, y) in positions.items():
            node = group.node(name)
            if node:
                node.setXYpos(x, y)
                if node.knob('postage_stamp'):
                    node['postage_stamp'].setValue(False)
        controller = group.node('ANALYSIS')
        controller['label'].setValue('Tracking settings')
        controller['hide_input'].setValue(True)
        for node in nuke.allNodes('Output'):
            if node.input(0) == group.node('RESULT'):
                node.setXYpos(0, 750)


def input_frame_range(source):
    if source is None:
        return int(nuke.root()['first_frame'].value()), int(nuke.root()['last_frame'].value())
    frames = source.frameRange()
    first, last = int(frames.first()), int(frames.last())
    if last < first:
        raise ValueError('The connected plate has an invalid frame range')
    return first, last


def reset_range(group):
    source = group.input(0)
    if source is None:
        return fail(group, ValueError('Connect a plate before resetting the frame range'))
    first, last = input_frame_range(source)
    controller = group.node('ANALYSIS')
    for name, value in [('first_frame', first), ('last_frame', last), ('reference_frame', first)]:
        controller[name].setValue(value)
    group['status'].setValue('Frame range: %d–%d (%d frames); Reference: %d' % (first, last, last-first+1, first))


def ensure_frame_controls(group):
    for name in ('cancelAnalysis', 'help'):
        if group.knob(name):
            try:
                group.removeKnob(group.knobs()[name])
            except ValueError:
                # New Groups expose Nuke's built-in help knob, not our old text.
                group.knobs()[name].setVisible(False)
    if not group.knob('frame_range_label'):
        knobs = group.knobs()
        names = list(knobs)
        tail = [knobs[name] for name in names[names.index('first_frame'):]]
        for knob in tail:
            group.removeKnob(knob)
        group.addKnob(nuke.Text_Knob('frame_range_label', 'Frame range', ' '))
        group.addKnob(knobs['first_frame'])
        group.addKnob(knobs['last_frame'])
        button(group, 'resetRange', 'Reset', 'reset_range')
        for knob in tail:
            if knob.name() not in ('first_frame', 'last_frame'):
                group.addKnob(knob)
    knobs = group.knobs()
    names = list(knobs)
    row = ['frame_range_label', 'first_frame', 'last_frame', 'resetRange', 'analyzeRange']
    start = names.index('frame_range_label')
    if names[start:start + len(row)] != row:
        tail = [knobs[name] for name in names[start:]]
        for knob in tail:
            group.removeKnob(knob)
        for name in row:
            group.addKnob(knobs[name])
        for knob in tail:
            if knob.name() not in row:
                group.addKnob(knob)
    knobs['frame_range_label'].setFlag(nuke.STARTLINE)
    knobs['frame_range_label'].setVisible(False)
    knobs['first_frame'].setLabel('Frame range')
    knobs['last_frame'].setLabel('')
    knobs['first_frame'].setFlag(nuke.STARTLINE)
    for name in ('last_frame', 'reference_frame', 'resetRange', 'analyzeRange'):
        knobs[name].clearFlag(nuke.STARTLINE)
    knobs['resetRange'].setTooltip('Read First and Last from the connected plate range, and set Reference to First. Respects Read timing and upstream frame-range changes.')


def ensure_aspect_controls(group):
    """Expose the original controller's ratio settings beside crop dimensions."""
    controller = group.node('ANALYSIS')
    if not controller.knob('aspect_preset'):
        matchmove()
        from sam3_matchmove_size import ensure_size_controls
        ensure_size_controls(controller)
    controls = [('aspect_preset', 'Aspect ratio'), ('aspect_lock', 'Lock ratio')]
    missing = [(name, label) for name, label in controls if not group.knob(name)]
    if missing:
        knobs = group.knobs()
        names = list(knobs)
        tail = [knobs[name] for name in names[names.index('output_width'):]]
        for knob in tail:
            group.removeKnob(knob)
        for name, label in missing:
            link(group, name, label, 'ANALYSIS')
        for knob in tail:
            group.addKnob(knob)
    knobs = group.knobs()
    names = list(knobs)
    row = ['output_width', 'output_height', 'aspect_preset', 'aspect_lock']
    start = min(names.index(name) for name in row)
    if names[start:start + len(row)] != row:
        tail = [knobs[name] for name in names[start:]]
        for knob in tail:
            group.removeKnob(knob)
        for name in row:
            group.addKnob(knobs[name])
        for knob in tail:
            if knob.name() not in row:
                group.addKnob(knob)
    group.knobs()['aspect_preset'].clearFlag(nuke.STARTLINE)
    group.knobs()['aspect_lock'].clearFlag(nuke.STARTLINE)
    group.knobs()['output_width'].setFlag(nuke.STARTLINE)
    group.knobs()['output_height'].clearFlag(nuke.STARTLINE)
    group.knobs()['output_width'].setLabel('Crop size')
    group.knobs()['output_height'].setLabel('×')
    group['refreshLive'].setFlag(nuke.STARTLINE)
    group['engineStatus'].clearFlag(nuke.STARTLINE)
    group['stopEngine'].clearFlag(nuke.STARTLINE)
    controller['aspect_preset'].setTooltip('Choose the crop ratio. Presets keep Crop height and adjust Crop width. New nodes start at 1:1, 720 x 720. No reanalysis is required.')
    # Only the controller owns the hidden ratio and its size-change callback.
    for name, _ in controls:
        group.knobs()[name].setTooltip(controller[name].tooltip())


def handle_crop_size(controller, name):
    import sam3_matchmove_size as size
    if name != 'aspect_preset':
        return size.handle_size_change(controller, name)
    key = controller.fullName()
    if key in size._CHANGING:
        return
    # Share the original controller's guard so linked-knob callbacks cannot
    # re-enter the width-based preset handler while the dimensions are updated.
    size._CHANGING.add(key)
    try:
        preset = controller['aspect_preset'].value()
        height = max(8, int(controller['output_height'].value()))
        if preset == 'Custom':
            ratio = max(8, int(controller['output_width'].value())) / height
        else:
            a, b = preset.split(':')
            ratio = float(a) / float(b)
        controller['aspect_ratio'].setValue(ratio)
        controller['aspect_lock'].setValue(True)
        controller['output_height'].setValue(height)
        controller['output_width'].setValue(max(8, int(height * ratio + 0.5)))
    finally:
        size._CHANGING.discard(key)


def upgrade_exports():
    for group in nuke.allNodes('Group', recurseGroups=True):
        if group.knob('sam3Unified'):
            ensure_export_controls(group)
            engine = group.node('INFERENCE')
            if engine.knob('playbackToken'):
                engine['playbackToken'].setValue('')
                group['status'].setValue('RAM masks are empty — Analyze the frame range')
    from sam3_memory import prune_batches
    prune_batches()


def register_callbacks():
    previous = getattr(nuke, '_sam3_export_load_callback', None)
    if previous is not None:
        nuke.removeOnScriptLoad(previous)
    nuke.addOnScriptLoad(upgrade_exports)
    nuke._sam3_export_load_callback = upgrade_exports


def live(group):
    group['maskSource'].setValue('Live')
    engine = group.node('INFERENCE')
    if engine.knob('playbackToken'):
        engine['playbackToken'].setValue('')
    from sam3_memory import prune_batches
    prune_batches()
    engine['cacheRevision'].setValue(int(engine['cacheRevision'].value()) + 1)
    group['analysisReady'].setValue(False)
    if group['outputMode'].value().startswith('Stabilized'):
        group['outputMode'].setValue('Mask')
    group['status'].setValue('RAM playback cleared — Analyze the frame range')


def engine_changed(engine, knob):
    try:
        name = knob.name()
        parent_name = engine.fullName().rsplit('.', 1)[0]
    except (ValueError, RuntimeError):
        return
    if name not in ('targetText', 'confidence', 'objectIndex', 'inputColorspace'):
        return
    with nuke.root():
        group = nuke.toNode(parent_name)
    if group is not None and group.knob('analysisReady'):
        if engine.knob('playbackToken'):
            engine['playbackToken'].setValue('')
        from sam3_memory import prune_batches
        prune_batches()
        group['maskSource'].setValue('Live')
        group['analysisReady'].setValue(False)
        if group['outputMode'].value().startswith('Stabilized'):
            group['outputMode'].setValue('Mask')
        group['status'].setValue('Settings changed - Analyze masks again, then Solve motion')


def changed(group, knob):
    try:
        if group.fullName() in MIGRATING_MOTION:
            return
        ready = group.knob('analysisReady')
        name = knob.name()
    except (ValueError, RuntimeError):
        return
    if not ready:
        return
    if name in ('aspect_preset', 'aspect_lock', 'output_width', 'output_height'):
        matchmove()
        handle_crop_size(group.node('ANALYSIS'), name)
    if name in ('targetText', 'confidence', 'objectIndex', 'inputColorspace'):
        engine_changed(group.node('INFERENCE'), knob)
    if name == 'reference_frame':
        from sam3_memory import BATCHES
        batch = BATCHES.get(group.node('INFERENCE')['playbackToken'].value())
        if batch and batch.get('temporal_tracking') and int(group['reference_frame'].value()) != batch['reference_frame']:
            group.node('INFERENCE')['playbackToken'].setValue('')
            from sam3_memory import prune_batches
            prune_batches()
            ready.setValue(False)
            group['status'].setValue('Reference frame changed - Analyze the target again, then Solve')
    if name in ('first_frame', 'last_frame', 'reference_frame', 'tracking_mode', 'smoothing_window', 'crop_margin', 'fixed_crop') and ready.value():
        ready.setValue(False)
        if group['outputMode'].value().startswith('Stabilized'):
            group['outputMode'].setValue('Mask')
        group['status'].setValue('Motion settings changed - Solve again using stored RAM masks')
    if name == 'maskSource' and group['maskSource'].value() == 'Analyzed':
        cached = group.node('CACHED_MASK')
        if cached.Class() == 'Read' and not cached['file'].value():
            group['maskSource'].setValue('Live')
            group['status'].setValue('Analyze the range to create the saved mask')
    if name == 'outputMode' and group['outputMode'].value().startswith('Stabilized'):
        if not group['analysisReady'].value():
            group['outputMode'].setValue('Mask')
            group['status'].setValue('Analyze the range before selecting stabilization')


def controller_changed(controller, knob):
    try:
        name = knob.name()
        parent_name = controller.fullName().rsplit('.', 1)[0]
    except (ValueError, RuntimeError):
        return
    if name not in ('first_frame', 'last_frame', 'reference_frame', 'tracking_mode', 'smoothing_window',
                    'crop_margin', 'fixed_crop', 'aspect_preset', 'aspect_lock', 'output_width', 'output_height'):
        return
    with nuke.root():
        group = nuke.toNode(parent_name)
    if group is not None:
        changed(group, knob)


def busy(group, enabled):
    for name in ('analyzeRange', 'solveMotion', 'exportTracker', 'targetText', 'confidence', 'objectIndex', 'inputColorspace',
                 'resetRange',
                 'first_frame', 'last_frame', 'reference_frame', 'tracking_mode', 'smoothing_window', 'crop_margin',
                 'aspect_preset', 'aspect_lock', 'output_width', 'output_height'):
        group[name].setEnabled(not enabled)


def analyze(group):
    if any(job['group'] == group for job in ACTIVE.values()):
        return
    from sam3_memory import analyze as memory_analyze
    return memory_analyze(group)


def solve(group):
    if any(job['group'] == group for job in ACTIVE.values()):
        return
    from sam3_memory import solve as memory_solve
    return memory_solve(group)


def fail(group, exc):
    group['status'].setValue(str(exc))
    if nuke.GUI:
        nuke.message('SAM3: ' + str(exc))
        return None
    raise exc


def poll(token):
    entry = ACTIVE.get(token)
    if entry is None:
        return True
    sm = matchmove()
    group = entry['group']
    try:
        controller = group.node('ANALYSIS')
        if controller is None:
            raise RuntimeError('Analysis node was removed')
        complete = sm.poll_job(token)
        group['status'].setValue(controller['status'].value())
        if not complete:
            return False
        result = controller['result_file'].value()
        if result and Path(result).is_file():
            build_internal_outputs(group, result)
            group['analysisReady'].setValue(True)
            group['maskSource'].setValue('Analyzed')
            group['status'].setValue('Ready — mask and motion saved. Choose Output or Export Tracker.')
        else:
            group['status'].setValue(controller['status'].value() + ' — saved mask remains available')
    except Exception as exc:
        try:
            group['status'].setValue('Analysis failed: ' + str(exc))
        except Exception:
            if token in sm._JOBS:
                sm.cancel(sm._JOBS[token]['node'])
                sm.poll_job(token)
    finally:
        # Keep the timer for a worker that is still running.
        if token not in sm._JOBS:
            entry = ACTIVE.pop(token, None)
            if entry and 'timer' in entry:
                entry['timer'].stop()
            try:
                busy(group, False)
            except Exception:
                pass
    return token not in ACTIVE


def build_internal_outputs(group, result):
    """Legacy callback: native motion/crop nodes are now created only on Export."""
    layout_internal(group)


def cancel(group):
    from sam3_memory import CANCELLED
    CANCELLED.add(group.fullName())
    matchmove().cancel(group.node('ANALYSIS'))
    group['status'].setValue('Cancelling motion analysis...')


def export_tracker(group):
    made = export_output(group, 'Tracker')
    return made[0] if made else None


def export_selected(group):
    return export_output(group, group['export_type'].value())


def result_data(group):
    if group.knob('analysisData') and group['analysisData'].value():
        payload = group['analysisData'].value()
        if payload.startswith('b64:'):
            payload = base64.b64decode(payload[4:]).decode('utf-8')
        return json.loads(payload)
    return json.loads(Path(group.node('ANALYSIS')['result_file'].value()).read_text(encoding='utf-8'))


def export_output(group, choice):
    pasted = []
    try:
        if choice == 'Mask Read':
            choice = 'Mask (RAM)'
        if choice != 'Mask (RAM)' and not group['analysisReady'].value():
            raise ValueError('Run Solve after Analyze before exporting nodes')
        # Retain programmatic compatibility for older saved button scripts only.
        if choice not in EXPORT_CHOICES and choice != 'Mask (RAM)':
            raise ValueError('Unknown export choice: ' + choice)
        if choice == 'Mask (RAM)':
            engine = group.node('INFERENCE')
            snapshots = [(engine.Class(), engine.writeKnobs(nuke.TO_SCRIPT | nuke.WRITE_USER_KNOB_DEFS), [], 'SAM3_RAM_Mask')]
        else:
            matchmove()
            controller = group.node('ANALYSIS')
            data = result_data(group)
            from sam3_matchmove_export import build_output
            with group:
                made = list(build_output(controller, data, choice, export_nk=False).values())
                try:
                    snapshots = []
                    for item in made:
                        links = [(made.index(item.input(i)) if item.input(i) in made else None) for i in range(item.inputs())]
                        snapshots.append((item.Class(), item.writeKnobs(nuke.TO_SCRIPT | nuke.WRITE_USER_KNOB_DEFS), links, item.name()))
                finally:
                    for item in made:
                        nuke.delete(item)
        parent_name = group.fullName().rsplit('.', 1)[0] if '.' in group.fullName() else ''
        with nuke.root():
            parent = nuke.toNode(parent_name) if parent_name else nuke.root()
            with parent:
                # Stack each export below SAM3 without covering prior exports.
                x, y = group.xpos(), group.ypos() + 110
                height = max(1, len(snapshots)) * 90
                while any(abs(item.xpos()-x) < 110 and y-50 < item.ypos() < y+height
                          for item in nuke.allNodes() if item != group):
                    y += 100
                for kind, script, links, name in snapshots:
                    node = getattr(nuke.nodes, kind)()
                    pasted.append(node)
                    node.readKnobs(script)
                    node.setName(name, uncollide=True)
                    node.setInput(0, None)
                for offset, (item, (_, _, links, _)) in enumerate(zip(pasted, snapshots)):
                    for index, source in enumerate(links):
                        if source is not None:
                            item.setInput(index, pasted[source])
                    if item.Class() in ('Tracker4', 'Transform') and choice in ('Tracker', 'Stabilize', 'Plate Stabilize Crop'):
                        item.setInput(0, group.input(0))
                    if choice == 'Mask (RAM)':
                        item['knobChanged'].setValue('')
                        configure_index(item)
                        item['onCreate'].setValue('import sam3_unified; sam3_unified.configure_index(nuke.thisNode())')
                        item['captureToken'].setValue('')
                        item['label'].setValue('RAM mask / exported settings')
                        item.setInput(0, group.input(0))
                    item.setXYpos(x, y + offset*90)
        group['status'].setValue('Exported ' + choice)
        return pasted
    except Exception as exc:
        for item in pasted:
            nuke.delete(item)
        return fail(group, exc)


def engine_status(group):
    import sam3_ofx_nuke
    sam3_ofx_nuke.status(group)


def stop_engine(group):
    import sam3_ofx_nuke
    sam3_ofx_nuke.stop(group)
