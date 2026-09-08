"""Export an existing similarity result through Nuke's native Tracker4."""
import math
import re


_DERIVED_HELP = ('Four virtual points are derived from the saved SAM3 translation, rotation and scale. '
                 'They are not independently measured feature tracks. Change the native Reference frame '
                 'and use the native Export menu to create linked or baked transforms. Linked exports '
                 'follow changes to this Tracker; baked exports keep their exported values.')


def _curve(rows, values):
    return '{curve L ' + ' '.join('x%d %.17g' % (row['frame'], value)
                                 for row, value in zip(rows, values)) + '}'


def _point(row, point, center):
    angle = math.radians(row['rotate'])
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y = point[0] - center[0], point[1] - center[1]
    scale = row['scale']
    return [center[0] + row['translate'][0] + scale * (cosine*x - sine*y),
            center[1] + row['translate'][1] + scale * (sine*x + cosine*y)]


def _track_table(tracker, data):
    """Serialize Table_Knob rows because add_track is GUI-dependent in Nuke.

    Keep the column definitions supplied by the installed Tracker. Refuse an
    unknown layout instead of silently writing coordinates into wrong columns.
    """
    template = tracker['tracks'].toScript()
    header = re.match(r'\s*\{\s*1\s+(\d+)\s+0\s*\}', template)
    expected = ('enable', 'name', 'track_x', 'track_y', 'offset_x', 'offset_y', 'T', 'R', 'S')
    columns = re.findall(r'\{\s*\d+\s+\d+\s+\d+\s+(\S+)\s+\S+\s+\d+\s*\}', template)
    if not header or int(header.group(1)) != 31 or tuple(columns[:9]) != expected:
        raise RuntimeError('This Nuke Tracker table layout is not supported.')
    # A new node has an empty final rows block. Replace its row count and fill it.
    rows_start = template.rfind('{')
    if rows_start <= header.end() or template[rows_start:].strip() != '{ \n}'.strip():
        # Whitespace differs between Nuke builds.
        if rows_start <= header.end() or not re.fullmatch(r'\{\s*\}\s*', template[rows_start:]):
            raise RuntimeError('Expected an empty native Tracker table.')
    prefix = re.sub(r'^\s*\{\s*1\s+31\s+0\s*\}', '{ 1 31 4 }', template[:rows_start], count=1)
    box = data.get('reference_box')
    if box is None:
        box = next(row['crop_box'] for row in data['frames'] if row['frame'] == data['reference_frame'])
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in box)):
        raise ValueError('The analyzed reference box must contain four finite coordinates.')
    left, bottom, right, top = box
    if right <= left or top <= bottom:
        raise ValueError('The analyzed reference box must be nonempty.')
    center = data['reference_center']
    points = ((left, bottom), (right, bottom), (right, top), (left, top))
    result = []
    for index, point in enumerate(points):
        coordinates = [_point(row, point, center) for row in data['frames']]
        cells = ['1', '"Derived %d"' % (index + 1),
                 _curve(data['frames'], [point[0] for point in coordinates]),
                 _curve(data['frames'], [point[1] for point in coordinates]),
                 '0', '0', '1', '1', '1', '0', '0', '0',
                 '-10', '-10', '10', '10', '-7', '-7', '7', '7',
                 '{}', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        result.append('{ ' + ' '.join(cells) + ' }')
    return prefix + '{\n' + '\n'.join(result) + '\n}\n'


def build_tracker(controller, data):
    """Create one native Tracker, leaving placement and saving to the caller."""
    import nuke
    import sam3_matchmove_nuke as sm

    sm.validate_result(data)
    reference = int(controller['reference_frame'].value())
    if reference not in {row['frame'] for row in data['frames']}:
        raise ValueError('Reference must be inside the analyzed range (%d to %d).' %
                         (data['frames'][0]['frame'], data['frames'][-1]['frame']))
    selection = list(nuke.selectedNodes())
    tracker = None
    try:
        source = controller.input(0)
        tracker = nuke.nodes.Tracker4(inputs=[source] if source is not None else [])
        tracker.setName('SAM3_Tracker', uncollide=True)
        tracker.setInput(0, source)
        if not tracker['tracks'].fromScript(_track_table(tracker, data)):
            raise RuntimeError('Nuke could not load the derived Tracker points.')
        tracker['tracks'].setTooltip(_DERIVED_HELP)
        tracker['label'].setValue('Derived from SAM3 motion')
        tracker['transform'].setValue('none')
        tracker['reference_frame'].setValue(reference)
        tracker['livelink_transform'].setValue(True)
        tracker['cornerPinOptions'].setValue('Transform (match-move)')
        # Force native validation, not image rendering, so export can use the
        # solved transform even before a Viewer requests this Tracker's pixels.
        tracker.forceValidate()
        return tracker
    except Exception:
        if tracker is not None:
            nuke.delete(tracker)
        raise
    finally:
        for item in nuke.selectedNodes():
            item['selected'].setValue(False)
        for item in selection:
            item['selected'].setValue(True)
