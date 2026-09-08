"""Create one selected native Nuke output, without importing ML dependencies."""
from pathlib import Path


OUTPUT_CHOICES = ('Tracker', 'Matchmove', 'Stabilize', 'Plate Stabilize Crop',
                  'Generated Crop Matchmove', 'Mask Read')
_CHOICE_ALIASES = {
    'Crop': 'Plate Stabilize Crop',
    'Square crop': 'Plate Stabilize Crop',
    'Crop to plate': 'Generated Crop Matchmove',
    'Square to plate': 'Generated Crop Matchmove',
}
_FILE_STEMS = {
    'Tracker': 'tracker',
    'Matchmove': 'matchmove',
    'Stabilize': 'stabilize',
    'Plate Stabilize Crop': 'crop',
    'Generated Crop Matchmove': 'crop_to_plate',
    'Mask Read': 'mask_read',
}


def normalize_choice(choice):
    """Retain output selections saved with earlier display names."""
    return _CHOICE_ALIASES.get(choice, choice)


def crop_mapping(row, width, height):
    """Expand a tracked square to the output aspect without stretching it.

    The shorter output edge covers the original square. The longer edge adds
    plate context equally on either side of the same tracked center.
    """
    left, bottom, right, top = row['crop_box']
    scale = (right - left) / min(width, height)
    return ([(left + right - width * scale) / 2,
             (bottom + top - height * scale) / 2], [scale, scale])


def _reserve_export_path(directory, stem):
    """Reserve a new filename; never overwrite a previous exported graph."""
    index = 0
    while True:
        suffix = '' if index == 0 else '_%03d' % index
        path = directory / (stem + suffix + '.nk')
        try:
            with path.open('x', encoding='utf-8'):
                pass
            return path
        except FileExistsError:
            index += 1


def build_output(controller, data, choice, export_nk=True):
    """Build only the selected Transform, crop pair, or recoverable Mask Read.

    Matchmove and crop-to-plate inputs remain disconnected for inserted
    footage. Stabilize and crop inputs use the controller's plate. Mask Read
    creates a separate Read without changing the controller's input connections.
    Returned dictionary values are the newly created native Nuke nodes only.
    """
    # Keep both imports lazy, including when the controller imports this module.
    import nuke
    import sam3_matchmove_nuke as sm

    choice = normalize_choice(choice)
    if choice not in OUTPUT_CHOICES:
        raise ValueError('Choose an output: ' + ', '.join(OUTPUT_CHOICES))
    sm.validate_result(data)
    width, height = sm.output_dimensions(controller) if choice in ('Plate Stabilize Crop', 'Generated Crop Matchmove') else (0, 0)
    source = controller.input(0)
    if choice in ('Tracker', 'Stabilize', 'Plate Stabilize Crop') and source is not None:
        fmt = source.format()
        if (fmt.width(), fmt.height()) != (data['width'], data['height']):
            raise ValueError('Current plate format differs from the analyzed result. Reconnect the matching plate.')
    directory = None
    result_file = controller['result_file'].value().strip()
    if export_nk or choice == 'Mask Read':
        if not result_file:
            raise ValueError('Analyze the plate or select a Result JSON before exporting nodes.')
        directory = Path(result_file).parent
        if not directory.is_dir():
            raise ValueError('The analysis output folder does not exist: %s' % directory)

    selection = list(nuke.selectedNodes())
    made = []
    nodes = {}
    export_path = None
    previous_mask = controller.input(1)
    previous_values = {key: controller[key].value() for key in
                       ('backend', 'mask_channel', 'result_file', 'input_info', 'status')
                       if controller.knob(key) is not None}
    # Every choice starts at the same place, including repeated exports.
    x = controller.xpos()
    y = controller.ypos() + 150

    def make(kind, key, name, source_node=None, row=0):
        item = getattr(nuke.nodes, kind)(inputs=[source_node] if source_node is not None else [])
        # Register immediately so a later naming/knob failure cleans it up too.
        made.append(item)
        nodes[key] = item
        if name:
            item.setName(name, uncollide=True)
        item.setInput(0, source_node)
        item.setXYpos(x, y + row * 80)
        return item

    try:
        if choice == 'Tracker':
            from sam3_matchmove_tracker import build_tracker
            tracker = build_tracker(controller, data)
            made.append(tracker)
            nodes['tracker'] = tracker
            tracker.setXYpos(x, y)
        elif choice == 'Mask Read':
            reader = sm._create_mask_read(controller, result_file)
            made.append(reader)
            nodes['mask_read'] = reader
            reader.setXYpos(x, y)
            # Legacy mask conversion can create a new result directory.
            directory = Path(controller['result_file'].value()).parent
        elif choice in ('Matchmove', 'Stabilize'):
            inverse = choice == 'Stabilize'
            key = 'stabilize' if inverse else 'matchmove'
            name = 'SAM3_Stabilize' if inverse else 'SAM3_Matchmove_Transform'
            transform = make('Transform', key, name, source if inverse else None)
            sm._animate_transform(transform, data, inverse=inverse)
            transform['label'].setValue('%s\nReference %d' % (choice, data['reference_frame']))
        else:
            crop = choice == 'Plate Stabilize Crop'
            key = 'crop_transform' if crop else 'uncrop'
            name = 'SAM3_PlateToCrop' if crop else 'SAM3_CropToPlate'
            transform = make('Transform', key, name, source if crop else None)
            transform['center'].setValue([0, 0])
            transform['invert_matrix'].setValue(crop)
            sm._keys(transform['translate'], data['frames'], lambda row: crop_mapping(row, width, height)[0], 2)
            sm._keys(transform['scale'], data['frames'],
                     lambda row: crop_mapping(row, width, height)[1], 2)
            if not crop:
                transform['label'].setValue('Connect %d x %d\ncropped / generated footage' % (width, height))
            reformat = make('Reformat', 'crop' if crop else 'uncrop_format',
                            None, transform, row=1)
            reformat['type'].setValue('to box')
            reformat['box_width'].setValue(width if crop else data['width'])
            reformat['box_height'].setValue(height if crop else data['height'])
            reformat['box_fixed'].setValue(True)
            reformat['resize'].setValue('none')
            reformat['center'].setValue(False)
        if export_nk:
            for item in nuke.selectedNodes():
                item['selected'].setValue(False)
            for item in made:
                item['selected'].setValue(True)
            export_path = _reserve_export_path(directory, _FILE_STEMS[choice])
            nuke.nodeCopy(str(export_path))
            if not export_path.is_file() or not export_path.stat().st_size:
                raise RuntimeError('Nuke did not write the exported node file.')
        controller['status'].setValue('Exported %s (%d frames)' % (choice, len(data['frames'])))
        return nodes
    except Exception:
        if choice == 'Mask Read':
            controller.setInput(1, previous_mask)
            for key, value in previous_values.items():
                controller[key].setValue(value)
        if export_path is not None and export_path.is_file():
            export_path.unlink()
        for item in reversed(made):
            nuke.delete(item)
        raise
    finally:
        for item in nuke.selectedNodes():
            item['selected'].setValue(False)
        for item in selection:
            item['selected'].setValue(True)
