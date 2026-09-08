"""Output dimensions and aspect-ratio controls for native Nuke exports."""
from __future__ import annotations

import math


ASPECT_PRESETS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2.39:1", "Custom")
_CONTROL_NAMES = ("aspect_preset", "aspect_lock", "output_width", "output_height", "aspect_ratio")
_CHANGING = set()


def _nuke():
    import nuke
    return nuke


def _key(node):
    return node.fullName()


def _ratio(preset):
    width, height = preset.split(":")
    return float(width) / float(height)


def _preset_for_ratio(ratio):
    for preset in ASPECT_PRESETS[:-1]:
        if math.isclose(ratio, _ratio(preset), rel_tol=1e-8):
            return preset
    return "Custom"


def _rounded(value):
    return max(8, int(math.floor(float(value) + 0.5)))


def output_dimensions(node):
    """Return valid integer export dimensions, supporting pre-upgrade nodes."""
    knobs = node.knobs()
    if "output_width" in knobs and "output_height" in knobs:
        values = (node["output_width"].value(), node["output_height"].value())
    elif "crop_resolution" in knobs:
        values = (node["crop_resolution"].value(),) * 2
    else:
        values = (720, 720)
    try:
        width, height = (int(value) for value in values)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Output width and height must be whole pixel values of at least 8.") from None
    if width < 8 or height < 8 or any(float(value) != int(value) for value in values):
        raise ValueError("Output width and height must be whole pixel values of at least 8.")
    return width, height


def ensure_size_controls(node, new=False):
    """Add dimensions once and retain all existing size values on later calls."""
    nuke = _nuke()
    key = _key(node)
    if key in _CHANGING:
        return
    _CHANGING.add(key)
    try:
        names = node.knobs()
        legacy_size = 720 if new else int(node["crop_resolution"].value()) if "crop_resolution" in names else 720
        width = int(node["output_width"].value()) if "output_width" in names else legacy_size
        height = int(node["output_height"].value()) if "output_height" in names else legacy_size
        ratio = float(width) / height if width > 0 and height > 0 else 1.0
        factories = {
            "aspect_preset": lambda: nuke.Enumeration_Knob("aspect_preset", "Aspect ratio", list(ASPECT_PRESETS)),
            "aspect_lock": lambda: nuke.Boolean_Knob("aspect_lock", "Lock ratio"),
            "output_width": lambda: nuke.Int_Knob("output_width", "Width"),
            "output_height": lambda: nuke.Int_Knob("output_height", "Height"),
            "aspect_ratio": lambda: nuke.Double_Knob("aspect_ratio", ""),
        }
        defaults = {"aspect_preset": _preset_for_ratio(ratio), "aspect_lock": True,
                    "output_width": width, "output_height": height, "aspect_ratio": ratio}
        controls = {}
        missing = False
        for name in _CONTROL_NAMES:
            if name in names:
                controls[name] = node[name]
            else:
                missing = True
                controls[name] = factories[name]()
                controls[name].setValue(defaults[name])
        if missing:
            # Insert beside the previous size setting, preserving all user knob
            # objects (including animation and custom scripts) in the tail.
            all_names = list(node.knobs())
            anchor = "crop_resolution" if "crop_resolution" in all_names else "fixed_crop"
            if anchor in all_names:
                tail = [node[name] for name in all_names[all_names.index(anchor) + 1:]
                        if name not in _CONTROL_NAMES]
                for name in all_names[all_names.index(anchor) + 1:]:
                    node.removeKnob(node[name])
                for name in _CONTROL_NAMES:
                    if name in node.knobs():
                        node.removeKnob(node[name])
                for knob in [controls[name] for name in _CONTROL_NAMES] + tail:
                    node.addKnob(knob)
            else:
                for name in _CONTROL_NAMES:
                    if name not in node.knobs():
                        node.addKnob(controls[name])
        for name in ("aspect_preset", "output_width"):
            node[name].setFlag(nuke.STARTLINE)
        for name in ("aspect_lock", "output_height"):
            node[name].clearFlag(nuke.STARTLINE)
        for name in ("output_width", "output_height"):
            node[name].setFlag(nuke.NO_ANIMATION)
            node[name].setRange(8, 16384)
        node["aspect_ratio"].setVisible(False)
        if "crop_resolution" in node.knobs():
            node["crop_resolution"].setVisible(False)
        if "fixed_crop" in node.knobs():
            node["fixed_crop"].setLabel("Fixed crop size")
        tooltips = {
            "aspect_preset": "Choose an output width-to-height ratio. A preset locks the ratio and adjusts Height using the current Width. Custom uses the current dimensions. Crop exports keep the image proportions and expand the crop to fit this ratio.",
            "aspect_lock": "Keep the chosen width-to-height ratio when editing either Width or Height. Disable for independent dimensions. Enabling the lock captures the current dimensions as the ratio.",
            "output_width": "Exported crop width in pixels, at least 8. With Lock ratio enabled, changing Width also adjusts Height. Size changes apply to the next Plate Stabilize Crop or Generated Crop Matchmove export; reanalysis is not required.",
            "output_height": "Exported crop height in pixels, at least 8. With Lock ratio enabled, changing Height also adjusts Width. Disable Lock ratio to edit the two dimensions independently.",
            "aspect_ratio": "Stored width-to-height ratio used by the dimension controls.",
            "crop_resolution": "Legacy square-size value retained for older scripts. Use Width and Height for new exports.",
        }
        for name, tooltip in tooltips.items():
            if name in node.knobs():
                node[name].setTooltip(tooltip)
    finally:
        _CHANGING.discard(key)


def handle_size_change(node, knob):
    """Respond to one committed knob edit; nested changes are ignored."""
    key = _key(node)
    name = knob.name() if hasattr(knob, "name") else str(knob)
    if key in _CHANGING or name not in _CONTROL_NAMES + ("crop_resolution",):
        return
    if not all(control in node.knobs() for control in _CONTROL_NAMES):
        return
    _CHANGING.add(key)
    try:
        if name == "aspect_ratio":
            return
        width, height = node["output_width"], node["output_height"]
        lock, ratio_knob, preset = node["aspect_lock"], node["aspect_ratio"], node["aspect_preset"]
        if name == "crop_resolution":
            size = _rounded(node["crop_resolution"].value())
            width.setValue(size)
            height.setValue(size)
            ratio_knob.setValue(1.0)
            preset.setValue("1:1")
            lock.setValue(True)
            return
        if name == "aspect_preset":
            ratio = (max(8, width.value()) / max(8, height.value())
                     if preset.value() == "Custom" else _ratio(preset.value()))
            ratio_knob.setValue(ratio)
            lock.setValue(True)
            name = "output_width"
        elif name == "aspect_lock":
            if lock.value():
                ratio = max(8, width.value()) / max(8, height.value())
                ratio_knob.setValue(ratio)
                preset.setValue(_preset_for_ratio(ratio))
            return
        if not lock.value():
            changed = width if name == "output_width" else height
            changed.setValue(_rounded(changed.value()))
            ratio = max(8, width.value()) / max(8, height.value())
            ratio_knob.setValue(ratio)
            preset.setValue(_preset_for_ratio(ratio))
            return
        ratio = ratio_knob.value()
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("The locked output aspect ratio must be positive.")
        if name == "output_width":
            value = max(_rounded(width.value()), int(math.ceil(8 * ratio)))
            width.setValue(value)
            height.setValue(_rounded(value / ratio))
        elif name == "output_height":
            value = max(_rounded(height.value()), int(math.ceil(8 / ratio)))
            height.setValue(value)
            width.setValue(_rounded(value * ratio))
    finally:
        _CHANGING.discard(key)
