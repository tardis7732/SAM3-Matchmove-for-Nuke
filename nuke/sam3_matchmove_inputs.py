"""Prepare existing image files or persistent Writes for explicit user rendering.

This module never renders and never reconnects a controller input. A prepared
Write remains in the graph so the artist chooses when its files are updated.
"""
from pathlib import Path


def _api():
    import sam3_matchmove_nuke as sm
    return sm, sm._nuke()


def _prepared_write(node, source, role, run_dir, first, last):
    sm, nuke = _api()
    owner = node["sam3_id"].value()
    for writer in nuke.allNodes("Write", recurseGroups=True):
        knobs = writer.knobs()
        if ("sam3_owner" in knobs and "sam3_role" in knobs
                and writer["sam3_owner"].value() == owner
                and writer["sam3_role"].value() == role
                and writer.input(0) == source):
            for key, value in (("use_limit", True), ("first", first), ("last", last)):
                writer[key].setValue(value)
            return writer, False

    folder = Path(run_dir) / "input"
    folder.mkdir(parents=True, exist_ok=True)
    mask = role == "mask"
    writer = sm._make("Write", "SAM3_Mask_Input_Write" if mask else "SAM3_Input_Write", inputs=[source])
    try:
        for key, value in (("sam3_owner", owner), ("sam3_role", role)):
            knob = nuke.String_Knob(key, "")
            writer.addKnob(knob)
            knob.setValue(value)
            knob.setVisible(False)
        pattern = folder / ("mask_input.%06d.png" if mask else "source.%06d.png")
        writer["file"].setValue(str(pattern).replace("\\", "/"))
        writer["file_type"].setValue("png")
        writer["channels"].setValue("rgba" if mask else "rgb")
        writer["datatype"].setValue("8 bit")
        writer["raw"].setValue(mask)
        if not mask:
            sm._srgb(writer)
        for key, value in (("use_limit", True), ("first", first), ("last", last)):
            writer[key].setValue(value)
        writer["label"].setValue("Render %s, then Analyze\n%d-%d" % ("mask" if mask else "plate", first, last))
        writer["file"].setTooltip("Render this PNG sequence, then click Analyze. Render again after changing the upstream graph.")
        writer.setXYpos(node.xpos() + (160 if mask else -160), node.ypos() - 70)
    except Exception:
        nuke.delete(writer)
        raise
    return writer, True


def _write_frames(writer, first, last):
    _, nuke = _api()
    if writer["file_type"].value() != "png":
        raise ValueError("Set the prepared input Write to PNG before analysis.")
    previous_frame = int(nuke.root()["frame"].value())
    rows = []
    try:
        for frame in range(first, last + 1):
            nuke.frame(frame)
            filename = nuke.filename(writer, nuke.REPLACE)
            if not filename:
                return None
            path = Path(filename)
            if path.suffix.lower() != ".png":
                raise ValueError("The prepared input Write needs a .png output filename.")
            if not path.is_file() or path.stat().st_size == 0:
                return None
            rows.append({"frame": frame, "path": str(path)})
    finally:
        nuke.frame(previous_frame)
    return rows


def _direct_masks(source, first, last):
    sm, _ = _api()
    rows, reason = sm._direct_source_frames(source, first, last)
    if rows is not None and all(Path(row["path"]).suffix.lower() in sm._DIRECT_IMAGES for row in rows):
        return rows, ""
    return None, reason.replace("Plate", "Mask") if reason else "A mask input needs image files or a rendered PNG Write."


def prepare_inputs(node, run_dir, first, last, backend):
    """Return resolved job frames, or None until all prepared Writes are rendered.

The caller validates formats, range, and mask-source availability. Existing
Write files are reused only for this controller, role, and exact input node.
Upstream edits require the artist to render that persistent Write again.
"""
    sm, nuke = _api()
    selection = list(nuke.selectedNodes())
    current_frame = int(nuke.root()["frame"].value())
    try:
        source = node.input(0)
        frames, reason = sm._direct_source_frames(source, first, last)
        direct = frames is not None
        created = False
        plate_writer = None
        if not direct:
            plate_writer, is_new = _prepared_write(node, source, "plate", run_dir, first, last)
            created = created or is_new

        mask_rows = None
        mask_writer = None
        if backend == "masks":
            mask_source = node.input(1)
            mask_rows, mask_reason = _direct_masks(mask_source, first, last)
            if mask_rows is None:
                mask_writer, is_new = _prepared_write(node, mask_source, "mask", run_dir, first, last)
                created = created or is_new
                reason = reason or mask_reason

        # Prepare both nodes before checking their files, so one Analyze click
        # exposes every required manual render in a Plate + intermediate-mask graph.
        if plate_writer is not None:
            frames = _write_frames(plate_writer, first, last)
        if mask_writer is not None:
            mask_rows = _write_frames(mask_writer, first, last)
        if frames is None or (backend == "masks" and mask_rows is None):
            sm._set_input_info(node, "Input requires conversion; Write created" if created else
                               "Render the input Write, then Analyze",
                               (reason + "\n\n" if reason else "") +
                               "Render the prepared input Write, then click Analyze. "
                               "Render again after changing the upstream graph.")
            return None

        frames = [dict(row) for row in frames]
        if mask_rows is not None:
            for plate, mask in zip(frames, mask_rows):
                plate["mask_path"] = mask["path"]
        sm._set_input_info(node, "Direct Read" if direct else "Rendered Write files ready")
        return {"frames": frames, "input_mode": "direct" if direct else "rendered_write",
                "analysis_colorspace": "source encoded RGB" if direct else plate_writer["colorspace"].value(),
                "render_reason": reason}
    finally:
        nuke.frame(current_frame)
        for item in nuke.selectedNodes():
            item["selected"].setValue(False)
        for item in selection:
            item["selected"].setValue(True)
