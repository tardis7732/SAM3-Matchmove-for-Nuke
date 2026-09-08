"""Nuke UI and native-node baking. No ML packages are imported into Nuke."""
from __future__ import annotations

import datetime
import functools
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

from sam3_matchmove_export import OUTPUT_CHOICES, normalize_choice

ROOT = Path(__file__).resolve().parents[1]
_JOBS = {}
_MENU_REGISTERED = False
_LOAD_REGISTERED = False
_SIZE_CALLBACK_REGISTERED = False
_DIRECT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_DIRECT_VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_INPUT_HELP = (
    "A supported Read connected directly to Plate is analyzed from the original files.\n"
    "Unsupported inputs create a PNG Write. Render it, then click Analyze again.\n"
    "Plate stays connected to the original input. Re-render the Write after upstream changes.\n"
    "Completed RGBA masks connect to Mask without changing Mask source.\n"
    "Select Input mask yourself to analyze the connected mask instead of running SAM3.\n\n"
    "Images: JPG, JPEG, PNG, BMP, TIFF, WEBP.\n"
    "Videos: MP4, MOV, AVI, MKV, WEBM (decoder support required).\n"
    "Custom video timing and proxy settings require a rendered input."
)
_INPUT_STATES = {"Automatic", "Direct Read", "Input requires conversion; Write created",
                 "Render the input Write, then Analyze", "Rendered Write files ready", "RGBA mask connected"}
_EXPORT_CHOICES = list(OUTPUT_CHOICES)


def _insert_after(node, name, knobs):
    names = list(node.knobs())
    tail = [node[key] for key in names[names.index(name) + 1:]]
    for knob in tail:
        node.removeKnob(knob)
    for knob in knobs + tail:
        node.addKnob(knob)


def _ensure_export_controls(node):
    nuke = _nuke()
    if "auto_bake" in node.knobs():
        node.removeKnob(node["auto_bake"])
    if "result_file" not in node.knobs():
        return
    if "export_type" not in node.knobs():
        if "bake" in node.knobs():
            node.removeKnob(node["bake"])
        _insert_after(node, "open_output" if "open_output" in node.knobs() else "result_file", [
            nuke.Text_Knob("export_section", "Export", ""),
            nuke.Enumeration_Knob("export_type", "Output", _EXPORT_CHOICES),
            _button("bake", "Export", "bake_button"),
        ])
    node["bake"].setLabel("Export")
    node["bake"].clearFlag(nuke.STARTLINE)
    previous = node["export_type"].value()
    previous = normalize_choice(previous)
    if node["export_type"].values() != _EXPORT_CHOICES:
        node["export_type"].setValues(_EXPORT_CHOICES)
        node["export_type"].setValue(previous if previous in _EXPORT_CHOICES else "Tracker")
    # A labelled knob uses Nuke's normal input column, including on old nodes.
    node["export_type"].setLabel("Output")
    node["export_type"].setFlag(nuke.STARTLINE)
    node["export_type"].setFlag(nuke.EXPAND_TO_WIDTH)


def _set_tooltips(node):
    descriptions = {
        "intro": "Track one object in the Plate image using SAM3 or an existing full-frame mask.",
        "backend": "SAM3 text generates masks from Target text. Input mask uses the connected Mask input without running SAM3. Analysis connects its RGBA mask without changing this setting.",
        "tracking_mode": "BBox estimates position and uniform scale from the mask, with zero rotation. Features tracks image points inside the mask and fits 2D translation, scale and rotation. This is not a 3D pose estimate.",
        "target_text": "Describe a visible target in English, such as face or red car. Used only in SAM3 text mode. New nodes default to face.",
        "object_index": "Choose a reference-frame detection by size: 0 is the largest, 1 the next largest. Its object identity remains locked across the shot.",
        "detection_threshold": "Minimum SAM3 detection score from 0 to 1. Lower values admit weaker detections. This is not a measure of tracking accuracy.",
        "mask_channel": "Channel used from the connected Input mask. Use alpha for Roto/RGBA mattes or red for grayscale images. The mask must match Plate coordinates and resolution.",
        "first_frame": "First timeline frame to analyze, inclusive. Reset reads the connected Plate output range.",
        "last_frame": "Last timeline frame to analyze, inclusive. Reset reads the connected Plate output range.",
        "reference_frame": "Analysis reference frame, where the target must be visible. Tracker exports start at this reference; use the native Tracker Reference control to change it after export without rerunning SAM3.",
        "use_frame": "Use the current Nuke timeline frame as Reference.",
        "smoothing_window": "Odd number of frames used for symmetric smoothing. 1 disables smoothing; larger values reduce jitter but soften fast movement.",
        "crop_margin": "Extra space around the tracked mask. 1 fits the mask; 1.2 adds 20 percent. Export expands this region to the selected output aspect ratio without stretching the image. Reanalyze after changing the margin.",
        "fixed_crop": "Use one crop size across the shot. Disable to let the crop follow the target size frame by frame. Reanalyze after changing this setting. Aspect ratio lock is a separate output setting.",
        "crop_resolution": "Legacy square output size retained for older scripts. Use the visible Width and Height settings for new exports.",
        "analyze": "Analyze the requested frames. If input conversion is needed, create a Write and wait for you to render it. Click Analyze again after rendering. Analysis connects an RGBA mask but does not export motion nodes.",
        "cancel": "Stop this controller's external worker and retain the previous tracking result.",
        "status": "Current preparation, analysis or export status. Detailed input instructions are available on the Input tooltip.",
        "result_file": "Saved tracking data used by Export. Select an existing result.json to export its motion without running analysis again.",
        "open_output": "Open the folder containing the current tracking result, or the configured output folder if no result is loaded.",
        "export_section": "Create native Nuke nodes from the loaded tracking data.",
        "export_type": "Tracker stores solved motion in native Nuke tracks with Reference and Export controls. Matchmove and Stabilize create baked Transforms. Plate Stabilize Crop extracts the tracked region from Plate. Generated Crop Matchmove places a generated or edited crop back onto Plate. Both crop outputs use Width and Height. Mask Read creates a separate saved-mask Read without changing any input connections.",
        "bake": "Create only the selected output below this node using Result JSON. Use Tracker for Nuke's native Reference and Export controls. Matchmove and Stabilize are baked Transform snapshots. Mask Read creates an unconnected Read. No analysis is rerun.",
        "python_exe": "External Python used by the worker. SAM3 requires the configured Python 3.12+ environment; do not use Nuke's bundled Python.",
        "checkpoint": "Local official SAM3 sam3.pt weights. Leave empty only if the worker environment can resolve the model through Hugging Face.",
        "output_root": "Folder for tracking results, RGBA masks, prepared input Writes and exported node scripts. Files referenced by Read nodes must remain available.",
        "check_environment": "Check worker dependencies and GPU availability for the selected Mask source without loading the model or running inference.",
        "help": "SAM3 inference runs outside Nuke. Input mask tracking requires NumPy, OpenCV and Pillow in the worker environment.",
    }
    for name, description in descriptions.items():
        if name in node.knobs():
            node[name].setTooltip(description)


def _nuke():
    import nuke
    return nuke


def _config():
    path = ROOT / "config.local.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _make(kind, name, **kwargs):
    node = getattr(_nuke().nodes, kind)(**kwargs)
    node.setName(name, uncollide=True)
    return node


def _in_controller_parent(callback):
    """Create graph nodes beside the controller, including from Group buttons."""
    @functools.wraps(callback)
    def wrapped(node, *args, **kwargs):
        nuke = _nuke()
        name = node.fullName()
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        previous = nuke.thisGroup()
        try:
            with nuke.root():
                parent = nuke.toNode(parent_name) if parent_name else nuke.root()
                if parent is None:
                    raise ValueError("The controller's parent group is unavailable.")
                with parent:
                    return callback(node, *args, **kwargs)
        finally:
            previous.begin()
    return wrapped


def _message(value):
    nuke = _nuke()
    (nuke.message if nuke.GUI else nuke.tprint)(str(value))


def _button(name, label, callback):
    knob = _nuke().PyScript_Knob(name, label)
    knob.setCommand("import sam3_matchmove_nuke as sm; sm.%s(nuke.thisNode())" % callback)
    return knob


def _ensure_range_controls(node):
    nuke = _nuke()
    if not all(name in node.knobs() for name in ("first_frame", "last_frame", "reference_frame")):
        return
    if "reset_range" not in node.knobs():
        # Nuke appends new knobs. Reattach the existing user knobs to insert the
        # button beside Last, retaining their values, expressions and commands.
        names = list(node.knobs())
        tail = [node[name] for name in names[names.index("last_frame") + 1:]]
        for knob in tail:
            node.removeKnob(knob)
        node.addKnob(_button("reset_range", "Reset", "reset_frame_range"))
        for knob in tail:
            node.addKnob(knob)
    node["first_frame"].setFlag(nuke.STARTLINE)
    node["last_frame"].clearFlag(nuke.STARTLINE)
    node["reset_range"].clearFlag(nuke.STARTLINE)
    node["reference_frame"].setFlag(nuke.STARTLINE)
    node["reset_range"].setTooltip("Set First and Last from the connected Plate input's output frame range. Upstream retiming and FrameRange nodes are respected. Reference is kept within this range.")


def ensure_controls(node, new=False):
    """Restore the original controls on controllers saved by either version."""
    nuke = _nuke()
    _ensure_range_controls(node)
    _ensure_export_controls(node)
    if "sam3_id" not in node.knobs():
        node.addKnob(nuke.String_Knob("sam3_id", ""))
        node["sam3_id"].setValue(uuid.uuid4().hex)
        node["sam3_id"].setVisible(False)
    for name in ("use_translate", "use_rotate", "use_scale", "export_mode", "transform_export"):
        if name in node.knobs():
            node.removeKnob(node[name])
    if "crop_margin" not in node.knobs():
        if not new:
            node.addKnob(nuke.Tab_Knob("crop_settings", "Crop"))
        node.addKnob(nuke.Double_Knob("crop_margin", "Crop margin"))
        node["crop_margin"].setRange(1, 3)
        node["crop_margin"].setValue(1.2)
    if "fixed_crop" not in node.knobs():
        node.addKnob(nuke.Boolean_Knob("fixed_crop", "Fixed crop size"))
        node["fixed_crop"].setValue(True)
    if "crop_resolution" not in node.knobs():
        node.addKnob(nuke.Int_Knob("crop_resolution", "Square output size"))
        node["crop_resolution"].setValue(720 if new else 512)
    for name in ("crop_margin", "fixed_crop"):
        node[name].setVisible(True)
    node["fixed_crop"].setLabel("Fixed crop size")
    from sam3_matchmove_size import ensure_size_controls
    ensure_size_controls(node, new=new)
    _register_size_callback()
    if "tracking_mode" in node.knobs():
        node["tracking_mode"].setLabel("Motion")
        node["tracking_mode"].setTooltip("BBox estimates position and scale, with zero rotation. Features tracks points inside the mask and fits 2D translation, uniform scale and rotation relative to Reference. This is not a 3D head pose estimate.")
    if "help" in node.knobs():
        node["help"].setValue("SAM3 needs a separate Python 3.12+ environment and approved Meta weights.\nInput mask mode needs NumPy, OpenCV and Pillow only.\nMotion is a 2D similarity estimate.")
    if "input_info" not in node.knobs():
        node.addKnob(nuke.Text_Knob("input_info", "Input", "Automatic"))
    if node["input_info"].value() not in _INPUT_STATES:
        node["input_info"].setValue("Automatic")
    node["input_info"].setTooltip(_INPUT_HELP)
    _set_tooltips(node)
    _upgrade_runtime(node)


def output_dimensions(node):
    from sam3_matchmove_size import output_dimensions as dimensions
    return dimensions(node)


def _size_knob_changed():
    nuke = _nuke()
    node, knob = nuke.thisNode(), nuke.thisKnob()
    try:
        ready = node is not None and knob is not None and node.knob("output_width") is not None
    except (ValueError, RuntimeError):
        # Nuke also dispatches callbacks as a Group is being deleted.
        return
    if ready:
        from sam3_matchmove_size import handle_size_change
        handle_size_change(node, knob)


def _register_size_callback():
    global _SIZE_CALLBACK_REGISTERED
    if _SIZE_CALLBACK_REGISTERED:
        return
    nuke = _nuke()
    previous = getattr(nuke, "_sam3_size_callback", None)
    if previous is not None:
        nuke.removeKnobChanged(previous, nodeClass="Group")
    nuke.addKnobChanged(_size_knob_changed, nodeClass="Group")
    nuke._sam3_size_callback = _size_knob_changed
    _SIZE_CALLBACK_REGISTERED = True


def _upgrade_runtime(node):
    """Move known previous install defaults to the installed worker; keep overrides."""
    if "python_exe" not in node.knobs():
        return
    cfg = _config()
    installed = cfg.get("python_exe", "")
    if not installed or not Path(installed).is_file():
        return
    normalize = lambda value: os.path.normcase(os.path.normpath(str(value)))
    previous = {normalize(value) for value in cfg.get("runtime_previous_python_exes", [])}
    current = node["python_exe"].value()
    if not current or normalize(current) in previous:
        node["python_exe"].setValue(installed.replace("\\", "/"))
    if ("checkpoint" in node.knobs() and not node["checkpoint"].value()
            and cfg.get("checkpoint")):
        node["checkpoint"].setValue(cfg["checkpoint"].replace("\\", "/"))


def upgrade_loaded_nodes():
    for node in _nuke().allNodes("Group", recurseGroups=True):
        knobs = node.knobs()
        if all(key in knobs for key in ("target_text", "result_file", "tracking_mode", "python_exe", "analyze")):
            if "sam3_matchmove_nuke" in node["analyze"].value():
                ensure_controls(node)


def create_node():
    nuke = _nuke()
    cfg = _config()
    selected = nuke.selectedNodes()
    source = selected[-1] if selected else None
    first = int(source.firstFrame() if source else nuke.root()["first_frame"].value())
    last = int(source.lastFrame() if source else nuke.root()["last_frame"].value())
    reference = min(last, max(first, int(nuke.root()["frame"].value())))
    node = _make("Group", "SAM3_Matchmove")
    node.begin()
    try:
        plate = _make("Input", "Plate")
        plate["number"].setValue(0)
        mask = _make("Input", "Mask")
        mask["number"].setValue(1)
        _make("Output", "Output", inputs=[plate])
    finally:
        node.end()
    if source:
        node.setInput(0, source)
        node.setXYpos(source.xpos(), source.ypos() + 90)
    node["tile_color"].setValue(0x358A91FF)
    node.addKnob(nuke.Tab_Knob("sam3", "SAM3 Matchmove"))
    node.addKnob(nuke.Text_Knob("intro", "", "Plate + optional full-frame Mask → 2D object motion"))
    for name, label, values in (
        ("backend", "Mask source", ["SAM3 text", "Input mask"]),
        ("tracking_mode", "Motion", ["BBox position + scale", "Features: position + scale + rotation"]),
    ):
        node.addKnob(nuke.Enumeration_Knob(name, label, values))
    node.addKnob(nuke.String_Knob("target_text", "Target text"))
    node["target_text"].setValue("face")
    node.addKnob(nuke.Int_Knob("object_index", "Object index"))
    node["object_index"].setTooltip("0 = largest detected object at the reference frame. Its SAM3 object ID stays locked across the shot.")
    node.addKnob(nuke.Double_Knob("detection_threshold", "Detection threshold"))
    node["detection_threshold"].setRange(0, 1)
    node["detection_threshold"].setValue(0.45)
    node.addKnob(nuke.Enumeration_Knob("mask_channel", "Input mask channel", ["alpha", "red", "green", "blue"]))
    node["mask_channel"].setTooltip("For Roto use alpha; for grayscale SAM3 footage use red. Must be in original plate coordinates, not a square cropped mask.")
    for name, label, value in (("first_frame", "First", first), ("last_frame", "Last", last)):
        node.addKnob(nuke.Int_Knob(name, label))
        node[name].setValue(value)
    node.addKnob(_button("reset_range", "Reset", "reset_frame_range"))
    node.addKnob(nuke.Int_Knob("reference_frame", "Reference"))
    node["reference_frame"].setValue(reference)
    node.addKnob(_button("use_frame", "Use Current Frame as Reference", "use_current_frame"))
    node.addKnob(nuke.Int_Knob("smoothing_window", "Smoothing window"))
    node["smoothing_window"].setValue(1)
    node["smoothing_window"].setTooltip("Odd frame count. 1 = off. Symmetric smoothing reduces jitter but can soften fast movement.")
    ensure_controls(node, new=True)
    node.addKnob(_button("analyze", "Analyze", "analyze"))
    node.addKnob(_button("cancel", "Cancel", "cancel"))
    node.addKnob(nuke.String_Knob("status", "Status"))
    node["status"].setValue("Ready")
    node["status"].setEnabled(False)
    node.addKnob(nuke.File_Knob("result_file", "Result JSON"))
    node.addKnob(_button("open_output", "Open Output Folder", "open_output"))
    node.addKnob(nuke.Tab_Knob("environment", "Environment"))
    for name, label, value in (
        ("python_exe", "Worker Python", cfg.get("python_exe", "")),
        ("checkpoint", "SAM3 checkpoint", cfg.get("checkpoint", "")),
        ("output_root", "Output folder", cfg.get("output_root", str(ROOT / "outputs"))),
    ):
        node.addKnob(nuke.File_Knob(name, label))
        node[name].setValue(str(value).replace("\\", "/"))
    node.addKnob(_button("check_environment", "Check Worker Environment", "check_environment"))
    node.addKnob(nuke.Text_Knob("help", "", ""))
    ensure_controls(node)
    node["label"].setValue("[value target_text]\n[value status]")
    if nuke.GUI:
        node.showControlPanel()
    return node


def use_current_frame(node):
    node["reference_frame"].setValue(int(_nuke().root()["frame"].value()))


def reset_frame_range(node):
    """Refresh the analysis range from the currently connected Plate output."""
    try:
        if _active(node):
            raise ValueError("Wait for the current analysis to finish before resetting the frame range.")
        source = node.input(0)
        if source is None:
            raise ValueError("Connect a source to Plate before resetting the frame range.")
        frame_range = source.frameRange()
        first, last = int(frame_range.first()), int(frame_range.last())
        if first > last:
            raise ValueError("The Plate input has an invalid frame range.")
        old_reference = int(node["reference_frame"].value())
        reference = min(last, max(first, old_reference))
        node["first_frame"].setValue(first)
        node["last_frame"].setValue(last)
        if reference != old_reference:
            node["reference_frame"].setValue(reference)
        node["status"].setValue("Frame range reset: %d-%d" % (first, last))
        return first, last
    except Exception as exc:
        _message(exc)
        return None


def worker_environment():
    env = dict(os.environ)
    blocked = {"PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "NUKE_PATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"}
    for key in list(env):
        if key.upper() in blocked:
            env.pop(key, None)
    key = next((k for k in env if k.upper() == "PATH"), "PATH")
    env[key] = os.pathsep.join(p for p in env.get(key, "").split(os.pathsep)
                              if not any(part.lower().startswith("nuke") for part in Path(p).parts))
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def _python(node):
    value = Path(os.path.expandvars(os.path.expanduser(node["python_exe"].value())))
    if not value.is_file():
        raise ValueError("Set Environment > Worker Python to a real external python.exe. See README.md.")
    return str(value)


def _busy(node, busy):
    for name in ("analyze", "check_environment", "bake", "reset_range", "reference_frame", "use_frame"):
        node[name].setEnabled(not busy)
    node["cancel"].setEnabled(busy)


def _active(node):
    return any(job["node"] == node for job in _JOBS.values())


def _open_progress(node, check=False):
    """Keep Nuke's native progress window alive for the external worker."""
    if not _nuke().GUI:
        return None
    task = _nuke().ProgressTask("SAM3 Environment" if check else node.name())
    task.setMessage("Checking worker environment" if check else "Starting analysis")
    task.setProgress(0)
    return task


def _update_progress(job, data=None, message=None, percent=None):
    task = job.get("progress_task")
    if task is None:
        return
    if data is not None:
        message = str(data.get("message", "Analyzing..."))
        current, total = data.get("current", 0), data.get("total", 0)
        # These ranges describe processing stages, not an estimate of time left.
        ranges = ({"validating": (0, 2), "sam3": (2, 76), "masks": (78, 10), "tracking": (88, 10)}
                  if job.get("sam3") else
                  {"validating": (0, 2), "masks": (2, 28), "tracking": (30, 68)})
        base, span = ranges.get(data.get("stage"), (job.get("progress_percent", 0), 0))
        fraction = max(0.0, min(1.0, float(current) / float(total))) if total else 0.0
        percent = min(98, int(base + span * fraction))
    if message is not None:
        task.setMessage(message)
    if percent is not None:
        # Reading masks and estimating motion can each report their own 0..N.
        # Do not move the overall bar backwards when a new stage starts.
        percent = max(job.get("progress_percent", 0), min(100, int(percent)))
        job["progress_percent"] = percent
        task.setProgress(percent)


def _close_progress(job):
    # ProgressTask closes when its final Python reference is released.
    job.pop("progress_task", None)


def _validate_runtime(python, backend, checkpoint=""):
    check = "--check-sam3" if backend == "SAM3 text" else "--check"
    report = subprocess.run([python, str(ROOT / "sam3_matchmove" / "worker.py"), check],
                            cwd=str(ROOT), env=worker_environment(), capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=60,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if report.returncode:
        raise ValueError("Worker dependencies are unavailable. Run Check Worker Environment.\n" + (report.stdout + report.stderr)[-2000:])
    try:
        info = json.loads(report.stdout)
    except ValueError as exc:
        raise ValueError("Worker environment check returned an invalid report: " + report.stdout[-1000:]) from exc
    if backend == "SAM3 text" and checkpoint:
        path = Path(os.path.expandvars(os.path.expanduser(checkpoint)))
        if not path.is_file():
            raise ValueError("SAM3 runtime is installed, but the model checkpoint is missing: %s\nApprove access at https://huggingface.co/facebook/sam3, then run scripts/auth_sam3.ps1 to log in and download sam3.pt. Or set Environment > SAM3 checkpoint to your existing sam3.pt." % path)
    return info


def _srgb(writer):
    choices = [c.split("\t", 1)[0] for c in writer["colorspace"].values()]
    for choice in ("sRGB", "sRGB - Texture", "Utility - sRGB - Texture", "srgb_tx", "sRGB Encoded Rec.709 (sRGB)"):
        if choice in choices:
            writer["colorspace"].setValue(choice)
            return choice
    raise ValueError("The current color config has no recognized sRGB texture output. Use a config with sRGB, or add that colorspace to the OCIO config.")


def _video_native_first(path):
    """Discover the decoder's frame origin, ignoring user-edited origfirst knobs.

    Nuke's AVI reader starts at 0, while mov64 can start at 1. A fresh Read
    resolves the file's native origin without creating rendered image files.
    """
    nuke = _nuke()
    selection = nuke.selectedNodes()
    probe = None
    try:
        probe = _make("Read", "SAM3_TemporaryVideoProbe")
        probe["file"].fromUserText(str(path).replace("\\", "/"))
        probe.forceValidate()
        return int(probe["origfirst"].value())
    finally:
        if probe is not None:
            nuke.delete(probe)
        for item in nuke.selectedNodes():
            item["selected"].setValue(False)
        for item in selection:
            item["selected"].setValue(True)


def _direct_source_frames(source, first, last):
    """Resolve a directly connected Read; let Nuke render other graph operations."""
    if source.Class() != "Read":
        return None, "Plate is connected through a %s node instead of a direct Read." % source.Class()
    if source["disable"].isAnimated() or source["disable"].value():
        return None, "The Read disable setting requires Nuke evaluation."
    if _nuke().root()["proxy"].value():
        return None, "Proxy settings require Nuke evaluation."
    frames = []
    video_origins = {}
    for frame in range(first, last + 1):
        try:
            paths = {str(path) for owner, files in source.fileDependencies(frame, frame)
                     if owner == source for path in files}
        except Exception:
            return None, "The Read input files could not be resolved."
        if len(paths) != 1:
            return None, "The frame requires multiple files or views, or has no resolved input."
        path = Path(next(iter(paths)))
        suffix = path.suffix.lower()
        if suffix not in _DIRECT_IMAGES | _DIRECT_VIDEOS:
            return None, "This format is not supported for direct input (%s)." % (suffix or "no extension")
        if not path.is_absolute() or not path.is_file():
            return None, "The source path requires Nuke evaluation."
        record = {"frame": frame, "path": str(path)}
        if suffix in _DIRECT_VIDEOS:
            if (source["frame_mode"].value() != "expression"
                    or source["frame"].value().strip() not in ("", "frame")
                    or not int(source["first"].value()) <= frame <= int(source["last"].value())):
                return None, "Video timing changes or out-of-range frames require Nuke evaluation."
            try:
                if path not in video_origins:
                    video_origins[path] = _video_native_first(path)
            except Exception:
                return None, "The native video frame range could not be resolved."
            record["source_frame"] = frame - video_origins[path]
            if record["source_frame"] < 0:
                return None, "The video start frame requires Nuke evaluation."
        # A Read can return a file dependency even when it outputs black outside its range.
        elif ((frame < int(source["first"].value()) and source["before"].value() == "black")
              or (frame > int(source["last"].value()) and source["after"].value() == "black")):
            return None, "Black frames outside the Read range require Nuke evaluation."
        frames.append(record)
    return frames, ""


def _set_input_info(node, message, detail=""):
    node["input_info"].setValue(message)
    node["input_info"].setTooltip((detail + "\n\n" if detail else "") + _INPUT_HELP)
    node["status"].setValue(message)
    print(message)


@_in_controller_parent
def _create_mask_read(node, result_path, connect=False):
    """Load saved RGBA masks; only analysis connects them to the controller."""
    nuke = _nuke()
    result_path = Path(result_path)
    data = validate_result(json.loads(result_path.read_text(encoding="utf-8-sig")))
    if data.get("mask_channels") != "rgba":
        destination = result_path.parent / ("rgba_" + uuid.uuid4().hex[:8])
        report = subprocess.run([_python(node), str(ROOT / "sam3_matchmove" / "worker.py"),
                                 "--rgba-result", str(result_path), "--output-dir", str(destination)],
                                cwd=str(ROOT), env=worker_environment(), capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=120,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if report.returncode:
            raise ValueError("Could not prepare RGBA masks: " + (report.stdout + report.stderr)[-2000:])
        payload = json.loads(report.stdout.strip().splitlines()[-1])
        result_path = Path(payload["result_path"])
        data = validate_result(json.loads(result_path.read_text(encoding="utf-8-sig")))
    paths = [Path(row["mask_path"]) for row in data["frames"]]
    if not all(path.is_file() for path in paths):
        raise ValueError("The result mask sequence is incomplete.")
    old_selection = nuke.selectedNodes()
    previous = node.input(1)
    read = _make("Read", "SAM3_Mask_Read")
    try:
        read["file"].setValue(str(paths[0].parent / "mask.%06d.png").replace(chr(92), "/"))
        for name, value in (("first", data["frames"][0]["frame"]), ("origfirst", data["frames"][0]["frame"]),
                            ("last", data["frames"][-1]["frame"]), ("origlast", data["frames"][-1]["frame"])):
            read[name].setValue(value)
        # setValue(file) leaves Read's displayed format at the Root default,
        # even when Read.format() already evaluates the correct PNG dimensions.
        read["format"].setValue(nuke.addFormat("%d %d 1" % (data["width"], data["height"])))
        read["raw"].setValue(True)
        if "premultiplied" in read.knobs():
            read["premultiplied"].setValue(True)
        read["label"].setValue("RGBA mask")
        read["file"].setTooltip("Full-frame mask with identical red, green, blue and alpha values.")
        x, y = node.xpos() + 160, node.ypos() - 80
        while any(item != read and abs(item.xpos() - x) < 120 and abs(item.ypos() - y) < 60
                  for item in nuke.allNodes()):
            x += 160
        read.setXYpos(x, y)
        node["result_file"].setValue(str(result_path).replace(chr(92), "/"))
        if connect:
            node.setInput(1, read)
            node["mask_channel"].setValue("alpha")
            _set_input_info(node, "RGBA mask connected")
        return read
    except Exception:
        if connect:
            node.setInput(1, previous)
        nuke.delete(read)
        raise
    finally:
        for item in nuke.selectedNodes():
            item["selected"].setValue(False)
        for item in old_selection:
            item["selected"].setValue(True)


def _connect_mask_result(node, result_path):
    """Connect an analysis result while retaining the original Plate graph."""
    return _create_mask_read(node, result_path, connect=True)


def export_inputs(node, run_dir):
    """Resolve original files or prepare a Write for a user-rendered input."""
    nuke = _nuke()
    ensure_controls(node)
    source = node.input(0)
    if source is None:
        raise ValueError("Connect the original plate to input 0 (Plate).")
    first, last, reference = [int(node[k].value()) for k in ("first_frame", "last_frame", "reference_frame")]
    if first > last or not first <= reference <= last:
        raise ValueError("First <= Reference <= Last is required.")
    window = int(node["smoothing_window"].value())
    if window < 1 or window % 2 != 1:
        raise ValueError("Smoothing window must be a positive odd number; use 1 for off.")
    margin = float(node["crop_margin"].value())
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("Crop margin must be at least 1.")
    output_dimensions(node)
    fmt = source.format()
    width, height = fmt.width(), fmt.height()
    aspect = fmt.pixelAspect()
    if abs(aspect - 1.0) > 1e-6:
        raise ValueError("Reformat the plate and mask to square pixels before analysis. Anamorphic feature rotation is not supported.")
    backend = "masks" if node["backend"].value() == "Input mask" else "sam3"
    if backend == "sam3" and not node["target_text"].value().strip():
        raise ValueError("Enter Target text for SAM3.")
    mask_source = node.input(1) if backend == "masks" else None
    if backend == "masks" and mask_source is None:
        raise ValueError("Connect a full-frame mask to input 1 (Mask). For a grayscale file choose mask channel red.")
    if mask_source:
        mask_fmt = mask_source.format()
        if (mask_fmt.width(), mask_fmt.height(), mask_fmt.pixelAspect()) != (width, height, aspect):
            raise ValueError("Plate and mask formats must match exactly. Cropped square masks must be restored to the plate first.")
    from sam3_matchmove_inputs import prepare_inputs
    prepared = prepare_inputs(node, Path(run_dir), first, last, backend)
    if prepared is None:
        return None
    return dict(prepared, schema_version=1, width=width, height=height,
                pixel_aspect=aspect, first_frame=first, last_frame=last,
                reference_frame=reference, backend=backend, prompt=node["target_text"].value(),
                checkpoint=node["checkpoint"].value(), object_index=int(node["object_index"].value()),
                detection_threshold=float(node["detection_threshold"].value()),
                tracking_mode="features" if node["tracking_mode"].value().startswith("Features") else "bbox",
                smoothing_window=window, crop_margin=margin, fixed_crop=bool(node["fixed_crop"].value()),
                mask_channel=node["mask_channel"].value(), output_dir=str(run_dir))


@_in_controller_parent
def _start(node, check=False):
    ensure_controls(node)
    if _active(node):
        raise ValueError("This node already has a running job.")
    run_dir = Path(os.path.expandvars(os.path.expanduser(node["output_root"].value()))).resolve()
    run_dir /= datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    run_dir.mkdir(parents=True)
    progress_task = None
    process = log = token = None
    _busy(node, True)
    try:
        if check:
            python = _python(node)
            command = [python, str(ROOT / "sam3_matchmove" / "worker.py")]
            command.append("--check-sam3" if node["backend"].value() == "SAM3 text" else "--check")
        else:
            job = export_inputs(node, run_dir)
            if job is None:
                _busy(node, False)
                return None
            python = _python(node)
            _validate_runtime(python, node["backend"].value(), node["checkpoint"].value())
            command = [python, str(ROOT / "sam3_matchmove" / "worker.py")]
            job_file = run_dir / "job.json"
            job_file.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
            command.extend(["--job", str(job_file)])
        progress_task = _open_progress(node, check)
        log = (run_dir / "worker.log").open("w", encoding="utf-8")
        report_file = (run_dir / "environment.json").open("w", encoding="utf-8") if check else None
        try:
            process = subprocess.Popen(command, cwd=str(ROOT), env=worker_environment(), stdout=report_file if check else log,
                                       stderr=log if check else subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            log.close()
            raise
        finally:
            if report_file is not None:
                report_file.close()
        token = uuid.uuid4().hex
        _JOBS[token] = dict(node=node, process=process, log=log, run_dir=run_dir, check=check, cancelled=False,
                            progress_task=progress_task, progress_percent=0,
                            sam3=node["backend"].value() == "SAM3 text")
        node["status"].setValue("Checking environment" if check else "Analyzing…")
        if _nuke().GUI:
            try:
                from PySide6 import QtCore
            except ImportError:
                from PySide2 import QtCore
            timer = QtCore.QTimer()
            timer.timeout.connect(lambda: poll_job(token))
            _JOBS[token]["timer"] = timer
            timer.start(500)
        return token
    except Exception:
        if token in _JOBS:
            failed_job = _JOBS.pop(token)
            if "timer" in failed_job:
                failed_job["timer"].stop()
            _close_progress(failed_job)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if log is not None and not log.closed:
            log.close()
        progress_task = None
        _busy(node, False)
        raise


def analyze(node):
    try:
        return _start(node)
    except Exception as exc:
        node["status"].setValue("Analysis failed: " + str(exc))
        _message(exc)


def check_environment(node):
    try:
        return _start(node, check=True)
    except Exception as exc:
        node["status"].setValue("Environment check failed")
        _message(exc)


def cancel(node):
    for job in list(_JOBS.values()):
        if job["node"] == node:
            job["cancelled"] = True
            if job["process"].poll() is None:
                job["process"].terminate()
            node["status"].setValue("Cancelling…")
            _update_progress(job, message="Cancelling analysis...")


def _read_worker_error(run_dir):
    """Read the worker's actionable error without exposing its traceback as UI."""
    try:
        error = json.loads((run_dir / "error.json").read_text(encoding="utf-8-sig"))
    except (ValueError, OSError):
        return None
    if isinstance(error, dict):
        for key in ("error", "message"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def poll_job(token):
    job = _JOBS.get(token)
    if job is None:
        return True
    node, process, run_dir = job["node"], job["process"], job["run_dir"]
    try:
        node.name()
    except Exception:
        if process.poll() is None:
            process.terminate()
        if "timer" in job:
            job["timer"].stop()
        job["log"].close()
        _close_progress(job)
        _JOBS.pop(token, None)
        return True
    if job.get("progress_task") is not None and job["progress_task"].isCancelled() and not job["cancelled"]:
        cancel(node)
    if process.poll() is None:
        progress = run_dir / "progress.json"
        if progress.is_file():
            try:
                data = json.loads(progress.read_text(encoding="utf-8"))
                if not job["cancelled"]:
                    node["status"].setValue(str(data.get("message", "Analyzing…")))
                    _update_progress(job, data)
            except (TypeError, ValueError, OSError):
                pass
        return False
    if "timer" in job:
        job["timer"].stop()
    job["log"].close()
    _JOBS.pop(token, None)
    try:
        if job["cancelled"]:
            node["status"].setValue("Cancelled; previous result retained")
            return True
        text = (run_dir / "worker.log").read_text(encoding="utf-8", errors="replace")
        if job["check"]:
            text = (run_dir / "environment.json").read_text(encoding="utf-8", errors="replace") + ("\n" + text if process.returncode else "")
        if process.returncode:
            log_path = str(run_dir / "worker.log")
            error = None if job["check"] else _read_worker_error(run_dir)
            if error:
                node["status"].setValue("Worker failed: " + " ".join(error.split()))
                notification = error + "\n\nWorker log: " + log_path
            else:
                node["status"].setValue("Worker failed; see " + log_path)
                notification = text[-3500:] or "Worker failed without a log."
                if not job["check"]:
                    notification += "\n\nWorker log: " + log_path
            _close_progress(job)
            _message(notification)
            return True
        if job["check"]:
            try:
                report = json.loads(text)
                if report.get("sam3_runtime", {}).get("ok"):
                    checkpoint = node["checkpoint"].value()
                    available = bool(checkpoint) and Path(os.path.expandvars(os.path.expanduser(checkpoint))).is_file()
                    node["status"].setValue("SAM3 runtime ready; checkpoint found" if available else "SAM3 runtime ready; model weights still required")
                else:
                    node["status"].setValue("Input mask worker ready")
            except ValueError:
                node["status"].setValue("Environment check finished; see report")
            _update_progress(job, message="Environment check complete", percent=100)
            _close_progress(job)
            _message(text[-5000:])
            return True
        result = run_dir / "result.json"
        if not result.is_file():
            node["status"].setValue("Worker failed: result.json missing")
            _close_progress(job)
            _message("Worker exited without result.json. See " + str(run_dir / "worker.log"))
            return True
        node["result_file"].setValue(str(result).replace("\\", "/"))
        _update_progress(job, message="Connecting RGBA mask...", percent=99)
        _connect_mask_result(node, result)
        node["status"].setValue("Tracking ready; choose an output and Export")
        _update_progress(job, message="Analysis complete", percent=100)
        return True
    except Exception as exc:
        node["status"].setValue("Finishing analysis failed: " + str(exc))
        _close_progress(job)
        _message(exc)
        return True
    finally:
        _close_progress(job)
        _busy(node, False)


def validate_result(data):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported result schema.")
    for key in ("width", "height"):
        if not isinstance(data.get(key), int) or data[key] < 1:
            raise ValueError("Invalid result " + key)
    if abs(float(data.get("pixel_aspect", 1)) - 1) > 1e-6:
        raise ValueError("Only square-pixel tracking results can be baked.")
    def numbers(values, length):
        if not isinstance(values, (tuple, list)) or len(values) != length or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("Invalid or non-finite tracking coordinates.")
    numbers(data.get("reference_center"), 2)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("Result has no frames.")
    ids = [row["frame"] for row in frames]
    if not all(type(f) is int for f in ids) or ids != list(range(ids[0], ids[-1] + 1)):
        raise ValueError("Tracking frame numbers must be contiguous and ordered.")
    if data.get("reference_frame") not in ids:
        raise ValueError("Reference frame is outside the result.")
    for row in frames:
        numbers(row.get("translate"), 2)
        numbers(row.get("center"), 2)
        numbers([row.get("scale"), row.get("rotate"), row.get("confidence")], 3)
        if row["scale"] <= 0:
            raise ValueError("Tracking scale must be positive.")
        numbers(row.get("crop_box"), 4)
        box = row["crop_box"]
        if box[2] <= box[0] or abs((box[2] - box[0]) - (box[3] - box[1])) > 1e-4:
            raise ValueError("Crop boxes must be nonempty squares.")
    return data


def _keys(knob, rows, getter, count):
    knob.setAnimated()
    for row in rows:
        value = getter(row)
        if count == 1:
            knob.setValueAt(float(value), row["frame"])
        else:
            for axis in range(count):
                knob.setValueAt(float(value[axis]), row["frame"], axis)
    for curve in knob.animations():
        curve.changeInterpolation(curve.keys(), _nuke().LINEAR)


def _animate_transform(node, data, inverse=False):
    node["center"].setValue(data["reference_center"])
    node["invert_matrix"].setValue(inverse)
    rows = data["frames"]
    _keys(node["translate"], rows, lambda r: r["translate"], 2)
    _keys(node["scale"], rows, lambda r: [r["scale"], r["scale"]], 2)
    _keys(node["rotate"], rows, lambda r: r["rotate"], 1)


def layout_outputs(node, nodes):
    """Compact only the supplied output nodes, preserving their graph and curves."""
    positions = {
        "mask_read": (-320, 150),
        "mask": (-320, 230),
        "stabilize": (-160, 150),
        "crop_transform": (0, 150),
        "crop": (0, 230),
        "insert": (160, 70),
        "matchmove": (160, 150),
        "uncrop": (320, 150),
        "uncrop_format": (320, 230),
    }
    for key, (dx, dy) in positions.items():
        item = nodes.get(key)
        if item is not None:
            item.setXYpos(node.xpos() + dx, node.ypos() + dy)
    insert, matchmove = nodes.get("insert"), nodes.get("matchmove")
    if insert is not None and matchmove is not None:
        # Terminal Nuke reports zero screen dimensions. Use the standard node
        # widths there so saved examples retain the same alignment as the GUI.
        match_width = matchmove.screenWidth() or 80
        dot_width = insert.screenWidth() or 12
        insert.setXpos(matchmove.xpos() + (match_width - dot_width) // 2)


@_in_controller_parent
def bake_result(node, result_path=None, export_nk=True):
    """Export only the selected output from existing tracking data."""
    ensure_controls(node)
    result_path = Path(result_path or node["result_file"].value())
    if not result_path.is_file():
        raise ValueError("Analyze first or select an existing Result JSON, then choose an output and click Export.")
    data = validate_result(json.loads(result_path.read_text(encoding="utf-8-sig")))
    from sam3_matchmove_export import build_output
    if not node["result_file"].value() or Path(node["result_file"].value()) != result_path:
        node["result_file"].setValue(str(result_path).replace("\\", "/"))
    return build_output(node, data, node["export_type"].value(), export_nk=export_nk)


def bake_button(node):
    try:
        return bake_result(node)
    except Exception as exc:
        node["status"].setValue("Export failed: " + str(exc))
        _message(exc)


def open_output(node):
    result = node["result_file"].value()
    path = Path(result).parent if result else Path(node["output_root"].value())
    if path.is_dir():
        if os.name == "nt":
            os.startfile(str(path))
        else:
            _message(path)


def register_menu():
    global _MENU_REGISTERED, _LOAD_REGISTERED
    if not _MENU_REGISTERED:
        _nuke().menu("Nodes").addCommand("AI/SAM3 Matchmove", create_node)
        _MENU_REGISTERED = True
    if not _LOAD_REGISTERED:
        _nuke().addOnScriptLoad(upgrade_loaded_nodes)
        _LOAD_REGISTERED = True
    upgrade_loaded_nodes()
