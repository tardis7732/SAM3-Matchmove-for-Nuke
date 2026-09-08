"""Official SAM 3 video inference adapter, loaded only by the external worker.

No Nuke or ML package is imported at module import time. See docs/references.md
for the inspected upstream revision and the coordinate/output conventions.
"""

import math
from contextlib import ExitStack
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import warnings


class TargetNotFoundError(ValueError):
    """The requested target has no usable mask on the reference frame."""

    def __init__(self, prompt, reference_frame, output_threshold):
        self.prompt = prompt
        self.reference_frame = reference_frame
        self.output_threshold = output_threshold
        super().__init__(
            "SAM 3 found no usable target for Target text %r on Nuke Reference "
            "frame %d (output threshold %.3g). Check Target text and choose a "
            "Reference frame where that target is visible."
            % (prompt, reference_frame, output_threshold)
        )


def _build_predictor(job):
    """Check the real inference runtime without affecting import-only clients."""
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Official SAM 3 requires external Python 3.12 or newer; this worker "
            "uses Python %s. Set the node's worker Python to a SAM 3 environment. "
            "Nuke's embedded Python does not need the ML packages."
            % sys.version.split()[0]
        )
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Cannot load PyTorch in the worker Python. Install CUDA-enabled "
            "PyTorch >= 2.7 in the external SAM 3 environment: %s" % exc
        ) from exc
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version and tuple(map(int, version.groups())) < (2, 7):
        raise RuntimeError("Official SAM 3 requires PyTorch >= 2.7; found %s." % torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SAM 3 video inference requires an NVIDIA CUDA GPU and a CUDA-enabled "
            "PyTorch build. torch.cuda.is_available() is false in %s." % sys.executable
        )
    cuda_version = re.match(r"(\d+)\.(\d+)", str(torch.version.cuda))
    if cuda_version and tuple(map(int, cuda_version.groups())) < (12, 6):
        raise RuntimeError("Official SAM 3 requires CUDA >= 12.6; PyTorch uses CUDA %s." % torch.version.cuda)
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Cannot import official SAM 3 in %s. Install facebookresearch/sam3 "
            "and its dependencies in this external Python environment. %s"
            % (sys.executable, exc)
        ) from exc
    checkpoint = job.get("checkpoint")
    # Load CPU frames synchronously so no loader thread can still read our
    # temporary input directory after a cancellation or an empty reference.
    kwargs = dict(gpus_to_use=[0], compile=False, async_loading_frames=False)
    if checkpoint:
        checkpoint = Path(os.path.expandvars(str(checkpoint))).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("SAM 3 checkpoint does not exist: %s" % checkpoint)
        kwargs["checkpoint_path"] = str(checkpoint)
    try:
        return build_sam3_video_predictor(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "SAM 3 model loading failed. Use the SAM 3 sam3.pt checkpoint "
            "(not SAM 3.1). If Checkpoint is empty, request access at "
            "https://huggingface.co/facebook/sam3 and authenticate in the "
            "worker environment with hf auth login. Original error: %s" % exc
        ) from exc


def _as_numpy(value, np):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _read_outputs(outputs, width, height, np):
    """Validate the official full-resolution mask payload before writing it."""
    if not isinstance(outputs, dict):
        raise ValueError("SAM 3 returned no outputs dictionary.")
    ids = _as_numpy(outputs.get("out_obj_ids", []), np).reshape(-1)
    if not len(ids):
        return [], np.empty((0, height, width), dtype=bool), np.empty(0)
    if "out_binary_masks" not in outputs or "out_probs" not in outputs:
        raise ValueError("SAM 3 outputs must contain out_binary_masks and out_probs.")
    masks = _as_numpy(outputs["out_binary_masks"], np)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.shape != (len(ids), height, width):
        raise ValueError(
            "SAM 3 mask size %s does not match (%d, %d, %d). "
            "Inference frames must use the job's full image dimensions."
            % (masks.shape, len(ids), height, width)
        )
    scores = _as_numpy(outputs["out_probs"], np).reshape(-1)
    if len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("SAM 3 returned invalid object confidence scores.")
    object_ids = [int(value) for value in ids]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("SAM 3 returned duplicate object IDs in one frame.")
    return object_ids, masks > 0, scores


def _validate_job(job):
    frames = list(job.get("frames", []))
    if not frames:
        raise ValueError("SAM 3 job requires at least one frame.")
    width, height = int(job["width"]), int(job["height"])
    if width <= 0 or height <= 0:
        raise ValueError("SAM 3 image width and height must be positive.")
    numbers = []
    for item in frames:
        frame = item["frame"]
        if isinstance(frame, bool) or int(frame) != frame:
            raise ValueError("Nuke frame numbers must be integers.")
        numbers.append(int(frame))
        source = Path(item["path"])
        if not source.is_absolute() or not source.is_file():
            raise FileNotFoundError("Frame path must be an existing absolute file: %s" % source)
    if len(set(numbers)) != len(numbers):
        raise ValueError("SAM 3 job contains duplicate Nuke frame numbers.")
    if numbers != sorted(numbers):
        raise ValueError("SAM 3 job frames must be ordered by increasing Nuke frame number.")
    reference = job.get("reference_frame", numbers[0])
    if reference not in numbers:
        raise ValueError("Reference frame %s is not in the job's frame list." % reference)
    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Enter a SAM 3 text prompt, for example 'head'.")
    object_index = job.get("object_index", 0)
    if isinstance(object_index, bool) or int(object_index) != object_index or object_index < 0:
        raise ValueError("Object index must be a non-negative integer.")
    threshold = float(job.get("detection_threshold", 0.45))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Detection threshold must be between 0 and 1.")
    return frames, numbers, width, height, numbers.index(reference), prompt, int(object_index), threshold


def generate_masks(job, output_dir, progress=None):
    """Return per-Nuke-frame binary PNG paths and one locked object identity.

    ``progress(current, total, message)`` may raise to cancel. A missing locked
    object produces a black full-frame mask, detected=False and its locked ID.
    No eligible reference detection raises TargetNotFoundError before exporting
    any masks or starting propagation.
    Reference candidates are confidence-filtered, then sorted by descending mask
    area (object ID breaks ties). No re-identification or mask union is performed.
    """
    frames, numbers, width, height, reference_index, prompt, object_index, threshold = _validate_job(job)
    try:
        import numpy as np
        from PIL import Image
        from sam3_matchmove.source_frames import SourceFrameReader, VIDEO_SUFFIXES
    except ImportError as exc:
        raise RuntimeError("The worker Python requires numpy and Pillow for SAM 3 masks.") from exc
    report = progress or (lambda current, total, message: None)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mask_dir = destination / "masks"
    mask_dir.mkdir(exist_ok=True)
    total = len(frames)
    result = {}
    predictor = None
    session_id = None
    stream = None
    locked_id = None
    blank = np.zeros((height, width), dtype=np.uint8)

    def write_result(index, mask, score=0.0, detected=False):
        number = numbers[index]
        path = mask_dir / ("mask.%06d.png" % number)
        with Image.fromarray(mask.astype(np.uint8) * 255) as matte:
            with Image.merge("RGBA", (matte, matte, matte, matte)) as rgba:
                rgba.save(str(path), format="PNG")
        result[number] = {
            "mask_path": str(path), "object_id": locked_id,
            "score": float(score), "detected": bool(detected),
        }

    def record(index, outputs):
        ids, masks, scores = _read_outputs(outputs, width, height, np)
        if locked_id not in ids:
            write_result(index, blank)
            return
        selected = ids.index(locked_id)
        score = float(scores[selected])
        detected = score >= threshold and bool(masks[selected].any())
        write_result(index, masks[selected] if detected else blank, score, detected)

    # Image sequences retain their source encoding in a privately ordered
    # hardlink/copy folder. Video records decode selected zero-based frames into
    # official SAM3's PIL-list input, without an intermediate PNG sequence.
    with tempfile.TemporaryDirectory(prefix="sam3_frames_", dir=str(destination)) as input_folder, ExitStack() as resources:
        use_memory_frames = any(Path(item["path"]).suffix.lower() in VIDEO_SUFFIXES for item in frames)
        resource_path = [] if use_memory_frames else input_folder
        with SourceFrameReader(width, height) as reader:
            for index, item in enumerate(frames):
                if use_memory_frames:
                    frame_image = reader.read_pil(item)
                    resources.callback(frame_image.close)
                    resource_path.append(frame_image)
                    continue
                reader.read_rgb(item)  # Validate decoding and exact dimensions before loading the model.
                source = Path(item["path"])
                suffix = source.suffix.lower()
                suffix = ".tiff" if suffix == ".tif" else suffix
                target = Path(input_folder) / ("%08d%s" % (index, suffix))
                try:
                    os.link(str(source), str(target))
                except OSError:
                    shutil.copy2(str(source), str(target))
        try:
            report(0, total, "Loading SAM 3 video model")
            predictor = _build_predictor(job)
            session_id = predictor.handle_request({
                "type": "start_session", "resource_path": resource_path,
                "offload_video_to_cpu": True, "offload_state_to_cpu": True,
            })["session_id"]
            response = predictor.handle_request({
                "type": "add_prompt", "session_id": session_id,
                "frame_index": reference_index, "text": prompt,
                "output_prob_thresh": threshold,
            })
            if int(response["frame_index"]) != reference_index:
                raise ValueError("SAM 3 returned a different reference frame index.")
            ids, masks, scores = _read_outputs(response["outputs"], width, height, np)
            candidates = [i for i in range(len(ids)) if scores[i] >= threshold and masks[i].any()]
            candidates.sort(key=lambda i: (-int(masks[i].sum()), ids[i]))
            if not candidates:
                raise TargetNotFoundError(prompt, numbers[reference_index], threshold)
            if object_index >= len(candidates):
                raise ValueError(
                    "Object index %d is unavailable: reference frame has %d eligible targets "
                    "(indices 0 to %d)." % (object_index, len(candidates), len(candidates) - 1)
                )
            locked_id = ids[candidates[object_index]]
            record(reference_index, response["outputs"])
            seen = {reference_index}
            report(1, total, "SAM 3 locked object %d on frame %d" % (locked_id, numbers[reference_index]))
            stream = predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session_id,
                "propagation_direction": "both", "start_frame_index": reference_index,
                "output_prob_thresh": threshold,
            })
            for response in stream:
                index = int(response["frame_index"])
                if not 0 <= index < total:
                    raise ValueError("SAM 3 returned out-of-range frame index %d." % index)
                # Preserve the mask used to choose the locked reference object.
                if index != reference_index:
                    record(index, response["outputs"])
                seen.add(index)
                report(len(seen), total, "SAM 3 frame %d / object %d" % (numbers[index], locked_id))
            for index in range(total):
                if numbers[index] not in result:
                    write_result(index, blank)
            report(total, total, "SAM 3 masks ready")
            return {number: result[number] for number in numbers}
        finally:
            active_error = sys.exc_info()[0] is not None
            cleanup_errors = []
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
            if predictor is not None:
                if session_id is not None:
                    try:
                        predictor.handle_request({"type": "close_session", "session_id": session_id})
                    except Exception as exc:
                        cleanup_errors.append(exc)
                try:
                    predictor.shutdown()
                except Exception as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                detail = "; ".join(str(error) for error in cleanup_errors)
                if active_error:
                    warnings.warn("SAM 3 cleanup also failed: %s" % detail, RuntimeWarning)
                else:
                    raise RuntimeError("SAM 3 cleanup failed: %s" % detail)
