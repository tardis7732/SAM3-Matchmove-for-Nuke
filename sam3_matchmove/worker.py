"""External Python entry point. Run: python worker.py --job job.json."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback

# Permit both `python -m sam3_matchmove.worker` and a direct script launch.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _replace_with_retry(source, destination, timeout=1.0):
    """Allow short Windows reader locks without exposing a partial file.

    Nuke polling can briefly open the previous progress file without delete
    sharing. Keep its complete contents in place until replacement succeeds.
    Persistent permission failures still propagate after the bounded deadline.
    """
    deadline = time.monotonic() + timeout
    delay = 0.01
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.05)


def prepare_worker_process():
    """Discard the host's inherited DLL directory in this child process only.

    On Windows, Nuke's SetDllDirectory setting survives CreateProcess even
    after PATH is cleaned. Loading Nuke's CUDA/PyTorch dependencies alongside
    the worker's PyTorch causes c10.dll initialization failure (WinError 1114).
    Reset before importing numpy, cv2 or torch; never change the Nuke process.
    """
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        reset = kernel.SetDllDirectoryW
        reset.argtypes = [ctypes.c_wchar_p]
        reset.restype = ctypes.c_int
        if not reset(None):
            raise ctypes.WinError(ctypes.get_last_error())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        _replace_with_retry(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def dependency_check():
    dependencies = {}
    for name in ("numpy", "cv2", "PIL"):
        try:
            module = importlib.import_module(name)
            dependencies[name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            dependencies[name] = {"available": False, "error": str(exc)}
    try:
        sam3_available = importlib.util.find_spec("sam3") is not None
    except (ImportError, ValueError):
        sam3_available = False
    return dict(ok=all(value["available"] for value in dependencies.values()),
                python=sys.executable, python_version=sys.version.split()[0],
                dependencies=dependencies, sam3_available=sam3_available)


def sam3_runtime_check():
    """Import the actual inference stack without loading models or weights."""
    result = dict(ok=False, torch_version=None, cuda_version=None,
                  cuda_available=False, device=None, sam3_import_ok=False)
    errors = []
    if sys.version_info < (3, 12):
        errors.append("Official SAM 3 requires Python >= 3.12; found %s."
                      % sys.version.split()[0])
    try:
        torch = importlib.import_module("torch")
        result["torch_version"] = str(torch.__version__)
        result["cuda_version"] = torch.version.cuda
        version = re.match(r"(\d+)\.(\d+)", result["torch_version"])
        if not version or tuple(map(int, version.groups())) < (2, 7):
            errors.append("Official SAM 3 requires PyTorch >= 2.7; found %s."
                          % result["torch_version"])
        cuda_version = re.match(r"(\d+)\.(\d+)", str(result["cuda_version"]))
        if not cuda_version or tuple(map(int, cuda_version.groups())) < (12, 6):
            errors.append("Official SAM 3 requires CUDA >= 12.6; PyTorch uses %s."
                          % result["cuda_version"])
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["device"] = torch.cuda.get_device_name(0)
        else:
            errors.append("CUDA is unavailable. Use an NVIDIA CUDA GPU and a CUDA-enabled PyTorch build.")
    except Exception as exc:
        errors.append("Cannot load the PyTorch CUDA runtime: %s: %s"
                      % (type(exc).__name__, exc))
    try:
        builder = importlib.import_module("sam3.model_builder")
        if not callable(getattr(builder, "build_sam3_video_predictor", None)):
            raise ImportError("sam3.model_builder.build_sam3_video_predictor is unavailable")
        result["sam3_import_ok"] = True
    except Exception as exc:
        errors.append("Cannot import official SAM 3 and its video dependencies: %s: %s"
                      % (type(exc).__name__, exc))
    result["ok"] = not errors
    if errors:
        result["error"] = " ".join(errors)
    return result


def write_csv(path, result):
    path = Path(path)
    handle, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    columns = ["frame", "translate_x", "translate_y", "scale", "rotate", "center_x", "center_y",
               "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
               "status", "confidence", "detected", "estimated", "object_id", "mask_path"]
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for frame in result["frames"]:
                writer.writerow([frame["frame"], *frame["translate"], frame["scale"], frame["rotate"],
                    *frame["center"], *(frame["bbox"] or [""]*4), *frame["crop_box"], frame["status"],
                    frame["confidence"], frame["detected"], frame["estimated"], frame["object_id"], frame["mask_path"]])
        _replace_with_retry(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_rgba_png(matte, destination):
    """Atomically write equal R/G/B/A channels, retaining 16-bit gray precision."""
    from PIL import Image
    destination = Path(destination)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    os.close(handle)
    try:
        if matte.mode in ("F", "I", "I;16", "I;16B", "I;16L"):
            import cv2
            import numpy as np
            values = np.asarray(matte)
            if matte.mode == "F":
                if not np.isfinite(values).all():
                    raise ValueError("Floating-point masks must contain finite values")
                values = np.rint(np.clip(values, 0.0, 1.0) * 65535.0)
            if values.min() < 0 or values.max() > 65535:
                raise ValueError("Grayscale mask values must be in the 16-bit PNG range")
            rgba = np.repeat(values.astype(np.uint16)[:, :, None], 4, axis=2)
            # All four channels are identical, so OpenCV's BGRA ordering is immaterial.
            success, encoded = cv2.imencode(".png", rgba)
            if not success:
                raise ValueError("Cannot encode 16-bit RGBA mask")
            Path(temporary).write_bytes(encoded.tobytes())
        else:
            with matte.convert("L") as channel:
                with Image.merge("RGBA", (channel, channel, channel, channel)) as rgba:
                    rgba.save(temporary, format="PNG")
        _replace_with_retry(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mask_channel(image, source, channel):
    """Extract a matte while preserving high-depth PNG/TIFF source values."""
    from PIL import Image
    if image.mode in ("L", "1", "F", "I", "I;16", "I;16B", "I;16L"):
        return image.copy()
    high_depth = False
    if image.format == "PNG":
        with open(source, "rb") as stream:
            header = stream.read(25)
        high_depth = len(header) == 25 and header[24] == 16
    elif image.format == "TIFF":
        bits = image.tag_v2.get(258, (8,))
        high_depth = max(bits if isinstance(bits, (list, tuple)) else (bits,)) > 8
    if high_depth:
        import cv2
        import numpy as np
        pixels = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.dtype not in (np.uint16, np.float32):
            raise ValueError("Cannot decode high-depth mask channels: %s" % source)
        if pixels.ndim == 2:
            return Image.fromarray(pixels)
        index = {"red": 2, "green": 1, "blue": 0, "alpha": 3}[channel]
        if index < pixels.shape[2]:
            return Image.fromarray(pixels[:, :, index].copy())
        # RGB files expose an opaque alpha, matching an ordinary Nuke Read.
        opaque = 1.0 if pixels.dtype == np.float32 else 65535
        return Image.fromarray(np.full(pixels.shape[:2], opaque, dtype=pixels.dtype))
    return image.convert("RGBA").getchannel({"red": "R", "green": "G", "blue": "B", "alpha": "A"}[channel])


def normalize_masks(job, output_dir, progress=None):
    """Write the complete output sequence while retaining soft input mattes.

    User image masks use the selected channel; generated SAM3 masks use red.
    Exported PNGs repeat the soft matte in R/G/B/A for a standalone Nuke Read.
    """
    from PIL import Image
    from sam3_matchmove.tracking import TrackingError
    mask_dir = Path(output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    width, height = job["width"], job["height"]
    channel = "red" if job.get("backend") == "sam3" else job.get("mask_channel", "red")
    channels = {"red": "R", "green": "G", "blue": "B", "alpha": "A"}
    if channel not in channels:
        raise TrackingError("mask_channel must be alpha, red, green, or blue")
    normalized = []
    for index, original in enumerate(job["frames"]):
        record = dict(original)
        source = record.get("mask_path")
        if source is None or record.get("detected") is False:
            matte = Image.new("L", (width, height), 0)
        else:
            try:
                with Image.open(source) as image:
                    if image.format not in ("PNG", "JPEG", "BMP", "TIFF", "WEBP"):
                        raise TrackingError(f"Unsupported mask image format: {source}")
                    if image.size != (width, height):
                        raise TrackingError(f"Dimension mismatch for {source}: {image.size}, expected {(width, height)}")
                    matte = _mask_channel(image, source, channel)
            except TrackingError:
                raise
            except Exception as exc:
                raise TrackingError(f"Cannot read mask image {source}: {exc}") from exc
        destination = mask_dir / ("mask.%06d.png" % record["frame"])
        try:
            # Source handles close before an atomic replacement of the same path.
            _write_rgba_png(matte, destination)
        finally:
            matte.close()
        record["mask_path"] = str(destination)
        normalized.append(record)
        if progress:
            progress("masks", index + 1, len(job["frames"]), f"Exported mask for frame {record['frame']}")
    return dict(job, frames=normalized, mask_channels="rgba")


def convert_result_masks(result_path, output_dir):
    """Copy a saved result into fresh RGBA masks without rerunning inference.

    Source JSON, masks and tracking values remain unchanged. Existing output
    results/mask directories are never overwritten, including an accidental
    request to convert directly into the original result's directory.
    """
    result_path = Path(result_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    with result_path.open("r", encoding="utf-8-sig") as stream:
        original = json.load(stream)
    if (not isinstance(original, dict) or type(original.get("schema_version")) is not int
            or original.get("schema_version") != 1):
        raise ValueError("Expected a schema-version-1 tracking result")
    for key in ("width", "height"):
        if type(original.get(key)) is not int or original[key] <= 0:
            raise ValueError("Result %s must be a positive integer" % key)
    frames = original.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Result must contain frames")
    numbers = set()
    for frame in frames:
        if not isinstance(frame, dict) or type(frame.get("frame")) is not int:
            raise ValueError("Result frame numbers must be integers")
        if frame["frame"] in numbers:
            raise ValueError("Result contains duplicate frame numbers")
        numbers.add(frame["frame"])
        source = frame.get("mask_path")
        if frame.get("detected") is not False:
            if not isinstance(source, str) or not Path(source).is_absolute() or not Path(source).is_file():
                raise ValueError("Missing absolute mask file at frame %s" % frame["frame"])
    result_file = destination / "result.json"
    if destination == result_path.parent or result_file.exists() or (destination / "masks").exists():
        raise FileExistsError("RGBA conversion needs a new output directory; original results are preserved")
    destination.mkdir(parents=True, exist_ok=True)
    # Existing analysis masks are defined by red, independently of any historic
    # input-mask channel selection. This does not rerun SAM3 or tracking.
    normalized = normalize_masks(dict(original, backend="sam3"), destination)
    copied = dict(original, frames=normalized["frames"], mask_channels="rgba",
                  rgba_source_result=str(result_path))
    atomic_json(result_file, copied)
    return dict(status="done", result_path=str(result_file), source_result=str(result_path),
                mask_pattern=str(destination / "masks" / "mask.%06d.png"),
                frame_count=len(frames), mask_channels="rgba")


def run(job_path):
    output_dir = None
    try:
        path = Path(job_path).resolve()
        with path.open("r", encoding="utf-8-sig") as stream:
            job = json.load(stream)
        candidate = job.get("output_dir") if isinstance(job, dict) else None
        if isinstance(candidate, str) and Path(candidate).is_absolute():
            output_dir = Path(candidate)
            output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir is None:
            raise ValueError("job.output_dir must be an absolute path")

        def progress(stage, current=0, total=0, message=""):
            payload = dict(status="running", stage=stage, current=current, total=total, message=message)
            atomic_json(output_dir / "progress.json", payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

        progress("validating", message="Validating inputs and dependencies")
        from sam3_matchmove.tracking import track_job, validate_job
        validate_job(job, validate_images=job.get("backend", "masks") == "sam3")
        if job.get("backend", "masks") == "sam3":
            from sam3_matchmove.sam3_backend import generate_masks
            generated = generate_masks(job, str(output_dir),
                                       lambda current, total, message: progress("sam3", current, total, message))
            if not isinstance(generated, dict):
                raise ValueError("SAM3 backend must return a mapping keyed by Nuke frame")
            mapped = []
            for original in job["frames"]:
                item = dict(original)
                metadata = generated.get(original["frame"], generated.get(str(original["frame"])))
                if metadata is None:
                    item.update(mask_path=None, detected=False)
                elif not isinstance(metadata, dict):
                    raise ValueError(f"Invalid SAM3 mask metadata at frame {original['frame']}")
                else:
                    for key in ("mask_path", "object_id", "score", "detected"):
                        if key in metadata:
                            item[key] = metadata[key]
                mapped.append(item)
            job = dict(job, frames=mapped)
        job = normalize_masks(job, output_dir, progress)
        result = track_job(job, progress)
        result["mask_channels"] = "rgba"
        write_csv(output_dir / "tracks.csv", result)
        atomic_json(output_dir / "result.json", result)
        done = dict(status="done", stage="done", current=len(result["frames"]), total=len(result["frames"]),
                    message="Tracking complete", result_path=str(output_dir / "result.json"),
                    csv_path=str(output_dir / "tracks.csv"), warnings=result["warnings"])
        atomic_json(output_dir / "progress.json", done)
        print(json.dumps(done, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        error = dict(status="error", stage="error", error=str(exc), error_type=type(exc).__name__,
                     message=str(exc))
        if output_dir is not None:
            try:
                atomic_json(output_dir / "error.json", error)
                atomic_json(output_dir / "progress.json", error)
            except Exception:
                pass
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", help="Absolute or relative path to a schema-version-1 job JSON")
    group.add_argument("--check", action="store_true", help="Report Python and tracking dependency availability")
    group.add_argument("--check-sam3", action="store_true",
                       help="Import and check the SAM 3 CUDA runtime without loading model weights")
    group.add_argument("--rgba-result", help="Copy a saved result and its masks to RGBA without inference")
    parser.add_argument("--output-dir", help="New destination directory for --rgba-result")
    args = parser.parse_args(argv)
    if args.rgba_result and not args.output_dir:
        parser.error("--rgba-result requires --output-dir")
    if args.output_dir and not args.rgba_result:
        parser.error("--output-dir is only used with --rgba-result")
    prepare_worker_process()
    if args.rgba_result:
        try:
            converted = convert_result_masks(args.rgba_result, args.output_dir)
        except Exception as exc:
            print(json.dumps(dict(status="error", error=str(exc), error_type=type(exc).__name__),
                             ensure_ascii=False), file=sys.stderr, flush=True)
            return 1
        print(json.dumps(converted, ensure_ascii=False), flush=True)
        return 0
    if args.check or args.check_sam3:
        # Some optional ML dependencies print during import. Keep stdout a
        # single JSON document so callers can parse both diagnostic modes.
        with redirect_stdout(sys.stderr):
            result = dependency_check()
            if args.check_sam3:
                result["sam3_runtime"] = sam3_runtime_check()
                result["ok"] = result["ok"] and result["sam3_runtime"]["ok"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    return run(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
