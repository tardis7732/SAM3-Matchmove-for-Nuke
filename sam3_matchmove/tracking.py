"""Mask-guided 2D similarity tracking, independent of Nuke and SAM3.

Image bboxes use exclusive pixel edges. Public coordinates use Nuke's
bottom-left origin; optical-flow sample indices are converted to pixel centers.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from sam3_matchmove.source_frames import SourceFrameError, SourceFrameReader, VIDEO_SUFFIXES


class TrackingError(ValueError):
    """A job cannot produce a trustworthy reference-relative track."""


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise TrackingError(f"{name} must be >= {minimum}")
    return value


def _path(value, name, must_exist=True):
    if not isinstance(value, str) or not value.strip():
        raise TrackingError(f"{name} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise TrackingError(f"{name} must be absolute: {value}")
    if must_exist and not path.is_file():
        raise TrackingError(f"{name} does not exist or is not a file: {value}")
    return path


def _read_png(path, width, height, mask=False):
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise TrackingError(f"Expected PNG input: {path}")
            if image.size != (width, height):
                raise TrackingError(
                    f"Dimension mismatch for {path}: {image.size}, expected {(width, height)}"
                )
            if mask:
                # Nuke's mask export writes the matte to red. Never use alpha:
                # Write nodes often produce opaque alpha around a grayscale mask.
                if image.mode in ("L", "1", "I", "I;16", "F"):
                    array = np.asarray(image).copy()
                    if array.dtype == np.bool_:
                        return array
                    threshold = 32767 if array.dtype == np.uint16 else 127
                    return array > threshold
                return np.asarray(image.convert("RGB"))[:, :, 0] > 127
            return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    except TrackingError:
        raise
    except Exception as exc:
        raise TrackingError(f"Cannot read PNG {path}: {exc}") from exc


def validate_job(job, validate_images=True):
    """Validate the schema and source pixels without changing caller data."""
    if not isinstance(job, dict) or type(job.get("schema_version")) is not int or job.get("schema_version") != 1:
        raise TrackingError("job.schema_version must be 1")
    width = _integer(job.get("width"), "width", 1)
    height = _integer(job.get("height"), "height", 1)
    aspect = job.get("pixel_aspect", 1.0)
    if isinstance(aspect, bool) or not isinstance(aspect, (int, float)) or not math.isfinite(aspect) or aspect <= 0:
        raise TrackingError("pixel_aspect must be finite and positive")
    frames = job.get("frames")
    if not isinstance(frames, list) or not frames:
        raise TrackingError("frames must contain at least one frame")
    numbers = []
    try:
        with SourceFrameReader(width, height) as reader:
            for record in frames:
                if not isinstance(record, dict):
                    raise TrackingError("Every frame must be an object")
                number = _integer(record.get("frame"), "frame")
                numbers.append(number)
                path = _path(record.get("path"), f"frame {number} path")
                if path.suffix.lower() in VIDEO_SUFFIXES:
                    _integer(record.get("source_frame"), f"frame {number} source_frame", 0)
                mask_path = record.get("mask_path")
                if mask_path is not None:
                    _path(mask_path, f"frame {number} mask_path")
                if validate_images:
                    reader.read_rgb(record)
                    if mask_path is not None:
                        _read_png(mask_path, width, height, mask=True)
    except SourceFrameError as exc:
        raise TrackingError(str(exc)) from exc
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise TrackingError("Frames must be sorted, unique, and contiguous")
    reference = _integer(job.get("reference_frame"), "reference_frame")
    if reference not in numbers:
        raise TrackingError("reference_frame must be included in frames")
    if job.get("backend", "masks") not in ("masks", "sam3"):
        raise TrackingError("backend must be masks or sam3")
    if job.get("tracking_mode", "bbox") not in ("bbox", "features"):
        raise TrackingError("tracking_mode must be bbox or features")
    window = _integer(job.get("smoothing_window", 1), "smoothing_window", 1)
    if window % 2 != 1:
        raise TrackingError("smoothing_window must be odd")
    margin = job.get("crop_margin", 1.2)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 1:
        raise TrackingError("crop_margin must be finite and >= 1")
    if not isinstance(job.get("fixed_crop", True), bool):
        raise TrackingError("fixed_crop must be boolean")
    _integer(job.get("object_index", 0), "object_index", 0)
    threshold = job.get("detection_threshold", 0.45)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise TrackingError("detection_threshold must be in [0, 1]")
    _path(job.get("output_dir"), "output_dir", must_exist=False)
    return job


def mask_bbox(mask, height):
    """Return an exclusive-edge bbox [left,bottom,right,top], or None."""
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return [float(xx.min()), float(height - yy.max() - 1),
            float(xx.max() + 1), float(height - yy.min())]


def _box_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def similarity_matrix(center, scale, radians, reference_center):
    """Similarity mapping a reference pivot to center in Nuke coordinates."""
    c, s = math.cos(radians) * scale, math.sin(radians) * scale
    linear = np.array([[c, -s], [s, c]], dtype=float)
    result = np.eye(3)
    result[:2, :2] = linear
    result[:2, 2] = np.asarray(center) - linear @ np.asarray(reference_center)
    return result


def _bbox_transform(box, reference_box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    ref_area = (reference_box[2] - reference_box[0]) * (reference_box[3] - reference_box[1])
    return similarity_matrix(_box_center(box), math.sqrt(area / ref_area), 0, _box_center(reference_box))


def _image_to_nuke(matrix, height):
    pixels = np.array([[1, 0, 0.5], [0, -1, height - 0.5], [0, 0, 1]], dtype=float)
    return pixels @ matrix @ np.linalg.inv(pixels)


def _estimate_pair(source_gray, target_gray, source_mask, target_mask):
    """Estimate image-index similarity using mask-gated, bidirectional LK."""
    source_area = int(source_mask.sum())
    if source_area < 25 or int(target_mask.sum()) < 25:
        return None, 0.0, "small mask"
    feature_mask = source_mask.astype(np.uint8) * 255
    eroded = cv2.erode(feature_mask, np.ones((3, 3), np.uint8))
    if np.count_nonzero(eroded) >= 25:
        feature_mask = eroded
    points = cv2.goodFeaturesToTrack(source_gray, maxCorners=600, qualityLevel=0.01,
                                    minDistance=4, mask=feature_mask, blockSize=5)
    if points is None or len(points) < 6:
        return None, 0.0, "fewer than 6 textured features"
    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    forward, found, _ = cv2.calcOpticalFlowPyrLK(source_gray, target_gray, points, None, **lk)
    if forward is None:
        return None, 0.0, "optical flow failed"
    backward, back_found, _ = cv2.calcOpticalFlowPyrLK(target_gray, source_gray, forward, None, **lk)
    if backward is None:
        return None, 0.0, "backward optical flow failed"
    src, dst = points.reshape(-1, 2), forward.reshape(-1, 2)
    valid = found.ravel().astype(bool) & back_found.ravel().astype(bool)
    valid &= np.isfinite(dst).all(axis=1) & np.isfinite(backward.reshape(-1, 2)).all(axis=1)
    valid &= np.linalg.norm(src - backward.reshape(-1, 2), axis=1) <= 1.5
    height, width = target_mask.shape
    # Use nearest pixel for mask membership; retain subpixel coordinates for fitting.
    safe_dst = np.nan_to_num(dst, nan=-1, posinf=-1, neginf=-1)
    ix = np.clip(np.rint(safe_dst[:, 0]), 0, width - 1).astype(int)
    iy = np.clip(np.rint(safe_dst[:, 1]), 0, height - 1).astype(int)
    valid &= (dst[:, 0] >= 0) & (dst[:, 0] < width) & (dst[:, 1] >= 0) & (dst[:, 1] < height)
    valid &= target_mask[iy, ix]
    src, dst = src[valid], dst[valid]
    if len(src) < 6:
        return None, 0.0, "fewer than 6 consistent masked features"
    if min(np.linalg.eigvalsh(np.cov(src.T))) < 2:
        return None, 0.0, "degenerate feature distribution"
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=2.0, maxIters=3000, confidence=0.995, refineIters=20)
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return None, 0.0, "similarity fit failed"
    count = int(inliers.sum())
    fraction = float(count / len(src))
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    if count < 6 or fraction < 0.45 or not 0.5 <= scale <= 2.0:
        return None, fraction, "similarity fit rejected"
    result = np.eye(3)
    result[:2] = affine
    return result, fraction, None


def _parameters(matrix, reference_center):
    center = (matrix @ np.r_[reference_center, 1])[:2]
    return np.array([center[0], center[1], math.log(math.hypot(matrix[0, 0], matrix[1, 0])),
                     math.atan2(matrix[1, 0], matrix[0, 0])], dtype=float)


def _fill_missing(values, present, max_gap=3):
    """Interpolate short bounded gaps; hold other gaps. Returns gap labels."""
    result = values.copy()
    labels = {}
    valid = np.flatnonzero(present)
    if not len(valid):
        raise TrackingError("No nonempty target masks were detected")
    index = 0
    while index < len(present):
        if present[index]:
            index += 1
            continue
        start = index
        while index < len(present) and not present[index]:
            index += 1
        stop = index
        left, right = start - 1, stop
        interpolate = left >= 0 and right < len(present) and stop - start <= max_gap
        for missing in range(start, stop):
            if interpolate:
                fraction = (missing - left) / (right - left)
                result[missing] = result[left] * (1 - fraction) + result[right] * fraction
                labels[missing] = "missing_interpolated"
            else:
                result[missing] = result[left] if left >= 0 else result[right]
                labels[missing] = "missing_held"
    return result, labels


def _smooth(values, window):
    if window == 1:
        return values.copy()
    radius = window // 2
    # Truncated symmetric windows avoid invented observations at sequence edges.
    return np.array([values[max(0, i-radius):min(len(values), i+radius+1)].mean(axis=0)
                     for i in range(len(values))])


def track_job(job, progress: Callable | None = None):
    """Compute tracks. SAM3 generation is the worker's responsibility."""
    validate_job(job, validate_images=False)
    frames = job["frames"]
    width, height = job["width"], job["height"]
    reference_index = next(i for i, item in enumerate(frames) if item["frame"] == job["reference_frame"])
    mode = job.get("tracking_mode", "bbox")
    records, warnings = [], []
    boxes = []
    # Decode each source once. The forward/backward feature passes retain only
    # grayscale pixels so a video is not reopened and decoded for every pair.
    gray_frames = []
    try:
        with SourceFrameReader(width, height) as reader:
            for index, record in enumerate(frames):
                # Validate sources in bbox mode too; incorrect source mappings must fail.
                rgb = reader.read_rgb(record)
                if mode == "features":
                    gray_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
                mask_path = record.get("mask_path")
                mask = _read_png(mask_path, width, height, mask=True) if mask_path else None
                box = mask_bbox(mask, height) if mask is not None else None
                if record.get("detected") is False:
                    box = None
                boxes.append(box)
                if progress:
                    progress("tracking", index + 1, len(frames), f"Read mask for frame {record['frame']}")
    except SourceFrameError as exc:
        raise TrackingError(str(exc)) from exc
    present = np.array([box is not None for box in boxes], dtype=bool)
    if not present.any():
        raise TrackingError("No nonempty target masks were detected; no track was written")
    if not present[reference_index]:
        raise TrackingError(f"Reference frame {job['reference_frame']} has no target mask; choose a visible reference frame")
    reference_box = boxes[reference_index]
    reference_center = _box_center(reference_box)
    matrices = [None] * len(frames)
    statuses = ["missing"] * len(frames)
    confidence = np.zeros(len(frames), dtype=float)
    reasons = [None] * len(frames)
    fallback_anchors = [None] * len(frames)
    matrices[reference_index] = np.eye(3)
    statuses[reference_index] = "reference"
    confidence[reference_index] = 1.0

    def read_pair(index):
        record = frames[index]
        return (gray_frames[index],
                _read_png(record["mask_path"], width, height, mask=True))

    for direction in (1, -1):
        previous = reference_index
        previous_pixels = read_pair(previous) if mode == "features" else None
        stop = len(frames) if direction == 1 else -1
        for index in range(reference_index + direction, stop, direction):
            if not present[index]:
                continue
            score = frames[index].get("score", 1.0)
            score = float(score) if isinstance(score, (int, float)) and math.isfinite(score) else 1.0
            score = min(1.0, max(0.0, score))
            matrix = None
            pixels = None
            if mode == "features":
                pixels = read_pair(index)
                if abs(index - previous) <= 4:
                    pair, quality, reason = _estimate_pair(previous_pixels[0], pixels[0], previous_pixels[1], pixels[1])
                    reasons[index] = reason
                    if pair is not None:
                        matrix = _image_to_nuke(pair, height) @ matrices[previous]
                        statuses[index] = "features"
                        confidence[index] = quality * score
                        fallback_anchors[index] = fallback_anchors[previous]
                        if fallback_anchors[index] is not None:
                            reasons[index] = f"Feature transform inherits bbox fallback at frame {fallback_anchors[index]}"
                            confidence[index] = min(confidence[index], 0.5)
                else:
                    reasons[index] = "more than 3 missing frames since last observation"
            if matrix is None:
                matrix = _bbox_transform(boxes[index], reference_box)
                statuses[index] = "bbox" if mode == "bbox" else "bbox_fallback"
                confidence[index] = score if mode == "bbox" else 0.5 * score
                if mode == "features":
                    fallback_anchors[index] = frames[index]["frame"]
            matrices[index] = matrix
            previous = index
            previous_pixels = pixels
            if progress:
                progress("tracking", int(sum(m is not None for m in matrices)), len(frames),
                         f"Tracked frame {frames[index]['frame']} ({statuses[index]})")

    parameters = np.zeros((len(frames), 4), dtype=float)
    for index in np.flatnonzero(present):
        parameters[index] = _parameters(matrices[index], reference_center)
    # Unwrap only measured angles; placeholder zeroes must not introduce wraps.
    parameters[present, 3] = np.unwrap(parameters[present, 3])
    parameters, gap_labels = _fill_missing(parameters, present)
    for index, status in gap_labels.items():
        statuses[index] = status
        reasons[index] = "target mask absent"
    window = job.get("smoothing_window", 1)
    smoothed = _smooth(parameters, window)
    smoothed_matrices = [similarity_matrix(row[:2], math.exp(row[2]), row[3], reference_center) for row in smoothed]
    rebase = np.linalg.inv(smoothed_matrices[reference_index])
    final_matrices = [matrix @ rebase for matrix in smoothed_matrices]
    final_matrices[reference_index] = np.eye(3)  # guarantee exact reference identity
    sides = np.array([max(box[2]-box[0], box[3]-box[1]) if box is not None else 0.0 for box in boxes])
    sides, _ = _fill_missing(sides, present)
    margin = float(job.get("crop_margin", 1.2))
    if job.get("fixed_crop", True):
        crop_sides = np.full(len(frames), math.ceil(float(np.quantile(sides[present], 0.95)) * margin / 2) * 2)
    else:
        crop_sides = np.ceil(_smooth(sides, window) * margin / 2) * 2
    ref_corners = np.array([[reference_box[0], reference_box[1], 1],
                            [reference_box[2], reference_box[1], 1],
                            [reference_box[2], reference_box[3], 1],
                            [reference_box[0], reference_box[3], 1]], dtype=float)
    for index, record in enumerate(frames):
        matrix = final_matrices[index]
        center_x, center_y, log_scale, angle = _parameters(matrix, reference_center)
        center = np.array([center_x, center_y])
        side = max(2.0, float(crop_sides[index]))
        crop_box = [center_x-side/2, center_y-side/2, center_x+side/2, center_y+side/2]
        records.append(dict(frame=record["frame"], center=center.tolist(),
            translate=(center-reference_center).tolist(), scale=float(math.exp(log_scale)),
            rotate=float(math.degrees(smoothed[index, 3] - smoothed[reference_index, 3])),
            cornerpin=(matrix @ ref_corners.T).T[:, :2].tolist(),
            bbox=boxes[index], crop_box=crop_box, mask_path=record.get("mask_path"),
            status=statuses[index], confidence=float(confidence[index]),
            object_id=record.get("object_id", job.get("object_index", 0)),
            detected=bool(present[index]), estimated=not bool(present[index]) or fallback_anchors[index] is not None,
            reason=reasons[index], fallback_anchor_frame=fallback_anchors[index], smoothed=window > 1))
    missing = int((~present).sum())
    fallbacks = statuses.count("bbox_fallback")
    if missing:
        warnings.append(f"{missing} frame(s) have no target mask: bounded gaps of up to 3 frames interpolate; other gaps hold. Inspect estimated frames.")
    if fallbacks:
        warnings.append(f"{fallbacks} frame(s) use bbox fallback because masked feature tracking failed; rotation is not measured on those frames. Subsequent feature tracks inherit that uncertainty until the reference anchor.")
    if mode == "bbox":
        warnings.append("BBox mode estimates translation and uniform scale from mask bounds; it does not measure rotation or perspective.")
    if float(job.get("pixel_aspect", 1.0)) != 1.0:
        warnings.append("Tracking similarities use image pixel coordinates. For non-square pixels, undistort/reformat to square pixels before solving rotation.")
    return dict(schema_version=1, width=width, height=height, pixel_aspect=float(job.get("pixel_aspect", 1.0)),
        reference_frame=job["reference_frame"], first_frame=frames[0]["frame"], last_frame=frames[-1]["frame"],
        reference_center=reference_center.tolist(), reference_box=reference_box,
        tracking_mode=mode, smoothing_window=window, crop_margin=margin,
        confidence_note="Diagnostic quality only: bbox uses detector score, features uses RANSAC inlier fraction times detector score. SAM3 may reuse its original detection score; this is not a per-frame probability.",
        fixed_crop=job.get("fixed_crop", True), frames=records, warnings=warnings)
