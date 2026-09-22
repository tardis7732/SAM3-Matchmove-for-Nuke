"""RAM adapter for SAM3 Matchmove's mask-guided similarity tracking.

Tracking loop adapted from the installed SAM3-Matchmove-for-Nuke tracking.py;
uses its existing math/optical-flow helpers, without any file readers/writers.
"""
import json
import math
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
settings = json.loads((root / 'config/frontend.json').read_text(encoding='utf-8'))
sys.path.insert(0, settings['matchmove_root'])
from sam3_matchmove.tracking import (TrackingError, mask_bbox, _box_center,
    similarity_matrix, _bbox_transform, _estimate_pair, _image_to_nuke,
    _parameters, _fill_missing, _smooth)


def validate_memory_job(job):
    frames = job['frames']
    if not frames:
        raise TrackingError('No frames in RAM analysis')
    numbers = [row['frame'] for row in frames]
    if numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise TrackingError('Frames must be contiguous')
    if job['reference_frame'] not in numbers:
        raise TrackingError('Reference must be inside the analysis range')
    if job.get('tracking_mode', 'bbox') not in ('bbox', 'features', 'translation', 'translation_scale', 'translation_rotation', 'translation_scale_rotation'):
        raise TrackingError('Invalid motion mode')
    window = job.get('smoothing_window', 1)
    if type(window) is not int or window < 1 or window % 2 != 1:
        raise TrackingError('Smoothing window must be a positive odd number')
    margin = job.get('crop_margin', 1.2)
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 1:
        raise TrackingError('Crop margin must be at least 1')
    if type(job.get('fixed_crop', True)) is not bool:
        raise TrackingError('Fixed crop must be boolean')
    for row in frames:
        if row['mask'].shape != (job['height'], job['width']) or row['gray'].shape != row['mask'].shape:
            raise TrackingError('Frame dimensions changed during analysis')


def track_memory(job, progress=None):
    """Compute tracks. SAM3 generation is the worker's responsibility."""
    validate_memory_job(job)
    frames = job["frames"]
    width, height = job["width"], job["height"]
    reference_index = next(i for i, item in enumerate(frames) if item["frame"] == job["reference_frame"])
    mode = job.get("tracking_mode", "bbox")
    use_features = mode in ('features', 'translation_rotation', 'translation_scale_rotation')
    records, warnings = [], []
    boxes = []
    # Decode each source once. The forward/backward feature passes retain only
    # grayscale pixels so a video is not reopened and decoded for every pair.
    gray_frames = [record['gray'] for record in frames]
    for index, record in enumerate(frames):
        boxes.append(mask_bbox(record['mask'], height))
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
                record["mask"])

    for direction in (1, -1):
        previous = reference_index
        previous_pixels = read_pair(previous) if use_features else None
        stop = len(frames) if direction == 1 else -1
        for index in range(reference_index + direction, stop, direction):
            if not present[index]:
                continue
            score = frames[index].get("score", 1.0)
            score = float(score) if isinstance(score, (int, float)) and math.isfinite(score) else 1.0
            score = min(1.0, max(0.0, score))
            matrix = None
            pixels = None
            if use_features:
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
                statuses[index] = "bbox_fallback" if use_features else "bbox"
                confidence[index] = 0.5 * score if use_features else score
                if use_features:
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
    # Restrict the solved transform itself so every export uses the selected components.
    if mode in ('translation', 'translation_rotation'):
        smoothed[:, 2] = 0.0
    if mode in ('translation', 'translation_scale', 'bbox'):
        smoothed[:, 3] = 0.0
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
            rotate=float(math.degrees(angle)),
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
