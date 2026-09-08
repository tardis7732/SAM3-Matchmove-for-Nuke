"""Read worker source frames without writing intermediate image files.

Sources must already contain sRGB-encoded display RGB. Decoding does not apply
Nuke processing, OCIO transforms, embedded ICC profiles, or display transforms.
Masks retain their separate PNG-only contract.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))
VIDEO_SUFFIXES = frozenset((".mp4", ".mov", ".avi", ".mkv", ".webm"))


class SourceFrameError(ValueError):
    """A source frame cannot be decoded with the requested geometry or index."""


class SourceFrameReader:
    """Decode RGB uint8 frames, retaining one sequential video decoder.

    Records use ``path`` and the Nuke ``frame`` number for diagnostics. Video
    records additionally require ``source_frame``, a zero-based integer decoded
    frame index. Reading backwards reopens the video and decodes from frame zero
    instead of depending on codec-specific random-seek accuracy.
    """

    def __init__(self, width, height):
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (width, height)):
            raise SourceFrameError("Source width and height must be positive integers")
        self.width, self.height = width, height
        self._capture = None
        self._video_path = None
        self._next_index = 0
        self._last_rgb = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._video_path = None
        self._next_index = 0
        self._last_rgb = None

    def _check_size(self, rgb, label):
        if rgb.shape != (self.height, self.width, 3):
            actual = (rgb.shape[1], rgb.shape[0]) if rgb.ndim >= 2 else rgb.shape
            raise SourceFrameError(
                "Dimension mismatch for %s: %s, expected %s"
                % (label, actual, (self.width, self.height))
            )
        if rgb.dtype != np.uint8:
            raise SourceFrameError("Expected decoded uint8 RGB for %s" % label)
        return np.ascontiguousarray(rgb).copy()

    def _read_video(self, path, index, label):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise SourceFrameError(
                "%s requires source_frame as a non-negative zero-based video index" % label
            )
        if self._video_path == path and index == self._next_index - 1 and self._last_rgb is not None:
            return self._last_rgb.copy()
        if self._video_path != path or index < self._next_index or self._capture is None:
            self.close()
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise SourceFrameError("Cannot open video for %s; check the file and codec" % label)
            self._capture = capture
            self._video_path = path
        while self._next_index <= index:
            success, bgr = self._capture.read()
            if not success or bgr is None:
                failed_index = self._next_index
                self.close()
                raise SourceFrameError(
                    "Cannot decode source_frame %d for %s: video ended or decoding failed "
                    "at zero-based frame %d" % (index, label, failed_index)
                )
            self._next_index += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._last_rgb = self._check_size(rgb, label)
        return self._last_rgb.copy()

    def read_rgb(self, record):
        """Return an owned HxWx3 uint8 RGB array for one source record."""
        if not isinstance(record, dict):
            raise SourceFrameError("Source frame record must be a dictionary")
        value = record.get("path")
        if not isinstance(value, str) or not value.strip():
            raise SourceFrameError("Source frame path must be a nonempty absolute path")
        path = Path(value)
        label = "Nuke frame %s (%s)" % (record.get("frame", "?"), path)
        if not path.is_absolute() or not path.is_file():
            raise SourceFrameError("Source path must be an existing absolute file: %s" % label)
        suffix = path.suffix.lower()
        try:
            if suffix in VIDEO_SUFFIXES:
                return self._read_video(path, record.get("source_frame"), label)
            if suffix not in IMAGE_SUFFIXES:
                raise SourceFrameError("Unsupported source format %r for %s" % (suffix, label))
            with Image.open(path) as image:
                if image.mode in ("F", "I", "I;16", "I;16B", "I;16L"):
                    raise SourceFrameError(
                        "High-depth or floating-point grayscale source needs conversion "
                        "to sRGB RGB before direct analysis: %s" % label
                    )
                rgb = np.asarray(image.convert("RGB"))
            return self._check_size(rgb, label)
        except SourceFrameError:
            raise
        except Exception as exc:
            self.close()
            raise SourceFrameError("Cannot decode %s: %s" % (label, exc)) from exc

    def read_pil(self, record):
        """Return a detached PIL RGB image for official SAM3's list input."""
        return Image.fromarray(self.read_rgb(record))


def read_rgb(record, width, height):
    """Read one frame; reuse SourceFrameReader for a sequence of video frames."""
    with SourceFrameReader(width, height) as reader:
        return reader.read_rgb(record)


def read_pil(record, width, height):
    """Read one detached PIL RGB source frame."""
    with SourceFrameReader(width, height) as reader:
        return reader.read_pil(record)
