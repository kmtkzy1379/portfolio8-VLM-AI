"""Screen capture using mss (cross-platform, fast on Windows)."""

from __future__ import annotations

import threading
import time
from typing import Iterator

import cv2
import numpy as np
from mss import mss
from PIL import Image
import imagehash

from vlm.common.datatypes import CapturedFrame, FrameMetadata
from vlm.common.validators import validate_frame


class ScreenCapture:
    """Captures screen frames with perceptual hashing.

    Args:
        monitor: Monitor index (0=all, 1=primary, 2=secondary, ...).
        target_fps: Target capture rate. Actual rate may be lower.
        max_dimension: Downscale frames if any dimension exceeds this.
    """

    def __init__(
        self,
        monitor: int = 1,
        target_fps: float = 2.0,
        max_dimension: int = 1920,
    ):
        self._monitor = monitor
        self._interval = 1.0 / target_fps
        self._max_dim = max_dimension
        self._frame_counter = 0

    def capture_one(self) -> CapturedFrame:
        """Capture a single screen frame."""
        with mss() as sct:
            mon = sct.monitors[self._monitor]
            raw = sct.grab(mon)
            # mss returns BGRA; convert to BGR
            image = np.array(raw)[:, :, :3].copy()  # drop alpha

        image = self._maybe_downscale(image)
        validate_frame(image)

        phash = self._compute_phash(image)
        metadata = FrameMetadata(
            frame_id=self._frame_counter,
            timestamp_ms=time.monotonic() * 1000,
            source_width=image.shape[1],
            source_height=image.shape[0],
        )
        self._frame_counter += 1

        return CapturedFrame(metadata=metadata, image=image, phash=phash)

    def stream(self) -> Iterator[CapturedFrame]:
        """Yield frames at target FPS (blocking iterator)."""
        while True:
            t0 = time.monotonic()
            yield self.capture_one()
            elapsed = time.monotonic() - t0
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stream_until(self, stop_event: threading.Event) -> Iterator[CapturedFrame]:
        """Yield frames at target FPS, stopping when *stop_event* is set."""
        while not stop_event.is_set():
            t0 = time.monotonic()
            yield self.capture_one()
            elapsed = time.monotonic() - t0
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                stop_event.wait(timeout=sleep_time)

    def _maybe_downscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= self._max_dim:
            return image
        scale = self._max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _compute_phash(image: np.ndarray) -> int:
        """Compute 64-bit perceptual hash."""
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h = imagehash.phash(pil_img)
        return int(str(h), 16)
