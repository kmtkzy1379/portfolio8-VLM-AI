"""Two-stage change detection: fast pHash then SSIM confirmation."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from vlm.common.datatypes import CapturedFrame, ChangeLevel

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Decides whether a frame warrants processing.

    Stage 1 (fast): Compare perceptual hash hamming distance.
    Stage 2 (slow, only if stage 1 triggers): Compute SSIM on downscaled images.

    Args:
        phash_threshold: Hamming distance threshold (0-64). Higher = more sensitive.
        ssim_threshold_major: SSIM below this = MAJOR change.
        ssim_threshold_moderate: SSIM below this = MODERATE change.
        ssim_downscale: Downscale to NxN for SSIM computation.
        periodic_interval: Force MODERATE after N consecutive NONE frames.
    """

    def __init__(
        self,
        phash_threshold: int = 12,
        ssim_threshold_major: float = 0.50,
        ssim_threshold_moderate: float = 0.80,
        ssim_downscale: int = 256,
        periodic_interval: int = 10,
    ):
        self._phash_threshold = phash_threshold
        self._ssim_major = ssim_threshold_major
        self._ssim_moderate = ssim_threshold_moderate
        self._downscale = ssim_downscale
        self._periodic_interval = periodic_interval

        self._reference_frame: CapturedFrame | None = None
        self._reference_gray: np.ndarray | None = None  # downscaled grayscale
        self._idle_counter = 0

    def evaluate(self, frame: CapturedFrame) -> ChangeLevel:
        """Evaluate change level for a new frame."""
        # First frame: always MAJOR
        if self._reference_frame is None:
            self._update_reference(frame)
            return ChangeLevel.MAJOR

        # Stage 1: fast pHash comparison
        hamming = self._hamming_distance(self._reference_frame.phash, frame.phash)

        if hamming < self._phash_threshold:
            self._idle_counter += 1
            # Periodic forced check
            if self._idle_counter >= self._periodic_interval:
                self._idle_counter = 0
                logger.debug(
                    "frame=%d periodic_check (idle=%d)",
                    frame.metadata.frame_id,
                    self._periodic_interval,
                )
                return ChangeLevel.MODERATE
            return ChangeLevel.NONE

        # Stage 2: SSIM on downscaled grayscale
        current_gray = self._to_downscaled_gray(frame.image)
        ssim_score = ssim(self._reference_gray, current_gray)

        if ssim_score < self._ssim_major:
            level = ChangeLevel.MAJOR
        elif ssim_score < self._ssim_moderate:
            level = ChangeLevel.MODERATE
        else:
            level = ChangeLevel.MINOR

        logger.debug(
            "frame=%d hamming=%d ssim=%.3f level=%s",
            frame.metadata.frame_id,
            hamming,
            ssim_score,
            level.name,
        )

        # Update reference on significant change
        if level in (ChangeLevel.MAJOR, ChangeLevel.MODERATE):
            self._update_reference(frame)
            self._idle_counter = 0

        return level

    def force_update_reference(self, frame: CapturedFrame) -> None:
        """Manually update the reference frame."""
        self._update_reference(frame)
        self._idle_counter = 0

    def _update_reference(self, frame: CapturedFrame) -> None:
        self._reference_frame = frame
        self._reference_gray = self._to_downscaled_gray(frame.image)

    def _to_downscaled_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(
            gray, (self._downscale, self._downscale), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _hamming_distance(hash1: int, hash2: int) -> int:
        return bin(hash1 ^ hash2).count("1")
