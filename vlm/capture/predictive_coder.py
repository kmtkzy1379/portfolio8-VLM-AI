"""Predictive Coding - inspired by the brain's V2 cortex.

The brain predicts the next visual frame and only processes prediction errors
(unexpected changes). This module identifies WHICH REGIONS changed, not just
WHETHER the frame changed. Combined with ChangeDetector, this enables:
  - ChangeDetector: "Did the screen change?" (global level)
  - PredictiveCoder: "WHERE did it change?" (region level)

Only changed regions are sent to downstream detection/analysis, reducing
computation by 80-90% on mostly-static screens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChangedRegion:
    """A rectangular region where visual change was detected."""
    x: int
    y: int
    w: int
    h: int
    change_magnitude: float  # Mean pixel difference in region (0-255)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class PredictiveCoder:
    """Identifies changed regions between frames using frame differencing.

    Mimics the brain's predictive coding: the "prediction" is the previous
    frame, and the "prediction error" is the absolute difference.

    Args:
        diff_threshold: Pixel difference threshold (0-255) for binary mask.
        min_region_area: Minimum area (pixels) for a changed region.
        blur_kernel: Gaussian blur kernel for noise suppression.
        merge_distance: Max pixel distance to merge nearby changed regions.
    """

    def __init__(
        self,
        diff_threshold: int = 30,
        min_region_area: int = 500,
        blur_kernel: int = 5,
        merge_distance: int = 50,
    ):
        self._threshold = diff_threshold
        self._min_area = min_region_area
        self._blur_k = blur_kernel
        self._merge_dist = merge_distance
        self._prev_gray: Optional[np.ndarray] = None

    def compute_change_regions(
        self, image: np.ndarray
    ) -> list[ChangedRegion]:
        """Compute regions that changed since the previous frame.

        Args:
            image: Current frame (BGR uint8).

        Returns:
            List of ChangedRegion, sorted by change_magnitude descending.
            Empty list if no significant changes or first frame.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []  # First frame: no comparison possible

        # Prediction error = |current - predicted(=previous)|
        diff = cv2.absdiff(self._prev_gray, gray)

        # Binary mask of changed pixels
        _, mask = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean noise and connect nearby changes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours of changed regions
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < self._min_area:
                continue

            # Compute mean change magnitude in this region
            region_diff = diff[y : y + h, x : x + w]
            magnitude = float(region_diff.mean())

            regions.append(ChangedRegion(
                x=x, y=y, w=w, h=h, change_magnitude=magnitude,
            ))

        # Merge nearby regions
        regions = self._merge_nearby(regions)

        # Sort by magnitude (most changed first)
        regions.sort(key=lambda r: r.change_magnitude, reverse=True)

        # Update prediction (previous frame)
        self._prev_gray = gray

        logger.debug("Found %d changed regions", len(regions))
        return regions

    def compute_change_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask (0/255) of changed pixels.

        Useful for visualization and saliency combination.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros(gray.shape, dtype=np.uint8)

        diff = cv2.absdiff(self._prev_gray, gray)
        _, mask = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)
        # Don't update prev_gray here (compute_change_regions handles it)
        return mask

    def reset(self) -> None:
        """Reset prediction state (e.g. on scene cut)."""
        self._prev_gray = None

    def _merge_nearby(self, regions: list[ChangedRegion]) -> list[ChangedRegion]:
        """Merge regions that are close together."""
        if len(regions) <= 1:
            return regions

        merged = True
        while merged:
            merged = False
            new_regions = []
            used = set()

            for i, r1 in enumerate(regions):
                if i in used:
                    continue
                for j, r2 in enumerate(regions):
                    if j <= i or j in used:
                        continue
                    if self._should_merge(r1, r2):
                        new_regions.append(self._merge_two(r1, r2))
                        used.add(i)
                        used.add(j)
                        merged = True
                        break
                if i not in used:
                    new_regions.append(r1)

            regions = new_regions

        return regions

    def _should_merge(self, r1: ChangedRegion, r2: ChangedRegion) -> bool:
        """Check if two regions are close enough to merge."""
        # Distance between closest edges
        dx = max(0, max(r1.x, r2.x) - min(r1.x + r1.w, r2.x + r2.w))
        dy = max(0, max(r1.y, r2.y) - min(r1.y + r1.h, r2.y + r2.h))
        return (dx * dx + dy * dy) <= self._merge_dist * self._merge_dist

    @staticmethod
    def _merge_two(r1: ChangedRegion, r2: ChangedRegion) -> ChangedRegion:
        """Merge two regions into their bounding rectangle."""
        x = min(r1.x, r2.x)
        y = min(r1.y, r2.y)
        x2 = max(r1.x + r1.w, r2.x + r2.w)
        y2 = max(r1.y + r1.h, r2.y + r2.h)
        mag = max(r1.change_magnitude, r2.change_magnitude)
        return ChangedRegion(x=x, y=y, w=x2 - x, h=y2 - y, change_magnitude=mag)
