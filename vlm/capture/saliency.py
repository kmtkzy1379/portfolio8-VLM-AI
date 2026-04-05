"""Saliency Detection - inspired by the brain's V1 attention mechanism.

The brain does not process the entire visual field equally. Bottom-up
attention (saliency) highlights regions that "stand out" due to contrast,
color, or structural uniqueness. This module identifies salient regions
and assigns priority scores to ROIs.

Combined with PredictiveCoder:
  - PredictiveCoder: "WHERE did things change?" (temporal saliency)
  - SaliencyDetector: "WHAT is visually prominent?" (spatial saliency)
  - Union: changed + salient = highest processing priority
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from vlm.capture.predictive_coder import ChangedRegion

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoredRegion:
    """A region with combined saliency and change scores."""
    x: int
    y: int
    w: int
    h: int
    saliency_score: float      # 0.0-1.0, spatial prominence
    change_score: float        # 0.0-1.0, temporal change (0 if static)
    combined_score: float      # Weighted combination

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class SaliencyDetector:
    """Computes spatial saliency maps using Spectral Residual method.

    The Spectral Residual approach (Hou & Zhang, CVPR 2007) identifies
    visually salient regions without any training, by analyzing the
    spectral characteristics of the image. Regions with unusual spectral
    content "pop out", similar to bottom-up attention in V1.

    Args:
        saliency_threshold: Normalized threshold (0-1) for binary saliency.
        min_region_area: Minimum area for a salient region.
        change_weight: Weight of temporal change in combined score.
        saliency_weight: Weight of spatial saliency in combined score.
    """

    def __init__(
        self,
        saliency_threshold: float = 0.4,
        min_region_area: int = 500,
        change_weight: float = 0.6,
        saliency_weight: float = 0.4,
    ):
        self._threshold = saliency_threshold
        self._min_area = min_region_area
        self._change_w = change_weight
        self._saliency_w = saliency_weight

    def compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """Compute normalized saliency map (0.0-1.0 float32).

        Implements Spectral Residual saliency (Hou & Zhang, CVPR 2007)
        directly, avoiding dependency on opencv-contrib.

        Algorithm:
          1. Convert to grayscale, resize to 64×64 for FFT
          2. Compute log amplitude spectrum
          3. Spectral residual = log_amplitude - averaged_log_amplitude
          4. Reconstruct saliency map from residual + original phase
          5. Gaussian blur and normalize

        Args:
            image: BGR uint8 image.

        Returns:
            Saliency map as float32 array, same H×W as input.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Resize to small fixed size for efficient FFT
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        # 2D FFT
        fft = np.fft.fft2(small)
        amplitude = np.abs(fft)
        phase = np.angle(fft)

        # Log amplitude spectrum
        log_amp = np.log(amplitude + 1e-10)

        # Spectral residual: subtract locally averaged spectrum
        avg_kernel = np.ones((3, 3)) / 9.0
        avg_log_amp = cv2.filter2D(log_amp, -1, avg_kernel)
        spectral_residual = log_amp - avg_log_amp

        # Reconstruct from residual amplitude + original phase
        saliency_fft = np.exp(spectral_residual) * np.exp(1j * phase)
        saliency_small = np.abs(np.fft.ifft2(saliency_fft)) ** 2

        # Gaussian blur to smooth
        saliency_small = cv2.GaussianBlur(
            saliency_small.astype(np.float32), (5, 5), 2.5
        )

        # Resize back to original dimensions
        smap = cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # If variance is near zero (uniform image), saliency is uniformly low
        if smap.var() < 1e-10:
            return np.zeros((h, w), dtype=np.float32)

        # Normalize to 0-1
        min_val = smap.min()
        max_val = smap.max()
        if max_val - min_val > 0:
            smap = (smap - min_val) / (max_val - min_val)
        else:
            return np.zeros((h, w), dtype=np.float32)

        return smap

    def find_salient_regions(
        self, image: np.ndarray
    ) -> list[ScoredRegion]:
        """Find spatially salient regions in the image.

        Args:
            image: BGR uint8 image.

        Returns:
            List of ScoredRegion sorted by saliency_score descending.
        """
        smap = self.compute_saliency_map(image)

        # Binary threshold
        binary = (smap > self._threshold).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < self._min_area:
                continue

            # Mean saliency within region
            region_saliency = float(smap[y : y + h, x : x + w].mean())

            regions.append(ScoredRegion(
                x=x, y=y, w=w, h=h,
                saliency_score=region_saliency,
                change_score=0.0,
                combined_score=region_saliency * self._saliency_w,
            ))

        regions.sort(key=lambda r: r.saliency_score, reverse=True)
        return regions

    def combine_with_changes(
        self,
        image: np.ndarray,
        changed_regions: list[ChangedRegion],
    ) -> list[ScoredRegion]:
        """Combine spatial saliency with temporal change regions.

        Produces a unified priority-scored list of regions:
        - Regions that are BOTH salient AND changed get highest priority
        - Changed-only regions get medium priority
        - Salient-only regions get lower priority

        Args:
            image: Current BGR frame.
            changed_regions: From PredictiveCoder.

        Returns:
            Unified list of ScoredRegion, sorted by combined_score descending.
        """
        smap = self.compute_saliency_map(image)
        scored: list[ScoredRegion] = []

        # Score changed regions with saliency
        for cr in changed_regions:
            x, y, w, h = cr.x, cr.y, cr.w, cr.h
            # Clamp to image bounds
            y2 = min(y + h, smap.shape[0])
            x2 = min(x + w, smap.shape[1])
            if y2 <= y or x2 <= x:
                continue

            region_saliency = float(smap[y:y2, x:x2].mean())
            change_norm = min(cr.change_magnitude / 255.0, 1.0)

            combined = (
                self._change_w * change_norm
                + self._saliency_w * region_saliency
            )

            scored.append(ScoredRegion(
                x=x, y=y, w=w, h=h,
                saliency_score=region_saliency,
                change_score=change_norm,
                combined_score=combined,
            ))

        # Also find salient-only regions not covered by changes
        salient_only = self.find_salient_regions(image)
        for sr in salient_only:
            if not self._overlaps_any(sr, scored):
                sr.combined_score = self._saliency_w * sr.saliency_score
                scored.append(sr)

        scored.sort(key=lambda r: r.combined_score, reverse=True)
        return scored

    @staticmethod
    def _overlaps_any(region: ScoredRegion, others: list[ScoredRegion]) -> bool:
        """Check if region significantly overlaps any region in the list."""
        rx1, ry1, rx2, ry2 = region.bbox
        for o in others:
            ox1, oy1, ox2, oy2 = o.bbox
            # Compute IoU
            ix1 = max(rx1, ox1)
            iy1 = max(ry1, oy1)
            ix2 = min(rx2, ox2)
            iy2 = min(ry2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_r = (rx2 - rx1) * (ry2 - ry1)
            if area_r > 0 and inter / area_r > 0.3:
                return True
        return False
