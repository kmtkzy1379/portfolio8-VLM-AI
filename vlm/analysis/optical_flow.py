"""Optical Flow Motion Detection - inspired by brain's MT/V5 area.

The MT (middle temporal) area of the visual cortex processes motion at
pixel level, detecting speed, direction, and complex motion patterns.
This module uses Farneback dense optical flow to compute per-pixel motion
vectors, then aggregates them per tracked entity for precise motion
understanding.

Advantages over simple bbox-center motion:
  - Detects internal motion (e.g., arm waving while body is still)
  - Distinguishes rotation from translation
  - Detects speed variations within an entity
  - More robust to bbox jitter from detection noise
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from vlm.common.datatypes import BoundingBox, MotionData

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FlowField:
    """Dense optical flow result for a full frame."""
    flow: np.ndarray       # H x W x 2 (dx, dy per pixel)
    magnitude: np.ndarray  # H x W (speed per pixel)
    angle: np.ndarray      # H x W (direction per pixel, radians)


class OpticalFlowMotion:
    """Computes motion from dense optical flow (Farneback method).

    Replaces simple bbox-center displacement with pixel-level motion
    analysis per tracked entity.

    Args:
        pyr_scale: Pyramid scale factor for Farneback.
        levels: Number of pyramid levels.
        winsize: Averaging window size.
        iterations: Number of iterations per level.
        poly_n: Pixel neighborhood size for polynomial expansion.
        poly_sigma: Gaussian sigma for polynomial expansion.
    """

    # Action classification thresholds (mean magnitude in pixels/frame)
    THRESHOLDS = {
        "stationary": 1.5,
        "slow_move": 5.0,
        "walking": 15.0,
        # above walking = running
    }

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        self._params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._last_flow: Optional[FlowField] = None

    def update_frame(self, image: np.ndarray) -> Optional[FlowField]:
        """Compute dense optical flow between previous and current frame.

        Must be called once per frame BEFORE compute_entity_motion.

        Args:
            image: Current frame (BGR uint8).

        Returns:
            FlowField, or None if this is the first frame.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._last_flow = None
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, **self._params
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        self._prev_gray = gray
        self._last_flow = FlowField(
            flow=flow,
            magnitude=magnitude.astype(np.float32),
            angle=angle.astype(np.float32),
        )
        return self._last_flow

    def compute_entity_motion(
        self,
        track_id: int,
        bbox: BoundingBox,
    ) -> MotionData:
        """Compute motion for a specific tracked entity using flow field.

        Args:
            track_id: Stable track ID from IDAuthority.
            bbox: Entity bounding box in current frame.

        Returns:
            MotionData with flow-based velocity and action label.
        """
        if self._last_flow is None:
            return MotionData(
                track_id=track_id,
                velocity=(0.0, 0.0),
                acceleration=(0.0, 0.0),
                action_label="new",
                displacement_since_last=0.0,
            )

        flow = self._last_flow
        h, w = flow.flow.shape[:2]

        # Extract flow within entity bbox (clamped to frame bounds)
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return MotionData(
                track_id=track_id,
                velocity=(0.0, 0.0),
                acceleration=(0.0, 0.0),
                action_label="stationary",
                displacement_since_last=0.0,
            )

        # Flow vectors within entity region
        region_flow = flow.flow[y1:y2, x1:x2]
        region_mag = flow.magnitude[y1:y2, x1:x2]

        # Mean velocity (dx, dy)
        mean_vx = float(region_flow[..., 0].mean())
        mean_vy = float(region_flow[..., 1].mean())

        # Mean magnitude (overall speed)
        mean_speed = float(region_mag.mean())

        # Max magnitude (peak motion, detects fast-moving parts)
        max_speed = float(region_mag.max())

        # Internal motion variance (high = complex motion like gesturing)
        motion_variance = float(region_mag.var())

        # Action classification based on mean speed
        action = self._classify_action(mean_speed)

        return MotionData(
            track_id=track_id,
            velocity=(round(mean_vx, 1), round(mean_vy, 1)),
            acceleration=(0.0, 0.0),  # Could track velocity history
            action_label=action,
            displacement_since_last=round(mean_speed, 1),
        )

    def update_reference_only(self, image: np.ndarray) -> None:
        """Update the reference frame without computing optical flow.

        Use this for MINOR changes where full flow computation is unnecessary,
        but the reference frame must be kept current for the next MODERATE/MAJOR frame.

        Args:
            image: Current frame (BGR uint8).
        """
        self._prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._last_flow = None

    def reset(self) -> None:
        """Reset flow state (e.g. on scene cut)."""
        self._prev_gray = None
        self._last_flow = None

    @property
    def has_flow(self) -> bool:
        return self._last_flow is not None

    def _classify_action(self, mean_speed: float) -> str:
        if mean_speed < self.THRESHOLDS["stationary"]:
            return "stationary"
        elif mean_speed < self.THRESHOLDS["slow_move"]:
            return "slow_move"
        elif mean_speed < self.THRESHOLDS["walking"]:
            return "walking"
        else:
            return "running"
