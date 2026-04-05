"""Motion detection from bounding box and keypoint deltas."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from vlm.common.datatypes import BoundingBox, MotionData, SkeletonData


class MotionDetector:
    """Computes motion metrics by comparing current vs previous features.

    Motion is derived from:
    1. Bounding box center displacement
    2. Skeleton keypoint displacement (if available)
    """

    # Velocity thresholds (pixels/frame) for action classification
    VELOCITY_THRESHOLDS = {
        "stationary": 3.0,
        "slow_move": 15.0,
        "walking": 40.0,
        # above walking = running
    }

    def compute(
        self,
        track_id: int,
        current_bbox: BoundingBox,
        prev_bbox: Optional[BoundingBox],
        current_skeleton: Optional[SkeletonData] = None,
        prev_skeleton: Optional[SkeletonData] = None,
    ) -> MotionData:
        """Compute motion between current and previous frame.

        Args:
            track_id: Stable track ID from IDAuthority.
            current_bbox: Current bounding box.
            prev_bbox: Previous bounding box (None if new entity).
            current_skeleton: Current skeleton (optional).
            prev_skeleton: Previous skeleton (optional).

        Returns:
            MotionData with velocity, acceleration, and action label.
        """
        if prev_bbox is None:
            return MotionData(
                track_id=track_id,
                velocity=(0.0, 0.0),
                acceleration=(0.0, 0.0),
                action_label="new",
                displacement_since_last=0.0,
            )

        # Bounding box center displacement
        cx, cy = current_bbox.center
        px, py = prev_bbox.center
        vx = cx - px
        vy = cy - py
        displacement = math.sqrt(vx * vx + vy * vy)

        # If skeleton data available, use mean keypoint displacement for precision
        if (
            current_skeleton is not None
            and prev_skeleton is not None
            and current_skeleton.keypoints.shape == prev_skeleton.keypoints.shape
        ):
            # Only use visible keypoints (confidence > 0.5)
            visible = (
                (current_skeleton.keypoints[:, 2] > 0.5)
                & (prev_skeleton.keypoints[:, 2] > 0.5)
            )
            if visible.any():
                diffs = (
                    current_skeleton.keypoints[visible, :2]
                    - prev_skeleton.keypoints[visible, :2]
                )
                mean_diff = diffs.mean(axis=0)
                vx = float(mean_diff[0])
                vy = float(mean_diff[1])
                displacement = float(np.linalg.norm(diffs, axis=1).mean())

        action = self._classify_action(displacement)

        return MotionData(
            track_id=track_id,
            velocity=(round(vx, 1), round(vy, 1)),
            acceleration=(0.0, 0.0),  # Phase 5: compute from velocity history
            action_label=action,
            displacement_since_last=round(displacement, 1),
        )

    def _classify_action(self, displacement: float) -> str:
        if displacement < self.VELOCITY_THRESHOLDS["stationary"]:
            return "stationary"
        elif displacement < self.VELOCITY_THRESHOLDS["slow_move"]:
            return "slow_move"
        elif displacement < self.VELOCITY_THRESHOLDS["walking"]:
            return "walking"
        else:
            return "running"
