"""Orchestrates per-entity analysis: pose, expression, motion."""

from __future__ import annotations

import logging
from typing import Optional

from vlm.analysis.expression import ExpressionDetector
from vlm.analysis.motion import MotionDetector
from vlm.analysis.pose import PoseEstimator
from vlm.common.datatypes import EntityFeatures, TrackedEntity

logger = logging.getLogger(__name__)


class PerIDAnalyzer:
    """Runs all analysis models on tracked entity crops.

    All analysis uses crops and IDs provided by the IDAuthority.
    This class never generates its own IDs or detections.

    Args:
        pose_estimator: PoseEstimator instance.
        expression_detector: ExpressionDetector instance.
        skip_iou_threshold: Skip re-analysis if bbox IoU > this.
        skip_min_frames: Minimum frames between re-analysis.
    """

    def __init__(
        self,
        pose_estimator: PoseEstimator,
        expression_detector: ExpressionDetector,
        skip_iou_threshold: float = 0.9,
        skip_min_frames: int = 5,
        expression_enabled: bool = True,
    ):
        self._pose = pose_estimator
        self._expr = expression_detector
        self._motion = MotionDetector()
        self._skip_iou = skip_iou_threshold
        self._skip_frames = skip_min_frames
        self._expr_enabled = expression_enabled

    def analyze(
        self,
        entity: TrackedEntity,
        frame_id: int,
        prev_features: Optional[EntityFeatures] = None,
    ) -> EntityFeatures:
        """Run full analysis on one tracked entity.

        Args:
            entity: TrackedEntity with crop from IDAuthority.
            frame_id: Current frame ID.
            prev_features: Previous features for this entity (for delta/motion).

        Returns:
            EntityFeatures with all analysis results.
        """
        # Check if we can skip (entity barely moved since last analysis)
        if self._should_skip(entity, frame_id, prev_features):
            return self._reuse_with_updated_bbox(entity, frame_id, prev_features)

        # Pose estimation
        skeleton = None
        if entity.crop is not None and entity.class_name == "person":
            skeleton = self._pose.estimate(entity.crop, entity.track_id)

        # Expression detection (only for person class, gated by config)
        expression = None
        if self._expr_enabled and entity.crop is not None and entity.class_name == "person":
            expression = self._expr.analyze(entity.crop, entity.track_id)

        # Motion computation
        prev_bbox = prev_features.bbox if prev_features else None
        prev_skeleton = prev_features.skeleton if prev_features else None
        motion = self._motion.compute(
            entity.track_id, entity.bbox, prev_bbox,
            skeleton, prev_skeleton,
        )

        return EntityFeatures(
            track_id=entity.track_id,
            frame_id=frame_id,
            bbox=entity.bbox,
            skeleton=skeleton,
            expression=expression,
            motion=motion,
            attributes={"class": entity.class_name},
        )

    def _should_skip(
        self,
        entity: TrackedEntity,
        frame_id: int,
        prev: Optional[EntityFeatures],
    ) -> bool:
        if prev is None:
            return False
        if frame_id - prev.frame_id < self._skip_frames:
            if entity.bbox.iou(prev.bbox) > self._skip_iou:
                return True
        return False

    @staticmethod
    def _reuse_with_updated_bbox(
        entity: TrackedEntity,
        frame_id: int,
        prev: EntityFeatures,
    ) -> EntityFeatures:
        """Reuse previous analysis but update bbox and frame_id."""
        return EntityFeatures(
            track_id=entity.track_id,
            frame_id=frame_id,
            bbox=entity.bbox,
            skeleton=prev.skeleton,
            expression=prev.expression,
            motion=prev.motion,
            attributes=prev.attributes,
        )

    def close(self) -> None:
        self._pose.close()
