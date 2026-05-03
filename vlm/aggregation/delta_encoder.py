"""Delta encoding: only emit changes between frames."""

from __future__ import annotations

import logging
from typing import Optional

from vlm.aggregation.feature_store import FeatureStore
from vlm.common.datatypes import (
    ChangeLevel,
    EntityDelta,
    EntityFeatures,
    FrameDelta,
    TrackingState,
)

logger = logging.getLogger(__name__)


class DeltaEncoder:
    """Computes per-entity deltas and formats compact output.

    Only reports fields that actually changed since last report,
    minimizing tokens sent to the LLM.

    Args:
        feature_store: FeatureStore for historical lookups.
        coordinate_precision: Round coordinates to nearest N pixels.
        min_movement: Ignore movements below N pixels.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        coordinate_precision: int = 5,
        min_movement: float = 10.0,
        min_lifetime: int = 0,
    ):
        self._store = feature_store
        self._precision = coordinate_precision
        self._min_move = min_movement
        self._min_lifetime = min_lifetime
        # Last reported features per entity (for delta computation)
        self._last_reported: dict[int, EntityFeatures] = {}

    def encode(
        self,
        tracking_state: TrackingState,
        features: dict[int, EntityFeatures],
        change_level: ChangeLevel,
        scene_label: Optional[str] = None,
        timestamp_ms: float = 0.0,
        change_magnitude_avg: float = 0.0,
        change_magnitude_max: float = 0.0,
        n_regions: int = 0,
        change_area_ratio: float = 0.0,
    ) -> FrameDelta:
        """Compute deltas for all entities in current frame.

        Args:
            tracking_state: Current tracking state from IDAuthority.
            features: EntityFeatures per track_id from PerIDAnalyzer.
            change_level: Change level from ChangeDetector.
            scene_label: Optional scene classification.
            timestamp_ms: Capture timestamp in milliseconds (monotonic clock).
            change_magnitude_avg: Area-weighted mean of region change_magnitude (0-255).
                Visual surprise proxy; not a true neuroscientific prediction error.
            change_magnitude_max: Max region change_magnitude (0-255).
            n_regions: Number of changed regions in this frame.
            change_area_ratio: Sum of region area / total image pixels (0-1).

        Returns:
            FrameDelta with only changed information.
        """
        deltas: list[EntityDelta] = []

        # New entities
        for eid in tracking_state.new_ids:
            feat = features.get(eid)
            if feat is None:
                continue
            # Skip entities below minimum lifetime
            if self._min_lifetime > 0:
                entity = tracking_state.entities.get(eid)
                if entity and entity.frames_alive < self._min_lifetime:
                    continue
            delta = EntityDelta(
                track_id=eid,
                class_name=feat.attributes.get("class", "unknown"),
                changed_fields=self._all_fields(feat),
                is_new=True,
            )
            deltas.append(delta)
            self._last_reported[eid] = feat

        # Updated entities (not new)
        for eid, feat in features.items():
            if eid in tracking_state.new_ids:
                continue  # Already handled above

            prev = self._last_reported.get(eid)
            changed = self._compute_delta(feat, prev)
            if changed:
                delta = EntityDelta(
                    track_id=eid,
                    class_name=feat.attributes.get("class", "unknown"),
                    changed_fields=changed,
                )
                deltas.append(delta)
                self._last_reported[eid] = feat

        # Lost entities
        for eid in tracking_state.lost_ids:
            prev = self._last_reported.get(eid)
            # If never reported (below lifetime), silently drop
            if prev is None:
                continue
            # Check lifetime for lost entities too
            if self._min_lifetime > 0:
                entity = tracking_state.entities.get(eid)
                if entity and entity.frames_alive < self._min_lifetime:
                    self._last_reported.pop(eid, None)
                    continue
            cls_name = prev.attributes.get("class", "unknown")
            deltas.append(
                EntityDelta(
                    track_id=eid,
                    class_name=cls_name,
                    changed_fields={},
                    is_lost=True,
                )
            )
            self._last_reported.pop(eid, None)

        return FrameDelta(
            frame_id=tracking_state.frame_id,
            timestamp_ms=timestamp_ms,
            change_level=change_level,
            change_magnitude_avg=change_magnitude_avg,
            change_magnitude_max=change_magnitude_max,
            n_regions=n_regions,
            change_area_ratio=change_area_ratio,
            scene_label=scene_label,
            entity_deltas=deltas,
        )

    def to_temporal_text(self, deltas: list[FrameDelta]) -> str:
        """Format multiple FrameDeltas as a timeline for LLM consumption.

        Shows temporal progression across frames so the LLM can infer
        causal relationships (e.g. "person picked up cup and walked away").

        Skips frames with no entity deltas to reduce noise.
        Falls back to to_compact_text for single-delta lists.

        Args:
            deltas: List of FrameDeltas in chronological order.

        Returns:
            Timeline-formatted text string.
        """
        if not deltas:
            return ""
        if len(deltas) == 1:
            return self.to_compact_text(deltas[0])

        # Filter out empty deltas
        non_empty = [d for d in deltas if d.entity_deltas]
        if not non_empty:
            return ""
        if len(non_empty) == 1:
            return self.to_compact_text(non_empty[0])

        first_fid = non_empty[0].frame_id
        last_fid = non_empty[-1].frame_id
        lines: list[str] = [
            f"TIMELINE ({len(non_empty)} frames, f{first_fid}→f{last_fid}):"
        ]

        for d in non_empty:
            lines.append(f"[f{d.frame_id}] change={d.change_level.name.lower()}")
            for ed in d.entity_deltas:
                if ed.is_new:
                    lines.append(f"  {self._format_new_entity(ed)}")
                elif ed.is_lost:
                    lines.append(f"  -E{ed.track_id}[{ed.class_name}]: lost")
                else:
                    lines.append(f"  {self._format_updated_entity(ed)}")

        return "\n".join(lines)

    def to_compact_text(self, delta: FrameDelta) -> str:
        """Format FrameDelta as compact text for LLM consumption.

        Format:
            SCENE: label | change=level
            ENTITIES(N active, M new, K lost):
            +E14[person]: (200,100,380,500) stand neutral
            ~E12[person]: move(+30,+5) sit->stand happy
            -E7[person]: lost
        """
        lines: list[str] = []

        # Scene line
        scene = delta.scene_label or "unknown"
        lines.append(f"SCENE: {scene} | change={delta.change_level.name.lower()}")

        # Count entities
        new_count = sum(1 for d in delta.entity_deltas if d.is_new)
        lost_count = sum(1 for d in delta.entity_deltas if d.is_lost)
        updated = len(delta.entity_deltas) - new_count - lost_count
        lines.append(f"ENTITIES({new_count} new, {updated} updated, {lost_count} lost):")

        # Entity lines
        for ed in delta.entity_deltas:
            if ed.is_new:
                line = self._format_new_entity(ed)
            elif ed.is_lost:
                line = f"-E{ed.track_id}[{ed.class_name}]: lost"
            else:
                line = self._format_updated_entity(ed)
            lines.append(line)

        return "\n".join(lines)

    def _compute_delta(
        self, current: EntityFeatures, prev: Optional[EntityFeatures]
    ) -> dict[str, object]:
        """Compare current vs previous and return only changes."""
        if prev is None:
            return self._all_fields(current)

        changes: dict[str, object] = {}

        # BBox movement
        cx, cy = current.bbox.center
        px, py = prev.bbox.center
        dx = self._round(cx - px)
        dy = self._round(cy - py)
        if abs(dx) >= self._min_move or abs(dy) >= self._min_move:
            changes["move"] = (dx, dy)
            changes["bbox"] = self._round_bbox(current.bbox)

        # Pose change
        if current.skeleton and prev.skeleton:
            if current.skeleton.pose_label != prev.skeleton.pose_label:
                changes["pose"] = f"{prev.skeleton.pose_label}->{current.skeleton.pose_label}"
        elif current.skeleton and not prev.skeleton:
            changes["pose"] = current.skeleton.pose_label

        # Expression change
        if current.expression and prev.expression:
            if current.expression.dominant_emotion != prev.expression.dominant_emotion:
                changes["expr"] = f"{prev.expression.dominant_emotion}->{current.expression.dominant_emotion}"
        elif current.expression and not prev.expression:
            changes["expr"] = current.expression.dominant_emotion

        # Motion action change
        if current.motion and prev.motion:
            if current.motion.action_label != prev.motion.action_label:
                changes["action"] = current.motion.action_label

        return changes

    def _all_fields(self, feat: EntityFeatures) -> dict[str, object]:
        """Return all fields for a new entity."""
        fields: dict[str, object] = {
            "bbox": self._round_bbox(feat.bbox),
        }
        if feat.skeleton and feat.skeleton.pose_label:
            fields["pose"] = feat.skeleton.pose_label
        if feat.expression:
            fields["expr"] = feat.expression.dominant_emotion
        if feat.motion and feat.motion.action_label:
            fields["action"] = feat.motion.action_label
        return fields

    def _format_new_entity(self, ed: EntityDelta) -> str:
        parts = [f"+E{ed.track_id}[{ed.class_name}]:"]
        bbox = ed.changed_fields.get("bbox")
        if bbox:
            parts.append(f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})")
        if "pose" in ed.changed_fields:
            parts.append(str(ed.changed_fields["pose"]))
        if "expr" in ed.changed_fields:
            parts.append(str(ed.changed_fields["expr"]))
        return " ".join(parts)

    def _format_updated_entity(self, ed: EntityDelta) -> str:
        parts = [f"~E{ed.track_id}[{ed.class_name}]:"]
        if "move" in ed.changed_fields:
            dx, dy = ed.changed_fields["move"]
            parts.append(f"move({dx:+.0f},{dy:+.0f})")
        if "pose" in ed.changed_fields:
            parts.append(str(ed.changed_fields["pose"]))
        if "expr" in ed.changed_fields:
            parts.append(str(ed.changed_fields["expr"]))
        if "action" in ed.changed_fields:
            parts.append(str(ed.changed_fields["action"]))
        return " ".join(parts)

    def _round(self, value: float) -> float:
        return round(value / self._precision) * self._precision

    def _round_bbox(self, bbox) -> tuple[float, float, float, float]:
        return (
            self._round(bbox.x1),
            self._round(bbox.y1),
            self._round(bbox.x2),
            self._round(bbox.y2),
        )
