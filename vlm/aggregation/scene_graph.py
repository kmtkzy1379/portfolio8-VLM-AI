"""Scene Graph Builder - inspired by brain's spatial cognition.

The brain does not perceive objects in isolation. Spatial relationships
(above, below, inside, near) are computed automatically by the parietal
cortex. This module builds an explicit graph of entity relationships,
enabling the LLM to reason about spatial structure.

Example output for LLM:
  RELATIONS: E0 above E1 | E2 inside E3 | E0 near E2
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from vlm.common.datatypes import BoundingBox, TrackedEntity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpatialRelation:
    """A directed spatial relationship between two entities."""
    subject_id: int     # E.g. E0
    relation: str       # "above", "below", "left_of", "right_of", "inside", "contains", "overlapping", "near"
    object_id: int      # E.g. E1


class SceneGraphBuilder:
    """Builds spatial relationship graph from tracked entities.

    Computes pairwise spatial relationships between all active entities.
    Only emits relationships that changed since last call (delta mode).

    Args:
        near_threshold: Max distance (pixels) between bbox centers for "near".
        overlap_iou_threshold: Min IoU for "overlapping".
        containment_threshold: Min ratio of area inside for "inside"/"contains".
    """

    def __init__(
        self,
        near_threshold: float = 200.0,
        overlap_iou_threshold: float = 0.15,
        containment_threshold: float = 0.7,
    ):
        self._near_thresh = near_threshold
        self._overlap_iou = overlap_iou_threshold
        self._contain_thresh = containment_threshold
        self._prev_relations: set[SpatialRelation] = set()

    def build(
        self, entities: dict[int, TrackedEntity]
    ) -> list[SpatialRelation]:
        """Compute all spatial relations between active entities.

        Args:
            entities: Active entities from IDAuthority.

        Returns:
            List of SpatialRelation.
        """
        active = {
            eid: e for eid, e in entities.items() if e.is_active
        }
        ids = sorted(active.keys())
        relations: list[SpatialRelation] = []

        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                bbox_a = active[id_a].bbox
                bbox_b = active[id_b].bbox
                rels = self._compute_relations(id_a, bbox_a, id_b, bbox_b)
                relations.extend(rels)

        return relations

    def build_delta(
        self, entities: dict[int, TrackedEntity]
    ) -> tuple[list[SpatialRelation], list[SpatialRelation]]:
        """Compute relation changes since last call.

        Returns:
            (added_relations, removed_relations)
        """
        current = set(self.build(entities))
        added = list(current - self._prev_relations)
        removed = list(self._prev_relations - current)
        self._prev_relations = current
        return added, removed

    def to_compact_text(self, relations: list[SpatialRelation]) -> str:
        """Format relations as compact text for LLM.

        Example: "RELATIONS: E0 above E1 | E2 inside E3 | E0 near E2"
        """
        if not relations:
            return ""
        parts = [
            f"E{r.subject_id} {r.relation} E{r.object_id}"
            for r in relations
        ]
        return "RELATIONS: " + " | ".join(parts)

    def to_delta_text(
        self,
        added: list[SpatialRelation],
        removed: list[SpatialRelation],
    ) -> str:
        """Format relation changes as compact text.

        Example: "REL_CHANGES: +E0 above E1 | -E2 near E3"
        """
        parts = []
        for r in added:
            parts.append(f"+E{r.subject_id} {r.relation} E{r.object_id}")
        for r in removed:
            parts.append(f"-E{r.subject_id} {r.relation} E{r.object_id}")
        if not parts:
            return ""
        return "REL_CHANGES: " + " | ".join(parts)

    def reset(self) -> None:
        self._prev_relations.clear()

    def _compute_relations(
        self,
        id_a: int,
        bbox_a: BoundingBox,
        id_b: int,
        bbox_b: BoundingBox,
    ) -> list[SpatialRelation]:
        """Compute spatial relations between two entities."""
        relations = []

        # Vertical relationship (above/below)
        cy_a = (bbox_a.y1 + bbox_a.y2) / 2
        cy_b = (bbox_b.y1 + bbox_b.y2) / 2
        vertical_gap = abs(cy_a - cy_b)
        min_height = min(bbox_a.height, bbox_b.height)

        if vertical_gap > min_height * 0.3:
            if cy_a < cy_b:
                relations.append(SpatialRelation(id_a, "above", id_b))
            else:
                relations.append(SpatialRelation(id_a, "below", id_b))

        # Horizontal relationship (left_of/right_of)
        cx_a = (bbox_a.x1 + bbox_a.x2) / 2
        cx_b = (bbox_b.x1 + bbox_b.x2) / 2
        horizontal_gap = abs(cx_a - cx_b)
        min_width = min(bbox_a.width, bbox_b.width)

        if horizontal_gap > min_width * 0.3:
            if cx_a < cx_b:
                relations.append(SpatialRelation(id_a, "left_of", id_b))
            else:
                relations.append(SpatialRelation(id_a, "right_of", id_b))

        # Containment (inside/contains)
        inter_x1 = max(bbox_a.x1, bbox_b.x1)
        inter_y1 = max(bbox_a.y1, bbox_b.y1)
        inter_x2 = min(bbox_a.x2, bbox_b.x2)
        inter_y2 = min(bbox_a.y2, bbox_b.y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        if bbox_a.area > 0 and inter_area / bbox_a.area > self._contain_thresh:
            relations.append(SpatialRelation(id_a, "inside", id_b))
        elif bbox_b.area > 0 and inter_area / bbox_b.area > self._contain_thresh:
            relations.append(SpatialRelation(id_a, "contains", id_b))

        # Overlapping (significant IoU but not containment)
        iou = bbox_a.iou(bbox_b)
        if iou > self._overlap_iou and not any(
            r.relation in ("inside", "contains") for r in relations
        ):
            relations.append(SpatialRelation(id_a, "overlapping", id_b))

        # Near (centers within threshold)
        dist = math.sqrt(
            (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
        )
        if dist < self._near_thresh and iou < self._overlap_iou:
            relations.append(SpatialRelation(id_a, "near", id_b))

        return relations
