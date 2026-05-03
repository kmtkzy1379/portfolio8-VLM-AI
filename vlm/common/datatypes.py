"""Core data types shared across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ── Frame-level types ──


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    frame_id: int
    timestamp_ms: float
    source_width: int
    source_height: int


@dataclass(slots=True)
class CapturedFrame:
    metadata: FrameMetadata
    image: np.ndarray  # H x W x 3, BGR uint8
    phash: int = 0     # 64-bit perceptual hash


class ChangeLevel(Enum):
    NONE = auto()       # No significant change
    MINOR = auto()      # <20% change
    MODERATE = auto()   # 20-50% change, small model
    MAJOR = auto()      # >50% change, mid model


# ── Detection types ──


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: int = -1  # ultralytics tracker ID (-1 = unassigned)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)

    def iou(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class DetectionResult:
    frame_id: int
    boxes: list[BoundingBox]
    model_tier: str  # "small" or "mid"
    inference_ms: float


# ── Tracking types ──


@dataclass(slots=True)
class TrackedEntity:
    track_id: int
    class_name: str
    bbox: BoundingBox
    crop: Optional[np.ndarray] = None
    frames_alive: int = 0
    frames_since_seen: int = 0
    is_active: bool = True
    appearance_embedding: Optional[np.ndarray] = None


@dataclass(slots=True)
class TrackingState:
    frame_id: int
    entities: dict[int, TrackedEntity]
    new_ids: list[int]
    lost_ids: list[int]
    recovered_ids: list[int]


# ── Per-ID analysis types ──


@dataclass(slots=True)
class SkeletonData:
    track_id: int
    keypoints: np.ndarray  # (N, 3) = x, y, confidence
    pose_label: Optional[str] = None


@dataclass(slots=True)
class ExpressionData:
    track_id: int
    dominant_emotion: str
    emotion_scores: dict[str, float] = field(default_factory=dict)
    face_bbox: Optional[BoundingBox] = None


@dataclass(slots=True)
class MotionData:
    track_id: int
    velocity: tuple[float, float]       # px/frame
    acceleration: tuple[float, float]
    action_label: Optional[str] = None
    displacement_since_last: float = 0.0


@dataclass(slots=True)
class EntityFeatures:
    """Complete feature set for one tracked entity at one frame."""
    track_id: int
    frame_id: int
    bbox: BoundingBox
    skeleton: Optional[SkeletonData] = None
    expression: Optional[ExpressionData] = None
    motion: Optional[MotionData] = None
    attributes: dict[str, str] = field(default_factory=dict)


# ── Delta encoding types ──


@dataclass(slots=True)
class EntityDelta:
    track_id: int
    class_name: str
    changed_fields: dict[str, object]
    is_new: bool = False
    is_lost: bool = False


@dataclass(slots=True)
class FrameDelta:
    frame_id: int
    timestamp_ms: float
    change_level: ChangeLevel
    # Phase 2: Visual surprise proxy fields (NOT a neuroscientific prediction error;
    # pixel-difference based approximation only).
    change_magnitude_avg: float = 0.0   # Area-weighted mean of region change_magnitude (0-255).
    change_magnitude_max: float = 0.0   # Max region change_magnitude (0-255).
    n_regions: int = 0                  # Number of changed regions (proxy reliability indicator).
    change_area_ratio: float = 0.0      # Sum of region area / total image pixels (0-1).
    scene_label: Optional[str] = None
    entity_deltas: list[EntityDelta] = field(default_factory=list)
    # Phase 3.1: ScoredRegion list (saliency precision weighting source).
    # Untyped on purpose to avoid circular import with vlm.capture.saliency.ScoredRegion.
    scored_regions: list = field(default_factory=list)


# ── Narration pipeline types ──


@dataclass(slots=True)
class NarrationRequest:
    deltas: list[FrameDelta]
    key_crops: list[tuple[int, object]] | None
    relations_text: str
    memory_text: str
    screenshot: Optional[np.ndarray]
    frame_id: int
    is_reset: bool = False  # sentinel for scene cut


# ── Phase 3 envelope ──


@dataclass
class NarrationResult:
    """Phase 3 envelope: queue 経由で Eve 側に渡す構造化ナレーション結果。

    Phase 1 の `(narration, change_level)` 2-tuple、Phase 2 の
    `(narration, change_level, meta_dict)` 3-tuple から格上げ。
    Bridge 側は legacy tuple / dict / NarrationResult のいずれも受信可能。
    VLM 単体起動時の `_drain_narration_results` も NarrationResult を扱う。
    """
    narration: str = ""
    change_level: ChangeLevel = ChangeLevel.NONE
    meta: dict = field(default_factory=dict)
    timestamp_ms: float = 0.0
    frame_id: int = 0
    type: str = "narration"
    version: str = "v2"

    def to_legacy_tuple(self) -> tuple:
        """旧 _custom_drain (Phase 1/2) 互換形式に変換 (テスト用)。"""
        return (self.narration, self.change_level, self.meta)


# ── Token budget ──


@dataclass(slots=True)
class TokenBudget:
    max_tokens: int = 4000
    scene_context_tokens: int = 500
    per_entity_tokens: int = 200
    history_tokens: int = 1000
    image_tokens: int = 1500
