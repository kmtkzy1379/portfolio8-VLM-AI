"""
Hierarchical surprise computation for FEP-inspired metacognition (Phase 3.2).

NOTE: This is a *bottom-up* observation of surprise across pre-defined levels.
Real predictive coding involves *bidirectional* inference (top-down predictions
+ bottom-up errors). Our implementation observes only the bottom-up half.
We deliberately label these as "level-N surprise observations" rather than
"prediction errors" to maintain conceptual honesty.

Inputs are STRUCTURED COUNTS from VisionMeta (NOT parsed from text). This
makes the implementation maintainable across narration prompt format changes.

Levels:
- L0: Global change (ChangeLevel categorical)
- L1: Pixel-level visual surprise proxy (saliency-weighted, normalized)
- L2: Entity-level surprise (new/lost/changed counts)
- L3: Relation/episode-level surprise (SceneGraph deltas + WorkingMemory episodes)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from config import Config

_CHANGE_LEVEL_TO_L0 = {
    "NONE": 0.0,
    "MINOR": 0.2,
    "MODERATE": 0.5,
    "MAJOR": 1.0,
}


@dataclass
class HierarchicalSurprise:
    """期間内の階層別 surprise observation 集計結果。"""
    l0_global: float = 0.0       # 0.0-1.0
    l1_pixel: float = 0.0        # 0.0-1.0 (saliency-weighted normalized)
    l2_entity: float = 0.0       # 0.0-1.0
    l3_relation: float = 0.0     # 0.0-1.0
    unified: float = 0.0         # 0.4*L1 + 0.3*L2 + 0.2*L3 + 0.1*L0

    n_entity_new: int = 0
    n_entity_lost: int = 0
    n_entity_changed: int = 0
    n_relation_delta: int = 0
    n_episodes: int = 0
    n_frames: int = 0


def compute_hierarchical_surprise(
    frames: list,
    weights: Optional[tuple[float, float, float, float]] = None,
) -> HierarchicalSurprise:
    """期間内 VisionFrame list から structured counts ベースで階層 surprise を集約。

    各 frame.meta は VisionMeta v2 で structured counts を持っている前提。
    v1 (Phase 2) や meta=None の frame は default 値 (0) として扱う (後方互換)。

    Args:
        frames: 期間内 VisionFrame のリスト (古い順)。
        weights: (L0, L1, L2, L3) の unified 計算用重み。None なら Config の baseline を使う。
            Phase 4.2: PrecisionState.weights を受け取って動的化可能。

    Returns:
        HierarchicalSurprise 集計結果。空入力でも default 値で返す。
    """
    if weights is None:
        weights = (
            Config.HS_WEIGHT_L0,
            Config.HS_WEIGHT_L1,
            Config.HS_WEIGHT_L2,
            Config.HS_WEIGHT_L3,
        )
    if not frames:
        return HierarchicalSurprise()

    # L0: ChangeLevel の期間最大 (期間中最も激しい変化)
    l0_values = []
    for f in frames:
        if f.meta and f.meta.change_level_name in _CHANGE_LEVEL_TO_L0:
            l0_values.append(_CHANGE_LEVEL_TO_L0[f.meta.change_level_name])
    l0_global = max(l0_values) if l0_values else 0.0

    # L1: saliency-weighted の期間平均 (0-255 -> 0-1)
    sw_avgs = [
        f.meta.saliency_weighted_avg
        for f in frames
        if f.meta and f.meta.saliency_weighted_avg > 0
    ]
    l1_pixel = (sum(sw_avgs) / len(sw_avgs) / 255.0) if sw_avgs else 0.0
    # 範囲 0-1 にクリップ (理論上 0-255 なので除算後 0-1)
    if l1_pixel > 1.0:
        l1_pixel = 1.0

    # L2: entity 出現/消失/変化を統合
    n_new = sum((f.meta.n_entity_new if f.meta else 0) for f in frames)
    n_lost = sum((f.meta.n_entity_lost if f.meta else 0) for f in frames)
    n_chg = sum((f.meta.n_entity_changed if f.meta else 0) for f in frames)
    # 期間内合計を 0-1 にスケール (出現は重大、消失次点、変化は軽め)
    weighted_count = (n_new * 1.0) + (n_lost * 0.9) + (n_chg * 0.4)
    l2_entity = min(weighted_count / 10.0, 1.0)

    # L3: relation delta + episode
    n_rel = sum(
        ((f.meta.n_relation_added + f.meta.n_relation_removed) if f.meta else 0)
        for f in frames
    )
    n_eps = sum((f.meta.n_memory_episodes if f.meta else 0) for f in frames)
    l3_relation = min(n_rel * 0.15 + n_eps * 0.25, 1.0)

    # Unified (Phase 4.4 統合: 重みは Config baseline または PrecisionState から動的)
    w0, w1, w2, w3 = weights
    unified = w1 * l1_pixel + w2 * l2_entity + w3 * l3_relation + w0 * l0_global

    return HierarchicalSurprise(
        l0_global=l0_global,
        l1_pixel=l1_pixel,
        l2_entity=l2_entity,
        l3_relation=l3_relation,
        unified=unified,
        n_entity_new=n_new,
        n_entity_lost=n_lost,
        n_entity_changed=n_chg,
        n_relation_delta=n_rel,
        n_episodes=n_eps,
        n_frames=len(frames),
    )
