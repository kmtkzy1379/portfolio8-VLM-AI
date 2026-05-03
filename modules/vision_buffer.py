"""
VisionBuffer - 常時最新の視覚情報を保持するリングバッファ

VLMパイプラインからのナレーション結果とスクリーンショットを蓄積し、
応答AIがいつでも最新の画面状況を参照できるようにする。

Phase 2 では各 VisionFrame に構造化メタデータ (VisionMeta) を保持し、
FEP-inspired feedback layer から visual surprise proxy / SceneGraph /
WorkingMemory / FrameDelta の集約済みテキストを参照できるようにする。
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisionMeta:
    """1 ナレーション分の構造化メタデータ。VLM 側で集約済み・整形済み。

    重要な注意:
        change_magnitude_avg / max は画素差分の (面積重み付き) 平均/最大値であり、
        神経科学的な階層的・精度重みづけされた予測誤差ではない。
        これは "visual surprise proxy" と呼ぶべき工学的近似である。
        FEP-inspired メタ認知の入力として使用するが、神経科学の prediction
        error そのものとして LLM に提示してはならない。

    フィールド (v1, Phase 2):
        version: envelope schema version。"v1"=Phase 2、"v2"=Phase 3 (saliency + structured counts)。
        change_magnitude_avg: 領域変化の面積重み付き平均 (0-255)。
        change_magnitude_max: 領域変化の最大値 (0-255)。
        n_regions_max: 期間ピーク時の検出領域数 (proxy 信頼性のデバッグ指標)。
        change_area_ratio_max: 期間ピーク時の画面に対する変化面積比 (0-1)。
        relations_text: SceneGraph.to_compact_text() 結果 ("RELATIONS: ...")。
        memory_text: WorkingMemory.get_episodes_text() 結果 ("MEMORY: ...")。
        entity_delta_text: DeltaEncoder.to_temporal_text() 結果 ("TIMELINE: ...")。
        n_deltas: 集約に使った FrameDelta の本数。
        frame_id: ナレーション生成時の参照 frame_id。

    フィールド (v2, Phase 3):
        saliency_weighted_avg/max: 面積×saliency 重み付き平均/最大 (0-255).
            change_score>0 の region のみで計算 (静的 salient-only 領域は除外)。
            precision-weighted prediction error の工学的近似 (saliency-aware proxy)。
        top_salient_regions: combined_score 降順 Top5 (静的領域も含む).
            各要素 dict: bbox, saliency, change, combined, is_static。
        n_movers: change_score>0 の region 数 (proxy 信頼性デバッグ用)。
        change_level_name: "NONE"/"MINOR"/"MODERATE"/"MAJOR" (L0 計算用)。
        n_entity_new/lost/changed: 期間内 entity 出現/消失/変化件数 (L2 計算用)。
        n_relation_added/removed: 期間内 SceneGraph 関係増減 (L3 計算用)。
        n_memory_episodes: 期間内 WorkingMemory に追加された episode 数 (L3 計算用)。
    """
    version: str = "v1"
    change_magnitude_avg: float = 0.0
    change_magnitude_max: float = 0.0
    n_regions_max: int = 0
    change_area_ratio_max: float = 0.0
    relations_text: str = ""
    memory_text: str = ""
    entity_delta_text: str = ""
    n_deltas: int = 0
    frame_id: int = 0
    # === Phase 3.1 (v2) saliency-aware ===
    saliency_weighted_avg: float = 0.0
    saliency_weighted_max: float = 0.0
    top_salient_regions: list = field(default_factory=list)
    n_movers: int = 0
    # === Phase 3.1 (v2) structured counts (Phase 3.2 hierarchical surprise input) ===
    change_level_name: str = "NONE"
    n_entity_new: int = 0
    n_entity_lost: int = 0
    n_entity_changed: int = 0
    n_relation_added: int = 0
    n_relation_removed: int = 0
    n_memory_episodes: int = 0


@dataclass
class VisionFrame:
    """1フレーム分の視覚情報"""
    timestamp: float
    narration: str
    screenshot: Optional[bytes] = None  # JPEG bytes
    change_tag: str = "none"  # "major" / "moderate" / "minor" / "none"
    reason: str = ""
    meta: Optional[VisionMeta] = None  # Phase 2: 集約済みメタデータ

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class VisionBuffer:
    """スレッドセーフな視覚情報リングバッファ"""

    def __init__(self, maxlen: int = 20):
        self._buffer: deque[VisionFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, frame: VisionFrame) -> None:
        """フレームを追加"""
        with self._lock:
            self._buffer.append(frame)

    def get_scene_journal(self, n: int = 5, max_age: float = 60.0) -> str:
        """直近n件のナレーションを変化レベル付きジャーナルとして返す

        Args:
            n: 取得件数
            max_age: これより古いフレームは除外（秒）

        Returns:
            構造化テキスト（800文字上限）
        """
        now = time.time()
        with self._lock:
            recent = [
                f for f in reversed(self._buffer)
                if (now - f.timestamp) < max_age
            ][:n]

        if not recent:
            return ""

        lines = []
        for f in recent:
            age = int(now - f.timestamp)
            time_label = "たった今" if age < 10 else f"{age}秒前"
            tag = f.change_tag.upper()
            lines.append(f"[{time_label}/{tag}] {f.narration}")

        return "\n".join(lines)[:800]

    def get_recent_summary(self, n: int = 3, max_age: float = 60.0) -> str:
        """get_scene_journal の互換ラッパー"""
        return self.get_scene_journal(n=n, max_age=max_age)

    def get_latest(self) -> Optional[VisionFrame]:
        """最新のフレームを返す"""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_recent_frames(self, n: int = 5) -> list[VisionFrame]:
        """直近n件のフレームをリストで返す（新しい順）"""
        with self._lock:
            return list(reversed(self._buffer))[:n]

    def get_frames_since(self, ts_epoch: float) -> list[VisionFrame]:
        """指定epoch時刻より後のフレームを古い順で返す（FeedbackLoop用）"""
        with self._lock:
            return [f for f in self._buffer if f.timestamp > ts_epoch]

    def get_latest_screenshot(self) -> Optional[bytes]:
        """最新のスクリーンショットJPEGバイトを返す"""
        with self._lock:
            for f in reversed(self._buffer):
                if f.screenshot:
                    return f.screenshot
        return None

    def clear(self) -> None:
        """バッファをクリア"""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
