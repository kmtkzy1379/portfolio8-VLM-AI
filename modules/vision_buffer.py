"""
VisionBuffer - 常時最新の視覚情報を保持するリングバッファ

VLMパイプラインからのナレーション結果とスクリーンショットを蓄積し、
応答AIがいつでも最新の画面状況を参照できるようにする。
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisionFrame:
    """1フレーム分の視覚情報"""
    timestamp: float
    narration: str
    screenshot: Optional[bytes] = None  # JPEG bytes
    change_tag: str = "none"  # "major" / "moderate" / "minor" / "none"
    reason: str = ""

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
