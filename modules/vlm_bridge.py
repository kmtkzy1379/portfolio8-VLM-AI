"""
VLMBridge - Eve <-> VLM Pipeline bridge module.

Wraps the VLM Pipeline to provide start/stop control and narration results
from a separate thread, suitable for integration with Eve's asyncio-based modes.

ルールベース分類: VLMパイプラインのchange_levelで分類し、
応答AIにタグ付きコンテキストを渡す。ImportanceAI不要。
"""
import difflib
import io
import logging
import queue
import threading
import time
from typing import Callable, Optional

from PIL import Image

from config import Config
from modules.vision_buffer import VisionBuffer, VisionFrame

logger = logging.getLogger(__name__)


class VLMBridge:
    """Bridge between Eve (asyncio) and VLM Pipeline (threading)."""

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path
        self._pipeline = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_narration: str = ""
        self._log_callback: Optional[Callable[[str, str], None]] = None
        self._running = False

        # VisionBuffer
        self.vision_buffer = VisionBuffer(maxlen=Config.VISION_BUFFER_SIZE)
        self._auto_push_callback: Optional[Callable[[VisionFrame], None]] = None

        # Watch mode: MODERATEもnudge対象にするフラグ
        self._watch_mode: bool = False
        self._watch_mode_expires_at: float = 0.0
        self._watch_mode_baseline_narration: str = ""  # watch mode開始時の画面状態
        self._watch_mode_nudge_count: int = 0  # watch mode内のnudge回数

        # Nudgeクールダウン: 連続nudge抑制
        self._last_nudge_at: float = 0.0
        self._nudge_cooldown: float = 15.0  # 秒（基本値）

        # ナレーション重複検出: 前回nudgeしたナレーションを記録
        self._last_nudged_narration: str = ""

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def set_log_callback(self, fn: Callable[[str, str], None]) -> None:
        self._log_callback = fn

    def set_auto_push_callback(self, fn: Callable[[VisionFrame], None]) -> None:
        """画面変化nudge時のコールバックを設定（TalkModeから呼ばれる）"""
        self._auto_push_callback = fn

    @property
    def _is_watch_mode_active(self) -> bool:
        if not self._watch_mode:
            return False
        if self._watch_mode_expires_at > 0 and time.time() > self._watch_mode_expires_at:
            self._watch_mode = False
            self._watch_mode_baseline_narration = ""
            self._watch_mode_nudge_count = 0
            self._log("VLM", "Watch mode expired, reset to normal")
            return False
        return True

    def set_watch_mode(self, enabled: bool = True, duration: float = 300.0) -> str:
        """画面変化の監視モードを切り替える。

        有効化時、現在の画面状態をベースラインとして保存する。
        以降、ベースラインと意味のある差異がある場合のみnudgeを発火する。
        同じ画面の微動（Live2Dアイドルアニメ等）では通知しない。

        Args:
            enabled: Trueで監視ON (MODERATEもnudge)、Falseで通常
            duration: 持続時間（秒）。0で無期限。

        Returns:
            設定結果の説明テキスト（tool結果として応答AIに返す）
        """
        self._watch_mode = enabled
        if not enabled:
            self._watch_mode_expires_at = 0.0
            self._watch_mode_baseline_narration = ""
            self._watch_mode_nudge_count = 0
            result = "画面監視モード: OFF (通常モード)"
        elif duration == 0:
            self._watch_mode_expires_at = float("inf")
            # 現在のナレーションをベースラインとして保存
            with self._lock:
                self._watch_mode_baseline_narration = self._latest_narration
            self._watch_mode_nudge_count = 0
            result = "画面監視モード: ON (無期限)"
        else:
            self._watch_mode_expires_at = time.time() + duration
            # 現在のナレーションをベースラインとして保存
            with self._lock:
                self._watch_mode_baseline_narration = self._latest_narration
            self._watch_mode_nudge_count = 0
            result = f"画面監視モード: ON ({duration:.0f}秒間)"

        self._log("VLM", result)
        return result

    def start(self) -> None:
        """Start VLM pipeline in a background thread."""
        if self.is_running:
            logger.warning("VLMBridge already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_pipeline, name="vlm-bridge", daemon=True
        )
        self._thread.start()
        logger.info("VLMBridge started")

    def stop(self) -> None:
        """Stop VLM pipeline gracefully."""
        if not self._running:
            return

        self._running = False
        if self._pipeline:
            self._pipeline._running = False
            self._pipeline._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("VLM thread did not exit in time")

        self._pipeline = None
        self._thread = None
        logger.info("VLMBridge stopped")

    def get_scene_description(self) -> str:
        """Get VisionBuffer scene journal for LLM context injection."""
        return self.vision_buffer.get_scene_journal(n=5)

    def get_latest_narration(self) -> str:
        """Get the latest narration result."""
        with self._lock:
            return self._latest_narration

    def get_detailed_info(self, n: int = 5) -> str:
        """Function Calling用: 詳細な視覚情報テキストを返す"""
        frames = self.vision_buffer.get_recent_frames(n=n)
        if not frames:
            return "画面情報なし（VLMが未起動またはデータなし）"

        now = time.time()
        lines = []
        for f in frames:
            age = int(now - f.timestamp)
            tag = f.change_tag.upper()
            lines.append(f"[{age}s前/{tag}] {f.narration}")
        return "\n".join(lines)

    def get_latest_screenshot_jpeg(self) -> Optional[bytes]:
        """最新のスクリーンショットJPEGバイトを返す"""
        screenshot = self.vision_buffer.get_latest_screenshot()
        if screenshot:
            return screenshot
        return self._capture_screenshot_jpeg()

    def _capture_screenshot_jpeg(self) -> Optional[bytes]:
        """Pipeline._latest_frame_imageからJPEGバイトを生成"""
        if self._pipeline is None:
            return None

        frame_image = getattr(self._pipeline, '_latest_frame_image', None)
        if frame_image is None:
            return None

        try:
            pil_img = Image.fromarray(frame_image)
            max_dim = 960
            w, h = pil_img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.LANCZOS
                )
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=60)
            return buf.getvalue()
        except Exception as e:
            logger.warning("Screenshot capture failed: %s", e)
            return None

    def _run_pipeline(self) -> None:
        """Thread target: create and run the VLM Pipeline."""
        try:
            self._log("VLM", "Loading VLM pipeline...")
            from vlm.main import Pipeline
            self._pipeline = Pipeline(self._config_path)

            self._pipeline._drain_narration_results = self._custom_drain
            self._wrap_narration_logging()

            self._log("VLM", "VLM pipeline started")
            self._pipeline.run()
        except Exception as e:
            logger.exception("VLM pipeline error")
            self._log("VLM_Error", f"VLM Error: {e}")
        finally:
            self._running = False
            self._log("VLM", "VLM pipeline stopped")

    def _wrap_narration_logging(self) -> None:
        """NarrationEngineの_call_llmをラップしてエラーをUIログに表示"""
        narration_engine = self._pipeline._narration
        original_call = narration_engine._call_llm
        bridge_log = self._log

        def wrapped_call(messages):
            result = original_call(messages)
            if result is None:
                bridge_log("VLM_Error", "Narration LLM call failed (check API key / model)")
            else:
                bridge_log("VLM", f"Narration generated ({len(result)} chars)")
            return result

        narration_engine._call_llm = wrapped_call

    def _is_narration_duplicate(self, current: str, previous: str) -> bool:
        """前回nudgeしたナレーションと内容が実質同じかどうかを判定する。

        VLMナレーションはUI説明部分が常に共通するため、
        単純な全体類似度だけでなく非一致部分の割合も考慮する。

        判定ロジック:
        1. 全体類似度 ratio <= 0.60 → 明らかに異なる (not duplicate)
        2. 全体類似度 ratio > 0.60 かつ非一致部分 >= 15% → 十分な差異あり (not duplicate)
        3. 全体類似度 ratio > 0.60 かつ非一致部分 < 15% → 実質同じ (duplicate)
        """
        if not previous:
            return False

        matcher = difflib.SequenceMatcher(None, current, previous)
        ratio = matcher.ratio()

        if ratio <= 0.60:
            return False

        # 非一致部分の割合を計算: 一致した文字数 / 長い方の文字数
        matched_chars = sum(block.size for block in matcher.get_matching_blocks())
        total_chars = max(len(current), len(previous))
        unmatched_ratio = 1.0 - (matched_chars / total_chars) if total_chars > 0 else 0

        # 全体類似度が高くても15%以上の非一致部分があれば変化ありとみなす
        if unmatched_ratio >= 0.15:
            return False

        return True

    def _classify_change(self, narration: str, change_level) -> str:
        """change_levelベースでタグを決定。Pipelineなしの場合はヒューリスティック。"""
        if change_level is not None:
            try:
                from vlm.common.datatypes import ChangeLevel
                return {
                    ChangeLevel.MAJOR: "major",
                    ChangeLevel.MODERATE: "moderate",
                    ChangeLevel.MINOR: "minor",
                    ChangeLevel.NONE: "none",
                }.get(change_level, "none")
            except ImportError:
                pass
        # Fallback: ナレーション長ベース
        if len(narration) > 200:
            return "moderate"
        return "minor"

    def _custom_drain(self) -> None:
        """Pipeline._drain_narration_results の差し替え。

        ナレーション + change_level を受け取り、ルールベースで分類。
        全フレームをVisionBufferに追加し、nudge対象ならコールバック発火。
        """
        if self._pipeline is None:
            return

        while True:
            try:
                result = self._pipeline._narration_result_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(result, tuple):
                narration, change_level = result
            else:
                narration, change_level = result, None

            with self._lock:
                self._latest_narration = narration

            tag = self._classify_change(narration, change_level)

            # MAJOR/MODERATEのみスクリーンショット取得（コスト削減）
            screenshot_jpeg = self._capture_screenshot_jpeg() if tag in ("major", "moderate") else None

            frame = VisionFrame(
                timestamp=time.time(),
                narration=narration,
                screenshot=screenshot_jpeg,
                change_tag=tag,
                reason=f"change_level={change_level}",
            )
            self.vision_buffer.add(frame)
            self._log("VLM", f"[{tag}] {narration}")

            # 「変化なし」を含むナレーションはnudge対象外
            has_change = "変化なし" not in narration

            # Nudge判定: MAJOR常時 or MODERATE(watch_mode時)、かつ実際に変化あり
            is_watch_active = self._is_watch_mode_active
            should_nudge = has_change and (
                tag == "major"
                or (tag == "moderate" and is_watch_active)
            )

            # Watch modeベースライン比較: ベースラインと類似なら抑制
            is_baseline_dup = False
            if should_nudge and tag != "major" and is_watch_active and self._watch_mode_baseline_narration:
                if self._is_narration_duplicate(narration, self._watch_mode_baseline_narration):
                    should_nudge = False
                    is_baseline_dup = True
                    self._log("VLM", "[baseline] similar to watch baseline, suppressed")

            # ナレーション重複検出: 前回nudgeと類似なら抑制（MAJORはバイパス）
            is_dedup = False
            if should_nudge and tag != "major" and self._last_nudged_narration:
                if self._is_narration_duplicate(narration, self._last_nudged_narration):
                    should_nudge = False
                    is_dedup = True
                    self._log("VLM", "[dedup] narration too similar to last nudge, suppressed")

            print(f"[DEBUG] drain: tag={tag}, watch={self._watch_mode}, should_nudge={should_nudge}, "
                  f"has_change={has_change}, dedup={is_dedup}, baseline_dup={is_baseline_dup}")

            if should_nudge and self._auto_push_callback:
                # プログレッシブクールダウン: watch mode内のnudge回数に応じて延長
                if is_watch_active and self._watch_mode_nudge_count > 0:
                    effective_cooldown = min(
                        self._nudge_cooldown * (2 ** self._watch_mode_nudge_count),
                        120.0  # 最大2分
                    )
                else:
                    effective_cooldown = self._nudge_cooldown

                now = time.time()
                if now - self._last_nudge_at >= effective_cooldown:
                    self._last_nudge_at = now
                    self._last_nudged_narration = narration
                    # Watch mode: nudge成功後にベースラインを更新（次の変化を検出するため）
                    if is_watch_active:
                        self._watch_mode_baseline_narration = narration
                        self._watch_mode_nudge_count += 1
                    self._auto_push_callback(frame)
                else:
                    self._log("VLM", f"[cooldown] nudge skipped ({tag}, {now - self._last_nudge_at:.1f}s/{effective_cooldown:.0f}s)")

    def _log(self, role: str, message: str) -> None:
        """Send log to callback if set."""
        if self._log_callback:
            try:
                self._log_callback(role, message)
            except Exception:
                pass
