"""
共通基底クラス - 全モードで共有する処理を提供
"""
import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, Optional
from colorama import Fore

from config import Config
from modules import (
    LLMHandler, TTSHandler, AudioPlayer,
    FeedbackHandler, RAGHandler, ConversationCache
)


class ResponseStage(Enum):
    """応答パイプラインの進行ステージ。

    入力2 到着時のマージ判断や is_speaking() 判定に使う。
    STT は独立 AudioListener に分離されたためステージから除外。
    """
    IDLE = auto()
    LLM_PENDING = auto()
    LLM_STREAMING = auto()
    TTS_QUEUED = auto()
    TTS_PLAYING = auto()


class BaseMode(ABC):
    """全モードの基底クラス"""

    def __init__(self, system_prompt: str, log_callback: Optional[Callable[[str, str, str], None]] = None):
        """
        Args:
            system_prompt: このモード用のシステムプロンプト
            log_callback: UIにログを送るコールバック関数 (role, message, level) -> None
                          level は "info" | "debug"。debug は UI 側で Show debug 時のみ表示される。
        """
        self.system_prompt = system_prompt
        self.log_callback = log_callback
        self.running = False
        self.stop_requested = False
        self.is_processing = False

        # 共通コンポーネント（初期化はinitialize()で行う）
        self.player: Optional[AudioPlayer] = None
        self.llm: Optional[LLMHandler] = None
        self.tts: Optional[TTSHandler] = None
        self.rag: Optional[RAGHandler] = None
        self.feedback: Optional[FeedbackHandler] = None
        self.conversation_cache: Optional[ConversationCache] = None

        # VLM bridge (optional)
        self.vlm_bridge = None

        # asyncio event loop reference (set in initialize(), used for thread-safe queue ops)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 内部タスク
        self._player_task: Optional[asyncio.Task] = None
        self._feedback_task: Optional[asyncio.Task] = None

        # Step 1: 応答パイプラインのステージ管理
        # - _stage: 現在の応答ステージ (Step 1 では観測のみ、Step 4 で is_speaking() に使う)
        # - _current_response_task: process_input の Task 化 (Step 2 で実装)
        # - _response_lock: 同時応答実行を排除 (Step 2)
        # - _pending_input_queue: AudioListener が STT 結果を入れるキュー
        # - _pending_lock / _pending_input: マージ判断用 (Step 4)
        self._stage: ResponseStage = ResponseStage.IDLE
        self._current_response_task: Optional[asyncio.Task] = None
        self._response_lock: asyncio.Lock = asyncio.Lock()
        self._pending_input_queue: asyncio.Queue = asyncio.Queue()
        self._pending_lock: asyncio.Lock = asyncio.Lock()
        self._pending_input: Optional[str] = None

    async def initialize(self):
        """共通コンポーネントの初期化"""
        self._loop = asyncio.get_running_loop()
        self.log("System", "Initializing components...", level="debug")

        # コンポーネント初期化
        self.player = AudioPlayer()
        # Step 2: AudioPlayer にメインループを渡す（別スレッドからの interrupt() 用）
        self.player.set_loop(self._loop)
        self.llm = LLMHandler()
        self.tts = TTSHandler()
        self.rag = RAGHandler()
        self.conversation_cache = ConversationCache(Config.HISTORY_FILE)

        # システムプロンプトを設定
        self.llm.system_prompt = self.system_prompt
        self.llm.history = [{"role": "system", "content": self.system_prompt}]

        # AI2フィードバック用のコールバック設定
        def update_memory_func(memory_text):
            self.llm.ai2_feedback = memory_text[:Config.FB_LOOP_MAX_CHARS]
            self.log("System", f"AI2 Feedback Updated ({len(memory_text)} chars)", level="debug")
        self.llm.update_memory = update_memory_func
        self.llm.rag = self.rag

        # FeedbackHandler初期化 (ループ型FEPメタ認知)
        self.feedback = FeedbackHandler(self.llm, conversation_cache=self.conversation_cache)
        # Phase 4.3.1: feedback ループから episode を RAG へ保存できるよう注入
        self.feedback.set_rag(self.rag)

        # RAGと会話キャッシュを初期化
        await self.rag.initialize()
        await self.conversation_cache.initialize()

        # バックグラウンドタスク開始
        self._player_task = asyncio.create_task(self.player.play_worker())
        self._feedback_task = asyncio.create_task(self.feedback.start_loop())

        self.running = True

        # Race-safe: set_vlm_bridge() が initialize() 中に呼ばれて
        # self.llm が None だった場合の自動再適用。
        self._apply_pending_vlm_bridge()

        self.log("System", "Initialization complete")

    async def shutdown(self):
        """終了処理"""
        self.log("System", "Shutting down...", level="debug")
        self.stop_requested = True
        self.running = False

        # 処理中の場合は完了を待つ
        while self.is_processing:
            await asyncio.sleep(0.1)

        # 音声再生完了を待つ
        if self.player:
            while self.player.is_playing or not self.player.queue.empty():
                await asyncio.sleep(0.1)

        # バックグラウンドタスクをキャンセル
        if self._player_task:
            self._player_task.cancel()
            try:
                await self._player_task
            except asyncio.CancelledError:
                pass

        # フィードバックループは停止要求を先に出し、進行中の LLM 呼出を中断しない
        if self.feedback:
            try:
                await self.feedback.stop()
            except Exception:
                pass

        if self._feedback_task:
            self._feedback_task.cancel()
            try:
                await self._feedback_task
            except asyncio.CancelledError:
                pass

        # リソース解放
        if self.rag:
            await self.rag.shutdown()
        if self.conversation_cache:
            await self.conversation_cache.shutdown()

        self.log("System", "Shutdown complete")

    async def process_input(self, input_text: str) -> str:
        """
        テキスト入力を処理してAI応答を生成（外部 API は完全互換）。

        Step 2: 内部で cancel 可能な Task を保持しつつ、必ず await で str を返す。
        - Task 即 return は禁止（game_mode/youtube_mode の `await self.process_input(...)` を維持）
        - `_response_lock` で同時応答実行を排除（length-based history rollback の安全性）
        - 直前の task が生存中なら cancel + 完了待ちしてから新しい応答を開始

        Args:
            input_text: 入力テキスト

        Returns:
            AI応答テキスト（cancel 時は空文字列）
        """
        if self.stop_requested or not self.running:
            return ""

        # 直前の task が生存中なら cancel + 完了待ち（_response_lock を取る前）
        await self._cancel_current_response_if_any()

        # 単一応答保証: 同時に複数の _run_response_pipeline が走らないようにする
        async with self._response_lock:
            self._current_response_task = asyncio.create_task(
                self._run_response_pipeline(input_text)
            )
            try:
                # 必ず await で完了待ち（Task 即 return 禁止）
                return await self._current_response_task
            except asyncio.CancelledError:
                # 外部 cancel 伝播時は空応答返却（呼び出し側コードを壊さない）
                return ""
            finally:
                self._current_response_task = None

    async def _cancel_current_response_if_any(self) -> None:
        """直前の応答 task が生存中なら cancel + 完了待ち（タイムアウト 2.0 秒）。"""
        task = self._current_response_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            self.log("System", "Cancel timeout - forcing continue", level="debug")
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _run_response_pipeline(self, input_text: str) -> str:
        """応答パイプライン本体（Task 化されて cancel 可能）。

        - 入口で `history_len_before = len(self.llm.history)` を snapshot
        - cancel 時は length-based restore で history を完全 rollback（user/assistant/tool すべて削除）
        - cancel 時に TTS 完了レース対策で `player.queue` を drain
        - finally で `_stage = IDLE` に戻す
        """
        if self.stop_requested or not self.running:
            return ""

        self.is_processing = True
        self._stage = ResponseStage.LLM_PENDING
        history_len_before = len(self.llm.history)
        pending_tts_tasks: list[asyncio.Task] = []
        start_time = time.time()

        try:
            self.log("User", input_text)

            # Step 3: ConversationCache への記録は正常完了時のみ atomic に行う
            # （cancel 時に user だけ残る穴を防ぐため、ここでは add_user_message しない）

            # RAG検索（800msタイムアウト）
            rag_memories = []
            try:
                # Phase 4.3.2: top_k は Config.FB_RAG_TOP_K (default 4) を使う
                # MMR による多様性確保が効くため top_k 拡大しても重複しない
                rag_task = asyncio.create_task(self.rag.search_similar(input_text))
                rag_memories = await asyncio.wait_for(rag_task, timeout=1.5)
                if rag_memories:
                    self.log("System", f"RAG found: {len(rag_memories)} memories", level="debug")
            except asyncio.TimeoutError:
                self.log("System", "RAG timeout - proceeding without memories")
            except Exception as e:
                self.log("System", f"RAG Error: {e}")

            # RAGと直近ターンをLLMに設定
            # exclude_ellipsis=True: 内部見守り発話「…」を recent_turns から除外
            #   (沈黙時に直前の実会話が押し出される問題への対策)
            self.llm.rag_memories = rag_memories
            recent_turns = await self.conversation_cache.get_recent_turns(
                count=5, exclude_ellipsis=True,
            )
            self.llm.recent_turns = recent_turns
            # 沈黙サマリ (「…」除外で失われた情報を別経路で AI1 に渡す)
            try:
                self.llm.silence_summary = (
                    await self.conversation_cache.get_silence_summary()
                )
            except Exception as e:
                self.log("System", f"silence_summary failed: {e}")
                self.llm.silence_summary = None

            # VLM screen recognition context
            if self.vlm_bridge and self.vlm_bridge.is_running:
                vlm_desc = self.vlm_bridge.get_scene_description()
                self.llm.vlm_context = vlm_desc if vlm_desc else ""
            else:
                self.llm.vlm_context = ""

            # LLM応答生成
            ai_response = ""
            sentence_queue = []

            async def process_tts_batch():
                if not sentence_queue:
                    return
                batch = sentence_queue[:3]
                sentence_queue[:] = sentence_queue[3:]
                tasks = [self.tts.generate_audio(s) for s in batch]
                wavs = await asyncio.gather(*tasks, return_exceptions=True)
                for wav in wavs:
                    if wav and not isinstance(wav, Exception):
                        self.player.add_to_queue(wav)
                # TTS が player.queue に積まれたら TTS_QUEUED へ遷移
                if self._stage == ResponseStage.LLM_STREAMING and self.player.queue.qsize() > 0:
                    self._stage = ResponseStage.TTS_QUEUED

            self._stage = ResponseStage.LLM_STREAMING
            async for sentence in self.llm.generate_stream(input_text):
                if self.stop_requested or self.player.interrupt_signal:
                    break
                ai_response += sentence
                if sentence.strip():
                    sentence_queue.append(sentence)
                    if len(sentence_queue) >= 3:
                        await process_tts_batch()

            # 残りのセンテンスを処理
            while sentence_queue:
                await process_tts_batch()

            # 応答をログに出力
            if ai_response:
                self.log("AI", ai_response)
                # Step 3: 正常完了時のみ atomic に1ターン記録
                # cancel 時はここに到達しないため user/ai どちらも記録されない
                await self.conversation_cache.add_turn(input_text, ai_response)
                # Phase 4.3.1: 生ターン保存は撤去。
                # RAG 入力は feedback ループが Episode Summary 経由で行う (rag.add_episode)。
                # conversation_history.txt は別途 cache 経由で監査用に蓄積される。

            # フィードバックループに「ターン完了」シグナルを送る (Event.set のみ、同期・即時)
            if self.feedback:
                self.feedback.signal_turn_done()

            total_time = time.time() - start_time
            self.log("System", f"Response time: {total_time*1000:.1f}ms", level="debug")

            return ai_response

        except asyncio.CancelledError:
            # cleanup: pending TTS task を確実にキャンセル
            for t in pending_tts_tasks:
                if not t.done():
                    t.cancel()
            if pending_tts_tasks:
                await asyncio.gather(*pending_tts_tasks, return_exceptions=True)

            # TTS 完了レース対策: cancel 直前に generate_audio が完了して
            # player.queue.put_nowait(wav) するレースに備え、queue を破棄
            while not self.player.queue.empty():
                try:
                    self.player.queue.get_nowait()
                    self.player.queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # history を完全 restore (user/assistant/tool すべて削除)
            self.llm.history = self.llm.history[:history_len_before]
            self.log("System", "[Cancelled] LLM stream aborted, history rolled back", level="debug")
            # 部分応答は記録しない（add_ai_response 未呼び出しなので OK）
            raise

        finally:
            self.is_processing = False
            self._stage = ResponseStage.IDLE
            self.on_response_complete()

    def log(self, role: str, message: str, level: str = "info"):
        """ログ出力

        Args:
            role: User / AI / System / Comment / Game / Inject など
            message: ログ本文
            level: "info" (普段表示) または "debug" (Show debug 時のみ UI 表示)
                   コンソール出力には level の影響なし（常に出る）。
        """
        # コンソール出力（debug レベルもコンソールには出す）
        if role == "User":
            print(Fore.WHITE + f"\n{role}: {message}")
        elif role == "AI":
            print(Fore.CYAN + f"{role}: {message}")
        else:
            print(Fore.YELLOW + f"[{role}] {message}")

        # UIコールバック
        if self.log_callback:
            self.log_callback(role, message, level)

    def on_response_complete(self):
        """応答完了時のコールバック（サブクラスでオーバーライド可能）"""
        pass

    def set_vlm_bridge(self, bridge) -> None:
        """VLMBridgeを接続/解除する。

        Race-safe: self.llm がまだ None (initialize() 進行中) でも壊れない。
        その場合は self.vlm_bridge にだけ保存され、initialize() 末尾で
        再適用される。bridge=None も明示的に伝搬し、LLM 側の tools を
        正しく無効化する (停止時のリーク防止)。

        差し替え時は古い bridge の auto-push callback を必ず解除する。
        """
        import logging
        _logger = logging.getLogger(__name__)

        # 古い bridge の callback を解除 (異なる bridge への差し替え or None 化時)
        old_bridge = self.vlm_bridge
        if old_bridge is not None and old_bridge is not bridge:
            try:
                old_bridge.set_auto_push_callback(None)
            except Exception as e:
                _logger.warning("old bridge clear callback failed: %s", e)

        self.vlm_bridge = bridge

        # 新しい bridge に callback 登録
        if bridge is not None:
            try:
                bridge.set_auto_push_callback(self._on_vision_auto_push)
            except Exception as e:
                _logger.warning("set_auto_push_callback failed: %s", e)

        # LLM 側に伝搬。bridge=None でも set_vision_components(None) を呼ぶ。
        if self.llm is not None and hasattr(self.llm, 'set_vision_components'):
            try:
                self.llm.set_vision_components(bridge)
            except Exception as e:
                _logger.warning("set_vision_components failed: %s", e)
        else:
            # initialize() 完了前に呼ばれた場合: self.vlm_bridge に残しておき、
            # initialize() 末尾で _apply_pending_vlm_bridge() が再適用する。
            _logger.info("set_vlm_bridge: llm not ready yet, will reapply after initialize()")

        # フィードバックループにも同じ bridge を共有 (VLM フレームを取り込むため)
        if self.feedback is not None:
            self.feedback.set_vlm_bridge(bridge)

    def _apply_pending_vlm_bridge(self) -> None:
        """initialize() 末尾で呼び、set_vlm_bridge() が初期化中に空振りした場合の
        再適用を行う。すでに self.vlm_bridge が None なら何もしない。
        """
        if self.vlm_bridge is None:
            return
        # 自分自身を再度通すことで llm/feedback 両方に確実に伝搬する。
        # 既存 callback の再登録は VLMBridge 側が単純上書きなので副作用なし。
        self.set_vlm_bridge(self.vlm_bridge)

    def _on_vision_auto_push(self, vision_frame) -> None:
        """VLMBridge auto-push callback (called from VLM thread).

        Step 5: 旧経路（_vision_push_queue 経由の独立 process_input）を廃止し、
        VLM スレッドから loop.call_soon_threadsafe でメインループ上の
        `_on_vision_alert_main` を呼ぶ。alert は llm._vlm_alerts に蓄積され、
        次のユーザー応答 or idle 自発 nudge で自然に言及される。
        """
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._on_vision_alert_main, vision_frame)
        except RuntimeError:
            # Loop already closed during shutdown
            pass

    def _on_vision_alert_main(self, vision_frame) -> None:
        """VLM alert をメインループ内で処理する（Step 5）。

        - llm._vlm_alerts に蓄積（独立 process_input は走らない）
        - feedback.signal_vlm_event() で feedback ループに通知（既存挙動維持）
        """
        try:
            if self.llm is not None:
                self.llm.append_vlm_alert(vision_frame)
            if self.feedback is not None:
                self.feedback.signal_vlm_event()
            narration = getattr(vision_frame, "narration", "") or ""
            self.log(
                "System",
                f"Vision alert queued: {narration[:60]}...",
                level="debug",
            )
        except Exception as e:
            self.log("System", f"_on_vision_alert_main error: {e}", level="debug")

    def interrupt(self):
        """音声再生を中断"""
        if self.player:
            self.player.interrupt()

    def is_speaking(self) -> bool:
        """Eve が発話中（LLM 生成・TTS 生成・TTS 再生のいずれか）かを総合判定。

        Step 4: process_input は再生完了を待たずに返るため、_stage が IDLE に
        戻った後も player.is_playing == True の状態が確実に存在する。
        Dispatcher のマージ判断にも使う。
        """
        if self._stage in (
            ResponseStage.LLM_PENDING,
            ResponseStage.LLM_STREAMING,
            ResponseStage.TTS_QUEUED,
            ResponseStage.TTS_PLAYING,
        ):
            return True
        if self.player and (self.player.is_playing or not self.player.queue.empty()):
            return True
        return False

    @abstractmethod
    async def run(self):
        """モード固有のメインループ（サブクラスで実装）"""
        pass

    def is_safe_to_switch(self) -> bool:
        """モード切替が安全かどうかを判定"""
        if self.is_processing:
            return False
        if self.player and (self.player.is_playing or not self.player.queue.empty()):
            return False
        return True
