"""
共通基底クラス - 全モードで共有する処理を提供
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional
from colorama import Fore

from config import Config
from modules import (
    LLMHandler, TTSHandler, AudioPlayer,
    FeedbackHandler, RAGHandler, ConversationCache
)


class BaseMode(ABC):
    """全モードの基底クラス"""

    def __init__(self, system_prompt: str, log_callback: Optional[Callable[[str, str], None]] = None):
        """
        Args:
            system_prompt: このモード用のシステムプロンプト
            log_callback: UIにログを送るコールバック関数 (role, message) -> None
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

        # Vision auto-push queue (Phase 2)
        self._vision_push_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

        # asyncio event loop reference (set in initialize(), used for thread-safe queue ops)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 内部タスク
        self._player_task: Optional[asyncio.Task] = None
        self._feedback_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """共通コンポーネントの初期化"""
        self._loop = asyncio.get_running_loop()
        self.log("System", "Initializing components...")

        # コンポーネント初期化
        self.player = AudioPlayer()
        self.llm = LLMHandler()
        self.tts = TTSHandler()
        self.rag = RAGHandler()
        self.conversation_cache = ConversationCache(Config.HISTORY_FILE)

        # システムプロンプトを設定
        self.llm.system_prompt = self.system_prompt
        self.llm.history = [{"role": "system", "content": self.system_prompt}]

        # AI2フィードバック用のコールバック設定
        def update_memory_func(memory_text):
            self.llm.ai2_feedback = memory_text[:500]
            self.log("System", "AI2 Feedback Updated")
        self.llm.update_memory = update_memory_func
        self.llm.rag = self.rag

        # FeedbackHandler初期化
        self.feedback = FeedbackHandler(self.llm)

        # RAGと会話キャッシュを初期化
        await self.rag.initialize()
        await self.conversation_cache.initialize()

        # バックグラウンドタスク開始
        self._player_task = asyncio.create_task(self.player.play_worker())
        self._feedback_task = asyncio.create_task(self.feedback.start_loop())

        self.running = True
        self.log("System", "Initialization complete")

    async def shutdown(self):
        """終了処理"""
        self.log("System", "Shutting down...")
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
        テキスト入力を処理してAI応答を生成

        Args:
            input_text: 入力テキスト

        Returns:
            AI応答テキスト
        """
        if self.stop_requested or not self.running:
            return ""

        self.is_processing = True
        start_time = time.time()

        try:
            self.log("User", input_text)

            # 会話キャッシュに追加
            await self.conversation_cache.add_user_message(input_text)

            # RAG検索（800msタイムアウト）
            rag_memories = []
            try:
                rag_task = asyncio.create_task(self.rag.search_similar(input_text, top_k=2))
                rag_memories = await asyncio.wait_for(rag_task, timeout=1.5)
                if rag_memories:
                    self.log("System", f"RAG found: {len(rag_memories)} memories")
            except asyncio.TimeoutError:
                self.log("System", "RAG timeout - proceeding without memories")
            except Exception as e:
                self.log("System", f"RAG Error: {e}")

            # RAGと直近ターンをLLMに設定
            self.llm.rag_memories = rag_memories
            recent_turns = await self.conversation_cache.get_recent_turns(count=5)
            self.llm.recent_turns = recent_turns

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
                await self.conversation_cache.add_ai_response(ai_response)
                await self.rag.add_turn(input_text, ai_response)

            # フィードバック処理をトリガー
            asyncio.create_task(self.feedback.process_pending())

            total_time = time.time() - start_time
            self.log("System", f"Response time: {total_time*1000:.1f}ms")

            return ai_response

        finally:
            self.is_processing = False
            self.on_response_complete()

    def log(self, role: str, message: str):
        """ログ出力"""
        # コンソール出力
        if role == "User":
            print(Fore.WHITE + f"\n{role}: {message}")
        elif role == "AI":
            print(Fore.CYAN + f"{role}: {message}")
        else:
            print(Fore.YELLOW + f"[{role}] {message}")

        # UIコールバック
        if self.log_callback:
            self.log_callback(role, message)

    def on_response_complete(self):
        """応答完了時のコールバック（サブクラスでオーバーライド可能）"""
        pass

    def set_vlm_bridge(self, bridge) -> None:
        """VLMBridgeを接続"""
        self.vlm_bridge = bridge

        # Auto-push callback登録
        if bridge is not None:
            bridge.set_auto_push_callback(self._on_vision_auto_push)

            # LLMにvision componentsを設定（Phase 3）
            if hasattr(self.llm, 'set_vision_components'):
                self.llm.set_vision_components(bridge)

    def _on_vision_auto_push(self, vision_frame) -> None:
        """VLMBridge auto-push callback (called from VLM thread).

        Uses call_soon_threadsafe to safely enqueue from a non-asyncio thread.

        Args:
            vision_frame: VisionFrame with importance >= threshold
        """
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue_vision_frame, vision_frame)
        except RuntimeError:
            # Loop already closed during shutdown
            pass

    def _enqueue_vision_frame(self, vision_frame) -> None:
        """Enqueue a vision frame from the event loop thread (called via call_soon_threadsafe)."""
        try:
            self._vision_push_queue.put_nowait(vision_frame)
        except asyncio.QueueFull:
            pass  # Drop if queue is full

    async def _handle_vision_push(self, vision_frame) -> None:
        """VLMからのnudge（MAJOR変化 or watch_mode時MODERATE）を処理。

        vlm_contextにアラートを注入し、通常のprocess_inputで応答を生成。
        """
        alert = f"★[SCENE/{vision_frame.change_tag.upper()}] {vision_frame.narration}"

        if self.llm:
            existing = self.llm.vlm_context or ""
            self.llm.vlm_context = f"{alert}\n{existing}"[:800]

        self.log("System", f"Vision nudge: {alert[:100]}...")

        await self.process_input(
            f"[内部: 画面変化通知 - 前回と本当に違う場合のみリアクション] {vision_frame.narration[:200]}"
        )

        # 応答完了後、キューに溜まった残りフレームを破棄（1変化=1応答）
        drained = 0
        while not self._vision_push_queue.empty():
            try:
                self._vision_push_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            self.log("System", f"Vision queue drained: {drained} frames skipped")

    def interrupt(self):
        """音声再生を中断"""
        if self.player:
            self.player.interrupt()

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
