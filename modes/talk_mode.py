"""
対話モード - マイク入力によるリアルタイム対話
"""
import asyncio
import time
from typing import Callable, Optional

from modules import AudioInput, STTHandler
from prompts import TALK_SYSTEM_PROMPT
from .base_mode import BaseMode


class TalkMode(BaseMode):
    """対話モード：マイク入力 + 無言処理"""
    
    def __init__(self, log_callback: Optional[Callable[[str, str], None]] = None):
        super().__init__(TALK_SYSTEM_PROMPT, log_callback)
        self.stt: Optional[STTHandler] = None
        self.mic: Optional[AudioInput] = None
        
        # 無言処理用ステート
        self.state = {
            "last_user_event_ts": time.time(),
            "idle_since_ts": time.time(),
            "last_ellipsis_ts": 0.0,
            "busy_stt": False,
            "busy_llm": False,
        }
        
        self._idle_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """対話モード固有の初期化"""
        await super().initialize()

        # STT初期化
        self.log("System", "DEBUG: Creating STTHandler...")
        self.stt = STTHandler()

        # マイク初期化（音声開始時のコールバック設定）
        def on_speech_start():
            self.interrupt()
            self.state["last_user_event_ts"] = time.time()

        self.log("System", "DEBUG: Creating AudioInput...")
        self.mic = AudioInput(callback_on_speech_start=on_speech_start)

        self.log("System", "DEBUG: Starting mic stream...")
        self.mic.start()

        # 無言処理ループ開始
        self._idle_task = asyncio.create_task(self._idle_ellipsis_loop())

        self.log("System", "Talk Mode Ready - Speak now!")
    
    async def shutdown(self):
        """対話モード固有の終了処理"""
        # 無言処理ループを停止
        if self._idle_task:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        
        # マイクを停止
        if self.mic:
            self.mic.stop()
        
        await super().shutdown()
    
    async def _idle_ellipsis_loop(self):
        """無言処理ループ：5秒間アイドル状態が続いたら「…」を送信"""
        while self.running and not self.stop_requested:
            await asyncio.sleep(0.3)
            now = time.time()
            
            # 処理中はアイドル時刻をリセット
            if self.state["busy_stt"] or self.state["busy_llm"] or self.is_processing:
                self.state["idle_since_ts"] = now
                continue
            
            # 再生中はアイドル時刻をリセット
            if self.player and (self.player.is_playing or not self.player.queue.empty()):
                self.state["idle_since_ts"] = now
                continue
            
            # 8秒間アイドルが続いたら「…」を送信
            if now - self.state["idle_since_ts"] < 8:
                continue
            if now - self.state["last_ellipsis_ts"] < 8:
                continue
            
            self.state["last_ellipsis_ts"] = now
            await self._process_ellipsis()
    
    async def _process_ellipsis(self):
        """無言時の「…」処理 - VLM情報を活用した自然な話しかけ"""
        self.state["busy_llm"] = True
        try:
            # ランダムなRAG記憶を取得
            rag_memories = await self.rag.get_random_turns(count=2)
            self.llm.rag_memories = rag_memories

            recent_turns = await self.conversation_cache.get_recent_turns(
                count=5, exclude_ellipsis=True,
            )
            self.llm.recent_turns = recent_turns
            try:
                self.llm.silence_summary = (
                    await self.conversation_cache.get_silence_summary()
                )
            except Exception as e:
                self.log("System", f"silence_summary failed: {e}")
                self.llm.silence_summary = None

            # VLMコンテキストを最新に更新
            if self.vlm_bridge and self.vlm_bridge.is_running:
                vlm_desc = self.vlm_bridge.get_scene_description()
                if vlm_desc:
                    self.llm.vlm_context = vlm_desc

            await self.process_input("…")
        finally:
            self.state["busy_llm"] = False
    
    async def run(self):
        """対話モードのメインループ

        mic.get_audio() と _vision_push_queue.get() を asyncio.wait() で
        並行待ちし、先に完了した方をディスパッチする。
        これにより、ユーザー無言時でも画面変化に即座に反応できる。
        """
        self.log("System", "Starting Talk Mode main loop...")

        try:
            while self.running and not self.stop_requested:
                await self._dispatch_next_event()

        except asyncio.CancelledError:
            self.log("System", "Talk Mode cancelled")
        except Exception as e:
            self.log("System", f"Error in Talk Mode: {e}")

    async def _dispatch_next_event(self) -> None:
        """mic入力またはvision pushのどちらか先に来た方を処理する。

        両方を asyncio.Task として同時に待ち、先に完了した方をディスパッチ。
        未完了の方はキャンセルして次のループに備える。
        """
        mic_task = asyncio.ensure_future(self.mic.get_audio())
        vision_task = asyncio.ensure_future(self._vision_push_queue.get())

        try:
            done, pending = await asyncio.wait(
                {mic_task, vision_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # run() 自体がキャンセルされた場合、両タスクを確実にクリーンアップ
            mic_task.cancel()
            vision_task.cancel()
            await self._cancel_quietly(mic_task)
            await self._cancel_quietly(vision_task)
            raise

        # 未完了タスクをキャンセル
        for task in pending:
            task.cancel()
            await self._cancel_quietly(task)

        # Vision push が先に来た場合
        if vision_task in done:
            try:
                vision_frame = vision_task.result()
            except (asyncio.CancelledError, Exception):
                return
            await self._handle_vision_event(vision_frame)
            return

        # Mic入力が先に来た場合
        if mic_task in done:
            try:
                audio_bytes = mic_task.result()
            except (asyncio.CancelledError, Exception):
                return
            await self._handle_mic_event(audio_bytes)

    async def _handle_vision_event(self, vision_frame) -> None:
        """画面変化イベントを処理する。"""
        if self.is_processing or self.state["busy_stt"] or self.state["busy_llm"]:
            # 処理中なら今回は無視（次のループで別のイベントを拾う）
            return

        self.state["busy_llm"] = True
        try:
            await self._handle_vision_push(vision_frame)
        finally:
            self.state["busy_llm"] = False
            now = time.time()
            self.state["last_user_event_ts"] = now
            self.state["idle_since_ts"] = now

    async def _handle_mic_event(self, audio_bytes) -> None:
        """マイク入力イベントを処理する。"""
        if not audio_bytes:
            return
        if self.stop_requested:
            return

        # ステート更新
        now = time.time()
        self.state["last_user_event_ts"] = now
        self.state["idle_since_ts"] = now

        # STT処理
        self.state["busy_stt"] = True
        try:
            user_text = await self.stt.transcribe(audio_bytes)
        finally:
            self.state["busy_stt"] = False

        if not user_text:
            return

        # リセットコマンド
        if "リセット" in user_text:
            self.llm.history = [{"role": "system", "content": self.system_prompt}]
            self.log("System", "Context Reset")
            return

        # 入力処理
        self.state["busy_llm"] = True
        try:
            await self.process_input(user_text)
        finally:
            self.state["busy_llm"] = False
            self.state["last_user_event_ts"] = time.time()

    @staticmethod
    async def _cancel_quietly(task: asyncio.Task) -> None:
        """タスクをキャンセルし、CancelledError を静かに吸収する。"""
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
