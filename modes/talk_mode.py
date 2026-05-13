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
    
    def __init__(self, log_callback: Optional[Callable[[str, str, str], None]] = None):
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
            # Step 5: 直近 nudge で消費した VLM alert の高水位タイムスタンプ。
            # has_unseen_vlm_alerts(consumed_at) で「未対応の新着 alert」を判定。
            "_vlm_alerts_consumed_at": 0.0,
        }
        
        self._idle_task: Optional[asyncio.Task] = None

        # Step 4: dispatch_loop が fire-and-forget で起動した process_input task を保持
        # （GC で「Task was destroyed but it is pending!」警告を防ぐ）
        self._spawned_input_tasks: set = set()
    
    async def initialize(self):
        """対話モード固有の初期化"""
        await super().initialize()

        # STT初期化
        self.log("System", "DEBUG: Creating STTHandler...", level="debug")
        self.stt = STTHandler()

        # マイク初期化（音声開始時のコールバック設定）
        def on_speech_start():
            self.interrupt()
            self.state["last_user_event_ts"] = time.time()

        self.log("System", "DEBUG: Creating AudioInput...", level="debug")
        self.mic = AudioInput(callback_on_speech_start=on_speech_start)

        self.log("System", "DEBUG: Starting mic stream...", level="debug")
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
        """無言処理ループ：アイドル状態が続いたら自発発話。

        Step 5: 未対応の VLM alert があれば待機を 2 秒に短縮し、
        「…」ではなく「[内部: 画面に新しい変化があった]」を投入する。
        同じ alert を二重展開しないため `_vlm_alerts_consumed_at` を
        高水位として記録する。
        """
        while self.running and not self.stop_requested:
            await asyncio.sleep(0.3)
            now = time.time()

            # 厳密な idle 判定（is_speaking + busy_stt + interrupt_signal）
            if self.is_speaking() or self.state["busy_stt"]:
                self.state["idle_since_ts"] = now
                continue
            if self.player and self.player.interrupt_signal:
                self.state["idle_since_ts"] = now
                continue

            # VLM alert のチェック（新しい alert がある場合は待機短縮 + nudge）
            consumed_at = self.state.get("_vlm_alerts_consumed_at", 0.0)
            has_new_alert = (
                self.llm is not None
                and self.llm.has_unseen_vlm_alerts(consumed_at)
            )
            wait_threshold = 2.0 if has_new_alert else 8.0

            if now - self.state["idle_since_ts"] < wait_threshold:
                continue
            if now - self.state["last_ellipsis_ts"] < wait_threshold:
                continue

            self.state["last_ellipsis_ts"] = now
            if has_new_alert:
                # alert 消費の高水位を更新（二重展開防止）
                self.state["_vlm_alerts_consumed_at"] = now
                await self._process_idle_input(
                    "[内部: 画面に新しい変化があった - 自然にリアクションして]"
                )
            else:
                await self._process_idle_input("…")

    async def _process_idle_input(self, input_text: str):
        """無言時の自発発話処理（旧 _process_ellipsis）。

        ellipsis (…) と VLM nudge の両方で同じパスを使う。
        VLM alert は llm._vlm_alerts に蓄積されているので、
        process_input → _build_system_prompt で自然に展開される。
        """
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

            await self.process_input(input_text)
        finally:
            self.state["busy_llm"] = False
    
    async def run(self):
        """対話モードのメインループ

        Step 1: AudioListener と Dispatcher を独立 worker として起動する。
        Step 5: VLM nudge は _on_vision_alert_main 経由で llm._vlm_alerts に
        蓄積されるだけになり、独立 process_input は走らなくなる。
        VLM 発話は次のユーザー応答 or _idle_ellipsis_loop の自発 nudge で行われる。
        """
        self.log("System", "Starting Talk Mode main loop...", level="debug")

        audio_task = asyncio.create_task(self._audio_listener_loop())
        dispatch_task = asyncio.create_task(self._dispatch_loop())

        try:
            await asyncio.gather(audio_task, dispatch_task)
        except asyncio.CancelledError:
            self.log("System", "Talk Mode cancelled", level="debug")
        except Exception as e:
            self.log("System", f"Error in Talk Mode: {e}")
        finally:
            for task in (audio_task, dispatch_task):
                if not task.done():
                    task.cancel()
                await self._cancel_quietly(task)

    async def _audio_listener_loop(self):
        """マイク入力 → STT → _pending_input_queue を独立 worker で回す。

        Eve が LLM/TTS 中でも常時走り続けるため、入力2 がリアルタイムに STT 完了する。
        Dispatcher が後段で状態に応じてマージ判断する（Step 4）。
        """
        self.log("System", "Audio listener started", level="debug")
        while self.running and not self.stop_requested:
            try:
                audio_bytes = await self.mic.get_audio()
                if self.stop_requested:
                    break
                if not audio_bytes:
                    continue

                self.state["busy_stt"] = True
                try:
                    user_text = await self.stt.transcribe(audio_bytes)
                finally:
                    self.state["busy_stt"] = False

                if not user_text:
                    continue

                await self._pending_input_queue.put(user_text)
                self.log("System", f"[AudioListener] enqueued: {user_text[:40]}...", level="debug")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log("System", f"Audio listener error: {e}")

    async def _dispatch_loop(self):
        """_pending_input_queue から取り出してマージ判断 + process_input を呼ぶ。

        Step 4: is_speaking() と _current_response_task の状態に応じて分岐:
        - TTS 再生未開始 AND task 生存中: 入力1+入力2 をマージして再投入（cancel + merge）
        - TTS 再生中 or task 完了済みで再生残: interrupt_async + one_shot_context 注入 + 新応答
        - IDLE: 通常処理
        """
        self.log("System", "Dispatch loop started", level="debug")
        while self.running and not self.stop_requested:
            try:
                user_text = await self._pending_input_queue.get()
                if self.stop_requested:
                    break

                # リセットコマンド
                if "リセット" in user_text:
                    self.llm.history = [{"role": "system", "content": self.system_prompt}]
                    self.log("System", "Context Reset")
                    continue

                now = time.time()
                self.state["last_user_event_ts"] = now
                self.state["idle_since_ts"] = now

                # Step 4: マージ判断（_pending_lock 内で atomic に）
                should_inject_interrupt_marker = False

                async with self._pending_lock:
                    task = self._current_response_task
                    tts_started = self.player.is_playing or not self.player.queue.empty()

                    if task is not None and not task.done() and not tts_started:
                        # ケース1: 入力1 が処理中 AND TTS 未開始 → マージ
                        prev = self._pending_input or ""
                        merged = (
                            f"{prev}\n[追加: 続けて発話] {user_text}"
                            if prev
                            else user_text
                        )
                        input_to_process = merged
                        self._pending_input = merged  # 連続 3 発以上に備えて更新
                        self.log(
                            "System",
                            f"[Merge] combined inputs ({len(merged)} chars): {merged[:80]}...",
                            level="debug",
                        )
                    elif self.is_speaking():
                        # ケース2: TTS 再生中 or task 完了済みで再生残 → interrupt + 新応答
                        input_to_process = user_text
                        self._pending_input = user_text
                        should_inject_interrupt_marker = True
                    else:
                        # ケース3: IDLE → 通常処理
                        input_to_process = user_text
                        self._pending_input = user_text

                # interrupt は lock の外で実行（interrupt_async は内部で wait する）
                if should_inject_interrupt_marker:
                    await self.player.interrupt_async()
                    self.llm.one_shot_context = (
                        "[直前の応答が中断されました — ユーザーが話し始めたため]"
                    )
                    self.log(
                        "System",
                        "[Interrupt] TTS playing, switching to new response",
                        level="debug",
                    )

                # ★ 重要: process_input は fire-and-forget で起動する。
                # await しないことで dispatch_loop は次の queue.get に即戻り、
                # 入力1 処理中でも入力2 をリアルタイムにマージ判断できる。
                # process_input は内部で _cancel_current_response_if_any → _response_lock
                # を取るため、複数 task が並列実行されることはない。
                spawned = asyncio.create_task(
                    self._run_input_with_cleanup(input_to_process)
                )
                self._spawned_input_tasks.add(spawned)
                spawned.add_done_callback(self._spawned_input_tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log("System", f"Dispatch loop error: {e}")

    async def _run_input_with_cleanup(self, input_text: str):
        """process_input を呼んで、完了後に _pending_input をクリアする helper。

        dispatch_loop からは fire-and-forget で起動される。例外はログのみ。
        """
        try:
            await self.process_input(input_text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log("System", f"process_input error: {e}", level="debug")
        finally:
            async with self._pending_lock:
                # 自分の入力が _pending_input に残っていればクリア
                # （マージで別の入力に上書きされている場合は触らない）
                if self._pending_input == input_text:
                    self._pending_input = None
            self.state["last_user_event_ts"] = time.time()

    @staticmethod
    async def _cancel_quietly(task: asyncio.Task) -> None:
        """タスクをキャンセルし、CancelledError を静かに吸収する。"""
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
