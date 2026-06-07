"""
対話モード - マイク入力によるリアルタイム対話
"""
import asyncio
import time
from typing import Callable, Optional

from config import Config
from modules import AudioInput, STTHandler
from prompts import TALK_SYSTEM_PROMPT
from .base_mode import BaseMode


class TalkMode(BaseMode):
    """対話モード：マイク入力 + 無言処理"""
    
    def __init__(self, log_callback: Optional[Callable[[str, str, str], None]] = None):
        super().__init__(TALK_SYSTEM_PROMPT, log_callback)
        self.task_planning_enabled = True  # Plan-and-Execute: TalkMode のみ有効
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

        # A1/A2: 沈黙ストリークの自己管理（ephemeral、実ユーザターンで reset）
        self._nudge_spoken: list = []          # [{"category","text"}] 沈黙 nudge のみ記録
        self._nudge_streak_count: int = 0      # 沈黙 "…" nudge 発火数（指数バックオフの n）
        self._last_nudge_fire_ts: float = 0.0  # 沈黙 nudge 間バックオフのゲート
        
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

        # Step 1.5 (g): instruction の PENDING → ACTIVE 遷移時に自発発話する callback を登録。
        # ユーザーが黙っていても Eve が約束を履行できるようにする。
        if self.task_manager is not None:
            self.task_manager.register_instruction_activated_callback(
                self._on_instruction_activated
            )
            self.log("System", "Instruction-activated callback registered", level="debug")

        self.log("System", "Talk Mode Ready - Speak now!")

    def _on_instruction_activated(self, instruction: dict) -> None:
        """TaskManager から PENDING → ACTIVE 遷移時に呼ばれる callback（同期）。

        ユーザー沈黙中でも Eve に約束を履行させるため、内部 user 入力として
        `_pending_input_queue` に投入する。dispatch_loop が拾って _run_response_pipeline
        を起動 → system prompt に [**!! 期限超過 !!**] が出る → Eve が約束を履行する。

        Step 1.5 Fix Layer E E6: 前回 audit 失敗 (audit_reason) があれば nudge テキストに
        付与して、 Eve に「前回未達成 → 今度は意味的に履行」を促す。

        VLM nudge の "[内部: 画面に新しい変化があった ...]" 経路と同型。
        """
        try:
            iid = instruction.get("id", "?")
            text = instruction.get("instruction", "")
            audit_reason = instruction.get("audit_reason")
            if audit_reason:
                payload = (
                    f"[内部: 期限超過再判定 {iid} — 前回未達成「{audit_reason}」 "
                    f"今度は内容的に履行 — {text}]"
                )
            else:
                payload = f"[内部: 期限超過 {iid} を履行 — {text}]"
            # Fix-6 P1-a: dict payload で is_internal_nudge=True を明示
            self._pending_input_queue.put_nowait({
                "text": payload,
                "is_internal_nudge": True,
            })
            self.log(
                "System",
                f"Instruction-activated nudge enqueued: {iid}"
                + (" (re-judge)" if audit_reason else ""),
                level="debug",
            )
        except Exception as e:
            self.log("System", f"_on_instruction_activated error: {e}", level="debug")

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
                # alert 消費の高水位を更新（二重展開防止）。VLM nudge は沈黙 backoff と直交。
                self.state["_vlm_alerts_consumed_at"] = now
                await self._process_idle_input(
                    "[内部: 画面に新しい変化があった - 自然にリアクションして]",
                    is_silence_nudge=False,
                )
            else:
                # A2: 沈黙 "…" nudge は指数バックオフで間隔を伸ばす（深い沈黙ほど引く）。
                # last_ellipsis_ts は上で更新済 = 8s 再アームは維持しつつ、実発火はこのゲートで絞る。
                interval = min(
                    Config.IDLE_BASE_SEC
                    * (Config.IDLE_BACKOFF_FACTOR ** self._nudge_streak_count),
                    Config.IDLE_BACKOFF_CAP_SEC,
                )
                if now - self._last_nudge_fire_ts < interval:
                    continue
                self._last_nudge_fire_ts = now
                await self._process_idle_input("…", is_silence_nudge=True)

    async def _process_idle_input(self, input_text: str, is_silence_nudge: bool = False):
        """無言時の自発発話処理（旧 _process_ellipsis）。

        ellipsis (…) と VLM nudge の両方で同じパスを使う。
        is_silence_nudge=True は沈黙 "…" nudge（バックオフ・自己記憶の対象）、
        False は VLM 画面変化 nudge（沈黙 backoff と直交、自己記憶には残さない）。
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

            # A2: カテゴリ判定クロックは silence_seconds（最後の「実」ユーザ発話起点。
            # 内部/期限超過 nudge では巻き戻らない信頼クロック。cold-start は 0.0 = early）。
            sec = 0.0
            try:
                summary = await self.conversation_cache.get_silence_summary()
                if isinstance(summary, dict):
                    # 既存 [沈黙サマリ] の「(…) N回」表示を実 nudge 回数で正す
                    # （nudge は cache 非保存のため元の ellipsis_count は常に 0）。
                    summary["ellipsis_count"] = self._nudge_streak_count
                    sec = float(summary.get("silence_seconds") or 0.0)
                self.llm.silence_summary = summary
            except Exception as e:
                self.log("System", f"silence_summary failed: {e}")
                self.llm.silence_summary = None

            if sec < Config.IDLE_EARLY_SEC:
                category = "early"
            elif sec < Config.IDLE_LONG_SEC:
                category = "mid"
            else:
                category = "long"

            # A1: この沈黙ストリークで既に言ったことを LLM に見せる（反復防止）。
            self.llm.nudge_self_memory = list(self._nudge_spoken)
            # 確認質問は 1 ストリーク 1 回まで（既に [check] があれば許可しない）。
            check_done = any(m.get("category") == "check" for m in self._nudge_spoken)
            allow_check = is_silence_nudge and category == "mid" and not check_done

            # VLMコンテキストを最新に更新
            if self.vlm_bridge and self.vlm_bridge.is_running:
                vlm_desc = self.vlm_bridge.get_scene_description()
                if vlm_desc:
                    self.llm.vlm_context = vlm_desc

            # Fix-6 P1-a: 「…」と VLM nudge は内部 nudge 扱い
            # (conversation_cache / LLM history / RAG query を汚染しない)
            response = await self.process_input(input_text, is_internal_nudge=True)

            # A1: 沈黙 nudge の実発話のみ自己記憶に控える
            # （VLM 反応は vlm_bridge 側に dedup があるので残さない＝バッファ汚染回避）。
            if is_silence_nudge:
                spoken = (response or "").strip()
                if spoken and spoken != "…":
                    rec_cat = (
                        "check"
                        if (allow_check and spoken.endswith(("？", "?")))
                        else category
                    )
                    self._nudge_spoken.append({"category": rec_cat, "text": spoken[:80]})
                    if len(self._nudge_spoken) > 8:
                        self._nudge_spoken = self._nudge_spoken[-8:]
                # 沈黙 nudge 発火ごとにバックオフ階段を1段進める。
                self._nudge_streak_count += 1
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
                payload = await self._pending_input_queue.get()
                if self.stop_requested:
                    break

                # Fix-6 P1-b: payload は dict or str。 dict なら is_internal_nudge を抽出。
                # 後方互換: 文字列ペイロードは通常 user 発話扱い (is_internal_nudge=False)
                if isinstance(payload, dict):
                    user_text = payload.get("text", "")
                    is_internal_nudge = bool(payload.get("is_internal_nudge", False))
                else:
                    user_text = payload
                    is_internal_nudge = False

                # リセットコマンド
                if "リセット" in user_text:
                    self.llm.history = [{"role": "system", "content": self.system_prompt}]
                    self.log("System", "Context Reset")
                    continue

                # Fix-6 P1-e: 内部 nudge は並行実行回避。 現在応答中なら queue 末尾に
                # 再 put + 0.5 秒待機してリトライ (会話の二重化防止)。
                if is_internal_nudge:
                    if self.is_speaking() or (
                        self._current_response_task is not None
                        and not self._current_response_task.done()
                    ):
                        self.log(
                            "System",
                            "[Dispatch] internal nudge deferred (speaking, retry)",
                            level="debug",
                        )
                        await asyncio.sleep(0.5)
                        self._pending_input_queue.put_nowait({
                            "text": user_text,
                            "is_internal_nudge": True,
                        })
                        continue

                now = time.time()
                # A2: 実ユーザターンでのみ沈黙ストリーク状態をリセット（is_internal_nudge ラベルでガード）。
                # 期限超過などの内部 nudge では reset しない（カテゴリ階段の early 巻き戻り防止）。
                # last_user_event_ts は読み出しゼロの dead フィールドのため書き込みを撤去。
                if not is_internal_nudge:
                    self._nudge_streak_count = 0
                    self._last_nudge_fire_ts = 0.0
                    self._nudge_spoken = []
                    if self.llm is not None:
                        self.llm.nudge_self_memory = []
                self.state["idle_since_ts"] = now

                # Step 4: マージ判断（_pending_lock 内で atomic に）
                # Step 1.5 Fix A1: 期限超過 nudge ([内部: 期限超過 ...]) はマージ禁止、
                # 単独で処理する。is_speaking 中なら interrupt して優先発火させる。
                should_inject_interrupt_marker = False

                async with self._pending_lock:
                    if user_text.startswith("[内部: 期限超過"):
                        # Fix A1: 期限超過 nudge は絶対優先で単独処理
                        # Fix-6 P1-e: 並行実行回避は上で処理済み (interrupt しない)
                        input_to_process = user_text
                        self._pending_input = user_text
                        # 内部 nudge は interrupt しない (Fix-6 P1-e)
                        self.log(
                            "System",
                            "[Dispatch] overdue priority, skip merge",
                            level="debug",
                        )
                    else:
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
                # Fix-6 P1-b: is_internal_nudge を伝搬する。
                spawned = asyncio.create_task(
                    self._run_input_with_cleanup(
                        input_to_process,
                        is_internal_nudge=is_internal_nudge,
                    )
                )
                self._spawned_input_tasks.add(spawned)
                spawned.add_done_callback(self._spawned_input_tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log("System", f"Dispatch loop error: {e}")

    async def _run_input_with_cleanup(
        self, input_text: str, is_internal_nudge: bool = False
    ):
        """process_input を呼んで、完了後に _pending_input をクリアする helper。

        dispatch_loop からは fire-and-forget で起動される。例外はログのみ。
        Fix-6 P1-b: is_internal_nudge を process_input に伝搬する。
        """
        try:
            await self.process_input(input_text, is_internal_nudge=is_internal_nudge)
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
