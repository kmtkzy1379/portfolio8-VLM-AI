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

        # Plan-and-Execute Task Queue（モードごとに明示的に有効化、デフォルト False）
        # TalkMode のみ __init__ で True にする。
        self.task_planning_enabled: bool = False
        self.task_manager = None
        self.planner = None
        self.validator = None

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
        def update_memory_func(memory_text, *, affect=None, affect_set_at=None):
            self.llm.ai2_feedback = memory_text[:Config.FB_LOOP_MAX_CHARS]
            # Step 1.5: 更新時刻も記録（古い参考情報ラベルの経過時間表示用）
            from datetime import datetime as _dt
            self.llm._ai2_feedback_set_at = _dt.now().isoformat()
            # affect→tone (WEAK): 新しく算出された affect のみスタンプする。沈黙サイクルは
            # affect=None で来るのでタイムスタンプを更新せず、age が伸びて TTL で失効する
            # （古い気分が毎サイクル「新鮮」に再スタンプされて永遠に残るのを防ぐ）。
            if affect is not None:
                self.llm.affect = affect
                self.llm._affect_set_at = affect_set_at or _dt.now().isoformat()
            self.log("System", f"AI2 Feedback Updated ({len(memory_text)} chars)", level="debug")
        self.llm.update_memory = update_memory_func
        self.llm.rag = self.rag

        # FeedbackHandler初期化 (ループ型FEPメタ認知)
        self.feedback = FeedbackHandler(self.llm, conversation_cache=self.conversation_cache)
        # Phase 4.3.1: feedback ループから episode を RAG へ保存できるよう注入
        self.feedback.set_rag(self.rag)
        # Step 1.5: AI1 prompt に「過去の goal 提案履歴」を注入する callable を渡す
        self.llm.goal_history_provider = self.feedback.get_goal_history_for_prompt

        # RAGと会話キャッシュを初期化
        await self.rag.initialize()
        await self.conversation_cache.initialize()

        # バックグラウンドタスク開始
        self._player_task = asyncio.create_task(self.player.play_worker())
        self._feedback_task = asyncio.create_task(self.feedback.start_loop())

        # Plan-and-Execute: TaskManager のみを Step 1 で起動（mode フラグ + global kill switch）
        # Planner / Validator の生成・注入は後続 Step で追加する。
        if self.task_planning_enabled and Config.PLANNER_ENABLE:
            from modules.task_manager import TaskManager
            self.task_manager = TaskManager(Config.TASK_QUEUE_FILE)
            await self.task_manager.start()
            # Step 1.5: llm に TaskManager を注入 + META_TOOLS を有効化（active_instruction ツール）
            self.llm.set_task_manager(self.task_manager)
            if Config.AI1_INSTRUCTION_TOOL_ENABLE:
                self.llm.enable_meta_tools(True)
            # Fix-4 Phase 1 C2: RAG に TaskManager を注入（active 時の capability_denial penalty 用）
            if hasattr(self.rag, "set_task_manager"):
                self.rag.set_task_manager(self.task_manager)

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

    # Fix-9b: コミット対象外の非実質発話（相槌・沈黙）。
    _NONSUBSTANTIVE_FOR_COMMIT = frozenset(
        {"…", "うん", "うん。", "はい", "あ", "あ。", "ん", "へえ", "そう", "うんうん", "あー", "あーね"}
    )

    def _maybe_commit_fact(self, input_text: str, ai_response: str, is_internal_nudge: bool) -> None:
        """Fix-9b: Eve の実質回答を一貫性ストアに記録する（応答完了時, sync 入口）。

        関連 instruction がある実質回答のみ対象。task_manager の command queue に enqueue するだけで、
        conversation_cache / RAG には一切書かない（Fix-6 保全）。実際の defend/保存は
        TaskManager._handle_commit_fact が行う（同じ話題に active な答えがあれば最初の答えを守る）。
        """
        if self.task_manager is None:
            return
        answer = (ai_response or "").strip()
        if not answer or answer in self._NONSUBSTANTIVE_FOR_COMMIT:
            return
        import re as _re
        instruction_text = ""
        iid = None
        m = _re.search(r"\[内部: 期限超過(?:再判定)?\s+(i_[0-9a-f]+)", input_text)
        if m:
            # 督促 nudge: instruction id が input_text に埋め込まれている
            iid = m.group(1)
            inst = self.task_manager.get_instruction(iid)
            instruction_text = inst["instruction"] if inst else ""
        else:
            # idle / 通常ターン: 非終端 instruction が 1 件だけならそれに帰属（曖昧なら捕捉しない）
            cands = self.task_manager.get_active_instructions_for_prompt()
            if len(cands) != 1:
                return
            # Bug-A gate: 期限未到来 (derived PENDING) の予約には「答え」がまだ存在しない。
            # 予約受理の返事（「うん、30秒後に答えるね」）や待機中の雑談を fact として
            # コミットしない（ライブで answer_text が了解発話に汚染された事故の根本対策）。
            # 期限到来 (derived=active) の回答のみ帰属させる。督促 regex 経路は上で処理済み。
            if cands[0].get("derived_status") != "active":
                return
            iid = cands[0]["id"]
            instruction_text = cands[0]["instruction"]
        if not instruction_text:
            return
        topic_norm = self.task_manager._normalize_instruction(instruction_text)
        if not topic_norm:
            return
        self.task_manager.enqueue_command_nowait({
            "kind": "commit_fact",
            "scope": "eve",
            "topic_norm": topic_norm,
            "answer_text": answer[:120],
            "instruction_id": iid,
            "source": "nudge" if is_internal_nudge else "conversation",
        })

    @staticmethod
    def _is_greeting(text: str) -> bool:
        """発話冒頭が挨拶かどうか（Bug-E: nudge 自己記憶のタグ付け + 内部 nudge の再挨拶フィルタ用）。"""
        import re as _re
        return bool(_re.search(
            r"(こんにち[はわ]|こんばん[はわ]|おはよ|やっほ|ハロー|はろー|^やあ\b)",
            (text or "")[:20],
        ))

    def _greeting_already_done(self) -> bool:
        """このセッションで挨拶が既に交わされたか（recent_turns の user/ai どちらかに挨拶）。"""
        for t in (getattr(self.llm, "recent_turns", None) or []):
            if self._is_greeting(str(t.get("user", ""))) or self._is_greeting(str(t.get("ai", ""))):
                return True
        return False

    @staticmethod
    def _dedup_runaway(text: str) -> str:
        """暴走（同一発話が丸ごと二重に出力される degeneration）を検出してトリムする。
        保守的: 全体を中点付近の文末で二分割し、前半後半がほぼ同一(>=0.95)かつ前半が十分長い
        ときだけ前半に切る。正当な短い繰り返し（「うんうん」等）や別内容の連結は触らない。"""
        t = (text or "").strip()
        if len(t) < 30:
            return text
        mid = len(t) // 2
        span = max(8, len(t) // 5)
        best = None
        for d in range(0, span):
            for j in (mid - d, mid + d):
                if 0 < j < len(t) and t[j - 1] in "。！？!?":
                    best = j
                    break
            if best is not None:
                break
        if best is None:
            return text
        a, b = t[:best].strip(), t[best:].strip()
        if a and b and len(a) >= 12:
            from difflib import SequenceMatcher
            if SequenceMatcher(None, a, b).ratio() >= 0.95:
                return a
        return text

    def _auto_complete_overdue_if_needed(self, input_text: str, ai_response: str) -> None:
        """Safety net: 期限超過 nudge で Eve が実発話したのに clear(done) を呼ばなかったら、
        テキスト品質に依存せず done を自動確定する（タスクが未完で残る脆さを塞ぐ）。
        - LLM が既に clear 済 (PROVISIONAL_DONE/DONE) なら何もしない (status==active ガードで二重防止)。
        - 相槌/空発話/極端に短い応答では自動完了しない (A2 ゲート相当 + 長さガード)。
        - eve_response 付きで enqueue → PROVISIONAL_DONE → AI2 audit を通すので off-topic は弾ける。"""
        if self.task_manager is None:
            return
        import re as _re
        m = _re.search(r"\[内部: 期限超過(?:再判定)?\s+(i_[0-9a-f]+)", input_text or "")
        if not m:
            return
        iid = m.group(1)
        inst = self.task_manager.get_instruction(iid)
        if not inst or inst.get("status") != "active":
            return
        spoken = (ai_response or "").strip()
        if (not spoken) or (spoken in self._NONSUBSTANTIVE_FOR_COMMIT) or (len(spoken) < 4):
            return
        self.task_manager.enqueue_command_nowait({
            "kind": "clear_active_instruction",
            "id": iid,
            "status": "done",
            "eve_response": ai_response,
            "reason": "auto: 実発話で履行を自動確定（clear 未呼び出しの保険）",
        })
        self.log("System", f"[AutoClear] overdue {iid} auto-completed (substantive)", level="debug")

    async def process_input(self, input_text: str, is_internal_nudge: bool = False) -> str:
        """
        テキスト入力を処理してAI応答を生成（外部 API は完全互換）。

        Step 2: 内部で cancel 可能な Task を保持しつつ、必ず await で str を返す。
        - Task 即 return は禁止（game_mode/youtube_mode の `await self.process_input(...)` を維持）
        - `_response_lock` で同時応答実行を排除（length-based history rollback の安全性）

        Fix-6 P1-b: is_internal_nudge=True なら内部 nudge として扱う
        (conversation_cache / LLM history / RAG query を汚染しない)。
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
        # Fix-6 P1-b: is_internal_nudge を _run_response_pipeline に伝搬
        async with self._response_lock:
            self._current_response_task = asyncio.create_task(
                self._run_response_pipeline(input_text, is_internal_nudge=is_internal_nudge)
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

    async def _run_response_pipeline(
        self, input_text: str, is_internal_nudge: bool = False
    ) -> str:
        """応答パイプライン本体（Task 化されて cancel 可能）。

        - 入口で `history_len_before = len(self.llm.history)` を snapshot
        - cancel 時は length-based restore で history を完全 rollback（user/assistant/tool すべて削除）
        - cancel 時に TTS 完了レース対策で `player.queue` を drain
        - finally で `_stage = IDLE` に戻す

        Fix-6 P1-c: is_internal_nudge=True なら 4 箇所 (log/RAG/cache/llm) の汚染防止を有効化。
        """
        if self.stop_requested or not self.running:
            return ""

        self.is_processing = True
        self._stage = ResponseStage.LLM_PENDING
        history_len_before = len(self.llm.history)
        pending_tts_tasks: list[asyncio.Task] = []
        start_time = time.time()

        try:
            # Fix-6 P1-c (1) ログ表示: User vs Nudge を分ける
            if is_internal_nudge:
                self.log("Nudge", input_text[:80] + ("..." if len(input_text) > 80 else ""), level="debug")
            else:
                self.log("User", input_text)

            # Step 3: ConversationCache への記録は正常完了時のみ atomic に行う
            # （cancel 時に user だけ残る穴を防ぐため、ここでは add_user_message しない）

            # RAG検索（800msタイムアウト）
            rag_memories = []
            rag_query = ""
            try:
                # Phase 4.3.2: top_k は Config.FB_RAG_TOP_K (default 4) を使う
                # MMR による多様性確保が効くため top_k 拡大しても重複しない
                # Fix-6 P1-c (2) RAG 汚染防止: 内部 nudge は "[内部:...]" を query にしない、
                # instruction の中身 (末尾の "—" 以降) を抽出して query にする
                if is_internal_nudge and "—" in input_text:
                    rag_query = input_text.split("—")[-1].strip().rstrip("]")
                elif is_internal_nudge:
                    # VLM nudge や「…」のように "—" を含まない場合は、 RAG search を skip
                    rag_query = ""
                else:
                    rag_query = input_text
                if rag_query:
                    rag_task = asyncio.create_task(self.rag.search_similar(rag_query))
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
            # Fix(沈黙RAG): 内部 nudge で RAG search を skip した場合（idle「…」, rag_query=""）、
            # _process_idle_input が事前にセットした random RAG memories（沈黙時の話題タネ）を
            # [] で上書きして消さない。実際に search した時 / 通常ユーザターン時のみ上書きする。
            # これが無いと沈黙時に Eve が手札ゼロになり、不自然な無難発話になる。
            if rag_query or not is_internal_nudge:
                self.llm.rag_memories = rag_memories
            recent_turns = await self.conversation_cache.get_recent_turns(
                count=5, exclude_ellipsis=True,
            )
            self.llm.recent_turns = recent_turns
            # Fix-9b: 一貫性ストア（既にコミットした自分の答え/嗜好）を全経路で注入
            # （idle / 督促 / 通常ターンは全てこの pipeline を通る）。
            if self.task_manager is not None:
                try:
                    # 関連性ゲート: 嗜好が増えても、いまの話題に関係する fact だけ注入して
                    # プロンプトが無限に伸びないようにする（≤limit 件なら従来どおり全件）。
                    # 関連シグナルは現ターンの入力 + 直近会話（fact は会話の話題に紐づくため）。
                    _rel_parts = [input_text or ""]
                    for _t in (recent_turns or []):
                        _rel_parts.append(str(_t.get("user", "")))
                        _rel_parts.append(str(_t.get("ai", "")))
                    self.llm.committed_facts = self.task_manager.get_committed_facts_for_prompt(
                        relevance_text=" ".join(_rel_parts)
                    )
                except Exception:  # noqa: BLE001
                    self.llm.committed_facts = []
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
            # Bug-E(code gate): 内部 nudge で既に挨拶済みなら、応答の「先頭の挨拶文」を
            # TTS/記録の前に剥がす（再挨拶の決定論的遮断。プロンプト規則では不十分と Tier-3 で確認）。
            # 最初の非空文だけを検査するので、本文への影響は無い。実ユーザターンは対象外。
            _greet_filter_armed = is_internal_nudge and self._greeting_already_done()
            # Fix-6 P1-c (3) LLM history 汚染防止: is_internal_nudge を伝搬
            async for sentence in self.llm.generate_stream(input_text, is_internal_nudge=is_internal_nudge):
                if self.stop_requested or self.player.interrupt_signal:
                    break
                if sentence.strip() and _greet_filter_armed:
                    _greet_filter_armed = False  # 先頭の非空文のみ検査
                    if self._is_greeting(sentence):
                        self.log("System", f"[GreetFilter] 再挨拶を抑制: {sentence.strip()[:30]}", level="debug")
                        continue  # ai_response にも TTS にも乗せない
                ai_response += sentence
                if sentence.strip():
                    sentence_queue.append(sentence)
                    if len(sentence_queue) >= 3:
                        await process_tts_batch()

            # 残りのセンテンスを処理
            while sentence_queue:
                await process_tts_batch()

            # Dedup guard: 暴走（同一発話の丸ごと二重出力）を記録/コミット/clear の前にトリム。
            ai_response = self._dedup_runaway(ai_response)

            # 応答をログに出力
            if ai_response:
                self.log("AI", ai_response)
                # Fix-9b: コミット事実の捕捉（応答完了時）。conversation_cache/RAG には書かない＝Fix-6 保全。
                # 内部 nudge ターン（ドリフトが起きる場所）でも捕捉するが、 下の Fix-6 ガード本体には入らない。
                if self.task_manager is not None:
                    try:
                        self._maybe_commit_fact(input_text, ai_response, is_internal_nudge)
                    except Exception as e:  # noqa: BLE001
                        self.log("System", f"[CommitFact] capture failed: {e}", level="debug")
                # Step 3: 正常完了時のみ atomic に1ターン記録
                # cancel 時はここに到達しないため user/ai どちらも記録されない
                # Fix-6 P1-c (4) conversation_cache 汚染防止: 内部 nudge は履歴に書かない
                if not is_internal_nudge:
                    await self.conversation_cache.add_turn(input_text, ai_response)
                else:
                    # 内部 nudge ターンの応答は別 logger に記録 (recent_context 汚染防止)
                    import logging as _logging
                    _task_log = _logging.getLogger("eve.task")
                    _task_log.info("[Nudge-Response] %s -> %s", input_text[:40], ai_response[:60])
                # Phase 4.3.1: 生ターン保存は撤去。
                # RAG 入力は feedback ループが Episode Summary 経由で行う (rag.add_episode)。
                # conversation_history.txt は別途 cache 経由で監査用に蓄積される。

            # フィードバックループに「ターン完了」シグナルを送る (Event.set のみ、同期・即時)
            if self.feedback:
                self.feedback.signal_turn_done()

            # Step 1.5: active_instruction の状態遷移を確定（PENDING→ACTIVE/EXPIRED 等）。
            # 例外で応答パスを止めないよう必ず try/except で wrap する。
            # 次応答の system prompt 構築時に正しく ACTIVE/EXPIRED が反映される。
            if self.task_manager is not None:
                try:
                    await self.task_manager._reconcile_instruction_status()
                except Exception as e:
                    self.log("System", f"[Task] reconcile failed: {e}", level="debug")

                # Step 1.5 Fix Layer E E0+E2: clear_active_instruction(done) が
                # command_queue にあれば drain させ、 PROVISIONAL_DONE 化を確定させる。
                # その後 pop_pending_audits で未 audit の instructions を取り出し、
                # AI2 audit を fire-and-forget で spawn する（応答 latency に影響しない）。
                # Fix-6 P2-g: audit prompt に直近会話 + VLM ナレーションを渡して精度向上
                if self.feedback is not None:
                    try:
                        await self.task_manager.flush_command_queue(timeout=2.0)
                        # Safety net: 期限超過の履行ターンで Eve が実発話したのに clear(done) を
                        # 呼ばなかった場合、テキスト品質に依存せず done を自動確定する。
                        # PROVISIONAL_DONE → AI2 audit を通すので off-topic は audit が弾ける。
                        if is_internal_nudge:
                            try:
                                self._auto_complete_overdue_if_needed(input_text, ai_response)
                                await self.task_manager.flush_command_queue(timeout=2.0)
                            except Exception as e:  # noqa: BLE001
                                self.log("System", f"[AutoClear] error: {e}", level="debug")
                        pending = self.task_manager.pop_pending_audits()
                        if pending:
                            # Fix-6 P2-g: audit に渡す文脈情報を 1 回だけ取得
                            audit_recent_turns = None
                            audit_vlm_narration = None
                            try:
                                audit_recent_turns = await self.conversation_cache.get_recent_turns(
                                    count=5, exclude_ellipsis=True,
                                )
                            except Exception:
                                pass
                            try:
                                if self.vlm_bridge and self.vlm_bridge.is_running:
                                    audit_vlm_narration = self.vlm_bridge.get_scene_description() or None
                            except Exception:
                                pass
                            for inst_meta in pending:
                                asyncio.create_task(
                                    self.feedback.audit_provisional_instruction(
                                        instruction_id=inst_meta["id"],
                                        instruction_text=inst_meta["instruction"],
                                        eve_response=inst_meta["eve_response"],
                                        task_manager=self.task_manager,
                                        recent_turns=audit_recent_turns,
                                        vlm_narration=audit_vlm_narration,
                                    )
                                )
                            self.log(
                                "System",
                                f"[Audit] spawned {len(pending)} provisional audits (with context)",
                                level="debug",
                            )
                    except Exception as e:
                        self.log("System", f"[Audit] spawn failed: {e}", level="debug")

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
            # Step 1.5 Fix A2-5: cancel 経路でも pending_clear をリセット
            # （次応答に stale な clear が持ち越されて誤 flush するのを防ぐ）
            if hasattr(self.llm, "_pending_clear_instructions"):
                self.llm._pending_clear_instructions = []
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
