"""
FeedbackHandler - FEP ループ型メタ認知フィードバック

前回フィードバック終了 → 即次回開始、というイベント駆動常駐ループで
期間内の対話 + VLM観察を総合的にFEP分析する。応答LLMのレイテンシには
一切影響しない (別 asyncio task)。

活性度の3段階 (脳神経科学の時間スケール準拠):
- 高活性: 会話ターン発生 -> 即時起動 (wait_for が Event.set() で即 wake)
- 中活性: VLMイベントのみ -> 30秒 (ワーキングメモリ保持時間)
- 低活性: 完全無音 -> 120秒 (DMN / マインドワンダリング周期)

無音期間は専用プロンプト (SILENT_PROMPT_SYSTEM) に切り替わり、
"静寂そのものを観測対象" として解釈する。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from config import Config
from modules.feedback_llm import FeedbackLLMClient
from modules.hierarchical_surprise import (
    HierarchicalSurprise,
    compute_hierarchical_surprise,
)
from modules.feedback_prompts import (
    FEP_PROMPT_SYSTEM,
    SILENT_PROMPT_SYSTEM,
    build_fep_user_text,
    build_silent_user_text,
)
from modules.precision import (
    PrecisionState,
    apply_mode_adjustments,
    compute_alpha_from_affect,
    decide_mode,
    normalize_affect_label,
    transition_log,
    update_surprise_ema,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Phase 4.1: Top-down 予測 / 予測評価 抽出用の enum + regex 定数
# ──────────────────────────────────────────────────────────────────────

# policy enum: LLM が文脈から自由に書けるが、検証可能性のため固定セットに正規化
_VALID_POLICIES = {"speak", "watch", "nudge", "question", "share", "other"}
# expected_activity enum: 集計可能性のため固定セットに正規化
_VALID_ACTIVITIES = {
    "silent", "chat", "coding", "window_change", "video", "game", "other"
}

# 半角コロン and 全角コロンの両方に対応
_PRED_TAG_RE = re.compile(
    r"^\s*(user|screen|policy|expected_activity|confidence|rationale)"
    r"\s*[:：]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_EVAL_TAG_RE = re.compile(
    r"^\s*(user_hit|screen_hit|policy_hit|surprise_1_5|calibration_note)"
    r"\s*[:：]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# 「次期間予測」見出し以降を切り出す (DOTALL で改行越え)
_PRED_SECTION_RE = re.compile(r"次期間予測.*", re.DOTALL)
# 「予測評価」(日本語) または "prediction_evaluation" (英語タグ) どちらでも拾う
_EVAL_SECTION_RE = re.compile(
    r"(?:予測評価|prediction_evaluation)\s*[:：(].*",
    re.DOTALL | re.IGNORECASE,
)

# Phase 4.2: Affective Inference セクション抽出 (4 タグ: affect/intensity/valence/narrative)
_AFFECT_TAG_RE = re.compile(
    r"^\s*(affect|intensity|valence|narrative)\s*[:：]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_AFFECT_SECTION_RE = re.compile(
    r"(?:感情・注意のクオリア|affective).*",
    re.DOTALL | re.IGNORECASE,
)

# Phase 4.3.1: Episode Summary セクション抽出 (3 タグ: summary/topic_tags/importance)
_EPISODE_TAG_RE = re.compile(
    r"^\s*(summary|topic_tags|importance)\s*[:：]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_EPISODE_SECTION_RE = re.compile(
    r"(?:エピソード要約|episode\s*summary).*",
    re.DOTALL | re.IGNORECASE,
)

# Phase 4.3.3 (Batch 2): Goal Slot セクション抽出 (3 タグ: goal_short/goal_change/goal_long)
_VALID_GOAL_CHANGES = {"keep", "update", "pivot"}
_GOAL_TAG_RE = re.compile(
    r"^\s*(goal_short|goal_change|goal_long)\s*[:：]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_GOAL_SECTION_RE = re.compile(
    r"(?:一貫した行動目標|goal\s*slot).*",
    re.DOTALL | re.IGNORECASE,
)
# 第一防衛: substring 一致で禁止語 detect (sync, 軽量)
_GOAL_BOILERPLATE_PHRASES = [
    "維持する", "良好な共存", "ユーザーとの良好",
    "対話の具体性を高める", "長期的かつ良好",
]

# AI1 注入用「応答AIへの内省ノート」セクション抽出 (regex)
_AI1_NOTE_SECTION_RE = re.compile(
    r"応答AIへの内省ノート.*",
    re.DOTALL,
)


@dataclass
class _Window:
    """フィードバック1ループの入力ウィンドウ。"""
    start_ts: Optional[datetime]
    end_ts: datetime
    turns: list[dict] = field(default_factory=list)
    vlm_frames: list = field(default_factory=list)
    last_user_utterance: str = ""
    last_user_ts: Optional[datetime] = None
    last_vlm_narration: str = ""
    last_vlm_ts: Optional[float] = None

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def n_vlm(self) -> int:
        return len(self.vlm_frames)

    @property
    def is_silent(self) -> bool:
        return self.n_turns == 0 and self.n_vlm == 0


class FeedbackHandler:
    """ループ型フィードバック常駐タスク。"""

    def __init__(self, llm_handler, conversation_cache=None):
        self.llm_handler = llm_handler
        self.conversation_cache = conversation_cache
        self.vlm_bridge = None
        self.rag_handler = None  # Phase 4.3.1: set_rag() で後付け注入される
        self.history_file = Config.HISTORY_FILE
        self.summary_file = Config.SUMMARY_FILE

        self.client = FeedbackLLMClient(
            model=Config.AI2_MODEL,
            api_key_openai=Config.OPENAI_API_KEY,
            api_key_anthropic=Config.ANTHROPIC_API_KEY,
            fallback_model=Config.AI2_FALLBACK_MODEL,
        )

        # ループ制御
        self.running = True
        self._running = False
        self._wake_event: Optional[asyncio.Event] = None
        self._cutoff_ts: Optional[datetime] = None
        self._last_run_monotonic = 0.0
        self._last_activity = "low"  # "high" / "mid" / "low"
        self._prev_feedback_head = ""
        # Phase 2.4: 前回 AI2 出力から抽出した「応答AIへの内省ノート」本文。
        # 次回ループで AI2 に「覚えておくこと」を継承させるため user_text に渡す。
        self._prev_ai1_note: str = ""
        self._start_time: Optional[datetime] = None
        # Phase 2: silence streak counter for the silent prompt
        self._silence_streak: int = 0

        # Phase 4.1: Top-down 予測ループ
        # _prev_predictions: 前回ループで LLM が出力した「次期間予測」を保持。
        #   次のループで「予測 vs 観測」比較ブロックの生成に使用。
        # _prediction_hold_count: Silent 期間中、消費せず保留している連続ループ数。
        #   FB_PREDICTION_MAX_HOLD を超えたら破棄 (古い予測の使い回し防止)。
        self._prev_predictions: Optional[dict] = None
        self._prediction_hold_count: int = 0

        # Phase 4.2: Precision Dynamics + Affective Inference
        # _precision_state: 現在の precision モード + EMA + calibration 集計
        # _last_evaluation_for_narrative: 因果叙述用の直前評価
        # _last_predictions_for_narrative: 因果叙述用の直前予測 (calibration push にも使う)
        self._precision_state = PrecisionState()
        self._precision_state.surprise_ema = Config.FB_PRECISION_INITIAL_EMA
        self._precision_state.weights = (
            Config.HS_WEIGHT_L0, Config.HS_WEIGHT_L1,
            Config.HS_WEIGHT_L2, Config.HS_WEIGHT_L3,
        )
        self._last_evaluation_for_narrative: Optional[dict] = None
        self._last_predictions_for_narrative: Optional[dict] = None
        self._precision_history_file = Config.PRECISION_HISTORY_FILE

        # Phase 4.3.3 (Batch 2): Goal Slot 二層
        # _goal_short_history: 直近 N 件の goal_short embedding (劣化検出用)
        # _goal_quality_warn: 前ループ検出フラグの文言。次ループ precision_block で参照
        # _boilerplate_embedding: lazy 初期化される (本検出 2 用)
        self._goal_short_history: deque = deque(maxlen=Config.FB_GOAL_HISTORY_SIZE)
        self._goal_quality_warn: Optional[str] = None
        # 前ループの goal_short 文字列。完全同一 (内省していない) を検出するための比較用
        self._last_goal_short_text: Optional[str] = None
        self._boilerplate_embedding: Optional[list] = None

        # Phase 4.3.4 (Batch 3): Self-Evaluation Bias Detection
        # 短期 deque (mode collapse + over-update 検出用)
        self._affect_history: deque = deque(maxlen=Config.FB_BIAS_WINDOW)
        self._goal_change_history: deque = deque(maxlen=Config.FB_BIAS_WINDOW)
        # 累計 Counter (positivity bias 検出用、long-term 傾向把握)
        # 起動時に memory_summary 末尾 N 件から復元される (_restore_valence_counter_from_summary)
        self._valence_counter: Counter = Counter()
        self._valence_total: int = 0
        # 検出結果を次ループの precision_block へ渡す (Phase 4.2 calibration / 4.3.3 goal_warn と同型)
        self._self_eval_bias_warn: Optional[str] = None

    # ==================================================================
    # Public API
    # ==================================================================
    def set_vlm_bridge(self, bridge) -> None:
        """VLMBridge を注入する。応答側と同じインスタンスが渡される想定。"""
        self.vlm_bridge = bridge

    def set_rag(self, rag_handler) -> None:
        """Phase 4.3.1: RAGHandler を後付け注入する。

        BaseMode.initialize から `feedback.set_rag(self.rag)` で呼ぶ想定。
        未注入時は episode 保存はスキップ + warn ログで運転継続 (グレースフル)。
        """
        self.rag_handler = rag_handler

    def signal_turn_done(self) -> None:
        """会話1ターン完了シグナル。ループを即ウェイクさせる。"""
        if self._wake_event is not None:
            self._wake_event.set()

    def signal_vlm_event(self) -> None:
        """VLMイベント (MAJOR nudge 等) でループを即ウェイクさせる。"""
        if self._wake_event is not None:
            self._wake_event.set()

    # 後方互換: 旧 API `process_pending()` を呼んでいる箇所が残っていても動くようにする
    async def process_pending(self) -> None:
        self.signal_turn_done()

    # ==================================================================
    # Loop
    # ==================================================================
    async def start_loop(self) -> None:
        """常駐ループ。BaseMode.initialize から asyncio.create_task で起動される。"""
        self._wake_event = asyncio.Event()
        self._running = True
        self._start_time = datetime.now()
        self._cutoff_ts = self._start_time  # 起動前履歴は無視

        # 前回セッションの最終フィードバック + 予測があれば即座に復元
        last_fb, last_preds = self._load_last_feedback()
        if last_fb:
            self._prev_feedback_head = last_fb[:400]
            # Phase 2.4: 前回内省ノート本文を復元 (次回ループで AI2 に継承を促す)
            self._prev_ai1_note = self._extract_ai1_note(last_fb) or ""
            self._inject_to_llm(last_fb)
        if last_preds:
            # Phase 4.1: 前回予測を復元 → 起動後最初のループで比較ブロックが現れる
            self._prev_predictions = last_preds
            logger.info(
                "[Feedback] restored prev_predictions from memory_summary: "
                "policy=%s, expected_activity=%s, confidence=%.2f",
                last_preds.get("policy", "?"),
                last_preds.get("expected_activity", "?"),
                last_preds.get("confidence", 0.0),
            )

        # Phase 4.2: precision_history.jsonl 末尾から PrecisionState を復元
        self._load_last_precision_state()

        # Phase 4.3.4 (Batch 3): positivity bias 検出のための valence 累計復元
        self._restore_valence_counter_from_summary()

        # 起動直後の暴発防止に短く待機
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            self._running = False
            return

        try:
            while self._running:
                timeout = self._compute_next_timeout()

                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                finally:
                    self._wake_event.clear()

                if not self._running:
                    break

                # 最小インターバル (ユーザー設定でデフォルト 0)
                elapsed = time.monotonic() - self._last_run_monotonic
                if elapsed < Config.FB_LOOP_MIN_INTERVAL_SEC:
                    await asyncio.sleep(Config.FB_LOOP_MIN_INTERVAL_SEC - elapsed)

                try:
                    await self._run_once()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception("Feedback loop iteration failed: %s", e)
                    await asyncio.sleep(3)
                finally:
                    self._last_run_monotonic = time.monotonic()
        finally:
            self._running = False
            self.running = False

    async def stop(self) -> None:
        """ループ停止要求。_feedback_task.cancel() の前に呼ぶ。"""
        self._running = False
        self.running = False
        if self._wake_event is not None:
            self._wake_event.set()

    # ==================================================================
    # Internal
    # ==================================================================
    def _compute_next_timeout(self) -> float:
        """直前の活性度に応じて次回 wait_for の timeout を決める。"""
        if self._last_activity == "high":
            # 高活性時は event 駆動。Event が来なくても低活性タイマーで保険
            return Config.FB_LOOP_LOW_ACTIVITY_SEC
        if self._last_activity == "mid":
            return Config.FB_LOOP_MID_ACTIVITY_SEC
        # 低活性
        return Config.FB_LOOP_LOW_ACTIVITY_SEC

    async def _run_once(self) -> None:
        """1ループ分のフィードバック生成。"""
        window_end = datetime.now()
        window = await self._collect_window(self._cutoff_ts, window_end)

        # Phase 4.2: hierarchical surprise を precision state の重みで計算
        # (FB_PRECISION_DYNAMIC_WEIGHTS=false の間は state.weights == baseline と同じ)
        hs = (
            compute_hierarchical_surprise(
                window.vlm_frames, weights=self._precision_state.weights,
            )
            if window.vlm_frames else None
        )

        # ── Phase 4.2 修正 #1 (核): affect → alpha 変調 ──────────────────
        # 直前 affect (前ループで保存) から effective_alpha を計算 (Joffily&Coricelli)
        effective_alpha = compute_alpha_from_affect(
            Config.FB_PRECISION_EMA_ALPHA,
            self._precision_state.last_affect,
            Config.FB_AFFECT_ALPHA_DELTA_MAX,
        )

        # ── Phase 4.2 修正 #5: silent EMA drift も update_surprise_ema 内で処理 ─
        # 前回ループの prediction_evaluation を反映して EMA 更新 (silent 期は drift)
        update_surprise_ema(
            self._precision_state,
            surprise_1_5=(self._last_evaluation_for_narrative or {}).get("surprise_1_5"),
            hierarchical_unified=(hs.unified if hs else None),
            is_silent=window.is_silent,
            silence_streak=self._silence_streak,
            alpha=effective_alpha,
        )

        # ── Phase 4.2 修正 #6: calibration push (前ループの confidence × surprise) ─
        if self._last_predictions_for_narrative and self._last_evaluation_for_narrative:
            self._precision_state.calibration.push(
                confidence=self._last_predictions_for_narrative.get("confidence"),
                surprise_1_5=self._last_evaluation_for_narrative.get("surprise_1_5"),
            )

        # mode 決定 (hysteresis 込み) → 機械調整適用
        # NOTE: silent_streak はまだインクリメント前。silent 判定後にインクリメントするため、
        # decide_mode に渡す値は「現在のループまでの連続沈黙数」を表現する
        # (window.is_silent なら +1 した値で判定するのが自然)
        effective_streak = self._silence_streak + (1 if window.is_silent else 0)
        new_mode = decide_mode(self._precision_state, effective_streak)
        transition_msg = transition_log(
            self._precision_state, new_mode, datetime.now().isoformat()
        )
        if transition_msg:
            logger.info("[Feedback][Precision] %s", transition_msg)
        base_weights = (
            Config.HS_WEIGHT_L0, Config.HS_WEIGHT_L1,
            Config.HS_WEIGHT_L2, Config.HS_WEIGHT_L3,
        )
        apply_mode_adjustments(self._precision_state, new_mode, base_weights)

        # プロンプト選択
        if window.is_silent:
            self._last_activity = "low"
            self._silence_streak += 1
            # Phase 4.1 (GPT 評価 #2-3): silent パスは予測検証スキップだが「保留」する。
            # policy=speak 高信頼度で予測 → 実は沈黙、という最も検証したい外れケースを
            # 失わないため _prev_predictions は消費せず保留。長すぎたら破棄。
            if self._prev_predictions is not None:
                self._prediction_hold_count += 1
                if self._prediction_hold_count > Config.FB_PREDICTION_MAX_HOLD:
                    logger.info(
                        "[Feedback] holding prediction expired after %d silent loops, "
                        "dropping (predicted_at=%s)",
                        self._prediction_hold_count,
                        self._prev_predictions.get("predicted_at", "?"),
                    )
                    self._prev_predictions = None
                    self._prediction_hold_count = 0
            system_blocks = SILENT_PROMPT_SYSTEM
            idle_sec = 0.0
            if self._cutoff_ts is not None:
                idle_sec = (window_end - self._cutoff_ts).total_seconds()
            last_user_ago = None
            if window.last_user_ts is not None:
                last_user_ago = (window_end - window.last_user_ts).total_seconds()
            last_vlm_ago = None
            if window.last_vlm_ts is not None:
                last_vlm_ago = time.time() - window.last_vlm_ts

            n_pairs = Config.FB_PRE_SILENCE_DIALOG_PAIRS
            pre_dialog = await self._format_pre_silence_dialog(n=n_pairs)
            silent_vlm_block = (
                self._format_vlm_block(window.vlm_frames) if window.vlm_frames else ""
            )
            # Phase 3.3: 沈黙時にも軽量版 action_context を渡す
            recent_pairs = await self._count_recent_dialog_pairs(minutes=5)
            action_ctx_silent = {
                "silence_seconds": idle_sec,
                "silence_streak": self._silence_streak,
                "recent_dialog_pairs_5min": recent_pairs,
                "hierarchical_unified": hs.unified if hs else 0.0,
            }
            user_text = build_silent_user_text(
                idle_seconds=idle_sec,
                silence_streak=self._silence_streak,
                n_pre_silence_pairs=n_pairs,
                pre_silence_dialog=pre_dialog,
                last_user_utterance=window.last_user_utterance,
                last_user_ago_sec=last_user_ago,
                last_vlm_narration=window.last_vlm_narration,
                last_vlm_ago_sec=last_vlm_ago,
                prev_feedback_head=self._prev_feedback_head,
                vlm_block=silent_vlm_block,
                action_context_lite=action_ctx_silent,
                prev_ai1_note=self._prev_ai1_note,  # Phase 2.4: 世代継承
            )
        else:
            self._last_activity = "high" if window.n_turns > 0 else "mid"
            self._silence_streak = 0
            system_blocks = FEP_PROMPT_SYSTEM
            dialog_log_text = self._format_dialog_log(window.turns)
            vlm_block = self._format_vlm_block(window.vlm_frames)
            # Phase 3.3: 通常時の補助統計
            action_ctx = await self._build_action_context(window, hs)
            # Phase 4.1: 前回予測との比較ブロック (客観 hint 込み)
            prev_prediction_block = self._format_prediction_comparison(
                self._prev_predictions, window, hs
            )
            # Phase 4.2: precision モード + Affective Inference 用因果叙述 + 過信警告
            precision_block = self._format_precision_block(
                self._prev_predictions,
                self._last_evaluation_for_narrative,
                self._silence_streak,
            )
            # 沈黙サマリ (recent_turns から「…」を除外したことに対応)
            silence_sec = 0.0
            ellipsis_cnt = 0
            if self.conversation_cache is not None:
                try:
                    sil = await self.conversation_cache.get_silence_summary()
                    silence_sec = float(sil.get("silence_seconds") or 0.0)
                    ellipsis_cnt = int(sil.get("ellipsis_count") or 0)
                except Exception as e:
                    logger.warning("[Feedback] silence_summary failed: %s", e)
            user_text = build_fep_user_text(
                window_start=(self._cutoff_ts or self._start_time or window_end).isoformat(timespec="seconds"),
                window_end=window_end.isoformat(timespec="seconds"),
                n_turns=window.n_turns,
                n_vlm_frames=window.n_vlm,
                prev_feedback_head=self._prev_feedback_head,
                dialog_log_text=dialog_log_text,
                vlm_block=vlm_block,
                last_user_utterance=window.last_user_utterance,
                action_context=action_ctx,
                prev_prediction_block=prev_prediction_block,
                precision_block=precision_block,
                silence_seconds=silence_sec,
                ellipsis_count=ellipsis_cnt,
                prev_ai1_note=self._prev_ai1_note,  # Phase 2.4: 世代継承
            )

        # Phase 2/3: surface visual surprise proxy stats + hierarchical + action_ctx in the loop log
        proxy_summary = self._summarize_vlm_proxy(window.vlm_frames)
        hs_log = ""
        if hs:
            hs_log = (
                f", hierarchical: L0={hs.l0_global:.2f} L1={hs.l1_pixel:.2f} "
                f"L2={hs.l2_entity:.2f} L3={hs.l3_relation:.2f} unified={hs.unified:.2f}"
            )
        logger.info(
            "[Feedback] window=%d turns + %d vlm frames, silent=%s, "
            "silence_streak=%d, model=%s%s%s",
            window.n_turns, window.n_vlm, window.is_silent,
            self._silence_streak, Config.AI2_MODEL,
            f", {proxy_summary}" if proxy_summary else "",
            hs_log,
        )

        t0 = time.monotonic()
        try:
            result = await self.client.complete(
                system_blocks=system_blocks,
                user_text=user_text,
                max_tokens=Config.AI2_MAX_TOKENS,
                temperature=Config.AI2_TEMPERATURE,
            )
        except Exception as e:
            logger.error("[Feedback] LLM call failed after all fallbacks: %s", e)
            # cutoff は進めずに次ループで再挑戦
            return

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[Feedback] generated %d chars in %d ms", len(result or ""), elapsed_ms)

        if not result:
            return

        # Phase 4.1: 出力から新予測 + 前回予測に対する評価を抽出 (silent 時は両方スキップ)
        new_predictions: Optional[dict] = None
        new_evaluation: Optional[dict] = None
        new_affect: Optional[dict] = None
        if not window.is_silent:
            new_predictions = self._extract_predictions(result)
            # 前回予測があった場合のみ、今回の評価を抽出 (検証ループの輪を閉じる)
            if self._prev_predictions is not None:
                new_evaluation = self._extract_evaluation(result)
                if new_evaluation:
                    new_evaluation["evaluated_predictions_predicted_at"] = (
                        self._prev_predictions.get("predicted_at")
                    )
            # Phase 4.2: affect 抽出 + state 反映 (次ループで alpha 変調に使う)
            new_affect = self._extract_affect(result)
            if new_affect:
                self._precision_state.last_affect = new_affect
                logger.info(
                    "[Feedback][Precision] affect: %s (intensity=%.2f, valence=%s) "
                    "→ next-loop alpha will be modulated",
                    new_affect.get("affect", "?"),
                    new_affect.get("intensity", 0.0),
                    new_affect.get("valence", "?"),
                )
                # Phase 4.3.4 (Batch 3): mode collapse / positivity bias 検出のため履歴更新
                aff_label = new_affect.get("affect")
                if aff_label:
                    self._affect_history.append(aff_label)
                val = new_affect.get("valence")
                if val in {"pos", "neu", "neg"}:
                    self._valence_counter[val] += 1
                    self._valence_total += 1
            # 次ループの narrative + calibration push 用に保存
            self._last_evaluation_for_narrative = new_evaluation
            self._last_predictions_for_narrative = new_predictions

        # 保存 + 応答LLMへ注入 (Phase 4.2: 永続化を 2 系統に分離)
        await self._persist(result, window_end, new_predictions, new_evaluation, new_affect)
        await self._persist_precision_history(window_end, transition_msg)

        # Phase 4.3.1: Episode Summary を抽出して RAG に保存 (silent はスキップ)
        # dedupe 責務は RAGHandler.add_episode() 内蔵 (cosine > FB_RAG_DEDUP_THRESHOLD で skip)
        # source_turn_range は window_start/window_end タイムスタンプ (turn id 無いため)
        if not window.is_silent and self.rag_handler is not None:
            episode = self._extract_episode(result)
            if episode:
                window_start_iso = (
                    self._cutoff_ts or self._start_time or window_end
                ).isoformat(timespec="seconds")
                window_end_iso = window_end.isoformat(timespec="seconds")
                source_range = f"[{window_start_iso}, {window_end_iso}]"
                try:
                    saved = await self.rag_handler.add_episode(
                        summary=episode["summary"],
                        topic_tags=episode["topic_tags"],
                        importance=episode["importance"],
                        source_turn_range=source_range,
                        timestamp=window_end_iso,
                    )
                    if saved:
                        logger.info(
                            "[Feedback][RAG] episode saved: importance=%.2f, "
                            "tags=%s",
                            episode["importance"], episode["topic_tags"],
                        )
                    # saved=False は dedupe スキップ。add_episode 内で log 済み。
                except Exception as e:
                    logger.warning("[Feedback][RAG] add_episode failed: %s", e)
        elif not window.is_silent and self.rag_handler is None:
            logger.warning(
                "[Feedback][RAG] rag_handler not injected; episode skipped. "
                "Did BaseMode.initialize call set_rag()?"
            )

        # Phase 4.3.3 (Batch 2): Goal Slot 抽出 + 劣化検証 + SoT 更新 + RAG 保存
        # silent 期間でも抽出する (silent プロンプトにも goal セクションあり)
        new_goal = self._extract_goal(result)
        goal_warn: Optional[str] = None
        if new_goal:
            # 1) 劣化検証 (async, 例外は飲み込んで運転継続)
            try:
                goal_warn = await self._check_goal_quality(
                    new_goal.get("goal_short", "")
                )
            except Exception as e:
                logger.warning("[Feedback][Goal] quality check failed: %s", e)
            # 1.5) 完全同一の goal_short が連続したら劣化警告に追記 (内省していないシグナル)
            #      keep 自体は許容するが、文字列を一切動かさない keep は形骸化の指標
            if (
                self._last_goal_short_text
                and new_goal.get("goal_short") == self._last_goal_short_text
            ):
                suffix = "前回と完全同一の goal_short (内省していない可能性)"
                goal_warn = f"{goal_warn} | {suffix}" if goal_warn else suffix
            # 次ループ比較用に記録 (extraction 成功時のみ)
            self._last_goal_short_text = new_goal.get("goal_short")
            # 2) SoT 更新: in-memory + goal.txt atomic write (set_goal_slot 内)
            if hasattr(self.llm_handler, "set_goal_slot"):
                try:
                    self.llm_handler.set_goal_slot(new_goal)
                except Exception as e:
                    logger.warning("[Feedback][Goal] set_goal_slot failed: %s", e)
            # 3) update/pivot 時のみ RAG へ追加 (rag.add_goal は Batch 1 で実装済み)
            if (
                new_goal.get("goal_change") in {"update", "pivot"}
                and self.rag_handler is not None
            ):
                try:
                    await self.rag_handler.add_goal(
                        goal_short=new_goal["goal_short"],
                        goal_long=new_goal.get("goal_long", "維持"),
                        goal_change=new_goal["goal_change"],
                        importance=0.7,
                        timestamp=window_end.isoformat(),
                    )
                except Exception as e:
                    logger.warning("[Feedback][Goal] add_goal failed: %s", e)
            # Phase 4.3.4 (Batch 3): over-update 検出のため履歴更新
            change = new_goal.get("goal_change")
            if change in {"keep", "update", "pivot"}:
                self._goal_change_history.append(change)
            logger.info(
                "[Feedback][Goal] extracted: change=%s, short=%s%s",
                new_goal.get("goal_change", "?"),
                (new_goal.get("goal_short", "?") or "?")[:40],
                f", warn={goal_warn}" if goal_warn else "",
            )
        # 次ループの precision_block 用に warn を保持
        self._goal_quality_warn = goal_warn

        # Phase 4.3.4 (Batch 3): self-eval bias 検出 (3 種統合警告)
        # 例外飲み込み: 検出ループは feedback の本流ではないので運転継続を優先
        try:
            self._self_eval_bias_warn = self._check_self_eval_bias()
            if self._self_eval_bias_warn:
                logger.info(
                    "[Feedback][Bias] detected: %s",
                    self._self_eval_bias_warn.replace("\n", " | "),
                )
        except Exception as e:
            logger.warning("[Feedback][Bias] check failed: %s", e)
            self._self_eval_bias_warn = None

        self._prev_feedback_head = result[:400]
        # Phase 2.4: 次回ループ用に AI1 内省ノートも保持 (世代継承)
        self._prev_ai1_note = self._extract_ai1_note(result) or ""
        # Phase 4.1: 状態更新 — silent パスは _prev_predictions を消費せず保留 (上で処理済み)。
        # 非 silent パスでは新予測に置き換え + 保留カウンタリセット。
        if not window.is_silent:
            self._prev_predictions = new_predictions
            self._prediction_hold_count = 0
        self._inject_to_llm(result)

        # cutoff 更新
        self._cutoff_ts = window_end

        # Phase 4.1 ログ
        if new_predictions:
            logger.info(
                "[Feedback] new predictions: confidence=%.2f, policy=%s, "
                "expected_activity=%s",
                new_predictions.get("confidence", 0.0),
                new_predictions.get("policy", "?"),
                new_predictions.get("expected_activity", "?"),
            )
        if new_evaluation:
            logger.info(
                "[Feedback] prediction_evaluation: surprise=%s, user_hit=%s, "
                "screen_hit=%s, policy_hit=%s",
                new_evaluation.get("surprise_1_5"),
                new_evaluation.get("user_hit"),
                new_evaluation.get("screen_hit"),
                new_evaluation.get("policy_hit"),
            )

    # ------------------------------------------------------------------
    # Window collection
    # ------------------------------------------------------------------
    async def _collect_window(
        self,
        cutoff_ts: Optional[datetime],
        window_end: datetime,
    ) -> _Window:
        """cutoff_ts より後 〜 window_end までの対話と VLM フレームを収集。"""
        window = _Window(start_ts=cutoff_ts, end_ts=window_end)

        # --- 対話履歴 ---
        conversations = await self._load_conversation_entries()
        if conversations:
            cutoff_iso = cutoff_ts.isoformat() if cutoff_ts else None
            end_iso = window_end.isoformat()
            entries = [
                e for e in conversations
                if e.get("timestamp") and e["timestamp"] > (cutoff_iso or "")
                and e["timestamp"] <= end_iso
            ]
            window.turns = entries
            # 最後の user 発話を抽出
            for e in reversed(entries):
                if "user" in e and e.get("user"):
                    window.last_user_utterance = e["user"]
                    try:
                        window.last_user_ts = datetime.fromisoformat(e["timestamp"])
                    except Exception:
                        window.last_user_ts = None
                    break

            # ウィンドウ内に user が無かった場合は、全履歴から最新 user 発話を推定
            if not window.last_user_utterance:
                for e in reversed(conversations):
                    if "user" in e and e.get("user"):
                        window.last_user_utterance = e["user"]
                        try:
                            window.last_user_ts = datetime.fromisoformat(e["timestamp"])
                        except Exception:
                            window.last_user_ts = None
                        break

        # --- VLM フレーム ---
        if Config.FB_LOOP_ENABLE_VLM and self.vlm_bridge is not None:
            try:
                buf = getattr(self.vlm_bridge, "vision_buffer", None)
                if buf is not None:
                    cutoff_epoch = cutoff_ts.timestamp() if cutoff_ts else 0.0
                    frames = buf.get_frames_since(cutoff_epoch)
                    window.vlm_frames = frames
                    # 最新の narration
                    if frames:
                        latest = frames[-1]
                        window.last_vlm_narration = latest.narration
                        window.last_vlm_ts = latest.timestamp
                    else:
                        # ウィンドウ外でも直近観察があればフォールバック
                        latest = buf.get_latest()
                        if latest is not None:
                            window.last_vlm_narration = latest.narration
                            window.last_vlm_ts = latest.timestamp
            except Exception as e:
                logger.warning("[Feedback] VLM collect failed: %s", e)

        return window

    async def _load_conversation_entries(self) -> list[dict]:
        """conversation_history.txt を JSON 行形式で読み込む (非同期)。"""
        if not os.path.exists(self.history_file):
            return []

        def _load() -> list[dict]:
            out: list[dict] = []
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            # 旧テキスト形式との互換
                            if line.startswith("User: "):
                                out.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "user": line[len("User: "):].strip(),
                                })
                            elif line.startswith("AI: "):
                                out.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "ai": line[len("AI: "):].strip(),
                                })
            except Exception as e:
                logger.warning("[Feedback] history load failed: %s", e)
            return out

        return await asyncio.to_thread(_load)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_dialog_log(entries: list[dict]) -> str:
        """対話ログを整形。

        Phase 2.1: 内部見守り発話「…」のエントリは AI2 に見せない。
        沈黙の事実は build_fep_user_text の【沈黙状況】ブロックで別途渡されるため、
        ログ本文に「…」を残すと LLM が「直前は…と返した」と誤分析する原因になる。
        """
        if not entries:
            return ""
        lines: list[str] = []
        for e in entries:
            user_text = e.get("user", "")
            ai_text = e.get("ai", "")
            # 「…」は機械沈黙イベントなのでログから除外
            if "user" in e and user_text.strip() == "…":
                continue
            if "ai" in e and ai_text.strip() == "…":
                continue
            ts = e.get("timestamp", "")
            try:
                t = datetime.fromisoformat(ts).strftime("%H:%M:%S") if ts else ""
            except Exception:
                t = ""
            if "user" in e and user_text:
                prefix = f"[{t}] " if t else ""
                lines.append(f"{prefix}User: {user_text}")
            elif "ai" in e and ai_text:
                prefix = f"[{t}] " if t else ""
                lines.append(f"{prefix}AI: {ai_text}")
        return "\n".join(lines)

    @staticmethod
    def _classify_pe_score(magnitude: float) -> int:
        """visual surprise proxy (0-255) を 1〜5 のスコアに分類する。

        境界値は Config.PE_LEVEL_*_MAX で可変。これは画素差分ベースの近似であり、
        神経科学的な階層的予測誤差そのものではない。
        """
        if magnitude < Config.PE_LEVEL_1_MAX:
            return 1
        if magnitude < Config.PE_LEVEL_2_MAX:
            return 2
        if magnitude < Config.PE_LEVEL_3_MAX:
            return 3
        if magnitude < Config.PE_LEVEL_4_MAX:
            return 4
        return 5

    @staticmethod
    def _select_key_frames(frames: list) -> list:
        """期間内の代表 key frames を選出 (重複除去付き)。

        - 0 frames: []
        - 1 frame: [(label, frame)] スナップショット 1 件のみ
        - 2+ frames: 期間冒頭 / サプライズ最大時 / 期間末尾 の 3 点を id() で重複除去
        """
        metaful = [f for f in frames if f.meta is not None]
        if not metaful:
            return []
        if len(metaful) == 1:
            return [("期間スナップショット (1 ナレーション)", metaful[0])]

        first = metaful[0]
        last = metaful[-1]
        peak = max(metaful, key=lambda f: f.meta.change_magnitude_max)

        selected: list = []
        seen: set = set()
        for label, frame in [
            ("期間冒頭", first),
            ("サプライズ最大時", peak),
            ("期間末尾", last),
        ]:
            if id(frame) in seen:
                continue
            seen.add(id(frame))
            selected.append((label, frame))
        return selected

    @classmethod
    def _summarize_vlm_proxy(cls, frames: list) -> str:
        """visual surprise proxy の 1 行サマリ (ログ用)。Phase 3.1: saliency も含める。"""
        metaful = [f for f in frames if f.meta is not None]
        if not metaful:
            return ""
        avgs = [f.meta.change_magnitude_avg for f in metaful]
        maxs = [f.meta.change_magnitude_max for f in metaful]
        n_regs = [f.meta.n_regions_max for f in metaful]
        ratios = [f.meta.change_area_ratio_max for f in metaful]
        period_avg = sum(avgs) / len(avgs)
        period_max = max(maxs)
        # Phase 3.1: saliency-weighted も含める (v2 で値が入っている前提)
        sw_avgs = [f.meta.saliency_weighted_avg for f in metaful if f.meta.saliency_weighted_avg > 0]
        sw_maxs = [f.meta.saliency_weighted_max for f in metaful if f.meta.saliency_weighted_max > 0]
        movers_total = sum(f.meta.n_movers for f in metaful)
        sw_part = ""
        if sw_avgs or sw_maxs:
            sw_avg = sum(sw_avgs) / len(sw_avgs) if sw_avgs else 0.0
            sw_max = max(sw_maxs) if sw_maxs else 0.0
            sw_part = (
                f", saliency-weighted: avg={sw_avg:.1f}/255 ({cls._classify_pe_score(sw_avg)}/5), "
                f"max={sw_max:.1f}/255 ({cls._classify_pe_score(sw_max)}/5), "
                f"movers={movers_total}"
            )
        return (
            f"surprise proxy: avg={period_avg:.1f}/255 ({cls._classify_pe_score(period_avg)}/5), "
            f"max={period_max:.1f}/255 ({cls._classify_pe_score(period_max)}/5), "
            f"n_regions_max={max(n_regs)}, area_ratio={max(ratios)*100:.1f}%"
            f"{sw_part}"
        )

    @classmethod
    def _format_vlm_block(cls, frames: list) -> str:
        """VisionFrame list を FEP-inspired feedback 用に整形する。

        Phase 2 + Phase 3:
        - visual surprise proxy (面積重み)
        - saliency-weighted surprise proxy (Phase 3.1, change_score>0 限定)
        - Top Salient Regions (Phase 3.1, 静的領域は [STATIC] マーク)
        - Hierarchical Surprise Observation L0/L1/L2/L3/Unified (Phase 3.2)
        - ナレーション系列 (上限50)
        - key frames (期間冒頭/最大/末尾の3点を重複除去) の relations / memory / entity_delta
        """
        if not frames:
            return ""

        capped = frames
        if len(capped) > 50:
            priority = [f for f in capped if f.change_tag in ("major", "moderate")]
            if len(priority) >= 30:
                capped = priority[-50:]
            else:
                capped = capped[-50:]

        # Surprise proxy 集約 (Phase 2)
        metaful = [f for f in frames if f.meta is not None]
        avgs = [f.meta.change_magnitude_avg for f in metaful]
        maxs = [f.meta.change_magnitude_max for f in metaful]
        n_regs = [f.meta.n_regions_max for f in metaful]
        ratios = [f.meta.change_area_ratio_max for f in metaful]
        period_avg = sum(avgs) / len(avgs) if avgs else 0.0
        period_max = max(maxs) if maxs else 0.0
        n_regs_max = max(n_regs) if n_regs else 0
        area_ratio_max = max(ratios) if ratios else 0.0
        pe_avg_score = cls._classify_pe_score(period_avg)
        pe_max_score = cls._classify_pe_score(period_max)

        # Saliency-weighted 集約 (Phase 3.1)
        sw_avg_values = [
            f.meta.saliency_weighted_avg for f in metaful
            if f.meta.saliency_weighted_avg > 0
        ]
        sw_max_values = [
            f.meta.saliency_weighted_max for f in metaful
            if f.meta.saliency_weighted_max > 0
        ]
        sw_avg = sum(sw_avg_values) / len(sw_avg_values) if sw_avg_values else 0.0
        sw_max = max(sw_max_values) if sw_max_values else 0.0
        movers_total = sum(f.meta.n_movers for f in metaful)
        # Top Salient Regions: 全 frame の top_salient_regions を flatten して
        # combined_score 降順で top 5
        all_top: list = []
        for f in metaful:
            if f.meta.top_salient_regions:
                all_top.extend(f.meta.top_salient_regions)
        all_top_sorted = sorted(
            all_top, key=lambda r: r.get("combined", 0.0), reverse=True
        )[:5]

        lines: list[str] = [
            f"# 視覚観察期間 ({len(frames)} ナレーション)",
            "## Visual Surprise Proxy (画素差分ベース, 面積重み)",
            "※ これは visual surprise proxy であり、神経科学的な階層的予測誤差そのものではない。",
            "  画素レベルの近似値として、FEP-inspired feedback の入力に使う。",
            f"- 期間平均: {period_avg:.1f}/255 (スコア {pe_avg_score}/5)",
            f"- 期間最大: {period_max:.1f}/255 (スコア {pe_max_score}/5)",
            f"- 検出領域数 (期間ピーク): {n_regs_max}",
            f"- 画面に対する変化面積比 (期間ピーク): {area_ratio_max*100:.1f}%",
            "",
            "## Saliency-Weighted Surprise Proxy (注意重み, Phase 3.1)",
            "※ 面積×saliency 重み付き。change_score>0 の region のみで計算 (静的 salient-only は除外)。",
            "  precision-weighted prediction error の工学的近似 (saliency-aware proxy)。",
            "  Saliency は Hou&Zhang Spectral Residual の bottom-up proto-object 検出であり、",
            "  semantic importance ではない点に留意。",
            f"- 期間平均 (saliency 重み): {sw_avg:.1f}/255 (スコア {cls._classify_pe_score(sw_avg)}/5)",
            f"- 期間最大 (saliency 重み): {sw_max:.1f}/255 (スコア {cls._classify_pe_score(sw_max)}/5)",
            f"- 動いた領域数 (movers): {movers_total}",
            "- 単純面積平均と乖離が大きい場合、目立たない背景揺らぎで proxy が膨らんでいる可能性。",
        ]

        # Top Salient Regions
        if all_top_sorted:
            lines.append("")
            lines.append("## Top Salient Regions (注意の的, combined_score 降順)")
            lines.append("※ [STATIC] は変化していないが目立つ領域 (例: UI ロゴ・ステータス表示)。")
            for r in all_top_sorted:
                bbox = r.get("bbox", [0, 0, 0, 0])
                static_mark = " [STATIC]" if r.get("is_static") else ""
                lines.append(
                    f"- bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}){static_mark} "
                    f"saliency={r.get('saliency', 0):.2f} "
                    f"change={r.get('change', 0):.2f} "
                    f"combined={r.get('combined', 0):.2f}"
                )

        # Hierarchical Surprise Observation (Phase 3.2)
        hs = compute_hierarchical_surprise(frames)
        lines.append("")
        lines.append("## Hierarchical Surprise Observation (層別観測, Phase 3.2)")
        lines.append("※ bottom-up 観測のみ。真の階層的予測符号化 (top-down 予測 + bottom-up 誤差) ではない。")
        lines.append("  各階層で surprise が出た位置を識別することで、対応する内部モデルを更新する手がかり。")
        lines.append(f"- L0 Global   : {hs.l0_global:.2f} (ChangeLevel 期間最大)")
        lines.append(f"- L1 Pixel    : {hs.l1_pixel:.2f} (saliency-weighted proxy)")
        lines.append(
            f"- L2 Entity   : {hs.l2_entity:.2f}  "
            f"(new={hs.n_entity_new} lost={hs.n_entity_lost} changed={hs.n_entity_changed})"
        )
        lines.append(
            f"- L3 Relation : {hs.l3_relation:.2f}  "
            f"(rel_delta={hs.n_relation_delta} episodes={hs.n_episodes})"
        )
        lines.append(f"- Unified     : {hs.unified:.2f} (0.4*L1+0.3*L2+0.2*L3+0.1*L0)")

        lines.append("")
        lines.append("## ナレーション系列")
        for f in capped:
            try:
                t = datetime.fromtimestamp(f.timestamp).strftime("%H:%M:%S")
            except Exception:
                t = ""
            tag = (f.change_tag or "none").upper()
            narration = (f.narration or "").replace("\n", " ")
            prefix = f"[{t} | {tag}] " if t else f"[{tag}] "
            lines.append(prefix + narration)

        # Key frames (重複除去済み)
        for label, frame in cls._select_key_frames(frames):
            m = frame.meta
            if m is None:
                continue
            try:
                t = datetime.fromtimestamp(frame.timestamp).strftime("%H:%M:%S")
            except Exception:
                t = ""
            lines.append("")
            lines.append(
                f"## [{label}] {t} "
                f"(surprise: {m.change_magnitude_max:.0f}/255, regions={m.n_regions_max})"
            )
            if m.relations_text:
                lines.append(f"### 空間関係 (SceneGraph)\n{m.relations_text}")
            if m.memory_text:
                lines.append(f"### エピソード記憶 (WorkingMemory)\n{m.memory_text}")
            if m.entity_delta_text:
                etext = m.entity_delta_text
                if len(etext) > 800:
                    etext = etext[:800] + "…"
                lines.append(f"### エンティティ変化 (FrameDelta)\n{etext}")

        return "\n".join(lines)

    async def _format_pre_silence_dialog(self, n: int = 4) -> str:
        """conversation_history.txt 末尾から user/AI ペアを最大 n 組取って整形する。

        無言プロンプト用に、沈黙に入る直前の対話を文脈として注入するためのもの。

        Phase 2.2: 「…」だけのペア (user=='…' かつ ai=='…' or ai 不在) はスキップし、
        n は実会話ペアでカウントする。これにより長期沈黙時に AI2 が真の
        「沈黙開始前の最後の実会話 n ペア」を読めるようにする。
        """
        if n <= 0:
            return ""
        entries = await self._load_conversation_entries()
        if not entries:
            return ""

        pairs: list = []
        i = len(entries) - 1
        while i >= 0 and len(pairs) < n:
            ai_entry = None
            user_entry = None
            if "ai" in entries[i]:
                ai_entry = entries[i]
                if i - 1 >= 0 and "user" in entries[i - 1]:
                    user_entry = entries[i - 1]
                    i -= 2
                else:
                    i -= 1
            elif "user" in entries[i]:
                user_entry = entries[i]
                i -= 1
            else:
                i -= 1
                continue
            # 「…」だけのペアはスキップ (沈黙イベントは別経路で AI2 に渡る)
            is_ellipsis_user = (
                user_entry is not None
                and user_entry.get("user", "").strip() == "…"
            )
            is_ellipsis_ai = (
                ai_entry is not None
                and ai_entry.get("ai", "").strip() == "…"
            )
            if is_ellipsis_user and (ai_entry is None or is_ellipsis_ai):
                continue
            if user_entry or ai_entry:
                pairs.append((user_entry, ai_entry))
        pairs.reverse()

        out: list[str] = []
        for u, a in pairs:
            if u and u.get("user"):
                out.append(f"User: {u['user']}")
            if a and a.get("ai"):
                out.append(f"AI: {a['ai']}")
        return "\n".join(out)

    # ------------------------------------------------------------------
    # Phase 3.3: LLM-assisted EFE — 補助統計のみ提供
    # ------------------------------------------------------------------
    async def _build_action_context(
        self,
        window: "_Window",
        hs: Optional[HierarchicalSurprise],
    ) -> dict:
        """LLM が行動を選択するための補助統計を収集する。

        Python は機械的な行動選択を行わない。あくまで LLM の文脈解釈を
        支援する「客観的指標」を提供。LLM はこれを参照しても無視してもよい。
        """
        now = datetime.now()
        silence_sec: Optional[float] = None
        if window.last_user_ts is not None:
            silence_sec = (now - window.last_user_ts).total_seconds()

        recent_pair_count = await self._count_recent_dialog_pairs(minutes=5)
        proxy_summary = self._summarize_vlm_proxy(window.vlm_frames)
        rough_user_state = self._infer_rough_user_state(
            silence_sec, recent_pair_count, hs
        )

        return {
            "silence_seconds": silence_sec,
            "silence_streak": self._silence_streak,
            "n_turns_in_window": window.n_turns,
            "recent_dialog_pairs_5min": recent_pair_count,
            "vlm_proxy_summary": proxy_summary,
            "hierarchical_unified": hs.unified if hs else 0.0,
            "rough_user_state_hint": rough_user_state,
        }

    async def _count_recent_dialog_pairs(self, minutes: int = 5) -> int:
        """直近 N 分の user 発話数を数える。

        厳密な user-AI ペアではなく user 発話の出現回数を返す (簡易)。
        LLM の補助統計として「対話の頻度感」が分かれば十分。
        """
        entries = await self._load_conversation_entries()
        if not entries:
            return 0
        threshold = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        return sum(
            1 for e in entries
            if e.get("timestamp", "") > threshold and "user" in e
        )

    @staticmethod
    def _infer_rough_user_state(
        silence_sec: Optional[float],
        recent_pairs: int,
        hs: Optional[HierarchicalSurprise],
    ) -> str:
        """heuristic でユーザー状態を仮推定する。LLM が再解釈する前提。

        Returns: "active" | "busy" | "idle" | "unknown"
        """
        if silence_sec is None:
            return "unknown"
        # 沈黙が長め + VLM で動きあり → 作業中
        if silence_sec > 60 and hs and hs.unified > 0.3:
            return "busy"
        # 長時間沈黙 + VLM 静止 → 離席/休憩
        if silence_sec > 120 and (not hs or hs.unified < 0.1):
            return "idle"
        # 直近対話が活発 → 会話中
        if recent_pairs >= 3:
            return "active"
        return "unknown"

    # ------------------------------------------------------------------
    # Phase 4.1: Top-down 予測ループ
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_predictions(feedback_text: str) -> Optional[dict]:
        """フィードバック末尾の「次期間予測」セクションから 6 タグを抽出する。

        失敗時 (フォーマット崩れ・セクション欠如) は None を返してグレースフル運転継続。
        enum 値 (policy / expected_activity) は不正値を "other" に正規化。
        """
        m = _PRED_SECTION_RE.search(feedback_text)
        if not m:
            return None
        section = m.group(0)
        # 「予測評価」見出しが同一テキスト内に出てきた場合、そこで切る (二重抽出回避)
        cut = _EVAL_SECTION_RE.search(section)
        if cut:
            section = section[:cut.start()]
        extracted: dict = {}
        for tag, value in _PRED_TAG_RE.findall(section):
            key = tag.lower()
            if key not in extracted:  # 重複ヒットは最初を優先
                extracted[key] = value.strip()
        if not extracted:
            return None
        # confidence を float に
        try:
            extracted["confidence"] = float(extracted.get("confidence", "0.5"))
        except (ValueError, TypeError):
            extracted["confidence"] = 0.5
        # enum 値の正規化
        policy_raw = str(extracted.get("policy", "")).lower().strip()
        extracted["policy"] = policy_raw if policy_raw in _VALID_POLICIES else "other"
        act_raw = str(extracted.get("expected_activity", "")).lower().strip()
        extracted["expected_activity"] = (
            act_raw if act_raw in _VALID_ACTIVITIES else "other"
        )
        extracted["predicted_at"] = datetime.now().isoformat()
        return extracted

    @staticmethod
    def _extract_evaluation(feedback_text: str) -> Optional[dict]:
        """LLM 出力の「予測評価」(prediction_evaluation) セクションから抽出。

        日本語見出し / 英語タグ両対応 (GPT 評価 #2-4 反映)。
        数値タグは float / int に変換。失敗時 None。
        """
        m = _EVAL_SECTION_RE.search(feedback_text)
        if not m:
            return None
        section = m.group(0)
        extracted: dict = {}
        for tag, value in _EVAL_TAG_RE.findall(section):
            key = tag.lower()
            if key not in extracted:
                extracted[key] = value.strip()
        if not extracted:
            return None
        for k in ("user_hit", "screen_hit", "policy_hit"):
            if k in extracted:
                try:
                    extracted[k] = float(extracted[k])
                except (ValueError, TypeError):
                    extracted[k] = None
        if "surprise_1_5" in extracted:
            try:
                extracted["surprise_1_5"] = int(float(extracted["surprise_1_5"]))
            except (ValueError, TypeError):
                extracted["surprise_1_5"] = None
        return extracted

    @staticmethod
    def _count_ai_turns(window: "_Window") -> int:
        """期間内の AI 発話数を数える (User 発話を含めない)。

        GPT 評価 #2-2 反映: window.turns は user/AI 両方を含むため、
        policy 検証 (Eve が静観/発話を選んだか) には AI エントリ数のみ使う。
        """
        n = 0
        for e in window.turns:
            if isinstance(e, dict) and "ai" in e and e.get("ai"):
                n += 1
        return n

    @classmethod
    def _compute_objective_hints(
        cls,
        prev: Optional[dict],
        window: "_Window",
        hs: Optional[HierarchicalSurprise],
    ) -> list[str]:
        """Python 側で粗い客観指標を計算し、LLM の判断を支援する hint リスト。

        DeepMind "LLMs Cannot Self-Correct Reasoning Yet" の警告 (外部信号無しの
        自己反省は悪化し得る) への対策として、外部観測 (VLM/対話ログ) との
        粗い照合結果を hint として LLM に提示する。
        """
        if not prev:
            return []
        hints: list[str] = []
        n_ai_turns = cls._count_ai_turns(window)

        # screen mismatch: 沈黙予測なのに画面変化あり
        if prev.get("expected_activity") == "silent" and window.n_vlm > 0:
            if hs and hs.unified > 0.3:
                hints.append(
                    "screen mismatch: 「silent」予測だが期間内 VLM フレーム "
                    f"{window.n_vlm} 件、unified={hs.unified:.2f} で動きあり"
                )
        # policy=watch (静観) 予測なのに AI が発話 → 自己一貫性破綻
        if prev.get("policy") == "watch" and n_ai_turns > 0:
            hints.append(
                f"policy mismatch: 「watch」予測だが AI 発話数 {n_ai_turns} "
                "(自分が静観できていない)"
            )
        # policy=speak (発話) 予測なのに AI 発話ゼロ
        if prev.get("policy") == "speak" and n_ai_turns == 0:
            hints.append(
                "policy mismatch: 「speak」予測だが AI 発話数 0 (沈黙が続いた)"
            )
        # confidence high (>0.8) なのに大きな surprise
        conf = prev.get("confidence", 0.5)
        if isinstance(conf, (int, float)) and conf > 0.8 and hs and hs.unified > 0.5:
            hints.append(
                f"calibration warning: 高信頼度 ({conf:.2f}) で予測したが "
                f"hierarchical_unified={hs.unified:.2f} で大きな変化あり"
            )
        return hints

    # ------------------------------------------------------------------
    # Phase 4.3.1: Episode Summary 抽出 (RAG 入力用)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_episode(feedback_text: str) -> Optional[dict]:
        """LLM 出力の「■ エピソード要約」セクションから 3 タグ抽出 (Phase 4.3.1)。

        topic_tags は CSV → list 正規化、importance は float clamp [0, 1]、fallback 0.5。
        失敗時 (フォーマット崩れ・セクション欠如) は None を返してグレースフル運転継続。
        """
        m = _EPISODE_SECTION_RE.search(feedback_text)
        if not m:
            return None
        section = m.group(0)
        # 後続セクション (■ 〜) で切る (自身の見出しを skip して次の ■ を探す)
        cut = re.search(r"\n■", section[1:])
        if cut:
            section = section[: 1 + cut.start()]
        extracted: dict = {}
        for tag, value in _EPISODE_TAG_RE.findall(section):
            key = tag.lower()
            if key not in extracted:
                extracted[key] = value.strip()
        if not extracted or "summary" not in extracted:
            return None
        # topic_tags を CSV → list 正規化
        tags_raw = extracted.get("topic_tags", "")
        if tags_raw:
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            # 引用符・括弧除去
            tags = [t.strip("\"'`[](){}") for t in tags if t.strip("\"'`[](){}")]
            extracted["topic_tags"] = tags[:6]  # 上限 6 個
        else:
            extracted["topic_tags"] = []
        # importance float clamp
        try:
            v = float(extracted.get("importance", 0.5))
            extracted["importance"] = max(0.0, min(1.0, v))
        except (ValueError, TypeError):
            extracted["importance"] = 0.5
        return extracted

    # ------------------------------------------------------------------
    # Phase 4.3.3 (Batch 2): Goal Slot 抽出 + 劣化検証
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_goal(feedback_text: str) -> Optional[dict]:
        """LLM 出力の「■ 一貫した行動目標」セクションから 3 タグ抽出 (Phase 4.3.3)。

        goal_change は enum 正規化 (keep/update/pivot)、不正値は keep にフォールバック。
        goal_short が無ければ None (グレースフル)。
        """
        m = _GOAL_SECTION_RE.search(feedback_text)
        if not m:
            return None
        section = m.group(0)
        # 後続セクション (■ 〜) で切る
        cut = re.search(r"\n■", section[1:])
        if cut:
            section = section[: 1 + cut.start()]
        extracted: dict = {}
        for tag, value in _GOAL_TAG_RE.findall(section):
            key = tag.lower()
            if key not in extracted:
                extracted[key] = value.strip().strip("「」\"'`")
        if not extracted or "goal_short" not in extracted:
            return None
        raw_change = extracted.get("goal_change", "keep").lower().strip()
        extracted["goal_change"] = (
            raw_change if raw_change in _VALID_GOAL_CHANGES else "keep"
        )
        extracted.setdefault("goal_long", "維持")
        extracted["extracted_at"] = datetime.now().isoformat()
        return extracted

    @staticmethod
    def _extract_ai1_note(feedback_text: str) -> Optional[str]:
        """LLM 出力の「■ 応答AIへの内省ノート」セクションを抽出 (AI1 注入用)。

        成功時: 見出し行 (「応答AIへの内省ノート ...」) を除いた本文を返す。
        失敗時 (セクション欠如・空) は None。
        """
        m = _AI1_NOTE_SECTION_RE.search(feedback_text)
        if not m:
            return None
        section = m.group(0)
        # 後続セクション (■ 〜) で切る
        cut = re.search(r"\n■", section[1:])
        if cut:
            section = section[: 1 + cut.start()]
        # 1 行目 (見出し行) を削除して本文だけ返す
        lines = section.splitlines()
        if lines:
            lines = lines[1:]
        body = "\n".join(lines).strip()
        return body if body else None

    async def _check_goal_quality(self, goal_short: str) -> Optional[str]:
        """3 段階検証で劣化フラグの文言を返す。立たなければ None。

        第一防衛 (sync, 軽量) — 禁止語 substring match:
            goal_short が _GOAL_BOILERPLATE_PHRASES のどれかを含む → flag

        本検出 1 (async, embedding) — 連続 cosine 平均 > FB_GOAL_COSINE_HISTORY_THRESHOLD:
            直近 N 件 goal_short の embedding と現在 embedding の平均 cosine。
            履歴が 2 件未満 (cold-start) はスキップして履歴に追加するだけ。

        本検出 2 (async, embedding, feature flag で gate):
            FB_GOAL_BOILERPLATE_CHECK=true の時のみ。boilerplate phrase との
            cosine > FB_GOAL_COSINE_BOILERPLATE_THRESHOLD → flag。
            embedding 失敗時は graceful skip。
        """
        if not goal_short or not self.rag_handler:
            return None
        # 第一防衛 (sync)
        for phrase in _GOAL_BOILERPLATE_PHRASES:
            if phrase in goal_short:
                return f"goal_short が決まり文句 '{phrase}' を含んでいる"
        # 本検出 1: 連続 cosine
        try:
            emb = await self.rag_handler._generate_embedding(goal_short)
        except Exception as e:
            logger.warning("[Feedback][Goal] embedding failed in check: %s", e)
            emb = None
        if not emb:
            return None
        # 連続 cosine 判定 (>=2 件履歴があるとき)
        if len(self._goal_short_history) >= 2:
            sims = [
                self.rag_handler._cosine_similarity(emb, prev_emb)
                for prev_emb in self._goal_short_history
            ]
            avg = sum(sims) / len(sims)
            if avg > Config.FB_GOAL_COSINE_HISTORY_THRESHOLD:
                self._goal_short_history.append(emb)
                return (
                    f"直近 {len(self._goal_short_history)} 件の goal_short と "
                    f"連続 cosine 平均 = {avg:.2f} で劣化傾向 "
                    f"(>{Config.FB_GOAL_COSINE_HISTORY_THRESHOLD})"
                )
        self._goal_short_history.append(emb)
        # 本検出 2 (feature flag、lazy init)
        if Config.FB_GOAL_BOILERPLATE_CHECK:
            if self._boilerplate_embedding is None:
                try:
                    self._boilerplate_embedding = (
                        await self.rag_handler._generate_embedding(
                            "ユーザーとの長期的かつ良好な共存関係を維持し対話の具体性を高める"
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "[Feedback][Goal] boilerplate embedding init failed: %s", e
                    )
                    self._boilerplate_embedding = []  # 失敗 marker (再試行しない)
            if self._boilerplate_embedding:
                sim = self.rag_handler._cosine_similarity(
                    emb, self._boilerplate_embedding
                )
                if sim > Config.FB_GOAL_COSINE_BOILERPLATE_THRESHOLD:
                    return (
                        f"goal_short が決まり文句 embedding に近い "
                        f"(cosine {sim:.2f} > "
                        f"{Config.FB_GOAL_COSINE_BOILERPLATE_THRESHOLD})"
                    )
        return None

    # ------------------------------------------------------------------
    # Phase 4.3.4 (Batch 3): Self-Evaluation Bias Detection
    # ------------------------------------------------------------------
    def _check_self_eval_bias(self) -> Optional[str]:
        """直近の自己評価出力から 3 種 bias を検出 (Phase 4.3.4)。

        Phase 4.2 calibration_active_loop + Phase 4.3.3 goal_quality_warn と同型 (Pull 型)。
        各 flag は独立 feature flag で gate、3 種すべて立っても 1 つの統合警告に圧縮して返す。
        警告文は強制ではなく「再考と理由具体化」を促す形 (思考誘発型)。

        Returns:
            警告文 (複数 flag を改行区切りで連結) or None。検出ゼロなら None。
        """
        flags: list[str] = []

        # 1. Mode collapse (affect ラベル偏重) — top1_ratio または top2_ratio 判定
        if (
            Config.FB_BIAS_DETECT_MODE_COLLAPSE
            and len(self._affect_history) >= Config.FB_BIAS_WINDOW
        ):
            cnt = Counter(self._affect_history)
            n = len(self._affect_history)
            sorted_counts = sorted(cnt.values(), reverse=True)
            top1_ratio = sorted_counts[0] / n
            top2_ratio = sum(sorted_counts[:2]) / n
            if top1_ratio > Config.FB_BIAS_AFFECT_TOP1_THRESHOLD:
                top_label, top_count = cnt.most_common(1)[0]
                flags.append(
                    f"  - Mode collapse: 直近 {n} 件中 {top_count} 件が "
                    f"affect=\"{top_label}\" "
                    f"(top1 {top1_ratio*100:.0f}% > "
                    f"{Config.FB_BIAS_AFFECT_TOP1_THRESHOLD*100:.0f}%)。"
                    f"別候補も検討し、同じならその理由を具体化せよ。"
                )
            elif top2_ratio > Config.FB_BIAS_AFFECT_TOP2_THRESHOLD:
                top2_labels = [l for l, _ in cnt.most_common(2)]
                flags.append(
                    f"  - Mode collapse: 直近 {n} 件で上位 2 ラベル {top2_labels} が "
                    f"{top2_ratio*100:.0f}% を占めている "
                    f"(>{Config.FB_BIAS_AFFECT_TOP2_THRESHOLD*100:.0f}%)。"
                    f"3 ラベル目以降の候補も検討せよ。"
                )

        # 2. Over-update (goal_change 偏重)
        if (
            Config.FB_BIAS_DETECT_OVER_UPDATE
            and len(self._goal_change_history) >= Config.FB_BIAS_WINDOW
        ):
            n = len(self._goal_change_history)
            update_count = sum(
                1 for c in self._goal_change_history if c in {"update", "pivot"}
            )
            ratio = update_count / n
            if ratio > Config.FB_BIAS_UPDATE_MAX_RATIO:
                flags.append(
                    f"  - Over-update: 直近 {n} 件で goal_change=update/pivot が "
                    f"{update_count} 件 ({ratio*100:.0f}% > "
                    f"{Config.FB_BIAS_UPDATE_MAX_RATIO*100:.0f}%)。"
                    f"話題や方針が実質同じなら keep。update は方針差分が明確な時のみ。"
                )

        # 3. Positivity bias (累計 neg 比率) — long-term 傾向の検出
        if (
            Config.FB_BIAS_DETECT_POSITIVITY
            and self._valence_total >= Config.FB_BIAS_VALENCE_MIN_SAMPLES
        ):
            neg_count = self._valence_counter.get("neg", 0)
            neg_ratio = neg_count / self._valence_total
            if neg_ratio < Config.FB_BIAS_NEG_VALENCE_MIN_RATIO:
                flags.append(
                    f"  - Positivity bias: 累計 {self._valence_total} 件で neg valence が "
                    f"{neg_count} 件 ({neg_ratio*100:.1f}% < "
                    f"{Config.FB_BIAS_NEG_VALENCE_MIN_RATIO*100:.0f}%)。"
                    f"pos に逃げず、不確実性・違和感・懸念があるなら neu/neg を選べ。"
                )

        if not flags:
            return None
        return "\n".join(flags)

    def _restore_valence_counter_from_summary(self) -> None:
        """起動時に memory_summary.txt 末尾 N 件から累計 valence カウンタを復元 (Phase 4.3.4)。

        positivity bias は long-term 傾向のため、再起動の度に cold-start すると
        検出に時間がかかりすぎる。短期 deque (affect/goal_change) と異なり累計を保つ。
        """
        if not os.path.exists(self.summary_file):
            return
        try:
            with open(self.summary_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning("[Feedback][Bias] valence sweep failed: %s", e)
            return
        sweep_n = Config.FB_BIAS_VALENCE_SWEEP_RECENT
        for line in lines[-sweep_n:]:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                affect = entry.get("affect")
                if not isinstance(affect, dict):
                    continue
                v = affect.get("valence")
                if v in {"pos", "neu", "neg"}:
                    self._valence_counter[v] += 1
                    self._valence_total += 1
            except (json.JSONDecodeError, AttributeError):
                continue
        if self._valence_total:
            logger.info(
                "[Feedback][Bias] valence counter restored from summary: "
                "total=%d, pos=%d, neu=%d, neg=%d",
                self._valence_total,
                self._valence_counter.get("pos", 0),
                self._valence_counter.get("neu", 0),
                self._valence_counter.get("neg", 0),
            )

    # ------------------------------------------------------------------
    # Phase 4.2: Precision Dynamics + Affective Inference
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_affect(feedback_text: str) -> Optional[dict]:
        """LLM 出力の Affective Inference セクションから構造化抽出 (Phase 4.2)。

        v2 修正: affect ラベルを enum に正規化、intensity fallback を 0.5 に統一、
        valence の不正値は 'neu' フォールバック。
        """
        m = _AFFECT_SECTION_RE.search(feedback_text)
        if not m:
            return None
        section = m.group(0)
        extracted: dict = {}
        for tag, value in _AFFECT_TAG_RE.findall(section):
            key = tag.lower()
            if key not in extracted:
                extracted[key] = value.strip()
        if not extracted:
            return None
        # affect ラベル enum 正規化 (raw を保持)
        if "affect" in extracted:
            extracted["affect_raw"] = extracted["affect"]
            extracted["affect"] = normalize_affect_label(extracted["affect"])
        else:
            extracted["affect"] = "other"
        # intensity fallback 0.5 (confidence と同じ default)
        if "intensity" in extracted:
            try:
                v = float(extracted["intensity"])
                extracted["intensity"] = max(0.0, min(1.0, v))
            except (ValueError, TypeError):
                extracted["intensity"] = 0.5
        else:
            extracted["intensity"] = 0.5
        # valence enum 正規化
        v = (extracted.get("valence") or "neu").lower().strip()
        extracted["valence"] = v if v in ("pos", "neu", "neg") else "neu"
        return extracted

    def _format_precision_block(
        self,
        prev_predictions: Optional[dict],
        last_evaluation: Optional[dict],
        silence_streak: int,
    ) -> str:
        """Phase 4.2: Affective Inference 用【精度・感情状態】ブロック + 過信警告。

        cold-start (samples < min) ではモード未判定の旨を簡潔に書く。
        Affective narrative は last_evaluation が存在する場合のみ濃く展開。
        過信傾向検出時には【過信傾向検出】サブブロックを追加 (calibration loop 能動化)。
        """
        s = self._precision_state
        if s.samples_observed < Config.FB_PRECISION_MIN_SAMPLES:
            return (
                "■ 精度・感情状態 (Affective Inference, Phase 4.2)\n"
                f"  - precision モード: cold-start ({s.samples_observed}/"
                f"{Config.FB_PRECISION_MIN_SAMPLES} サンプル)\n"
                f"  - 直近 surprise EMA: {s.surprise_ema:.2f}/5\n"
                "  - サンプル不足のため normal 動作。クオリア言語化は控えめでよい。\n"
            )

        # 因果叙述の素材
        narrative_lines = []
        if last_evaluation and prev_predictions:
            narrative_lines.append(
                f"    - 直前の予測 (時刻 {prev_predictions.get('predicted_at', '?')}): "
                f"policy={prev_predictions.get('policy', '?')}, "
                f"expected_activity={prev_predictions.get('expected_activity', '?')}, "
                f"confidence={prev_predictions.get('confidence', 0.0):.2f}"
            )
            narrative_lines.append(
                f"    - 評価結果: surprise_1_5={last_evaluation.get('surprise_1_5', '?')}/5, "
                f"calibration_note: {last_evaluation.get('calibration_note', '(なし)')}"
            )
        else:
            narrative_lines.append("    - 直前の予測誤差データ: なし (初回もしくは silent 連続)")

        # 直前 affect (前ループで保存) を表示し、alpha がどう変調されたか見せる
        affect_line = ""
        if s.last_affect:
            aff = s.last_affect
            affect_line = (
                f"  - 直前の自己 affect: {aff.get('affect', '?')} "
                f"(intensity={aff.get('intensity', 0.0):.2f}, "
                f"valence={aff.get('valence', '?')}) "
                f"→ 次ループ alpha 変調済み\n"
            )

        # calibration 過信警告ブロック (能動化)
        overconf_block = ""
        if s.calibration.is_overconfident():
            ratio = s.calibration.overconfidence_ratio()
            overconf_block = (
                "\n  【過信傾向検出 (Calibration Active Loop)】\n"
                f"  - 過去 {len(s.calibration.history)} 件中、高 confidence (>0.7) で"
                f"大きく外れた (surprise>=4) ケースが {ratio*100:.0f}% を占めている。\n"
                "  - 次期間予測の `confidence` は **意識的に控えめ** "
                "(普段の 0.7 倍程度) に見積もれ。\n"
            )

        # Phase 4.3.3 (Batch 2): 前ループで検出された Goal 劣化フラグの警告ブロック
        # 改訂: 命令調 (「書き直せ」) ではなく再審査要求に変更し、keep の選択肢を明示する。
        # これにより安定した目標まで update/pivot に揺らぐ問題を抑制する。
        goal_warn_block = ""
        if self._goal_quality_warn:
            goal_warn_block = (
                "\n  【Goal 再審査要求 (Phase 4.3.3)】\n"
                f"  - 前ループの goal_short に劣化シグナル: {self._goal_quality_warn}\n"
                "  - これは「変えろ」ではなく「再審査せよ」。次のいずれかを選び理由を書け:\n"
                "    (a) keep: 芯は同じだが、goal_short の表現と焦点を直近観測に合わせて具体化する\n"
                "    (b) update: 話題が動いたので軸を更新する\n"
                "    (c) pivot: 方向転換が起きた\n"
                "  - keep を選ぶなら、なぜ keep が妥当か直近観測ベースの理由を 1 文添えること。\n"
                "  - 完全に同一の goal_short 文字列は「内省していない」シグナルとして扱われる。\n"
                "  - 「維持」「良好な共存」「対話の具体性を高める」など決まり文句の言い直しは禁止。\n"
            )

        # Phase 4.3.4 (Batch 3): Self-Evaluation Bias 警告サブブロック (3 種統合)
        # 警告文は強制ではなく「再考と理由具体化」を促す思考誘発型 (Verbalized Sampling 思想)
        self_eval_bias_block = ""
        if self._self_eval_bias_warn:
            self_eval_bias_block = (
                "\n  【自己評価バイアス警告 (Phase 4.3.4)】\n"
                f"{self._self_eval_bias_warn}\n"
            )

        L0, L1, L2, L3 = s.weights

        return (
            "■ 精度・感情状態 (Affective Inference, Phase 4.2)\n"
            f"  - precision モード: **{s.mode}**\n"
            f"  - 直近 surprise EMA: {s.surprise_ema:.2f}/5 (samples={s.samples_observed})\n"
            f"  - 連続沈黙ストリーク: {silence_streak}\n"
            "  - 直近の予測誤差の物語:\n"
            + "\n".join(narrative_lines) + "\n"
            + affect_line
            + f"  - 機械調整: PE_LEVEL 倍率 ×{s.pe_factor:.2f}, "
            f"階層重み L0/L1/L2/L3 = {L0:.2f}/{L1:.2f}/{L2:.2f}/{L3:.2f}"
            + (" (動的)" if Config.FB_PRECISION_DYNAMIC_WEIGHTS else " (静的, baseline)")
            + "\n"
            "  - 含意: このモードを「機械フラグ」ではなく、自分が今体感している\n"
            "    感情・注意の質 (クオリア) として言語化せよ (■ 感情・注意のクオリア セクション)。\n"
            "  - **重要**: あなたの valence/intensity は次ループの learning rate に投入される。\n"
            "    無責任な過大表現はメタ認知を歪ませる。"
            + overconf_block
            + goal_warn_block
            + self_eval_bias_block
        )

    def _format_prediction_comparison(
        self,
        prev: Optional[dict],
        window: "_Window",
        hs: Optional[HierarchicalSurprise],
    ) -> str:
        """前回予測 + 今回観測 + 客観 hint を user_text 注入用テキストにする。

        prev=None (初回・抽出失敗) の場合は空文字を返し、上位で省略される。
        """
        if not prev:
            return ""
        hints = self._compute_objective_hints(prev, window, hs)
        hints_block = ""
        if hints:
            hints_block = "\n【客観 hint (Python 側計算、LLM の判断補助)】\n"
            for h in hints:
                hints_block += f"  - {h}\n"
        return (
            "■ 前回予測と今回観測の比較 (Bottom-up Prediction Error, Phase 4.1)\n"
            f"【前回予測】 (時刻 {prev.get('predicted_at', '?')}, "
            f"信頼度 {prev.get('confidence', 0.5):.2f})\n"
            f"  - User予測          : {prev.get('user', '(未指定)')}\n"
            f"  - Screen予測        : {prev.get('screen', '(未指定)')}\n"
            f"  - Policy予測        : {prev.get('policy', '(未指定)')}\n"
            f"  - Expected activity : {prev.get('expected_activity', '(未指定)')}\n"
            f"  - 根拠              : {prev.get('rationale', '(未指定)')}\n\n"
            f"【今回観測】\n"
            f"  - Userの実発話 : {window.last_user_utterance or '(なし)'}\n"
            f"  - Screen実観測 : {(window.last_vlm_narration or '(なし)')[:200]}\n"
            f"  - 期間内ターン数: {window.n_turns}, VLMフレーム: {window.n_vlm}\n"
            f"  - Hierarchical unified: {hs.unified if hs else 0.0:.2f}\n"
            f"{hints_block}\n"
            "➡ 「予測評価 (prediction_evaluation)」セクションで構造化評価を出力すること (タグ必須)。\n"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    async def _persist(
        self,
        feedback: str,
        window_end: datetime,
        predictions: Optional[dict] = None,
        evaluation: Optional[dict] = None,
        affect: Optional[dict] = None,
    ) -> None:
        """memory_summary.txt に 1 エントリ追記する。

        Phase 4.1: predictions / prediction_evaluation を同梱可能。
        Phase 4.2: affect を同梱 (LLM の Affective Inference 出力を保存)。
        precision_state は別ファイル (precision_history.jsonl) に分離 (肥大化対策)。
        全フィールド None なら従来形式と完全一致 (後方互換)。
        """
        entry = {"timestamp": window_end.isoformat(), "feedback": feedback}
        if predictions:
            entry["predictions"] = predictions
        if evaluation:
            entry["prediction_evaluation"] = evaluation
        if affect:
            entry["affect"] = affect

        def _append() -> None:
            try:
                with open(self.summary_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning("[Feedback] persist failed: %s", e)

        await asyncio.to_thread(_append)

    async def _persist_precision_history(
        self,
        window_end: datetime,
        transition_msg: Optional[str],
    ) -> None:
        """Phase 4.2: precision_state を別ファイルに分離。mode 遷移時のみ書く。

        memory_summary.txt 肥大化防止 + reverse スキャン高速化。
        no-op 時はスキップ (transition_msg=None)。
        """
        if transition_msg is None:
            return
        s = self._precision_state
        entry = {
            "timestamp": window_end.isoformat(),
            "mode": s.mode,
            "surprise_ema": round(s.surprise_ema, 3),
            "samples_observed": s.samples_observed,
            "pe_factor": s.pe_factor,
            "weights": [round(w, 3) for w in s.weights],
            "transition_from": s.last_transition_from,
            "overconfidence_ratio": round(s.calibration.overconfidence_ratio(), 3),
            "calibration_window_size": len(s.calibration.history),
        }

        def _append() -> None:
            try:
                with open(self._precision_history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning("[Feedback] precision_history persist failed: %s", e)

        await asyncio.to_thread(_append)

    def _load_last_precision_state(self) -> None:
        """Phase 4.2: precision_history.jsonl 末尾から PrecisionState を復元。

        ファイル不在時は cold-start (initial EMA + mode=normal)。
        起動時に 1 回だけ呼ばれる。
        """
        if not os.path.exists(self._precision_history_file):
            return
        try:
            with open(self._precision_history_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning("[Feedback] precision_history load failed: %s", e)
            return
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                self._precision_state.mode = entry.get("mode", "normal")
                self._precision_state.surprise_ema = entry.get(
                    "surprise_ema", Config.FB_PRECISION_INITIAL_EMA
                )
                self._precision_state.samples_observed = entry.get("samples_observed", 0)
                self._precision_state.pe_factor = entry.get("pe_factor", 1.0)
                w = entry.get("weights")
                if isinstance(w, list) and len(w) == 4:
                    self._precision_state.weights = tuple(w)
                logger.info(
                    "[Feedback][Precision] restored from history: "
                    "mode=%s, ema=%.2f, samples=%d, pe_factor=%.2f",
                    self._precision_state.mode,
                    self._precision_state.surprise_ema,
                    self._precision_state.samples_observed,
                    self._precision_state.pe_factor,
                )
                return
            except json.JSONDecodeError:
                continue

    def _load_last_feedback(self) -> tuple[str, Optional[dict]]:
        """memory_summary.txt 末尾から最後のフィードバック本文と予測を返す。

        Returns:
            (feedback_text, predictions or None)。Phase 4.1 で戻り値が tuple に
            拡張された (旧 str → 新 tuple)。呼び出しは start_loop の 1 箇所のみで
            影響範囲は限定的。
        """
        if not os.path.exists(self.summary_file):
            return "", None
        try:
            with open(self.summary_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return "", None
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                fb = entry.get("feedback", "")
                preds = entry.get("predictions")  # 旧エントリは無し → None
                if fb:
                    return fb, preds
            except json.JSONDecodeError:
                continue
        return "", None

    # ------------------------------------------------------------------
    # Inject into response LLM
    # ------------------------------------------------------------------
    def _inject_to_llm(self, feedback_text: str) -> None:
        """AI2 出力から「応答AIへの内省ノート」だけを抽出して AI1 に注入する。

        失敗時 (新セクション未出力) は feedback_text 先頭 600 字でフォールバック。
        これにより内部用語まみれの長文が AI1 のシステムプロンプトを支配することを防ぐ。
        """
        note = self._extract_ai1_note(feedback_text)
        payload = note if note else feedback_text[:600]
        if hasattr(self.llm_handler, "update_memory"):
            try:
                self.llm_handler.update_memory(payload)
            except Exception as e:
                logger.warning("[Feedback] update_memory failed: %s", e)
