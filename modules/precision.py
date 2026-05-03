"""
Precision Dynamics — Phase 4.2 (v2)

Joffily & Coricelli 2013 Affective Inference Theory に忠実:
- valence = -d(free energy)/dt
- valence × intensity が learning rate (alpha) を変調する
v2 で「形だけ」リスク (Affective Hallucination) を脱却。

Modes (3 つに削減 — relaxed は将来検討):
- vigilant : 直近で予測が外れている → 観察を重視 (低層 PE gain 上昇 ≒ dopamine)
- normal   : ベースライン
- dmn      : 沈黙長い + surprise 低い → 内省/長期目標 (Default Mode Network)

References:
- Friston "predictive coding precision" + Frontiers Psychiatry 2025 10.3389/fpsyt.2025.1713833
- Seth & Friston 2016 "Active interoceptive inference and the emotional brain"
- Joffily & Coricelli 2013 (Affective Inference Theory)
- Steyvers & Peters 2025 "Metacognition and Uncertainty Communication in LLMs"
- DMC framework (AAAI 2025) for hallucination mitigation
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Optional

from config import Config

logger = logging.getLogger(__name__)


PrecisionMode = Literal["vigilant", "normal", "dmn"]
ValenceLabel = Literal["pos", "neu", "neg"]
AffectEnum = Literal[
    "curiosity", "surprise", "calm", "concern",
    "vigilance", "relief", "boredom", "confusion", "other",
]
_VALID_AFFECTS: set[str] = {
    "curiosity", "surprise", "calm", "concern",
    "vigilance", "relief", "boredom", "confusion", "other",
}
# 自由記述 → enum への粗マッピング (LLM が自然な日本語/英語を書いた場合の救済)
_AFFECT_ALIAS: dict[str, str] = {
    "戸惑い": "confusion", "困惑": "confusion", "当惑": "confusion",
    "驚き": "surprise", "軽い驚き": "surprise",
    "警戒": "vigilance", "緊張": "vigilance",
    "懸念": "concern", "不安": "concern",
    "落ち着き": "calm", "安心": "relief",
    "内省": "calm", "退屈": "boredom",
    "関心": "curiosity",
}


def normalize_affect_label(raw: Optional[str]) -> str:
    """自由記述 affect ラベルを enum に正規化。失敗時は 'other'。"""
    if not raw:
        return "other"
    s = raw.strip().lower()
    if s in _VALID_AFFECTS:
        return s
    # 日本語 alias チェック (大文字化前の生の raw を使う)
    for alias, canon in _AFFECT_ALIAS.items():
        if alias in raw:
            return canon
    return "other"


@dataclass
class CalibrationStats:
    """Phase 4.1 の confidence × surprise を集約して過信傾向を検出。

    過去 N 件で「高 confidence (>0.7) + 高 surprise (>=4)」が
    overconf_threshold 以上の比率になれば warn フラグを立てる。
    """
    history: deque = field(default_factory=lambda: deque(maxlen=Config.FB_CALIBRATION_WINDOW))

    def push(self, confidence: Optional[float], surprise_1_5: Optional[int]) -> None:
        if confidence is None or surprise_1_5 is None:
            return
        try:
            c = float(confidence)
            s = int(surprise_1_5)
        except (ValueError, TypeError):
            return
        self.history.append((c, s))

    def overconfidence_ratio(self) -> float:
        """過去 N 件で「conf > 0.7 かつ surprise >= 4」の比率。"""
        if not self.history:
            return 0.0
        n = sum(1 for c, s in self.history if c > 0.7 and s >= 4)
        return n / len(self.history)

    def is_overconfident(self) -> bool:
        return (
            len(self.history) >= max(3, Config.FB_CALIBRATION_WINDOW // 2)
            and self.overconfidence_ratio() >= Config.FB_CALIBRATION_OVERCONF_THRESHOLD
        )


@dataclass
class PrecisionState:
    """Phase 4.2 precision 状態。precision_history.jsonl に永続化される。"""
    mode: PrecisionMode = "normal"
    surprise_ema: float = 2.0   # v2: 初期は中点 2.5 ではなく 2.0 (LLM bias 対策)
    samples_observed: int = 0
    pe_factor: float = 1.0
    weights: tuple[float, float, float, float] = (0.1, 0.4, 0.3, 0.2)
    # v2: affect → alpha ループで使う直前 affect
    last_affect: Optional[dict] = None
    # v2: calibration 集計 (Phase 4.1 の confidence×surprise を蓄積)
    calibration: CalibrationStats = field(default_factory=CalibrationStats)
    # mode 遷移トレース (デバッグ・ログ用)
    last_transition_at: Optional[str] = None
    last_transition_from: Optional[PrecisionMode] = None


def compute_alpha_from_affect(
    base_alpha: float,
    affect: Optional[dict],
    delta_max: float,
) -> float:
    """v2 修正 #1 (核): valence × intensity → alpha 変調。

    Joffily & Coricelli 理論: valence は learning rate を変調する。
    - neg × high intensity → alpha boost (faster update, 「予測が外れて学習する必要」)
    - pos × high intensity → alpha 減衰 (trust current state, 「うまくいっているので世界観を維持」)
    - neu / 不明 → base_alpha そのまま

    Returns: 変調後の alpha (clamp [0.05, 0.95])。
    """
    if not affect or not Config.FB_AFFECT_DRIVES_ALPHA:
        return base_alpha
    valence = (affect.get("valence") or "neu").lower()
    intensity = affect.get("intensity")
    if not isinstance(intensity, (int, float)):
        return base_alpha
    intensity_clamped = max(0.0, min(1.0, float(intensity)))
    delta = delta_max * intensity_clamped
    if valence == "neg":
        new_alpha = base_alpha + delta
    elif valence == "pos":
        new_alpha = base_alpha - delta
    else:
        new_alpha = base_alpha
    return max(0.05, min(0.95, new_alpha))


def update_surprise_ema(
    state: PrecisionState,
    surprise_1_5: Optional[int],
    hierarchical_unified: Optional[float],
    is_silent: bool,
    silence_streak: int,
    alpha: Optional[float] = None,
) -> float:
    """v2: silent EMA drift + affect → alpha を統合した EMA 更新。

    優先順位:
    1. surprise_1_5 が利用可 → 主信号 (alpha は呼び出し側で affect 変調済みを渡す)
    2. hierarchical_unified が利用可 → fallback (`unified*4+1`)
    3. silent + streak>=2 → 中点 2.5 へ drift (DMN 自然移行)
    4. それ以外 → EMA 不変 (観測なし劣化防止)
    """
    if alpha is None:
        alpha = Config.FB_PRECISION_EMA_ALPHA

    if surprise_1_5 is not None and 1 <= int(surprise_1_5) <= 5:
        s = float(surprise_1_5)
        state.samples_observed += 1
        state.surprise_ema = (1 - alpha) * state.surprise_ema + alpha * s
        return state.surprise_ema

    if hierarchical_unified is not None:
        try:
            u = float(hierarchical_unified)
        except (ValueError, TypeError):
            u = 0.0
        s = max(1.0, min(5.0, u * 4.0 + 1.0))
        state.samples_observed += 1
        state.surprise_ema = (1 - alpha) * state.surprise_ema + alpha * s
        return state.surprise_ema

    # v2 修正 #5: silent + streak>=2 で中点 (2.5) へ drift
    if is_silent and silence_streak >= 2:
        drift = Config.FB_PRECISION_SILENT_DRIFT
        state.surprise_ema = (1 - drift) * state.surprise_ema + drift * 2.5
        return state.surprise_ema

    return state.surprise_ema


def decide_mode(
    state: PrecisionState,
    silence_streak: int,
    hysteresis: Optional[float] = None,
) -> PrecisionMode:
    """v2: 3 モード (vigilant/normal/dmn) を決定。

    優先順位 (上から評価): vigilant > dmn > normal
    - vigilant: ema >= 3.5 (with hysteresis)
    - dmn:      silence_streak >= 4 AND ema < 2.5 (with hysteresis)
    - normal:   それ以外
    cold-start (samples < min) は normal 固定。
    """
    if hysteresis is None:
        hysteresis = Config.FB_PRECISION_HYSTERESIS
    if state.samples_observed < Config.FB_PRECISION_MIN_SAMPLES:
        return "normal"

    ema = state.surprise_ema
    prev = state.mode
    h = hysteresis

    # Hysteresis: 現在の mode に居続ける方向にバッファ
    th_vigilant_in = 3.5 - (h if prev == "vigilant" else 0.0)
    th_dmn_ema     = 2.5 + (h if prev == "dmn" else 0.0)

    # 優先順位: vigilant > dmn > normal
    if ema >= th_vigilant_in:
        return "vigilant"
    if silence_streak >= 4 and ema < th_dmn_ema:
        return "dmn"
    return "normal"


def apply_mode_adjustments(
    state: PrecisionState,
    mode: PrecisionMode,
    base_weights: tuple[float, float, float, float],
) -> None:
    """v2: pe_factor は常に動的、重み調整は feature flag で gate。

    GPT 評価 #4 反映: 動的 pe_factor + 動的重みを同時に投入すると振動リスク。
    まず pe_factor だけ動的化して安定確認、後で重み動的化を有効に。
    """
    pe_factor_map = {
        "vigilant": Config.FB_PE_FACTOR_VIGILANT,
        "normal":   Config.FB_PE_FACTOR_NORMAL,
        "dmn":      Config.FB_PE_FACTOR_DMN,
    }
    state.pe_factor = pe_factor_map[mode]

    if not Config.FB_PRECISION_DYNAMIC_WEIGHTS:
        # feature flag OFF: baseline 重みをそのまま使う
        state.weights = base_weights
        return

    L0, L1, L2, L3 = base_weights
    if mode == "vigilant":
        # 低層 PE 重視 (dopamine 系のゆるい近似)
        L1 += 0.05
        L0 -= 0.05
    elif mode == "dmn":
        # 細部を抑え、global を見る
        L0 += 0.10
        L1 -= 0.10
    # normal は無調整

    # 再正規化 (負値の保護込み)
    L0 = max(0.0, L0)
    L1 = max(0.0, L1)
    L2 = max(0.0, L2)
    L3 = max(0.0, L3)
    total = L0 + L1 + L2 + L3
    state.weights = (L0/total, L1/total, L2/total, L3/total) if total > 0 else base_weights


def transition_log(
    state: PrecisionState,
    new_mode: PrecisionMode,
    iso_now: str,
) -> Optional[str]:
    """mode 遷移時のみ log メッセージを返す。同一 mode は None。"""
    if new_mode == state.mode:
        return None
    msg = (
        f"mode transition: {state.mode} -> {new_mode} "
        f"(ema={state.surprise_ema:.2f}, samples={state.samples_observed})"
    )
    state.last_transition_from = state.mode
    state.last_transition_at = iso_now
    state.mode = new_mode
    return msg
