import os
from pathlib import Path
from dotenv import load_dotenv


# この config.py と同じディレクトリの .env を必ず読む
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    # YouTube
    TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID")

    # Models
    AI1_MODEL = os.getenv("AI1_MODEL_NAME", "llama-3.3-70b-versatile")
    AI2_MODEL = os.getenv("AI2_MODEL_NAME", "claude-opus-4-5")
    AI2_FALLBACK_MODEL = os.getenv("AI2_FALLBACK_MODEL", "gpt-4o")
    AI2_MAX_TOKENS = int(os.getenv("AI2_MAX_TOKENS", "4000"))
    AI2_TEMPERATURE = float(os.getenv("AI2_TEMPERATURE", "0.6"))
    VISION_ANALYSIS_MODEL = os.getenv(
        "VISION_ANALYSIS_MODEL",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # Vision Buffer
    VISION_BUFFER_SIZE = int(os.getenv("VISION_BUFFER_SIZE", "20"))

    # Feedback Loop (FEP-inspired metacognition)
    FB_LOOP_MIN_INTERVAL_SEC = float(os.getenv("FB_LOOP_MIN_INTERVAL_SEC", "0"))
    FB_LOOP_MID_ACTIVITY_SEC = float(os.getenv("FB_LOOP_MID_ACTIVITY_SEC", "30"))
    FB_LOOP_LOW_ACTIVITY_SEC = float(os.getenv("FB_LOOP_LOW_ACTIVITY_SEC", "120"))
    FB_LOOP_MAX_CHARS = int(os.getenv("FB_LOOP_MAX_CHARS", "3000"))
    FB_LOOP_ENABLE_VLM = os.getenv("FB_LOOP_ENABLE_VLM", "true").lower() == "true"

    # Phase 2: Visual surprise proxy score boundaries (change_magnitude, 0-255)
    # NOTE: This is a pixel-difference proxy, NOT a true neuroscientific prediction error.
    PE_LEVEL_1_MAX = float(os.getenv("PE_LEVEL_1_MAX", "5"))
    PE_LEVEL_2_MAX = float(os.getenv("PE_LEVEL_2_MAX", "15"))
    PE_LEVEL_3_MAX = float(os.getenv("PE_LEVEL_3_MAX", "40"))
    PE_LEVEL_4_MAX = float(os.getenv("PE_LEVEL_4_MAX", "80"))
    # (Level 5 = magnitude > PE_LEVEL_4_MAX)

    FB_PRE_SILENCE_DIALOG_PAIRS = int(os.getenv("FB_PRE_SILENCE_DIALOG_PAIRS", "4"))

    # Phase 4.1: Top-down prediction loop
    # Silent 期間中、前回予測を消費せず保留する最大ループ数 (越えたら破棄)。
    FB_PREDICTION_MAX_HOLD = int(os.getenv("FB_PREDICTION_MAX_HOLD", "5"))

    # Phase 4.4 (4.2 統合): 階層 surprise の baseline 重み (.env 化)
    HS_WEIGHT_L0 = float(os.getenv("HS_WEIGHT_L0", "0.1"))
    HS_WEIGHT_L1 = float(os.getenv("HS_WEIGHT_L1", "0.4"))
    HS_WEIGHT_L2 = float(os.getenv("HS_WEIGHT_L2", "0.3"))
    HS_WEIGHT_L3 = float(os.getenv("HS_WEIGHT_L3", "0.2"))

    # Phase 4.2: Precision Dynamics + Affective Inference
    # EMA smoothing (alpha が大きいほど直近重視)
    FB_PRECISION_EMA_ALPHA = float(os.getenv("FB_PRECISION_EMA_ALPHA", "0.4"))
    # 初期 EMA (LLM bias 対策で中点 2.5 ではなく低め)
    FB_PRECISION_INITIAL_EMA = float(os.getenv("FB_PRECISION_INITIAL_EMA", "2.0"))
    # モード判定が安定化するまでの最小サンプル数 (cold-start 保護)
    FB_PRECISION_MIN_SAMPLES = int(os.getenv("FB_PRECISION_MIN_SAMPLES", "3"))
    # Hysteresis buffer
    FB_PRECISION_HYSTERESIS = float(os.getenv("FB_PRECISION_HYSTERESIS", "0.3"))
    # Silent 期間 EMA を中点へ drift する係数 (per silent loop)
    FB_PRECISION_SILENT_DRIFT = float(os.getenv("FB_PRECISION_SILENT_DRIFT", "0.05"))
    # モード別 PE_LEVEL 倍率 (vigilant 時は閾値を下げる→より敏感に観察)
    FB_PE_FACTOR_VIGILANT = float(os.getenv("FB_PE_FACTOR_VIGILANT", "0.7"))
    FB_PE_FACTOR_NORMAL   = float(os.getenv("FB_PE_FACTOR_NORMAL",   "1.0"))
    FB_PE_FACTOR_DMN      = float(os.getenv("FB_PE_FACTOR_DMN",      "1.5"))
    # 動的重みは feature flag で gate (段階導入で振動切り分け)
    FB_PRECISION_DYNAMIC_WEIGHTS = os.getenv("FB_PRECISION_DYNAMIC_WEIGHTS", "false").lower() == "true"
    # Affect → function ループ (Joffily & Coricelli 2013 への忠実化)
    FB_AFFECT_DRIVES_ALPHA = os.getenv("FB_AFFECT_DRIVES_ALPHA", "true").lower() == "true"
    # alpha 変調の最大幅 (alpha_base ± delta_max * intensity)
    FB_AFFECT_ALPHA_DELTA_MAX = float(os.getenv("FB_AFFECT_ALPHA_DELTA_MAX", "0.2"))
    # Calibration loop active 化 (過信傾向検出窓)
    FB_CALIBRATION_WINDOW = int(os.getenv("FB_CALIBRATION_WINDOW", "10"))
    FB_CALIBRATION_OVERCONF_THRESHOLD = float(os.getenv("FB_CALIBRATION_OVERCONF_THRESHOLD", "0.5"))
    # Affective Inference 出力 ON/OFF (実機で冗長過ぎなら false に)
    FB_AFFECT_OUTPUT_ENABLED = os.getenv("FB_AFFECT_OUTPUT_ENABLED", "true").lower() == "true"
    # precision_history.jsonl のパス (memory_summary 肥大化対策で分離)
    PRECISION_HISTORY_FILE = os.getenv("PRECISION_HISTORY_FILE", "precision_history.jsonl")

    # Phase 4.3 (Memory Architecture v4): RAG 改良
    # Episode 保存時の重複除去閾値 (前回 episode との cosine がこの値以上なら保存スキップ)
    FB_RAG_DEDUP_THRESHOLD = float(os.getenv("FB_RAG_DEDUP_THRESHOLD", "0.88"))
    # search_similar の戻り件数 (top_k=4 に拡大、多様性が効くので増量しても重複しない)
    FB_RAG_TOP_K = int(os.getenv("FB_RAG_TOP_K", "4"))
    # base_score 重み (importance × recency × relevance)
    FB_RAG_W_IMP = float(os.getenv("FB_RAG_W_IMP", "0.3"))
    FB_RAG_W_REC = float(os.getenv("FB_RAG_W_REC", "0.3"))
    FB_RAG_W_REL = float(os.getenv("FB_RAG_W_REL", "0.4"))
    # Recency time decay の τ (秒、1 時間で 1/e に減衰)
    FB_RAG_RECENCY_TAU = float(os.getenv("FB_RAG_RECENCY_TAU", "3600"))
    # MMR の trade-off λ (Carbonell&Goldstein 1998 原典: λ × base_score - (1-λ) × max_sim)
    FB_RAG_MMR_LAMBDA = float(os.getenv("FB_RAG_MMR_LAMBDA", "0.7"))
    # 完全一致 hard-cut (MMR alone は重複完全排除を保証しないため、事前に除去)
    FB_RAG_DUPLICATE_HARDCUT = float(os.getenv("FB_RAG_DUPLICATE_HARDCUT", "0.95"))

    # Phase 4.3.3 (Batch 2): Goal Slot 二層
    GOAL_FILE = os.getenv("GOAL_FILE", "goal.txt")
    # 連続 N 件 goal_short の cosine 平均がこの値以上で劣化フラグ (本検出 1)
    FB_GOAL_COSINE_HISTORY_THRESHOLD = float(os.getenv("FB_GOAL_COSINE_HISTORY_THRESHOLD", "0.85"))
    # boilerplate phrase との cosine がこの値以上で決まり文句フラグ (本検出 2)
    FB_GOAL_COSINE_BOILERPLATE_THRESHOLD = float(os.getenv("FB_GOAL_COSINE_BOILERPLATE_THRESHOLD", "0.7"))
    # 本検出 2 は段階導入 (default OFF、Batch 2 安定化後に true に切替)
    FB_GOAL_BOILERPLATE_CHECK = os.getenv("FB_GOAL_BOILERPLATE_CHECK", "false").lower() == "true"
    # 連続劣化判定の窓
    FB_GOAL_HISTORY_SIZE = int(os.getenv("FB_GOAL_HISTORY_SIZE", "5"))

    # Phase 4.3.4 (Batch 3): Self-Evaluation Bias Detection
    # 短期 deque の窓 (affect / goal_change 用)
    FB_BIAS_WINDOW = int(os.getenv("FB_BIAS_WINDOW", "5"))
    # Mode collapse: 直近 N 件で最頻ラベル占有率がこの値より大きいで flag
    FB_BIAS_AFFECT_TOP1_THRESHOLD = float(os.getenv("FB_BIAS_AFFECT_TOP1_THRESHOLD", "0.6"))
    # Mode collapse: 上位 2 ラベル合計占有率がこの値より大きいで flag
    FB_BIAS_AFFECT_TOP2_THRESHOLD = float(os.getenv("FB_BIAS_AFFECT_TOP2_THRESHOLD", "0.85"))
    # Over-update: 直近 N 件で goal_change=update/pivot 比率がこの値より大きいで flag
    FB_BIAS_UPDATE_MAX_RATIO = float(os.getenv("FB_BIAS_UPDATE_MAX_RATIO", "0.6"))
    # Positivity bias: 累計 valence の neg 比率がこの値未満で flag
    FB_BIAS_NEG_VALENCE_MIN_RATIO = float(os.getenv("FB_BIAS_NEG_VALENCE_MIN_RATIO", "0.15"))
    # Positivity bias 判定の最小サンプル数 (cold-start 保護)
    FB_BIAS_VALENCE_MIN_SAMPLES = int(os.getenv("FB_BIAS_VALENCE_MIN_SAMPLES", "10"))
    # 起動時 memory_summary 末尾 sweep する件数 (累計 Counter 復元用)
    FB_BIAS_VALENCE_SWEEP_RECENT = int(os.getenv("FB_BIAS_VALENCE_SWEEP_RECENT", "50"))
    # 各検出の独立 feature flag (段階的 rollout、振動時は個別 OFF 可)
    FB_BIAS_DETECT_MODE_COLLAPSE = os.getenv("FB_BIAS_DETECT_MODE_COLLAPSE", "true").lower() == "true"
    FB_BIAS_DETECT_OVER_UPDATE = os.getenv("FB_BIAS_DETECT_OVER_UPDATE", "true").lower() == "true"
    # Positivity bias は default OFF (cautious rollout、運用検証後に true へ手動切替)
    FB_BIAS_DETECT_POSITIVITY = os.getenv("FB_BIAS_DETECT_POSITIVITY", "false").lower() == "true"


    # Voicevox
    VV_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
    VV_SPEAKER = int(os.getenv("VOICEVOX_SPEAKER_ID", "1"))
    VV_SPEED = float(os.getenv("VOICEVOX_SPEED", "1.1"))
    VV_PITCH = float(os.getenv("VOICEVOX_PITCH", "0.0"))


    # Audio & VAD
    SAMPLE_RATE = 16000
    CHANNELS = 1
    VAD_THRESHOLD = 0.5
    SILENCE_LIMIT = 0.5


    # Paths
    HISTORY_FILE = "conversation_history.txt"
    SUMMARY_FILE = "memory_summary.txt"
    RAG_FILE = "rag_memory.jsonl"

    # Plan-and-Execute Task Queue
    TASK_QUEUE_FILE = os.getenv("TASK_QUEUE_FILE", "tasks.jsonl")
    TASK_MAX_ACTIVE = int(os.getenv("TASK_MAX_ACTIVE", "3"))
    PLANNER_ENABLE = os.getenv("PLANNER_ENABLE", "true").lower() == "true"
    VALIDATOR_ENABLE = os.getenv("VALIDATOR_ENABLE", "true").lower() == "true"
    VALIDATOR_WATCHDOG_SEC = float(os.getenv("VALIDATOR_WATCHDOG_SEC", "90"))
    VALIDATOR_REPLAN_THRESHOLD = float(os.getenv("VALIDATOR_REPLAN_THRESHOLD", "0.5"))

    # Step 1.5: AI1 active_instruction ツールと期限管理（timestamp ベース）
    AI1_INSTRUCTION_TOOL_ENABLE = os.getenv("AI1_INSTRUCTION_TOOL_ENABLE", "true").lower() == "true"
    ACTIVE_INSTRUCTION_DEFAULT_EXPIRE_SEC = float(os.getenv("ACTIVE_INSTRUCTION_DEFAULT_EXPIRE_SEC", "300"))
    INSTRUCTION_OVERDUE_GRACE_SEC = float(os.getenv("INSTRUCTION_OVERDUE_GRACE_SEC", "60"))
    GOAL_CHANGE_MIN_INTERVAL_SEC = float(os.getenv("GOAL_CHANGE_MIN_INTERVAL_SEC", "120"))

    # Idle/silence nudge escalation（無言時の自発発話。ユーザがライブ調整する想定）
    # カテゴリ判定は ConversationCache.get_silence_summary()["silence_seconds"]
    # （最後の「実」ユーザ発話起点。Eve の応答発話時間も含むため early は 20s 前後を既定）。
    IDLE_EARLY_SEC       = float(os.getenv("IDLE_EARLY_SEC", "20"))   # < これ → early（基本「…」）
    IDLE_LONG_SEC        = float(os.getenv("IDLE_LONG_SEC",  "90"))   # >= これ → long（深い沈黙）
    # 沈黙 "…" nudge 間の指数バックオフ: min(BASE * FACTOR**n, CAP)
    IDLE_BASE_SEC        = float(os.getenv("IDLE_BASE_SEC",        "8"))
    IDLE_BACKOFF_FACTOR  = float(os.getenv("IDLE_BACKOFF_FACTOR",  "1.8"))
    IDLE_BACKOFF_CAP_SEC = float(os.getenv("IDLE_BACKOFF_CAP_SEC", "90"))


    @staticmethod
    def validate():
        # デバッグ用に現在読み込まれているキーの情報を少しだけ表示
        if Config.GROQ_API_KEY:
            print(f"GROQ_API_KEY(head)={Config.GROQ_API_KEY[:8]}..., len={len(Config.GROQ_API_KEY)}")
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in .env")
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in .env")



