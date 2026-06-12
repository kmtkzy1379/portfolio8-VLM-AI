"""Phase 1: prompt-size audit (NO API) — measure per-block injection sizes by toggling.

Builds the REAL system prompt via LLMHandler._build_system_prompt with representative
live-session state, then re-renders with each context source disabled; the delta is that
block's cost. Repeats for the 4 turn types (normal / silence nudge / VLM nudge / overdue).

Run:  $env:PYTHONIOENCODING="utf-8"; venv\\Scripts\\python.exe tools\\prompt_audit_phase1.py
"""
import asyncio
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import Config  # noqa: E402
Config.GROQ_API_KEY = Config.GROQ_API_KEY or "x"
Config.OPENAI_API_KEY = Config.OPENAI_API_KEY or "x"

from prompts import TALK_SYSTEM_PROMPT  # noqa: E402
from modules.llm import LLMHandler  # noqa: E402
from modules.task_manager import TaskManager  # noqa: E402


def iso(dt: datetime) -> str:
    return dt.isoformat()


# ── 代表的なライブ状態（2026-06-11 セッション相当のスケール） ──────────────
AI2_NOTE = (
    "ユーザーは今夜はゲームの話題に乗り気で、テンポの良い短い相槌を好む傾向。"
    "さっきの約束（魚の話）は時間どおりに返せたが、少し早口になりがちだった。"
    "次は相手の沈黙の理由を先に考えてから話すこと。質問を重ねすぎない。"
    "画面共有中はゲームの進行を優先し、割り込みは最小限に。"
) * 2  # ~340 chars (FB ノートの現実的なスケール)

VLM_JOURNAL = (
    "[たった今/MAJOR] 画面がコードエディタに切り替わり、Pythonのコードが表示されている\n"
    "[1分前/MODERATE] ブラウザでニュース記事（宇宙開発の話題）を読んでいる\n"
    "[2分前/MINOR] ニュースサイトをスクロールしている"
)

RAG_MEMORIES = [
    {"user": "さっき新しいローグライク始めたんだ", "ai": "へえ、難しい系？死に覚えるやつ？"},
    {"user": "コーヒーは何派？", "ai": "ブラック派かな。苦いのが落ち着くんだよね"},
]

RECENT_TURNS = [
    {"user": "こんばんは", "ai": "こんばんは、えへへ。", "user_timestamp": "", "ai_timestamp": ""},
    {"user": "今日新しいゲーム買ったんだ", "ai": "お、何系？アクション？", "user_timestamp": "", "ai_timestamp": ""},
    {"user": "ローグライクってやつ", "ai": "死に覚えゲーだ。沼るよ、それ", "user_timestamp": "", "ai_timestamp": ""},
    {"user": "もう10回死んだ", "ai": "順調順調。最初はそんなもん", "user_timestamp": "", "ai_timestamp": ""},
    {"user": "ちょっと休憩する", "ai": "りょうかい、ゆっくりして", "user_timestamp": "", "ai_timestamp": ""},
]

COMMITTED_FACTS = [
    {"topic": "好きな動物", "answer": "猫だよ", "recent_expressions": ["猫だよ", "猫かな、気まぐれが好き", "やっぱり猫"]},
    {"topic": "好きな魚を答える", "answer": "さばかな", "recent_expressions": ["さばかな"]},
    {"topic": "好きな色", "answer": "青", "recent_expressions": ["青", "青かな、落ち着くから"]},
]

NUDGE_MEMORY = [
    {"category": "early", "text": "ローグライク、進んだ？"},
    {"category": "check", "text": "なんか集中してる？"},
]


async def build_llm(tm: TaskManager) -> LLMHandler:
    llm = LLMHandler()
    llm.system_prompt = TALK_SYSTEM_PROMPT
    llm.set_task_manager(tm)
    return llm


def apply_full_state(llm: LLMHandler) -> None:
    llm.ai2_feedback = AI2_NOTE
    llm._ai2_feedback_set_at = iso(datetime.now() - timedelta(seconds=40))
    llm.vlm_context = VLM_JOURNAL
    llm._vlm_alerts.clear()
    llm._vlm_alerts.append((time.time() - 60, "ブラウザでニュース記事（宇宙開発の話題）を読んでいる", "major"))
    llm._vlm_alerts.append((time.time() - 2, "画面がコードエディタに切り替わり、Pythonのコードが表示されている", "major"))
    llm.rag_memories = list(RAG_MEMORIES)
    llm.silence_summary = {"silence_seconds": 35, "ellipsis_count": 2, "last_real_user_ts": iso(datetime.now())}
    llm.nudge_self_memory = list(NUDGE_MEMORY)
    llm.committed_facts = [dict(f) for f in COMMITTED_FACTS]
    llm.affect = {"affect": "curiosity", "intensity": 0.7, "valence": "pos"}
    llm._affect_set_at = iso(datetime.now() - timedelta(seconds=30))
    llm.recent_turns = [dict(t) for t in RECENT_TURNS]
    llm.one_shot_context = ""


# (toggle_name, fn(llm) -> None で無効化)
TOGGLES = [
    ("ai2_feedback",     lambda l: (setattr(l, "ai2_feedback", ""), setattr(l, "_ai2_feedback_set_at", None))),
    ("vlm_context",      lambda l: setattr(l, "vlm_context", "")),
    ("vlm_alerts",       lambda l: l._vlm_alerts.clear()),
    ("rag_memories",     lambda l: setattr(l, "rag_memories", [])),
    ("silence_summary",  lambda l: setattr(l, "silence_summary", None)),
    ("nudge_self_memory", lambda l: setattr(l, "nudge_self_memory", [])),
    ("committed_facts",  lambda l: setattr(l, "committed_facts", [])),
    ("affect_tone",      lambda l: (setattr(l, "affect", None), setattr(l, "_affect_set_at", None))),
    ("recent_turns",     lambda l: setattr(l, "recent_turns", [])),
]


async def audit_turn_type(tm: TaskManager, label: str, user_text: str, suppress_pending: bool) -> dict:
    llm = await build_llm(tm)
    apply_full_state(llm)
    llm._suppress_pending_block = suppress_pending
    full = llm._build_system_prompt(user_text)
    sizes = {"TOTAL": len(full)}

    # persona+固定部 = 全コンテキスト OFF でのサイズ
    bare = await build_llm(tm)
    bare._suppress_pending_block = suppress_pending
    bare.rag_memories = []
    bare.recent_turns = []
    bare.silence_summary = None
    bare.committed_facts = []
    bare.nudge_self_memory = []
    bare.vlm_context = ""
    bare.one_shot_context = ""
    bare_len = len(bare._build_system_prompt(user_text))
    sizes["persona+static(talk_prompt+now+instr)"] = bare_len

    for name, off in TOGGLES:
        llm2 = await build_llm(tm)
        apply_full_state(llm2)
        llm2._suppress_pending_block = suppress_pending
        off(llm2)
        sizes[name] = len(full) - len(llm2._build_system_prompt(user_text))

    return sizes


async def main() -> None:
    fd, path = tempfile.mkstemp(suffix="_audit.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()
    tm = TaskManager(tasks_file=path)
    await tm.start()
    try:
        # 代表 instruction 状態: PENDING 1件 + 期限超過 ACTIVE 1件（最重量ケース）
        now = datetime.now()
        for instr, dl in (("好きな魚を答える", now + timedelta(seconds=300)),
                          ("好きな動物を答える", now + timedelta(seconds=30))):
            tm.enqueue_command_nowait({
                "kind": "set_active_instruction", "instruction": instr,
                "deadline_at": iso(dl), "reason": "audit", "created_by": "ai1",
            })
        await tm.flush_command_queue()
        iid = [i for i, v in tm._active_instructions.items() if v.instruction == "好きな動物を答える"][0]
        tm._active_instructions[iid].deadline_at = iso(now - timedelta(seconds=1))
        await tm._reconcile_instruction_status()

        turn_types = [
            ("normal user turn", "好きな魚って何だっけ？", False),
            ("silence nudge (…)", "…", True),
            ("VLM nudge", "[内部: 画面に新しい変化があった - 自然にリアクションして]", True),
            ("overdue nudge", f"[内部: 期限超過 {iid} を履行 — 好きな動物を答える]", True),
        ]
        results = {}
        for label, text, sup in turn_types:
            results[label] = await audit_turn_type(tm, label, text, sup)

        # ── 表出力 ──
        keys = ["TOTAL", "persona+static(talk_prompt+now+instr)"] + [n for n, _ in TOGGLES]
        col_w = max(len(k) for k in keys) + 2
        header = "block".ljust(col_w) + "".join(label.split(" ")[0].rjust(14) for label, _, _ in turn_types)
        print("\n=== System-prompt size by block (chars; JP chars ~= 1 token) ===")
        print(header)
        for k in keys:
            row = k.ljust(col_w)
            for label, _, _ in turn_types:
                row += str(results[label].get(k, "")).rjust(14)
            print(row)
        # 動的合計と注入比率
        print("\n--- share of dynamic injection (TOTAL - persona/static) ---")
        for label, _, _ in turn_types:
            r = results[label]
            dyn = r["TOTAL"] - r["persona+static(talk_prompt+now+instr)"]
            print(f"  {label}: persona/static={r['persona+static(talk_prompt+now+instr)']}  dynamic={dyn}  total={r['TOTAL']}")
        print(f"\n  (history はこの上に毎ターン蓄積: user+assistant 各ターン分が別途加算)")
        print(f"  talk_prompt 静的本体: {len(TALK_SYSTEM_PROMPT)} chars")
    finally:
        try:
            await tm.stop()
        except Exception:
            pass
        await asyncio.sleep(0.05)
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
