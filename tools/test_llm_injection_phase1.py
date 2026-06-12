"""Phase 1 Tier-1 injection test (NO API call — only builds the prompt string).

Verifies that _build_system_prompt actually renders:
- the Fix-9b committed-facts firm-consistency block, and
- the Phase-0 nudge self-memory block,
when the corresponding llm attributes are populated.

Run:  venv\\Scripts\\python.exe tools\\test_llm_injection_phase1.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Dummy keys so LLMHandler construction never needs real creds (no network is opened).
from config import Config  # noqa: E402
Config.GROQ_API_KEY = Config.GROQ_API_KEY or "x"
Config.OPENAI_API_KEY = Config.OPENAI_API_KEY or "x"

from modules.llm import LLMHandler  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))


def run() -> None:
    llm = LLMHandler()
    llm.system_prompt = "テスト用システムプロンプト"
    # The two contexts under test:
    llm.committed_facts = [{"topic": "好きな動物", "answer": "猫だよ"}]
    llm.nudge_self_memory = [{"category": "early", "text": "猫だよ"}]
    # Defensive defaults for other context inputs.
    llm.recent_turns = []
    llm.silence_summary = None
    llm.rag_memories = []
    llm.vlm_context = ""
    llm.one_shot_context = ""
    llm._task_manager = None

    print("\n--- Fix-9b: committed-facts block renders in the system prompt ---")
    try:
        prompt = llm._build_system_prompt("…")
    except Exception as e:  # noqa: BLE001
        check("_build_system_prompt runs", False, f"{type(e).__name__}: {e}")
        return
    check("_build_system_prompt runs without error", True)
    check("committed-facts block header present",
          "中身も理由も firm" in prompt)
    check("committed fact value rendered (好きな動物 = 猫だよ)",
          "好きな動物 = 猫だよ" in prompt)
    check("anti-drift wording present (猫→うさぎ禁止)",
          "猫→うさぎは禁止" in prompt)
    check("reason-consistency wording present (理由もでっち上げない)",
          "好きな理由も毎回でっち上げず一貫" in prompt)
    check("anti-verbatim wording present (全く同じ発話をしない)",
          "全く同じ発話をしない" in prompt and "一字一句そのまま言わない" in prompt)
    check("Phase-0 nudge self-memory block also renders",
          "この沈黙中に自分が既に言ったこと" in prompt)

    # When empty, the block must NOT appear (no noise).
    llm.committed_facts = []
    prompt2 = llm._build_system_prompt("…")
    check("no committed-facts block when store empty",
          "中身(substance)は firm" not in prompt2)


async def test_fix8_gate() -> None:
    import tempfile
    from datetime import datetime, timedelta
    from modules.task_manager import TaskManager
    print("\n--- Fix-8 code gate: PENDING block hidden on internal nudge, shown on user turn ---")
    fd, path = tempfile.mkstemp(suffix="_fix8.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()
    tm = TaskManager(tasks_file=path)
    await tm.start()
    try:
        tm.enqueue_command_nowait({
            "kind": "set_active_instruction", "instruction": "好きな動物を答える",
            "deadline_at": (datetime.now() + timedelta(seconds=60)).isoformat(),
            "reason": "t", "created_by": "ai1",
        })
        await tm.flush_command_queue()
        llm = LLMHandler()
        llm.system_prompt = "テスト用システムプロンプト"
        llm.recent_turns = []
        llm.silence_summary = None
        llm.committed_facts = []
        llm.nudge_self_memory = []
        llm.rag_memories = []
        llm.vlm_context = ""
        llm.one_shot_context = ""
        llm.set_task_manager(tm)

        # Real user turn: the PENDING reservation IS visible (so Eve can act if asked).
        llm._suppress_pending_block = False
        p_user = llm._build_system_prompt("好きな動物おしえて")
        check("PENDING reservation visible on real user turn", "好きな動物を答える" in p_user)

        # Internal / silence nudge: the PENDING reservation is HIDDEN (no early fulfillment).
        llm._suppress_pending_block = True
        p_nudge = llm._build_system_prompt("…")
        check("PENDING reservation hidden on internal/silence nudge",
              "好きな動物を答える" not in p_nudge)
    finally:
        try:
            await tm.stop()
        except Exception:
            pass
        await asyncio.sleep(0.02)
        try:
            os.remove(path)
        except OSError:
            pass


def test_proactive_block() -> None:
    print("\n--- ① Proactivity: 能動判断ブロックは沈黙『…』時のみ（gate） ---")
    llm = LLMHandler()
    llm.system_prompt = "テスト用システムプロンプト"
    llm.recent_turns = []
    llm.silence_summary = None
    llm.committed_facts = []
    llm.nudge_self_memory = []
    llm.rag_memories = []
    llm.vlm_context = ""
    llm.one_shot_context = ""
    llm._task_manager = None
    MARK = "沈黙時の能動判断"
    p_silence = llm._build_system_prompt("…")
    check("proactive block present on silence '…'", MARK in p_silence)
    check("bans generic filler (今日は何する？/定型句)",
          "今日は何する？" in p_silence and "定型句" in p_silence)
    check("defers to busy/away/'黙ってて' -> 見守る",
          "黙ってて" in p_silence and "見守る" in p_silence)
    check("Bug-E: no-re-greeting rule present (言い換え含む)",
          "セッションで1回まで" in p_silence and "どんな言い換えでも再挨拶しない" in p_silence)
    # [greeting] タグルールは nudge_self_memory がある時のみ（nudge_memory_context 内）
    llm.nudge_self_memory = [{"category": "greeting", "text": "こんばんは、えへへ。"}]
    p_with_greet = llm._build_system_prompt("…")
    check("Bug-E: [greeting] tag rule present in nudge memory block",
          "[greeting] が既にあれば挨拶系は一切出さない" in p_with_greet)
    llm.nudge_self_memory = []
    # Bug-F regression: RAG legacy_turn 描画が引数 user_text を shadow して proactive を
    # 消していた（RAGあり+「…」の組合せが全テストで未検証だった穴）。必ず両立を確認する。
    llm.rag_memories = [{"user": "コーヒーは何派？", "ai": "ブラック派かな"}]
    p_rag = llm._build_system_prompt("…")
    check("proactive block present on '…' EVEN WITH rag memories (Bug-F)", MARK in p_rag)
    check("rag block also present (both coexist)", "Long-term Memory" in p_rag)
    llm.rag_memories = []

    # Gate: must NOT appear on real user turns / VLM nudges / overdue nudges.
    check("absent on real user turn", MARK not in llm._build_system_prompt("好きな動物おしえて"))
    check("absent on VLM nudge",
          MARK not in llm._build_system_prompt("[内部: 画面に新しい変化があった - 自然にリアクションして]"))
    check("absent on overdue nudge",
          MARK not in llm._build_system_prompt("[内部: 期限超過 i_abc を履行 — 好きな動物を答える]"))


async def _amain() -> None:
    run()
    test_proactive_block()
    await test_fix8_gate()


if __name__ == "__main__":
    asyncio.run(_amain())
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
