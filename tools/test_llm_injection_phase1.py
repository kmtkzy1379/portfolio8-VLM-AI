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
          "中身(substance)は firm" in prompt)
    check("committed fact value rendered (好きな動物 = 猫だよ)",
          "好きな動物 = 猫だよ" in prompt)
    check("anti-drift wording present (猫→うさぎ禁止)",
          "猫と言ったのに後でうさぎ" in prompt)
    check("expression-diversity wording present (表現は毎回変える)",
          "表現は毎回変える" in prompt)
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


async def _amain() -> None:
    run()
    await test_fix8_gate()


if __name__ == "__main__":
    asyncio.run(_amain())
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
