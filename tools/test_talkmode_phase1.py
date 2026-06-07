"""Phase 1 Tier-1 test for Fix-9a (NO audio / NO LLM / NO hardware).

Verifies the dispatch-time streak-state logic (extracted into
TalkMode._apply_dispatch_streak_state):
- An internal nudge (督促 / 期限超過) CARRIES the silence-streak self-memory into
  llm.nudge_self_memory, so overdue fulfillment sees what it already said
  (prevents the 猫→うさぎ answer drift), and does NOT reset the streak.
- A real user turn RESETS streak state and clears nudge_self_memory.

Run:  venv\\Scripts\\python.exe tools\\test_talkmode_phase1.py
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from modes.talk_mode import TalkMode  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))


def run_tests() -> None:
    # Construct WITHOUT initialize() => no mic / LLM / TTS / hardware created.
    mode = TalkMode()
    mode.llm = SimpleNamespace(nudge_self_memory=None)
    streak = [{"category": "early", "text": "猫だよ"}, {"category": "mid", "text": "猫が好き"}]
    mode._nudge_spoken = list(streak)
    mode._nudge_streak_count = 2
    mode._last_nudge_fire_ts = 123.0

    print("\n--- Fix-9a: overdue/internal nudge CARRIES silence self-memory ---")
    mode._apply_dispatch_streak_state(is_internal_nudge=True)
    check("overdue nudge sets llm.nudge_self_memory from _nudge_spoken",
          mode.llm.nudge_self_memory == streak, f"got={mode.llm.nudge_self_memory}")
    check("overdue nudge does NOT reset streak count",
          mode._nudge_streak_count == 2, f"streak={mode._nudge_streak_count}")
    check("overdue nudge does NOT clear _nudge_spoken",
          mode._nudge_spoken == streak)
    check("nudge_self_memory is a separate list (copy, not alias)",
          mode.llm.nudge_self_memory is not mode._nudge_spoken)

    print("\n--- Real user turn: streak fully resets ---")
    mode._apply_dispatch_streak_state(is_internal_nudge=False)
    check("real user turn clears nudge_self_memory", mode.llm.nudge_self_memory == [])
    check("real user turn resets streak count", mode._nudge_streak_count == 0)
    check("real user turn clears _nudge_spoken", mode._nudge_spoken == [])
    check("real user turn resets _last_nudge_fire_ts", mode._last_nudge_fire_ts == 0.0)


async def _amain() -> None:
    run_tests()


if __name__ == "__main__":
    asyncio.run(_amain())
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
