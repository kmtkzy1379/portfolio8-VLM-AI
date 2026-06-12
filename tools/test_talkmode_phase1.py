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


def run_greeting_tests() -> None:
    print("\n--- Bug-E: _is_greeting matrix (tag-only detector) ---")
    g = TalkMode._is_greeting
    for text, expect in [
        ("こんばんは！", True), ("こんばんわ、えへへ", True), ("こんにちは", True),
        ("おはよー", True), ("やっほー、来たよ", True), ("ハロー！", True),
        ("うん", False), ("そうなんだ", False), ("猫だよ、強い", False),
        ("ローグライク進んだ？", False), ("", False),
    ]:
        check(f"_is_greeting({text!r}) == {expect}", g(text) == expect)

    print("\n--- Bug-E: greeting spoken on a silence nudge is tagged [greeting] ---")

    def _aret(v):
        async def f():
            return v
        return f()

    class GreetLLM:
        def __init__(self):
            self.history = []
            self.rag_memories = []
            self.recent_turns = []
            self.silence_summary = None
            self.committed_facts = []
            self.nudge_self_memory = []
            self.vlm_context = ""
            self.one_shot_context = ""

        def has_unseen_vlm_alerts(self, *_a, **_k):
            return False

        async def generate_stream(self, text, is_internal_nudge=False):
            yield "こんばんは、えへへ。"

    mode = TalkMode()
    mode.llm = GreetLLM()
    mode.rag = SimpleNamespace(
        search_similar=lambda q, *a, **k: _aret([]),
        get_random_turns=lambda count=2: _aret([]),
    )
    mode.conversation_cache = SimpleNamespace(
        get_recent_turns=lambda count=5, exclude_ellipsis=False: _aret([]),
        get_silence_summary=lambda: _aret({"silence_seconds": 12, "ellipsis_count": 0, "last_real_user_ts": None}),
        add_turn=lambda u, a: _aret(None),
    )
    mode.tts = SimpleNamespace(generate_audio=lambda s: _aret(b""))
    mode.player = SimpleNamespace(
        add_to_queue=lambda w: None,
        queue=SimpleNamespace(qsize=lambda: 0, empty=lambda: True),
        is_playing=False, interrupt_signal=False,
    )
    mode.feedback = SimpleNamespace(signal_turn_done=lambda: None)
    mode.task_manager = None
    mode.vlm_bridge = None
    mode.running = True
    asyncio.get_event_loop()  # noqa: F841 (loop already running via _amain)
    return mode


async def _greeting_tag_async(mode) -> None:
    await mode._process_idle_input("…", is_silence_nudge=True)
    check("greeting response recorded with [greeting] tag",
          mode._nudge_spoken and mode._nudge_spoken[-1]["category"] == "greeting",
          f"spoken={mode._nudge_spoken}")


async def _amain() -> None:
    run_tests()
    mode = run_greeting_tests()
    await _greeting_tag_async(mode)


if __name__ == "__main__":
    asyncio.run(_amain())
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
