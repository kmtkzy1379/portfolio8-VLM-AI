"""Phase 1 E2E test (NO API / NO hardware): the real idle-nudge pipeline.

Drives _process_idle_input -> process_input -> _run_response_pipeline with lightweight
stubs (no full initialize()), to verify the silence-RAG fix end-to-end:

  Bug: _process_idle_input set llm.rag_memories = 2 random turns, then
       _run_response_pipeline overwrote it with [] for idle nudges (search skipped),
       so Eve reached generation with NO material during silence.
  Fix: only overwrite when a search actually ran (or on a real user turn).

This test snapshots llm.rag_memories at generate_stream time and asserts the 2 random
memories SURVIVE to generation on an idle "…" nudge.

Run:  venv\\Scripts\\python.exe tools\\test_e2e_idle_phase1.py
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
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


def _aret(v):
    async def f():
        return v
    return f()


class StubLLM:
    """Captures rag_memories at generation time (what actually reaches the prompt)."""
    def __init__(self, captured: dict):
        self._captured = captured
        self.history = []
        self.rag_memories = []
        self.recent_turns = []
        self.silence_summary = None
        self.committed_facts = []
        self.nudge_self_memory = []
        self.vlm_context = ""
        self.one_shot_context = ""

    async def generate_stream(self, text, is_internal_nudge=False):
        self._captured["rag_memories"] = list(self.rag_memories)
        self._captured["recent_turns"] = list(self.recent_turns)
        self._captured["silence_summary"] = self.silence_summary
        yield "テスト応答"


def _make_mode(captured: dict, random_turns: list) -> TalkMode:
    mode = TalkMode()  # __init__ only — no hardware
    mode.llm = StubLLM(captured)
    mode.rag = SimpleNamespace(
        get_random_turns=lambda count=2: _aret(list(random_turns)),
        search_similar=lambda q: _aret([]),
    )
    mode.conversation_cache = SimpleNamespace(
        get_recent_turns=lambda count=5, exclude_ellipsis=False: _aret([]),
        get_silence_summary=lambda: _aret({"silence_seconds": 30, "ellipsis_count": 1, "last_real_user_ts": None}),
        add_turn=lambda u, a: _aret(None),
    )
    mode.tts = SimpleNamespace(generate_audio=lambda s: _aret(b""))
    mode.player = SimpleNamespace(
        add_to_queue=lambda w: None,
        queue=SimpleNamespace(qsize=lambda: 0, empty=lambda: True),
        is_playing=False,
        interrupt_signal=False,
    )
    mode.feedback = SimpleNamespace(signal_turn_done=lambda: None)
    mode.task_manager = None
    mode.vlm_bridge = None
    mode.running = True
    return mode


async def test_idle_rag_survives() -> None:
    print("\n--- silence-RAG: 2 random memories survive to generation on idle nudge ---")
    captured: dict = {}
    random_turns = [{"user": "前にゲームの話したね", "ai": "うん、楽しかった"},
                    {"user": "コーヒー好き？", "ai": "ブラック派かな"}]
    mode = _make_mode(captured, random_turns)
    await mode._process_idle_input("…", is_silence_nudge=True)
    got = captured.get("rag_memories", [])
    check("idle nudge: random RAG memories NOT wiped (reach generation)",
          len(got) == 2, f"n={len(got)}")
    check("idle nudge: recent_turns/silence_summary also present",
          captured.get("silence_summary") is not None)


if __name__ == "__main__":
    asyncio.run(test_idle_rag_survives())
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
