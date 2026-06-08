"""Phase 1 Tier-1 test: proactive silence MATERIAL (Lever 1) — NO API / NO hardware.

Drives the real `_process_idle_input` with stubs and asserts the silence RAG material is now
RELEVANT (search_similar queried by the last real user turn / VLM scene) instead of random,
that the relevant memories SURVIVE the internal-nudge pipeline (base_mode:420 property), and
that cold-start / search failure degrade gracefully to the random fallback.

Run:  venv\\Scripts\\python.exe tools\\test_proactivity_phase1.py
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
RELEVANT = [{"user": "rel-q", "ai": "rel-a"}]   # sentinel returned by search_similar
RANDOM = [{"user": "rnd-q", "ai": "rnd-a"}]      # sentinel returned by get_random_turns


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


def _aret(v):
    async def f():
        return v
    return f()


class StubLLM:
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
        yield "…"


class RecordingRAG:
    def __init__(self, raise_on_search=False):
        self.queries: list[str] = []
        self.random_calls = 0
        self.raise_on_search = raise_on_search

    async def search_similar(self, query, *a, **k):
        self.queries.append(query)
        if self.raise_on_search:
            raise RuntimeError("boom")
        return list(RELEVANT)

    async def get_random_turns(self, count=2):
        self.random_calls += 1
        return list(RANDOM)


def _make_mode(rag, recent_turns) -> TalkMode:
    mode = TalkMode()
    mode.llm = StubLLM()
    mode.rag = rag
    mode.conversation_cache = SimpleNamespace(
        get_recent_turns=lambda count=5, exclude_ellipsis=False: _aret(list(recent_turns)),
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
    return mode


async def test_relevant_query_used_and_survives() -> None:
    print("\n--- silence material: relevant RAG queried by last user turn, survives pipeline ---")
    rag = RecordingRAG()
    mode = _make_mode(rag, [{"user": "さっきローグライク始めたんだ", "ai": "へえ"}])
    await mode._process_idle_input("…", is_silence_nudge=True)
    check("search_similar was queried", len(rag.queries) == 1, f"queries={rag.queries}")
    check("query built from last real user turn",
          rag.queries and "ローグライク" in rag.queries[0], f"q={rag.queries[:1]}")
    check("relevant memories survived to llm (not random, not wiped)",
          mode.llm.rag_memories == RELEVANT, f"got={mode.llm.rag_memories}")
    check("random fallback NOT used when relevant hit", rag.random_calls == 0)


async def test_coldstart_falls_back_to_random() -> None:
    print("\n--- silence material: cold-start (no real turn, no VLM) -> random fallback ---")
    rag = RecordingRAG()
    mode = _make_mode(rag, [])  # no recent turns
    await mode._process_idle_input("…", is_silence_nudge=True)
    check("no query built when nothing to query", len(rag.queries) == 0, f"queries={rag.queries}")
    check("random fallback used", rag.random_calls == 1 and mode.llm.rag_memories == RANDOM,
          f"got={mode.llm.rag_memories}")


async def test_search_error_falls_back_to_random() -> None:
    print("\n--- silence material: search_similar error -> random fallback (no crash) ---")
    rag = RecordingRAG(raise_on_search=True)
    mode = _make_mode(rag, [{"user": "今日は疲れた", "ai": "おつかれ"}])
    await mode._process_idle_input("…", is_silence_nudge=True)
    check("search attempted then failed", len(rag.queries) == 1)
    check("random fallback used on error", rag.random_calls == 1 and mode.llm.rag_memories == RANDOM,
          f"got={mode.llm.rag_memories}")


async def main() -> None:
    await test_relevant_query_used_and_survives()
    await test_coldstart_falls_back_to_random()
    await test_search_error_falls_back_to_random()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
