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
    def __init__(self, raise_on_search=False, raise_on_random=False):
        self.queries: list[str] = []
        self.random_calls = 0
        self.raise_on_search = raise_on_search
        self.raise_on_random = raise_on_random

    async def search_similar(self, query, *a, **k):
        self.queries.append(query)
        if self.raise_on_search:
            raise RuntimeError("boom")
        return list(RELEVANT)

    async def get_random_turns(self, count=2):
        self.random_calls += 1
        if self.raise_on_random:
            raise RuntimeError("boom-random")
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


async def test_random_seed_used_and_survives() -> None:
    # 2026-06-14 改訂: 自律的発話は「直近の続き(=二重返事)」でなく「別の新しい話題」を狙う。
    # そのため沈黙時の RAG 材料は last-turn 関連検索ではなく“少し前のランダムな記憶”を種にする。
    print("\n--- silence material: random (non-recent) memory seed, survives pipeline ---")
    rag = RecordingRAG()
    mode = _make_mode(rag, [{"user": "さっきローグライク始めたんだ", "ai": "へえ"}])
    await mode._process_idle_input("…", is_silence_nudge=True)
    check("random memory fetched as new-topic seed", rag.random_calls == 1)
    check("search_similar NOT used for silence (avoids continuation)", len(rag.queries) == 0,
          f"queries={rag.queries}")
    check("random memories survived to llm (not wiped)",
          mode.llm.rag_memories == RANDOM, f"got={mode.llm.rag_memories}")


async def test_random_fetch_error_empty() -> None:
    print("\n--- silence material: random fetch error -> empty, no crash ---")
    rag = RecordingRAG(raise_on_random=True)
    mode = _make_mode(rag, [{"user": "今日は疲れた", "ai": "おつかれ"}])
    await mode._process_idle_input("…", is_silence_nudge=True)
    check("random fetch attempted", rag.random_calls == 1)
    check("error -> rag_memories empty (no crash, no continuation seed)",
          mode.llm.rag_memories == [], f"got={mode.llm.rag_memories}")


async def main() -> None:
    await test_random_seed_used_and_survives()
    await test_random_fetch_error_empty()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
