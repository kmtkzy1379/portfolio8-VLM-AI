"""Diagnostic (NO real API): does a SILENCE nudge actually inject RAG + screen into the prompt?

Drives the REAL pipeline (_process_idle_input -> process_input -> generate_stream ->
_build_system_prompt) with a fake LLM client that captures the exact system prompt the
model would receive. Confirms whether screen (VLM) + 2 RAG memories survive into the
silence-nudge prompt (the user's hypothesis: they don't, so Eve never talks from them).

Run:  $env:PYTHONIOENCODING="utf-8"; venv\\Scripts\\python.exe tools\\diag_silence_inject.py
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import Config  # noqa: E402
Config.GROQ_API_KEY = Config.GROQ_API_KEY or "x"
Config.OPENAI_API_KEY = Config.OPENAI_API_KEY or "x"

from modules.llm import LLMHandler  # noqa: E402
from modes.talk_mode import TalkMode  # noqa: E402

RAG_RELEVANT = [{"user": "ローグライクの話", "ai": "死に覚えゲーだね"},
                {"user": "コーヒーは？", "ai": "ブラック派"}]
RAG_RANDOM = [{"user": "rnd1", "ai": "a1"}, {"user": "rnd2", "ai": "a2"}]
SCREEN = "[たった今/MAJOR] コードエディタでPythonを編集している"


def _aret(v):
    async def f():
        return v
    return f()


class FakeClient:
    """OpenAI-compatible: records the system prompt, returns a no-tool '…' response."""
    def __init__(self, sink):
        self._sink = sink
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        msgs = kw.get("messages") or []
        if msgs and msgs[0].get("role") == "system":
            self._sink["system_prompt"] = msgs[0]["content"]
        msg = SimpleNamespace(content="…", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _make_mode(sink, recent_turns, vlm_on=True):
    llm = LLMHandler()
    llm.enable_meta_tools(True)            # route through _tool_augmented_stream (1 non-stream call)
    llm.set_task_manager(None)
    llm.client = FakeClient(sink)
    llm._fallback_client = FakeClient(sink)

    mode = TalkMode()
    mode.llm = llm
    mode.rag = SimpleNamespace(
        search_similar=lambda q, *a, **k: _aret(list(RAG_RELEVANT)),
        get_random_turns=lambda count=2: _aret(list(RAG_RANDOM)),
    )
    mode.conversation_cache = SimpleNamespace(
        get_recent_turns=lambda count=5, exclude_ellipsis=False: _aret(list(recent_turns)),
        get_silence_summary=lambda: _aret({"silence_seconds": 30, "ellipsis_count": 1, "last_real_user_ts": None}),
        add_turn=lambda u, a: _aret(None),
    )
    mode.tts = SimpleNamespace(generate_audio=lambda s: _aret(b""))
    mode.player = SimpleNamespace(add_to_queue=lambda w: None,
                                  queue=SimpleNamespace(qsize=lambda: 0, empty=lambda: True),
                                  is_playing=False, interrupt_signal=False)
    mode.feedback = SimpleNamespace(signal_turn_done=lambda: None)
    mode.task_manager = None
    mode.vlm_bridge = (SimpleNamespace(is_running=True, get_scene_description=lambda: SCREEN)
                       if vlm_on else None)
    mode.running = True
    return mode


async def _check(label, recent_turns, vlm_on):
    sink = {}
    mode = _make_mode(sink, recent_turns, vlm_on)
    await mode._process_idle_input("…", is_silence_nudge=True)
    p = sink.get("system_prompt", "")
    rag_in = "[Long-term Memory from RAG]" in p
    screen_in = "Screen Recognition" in p
    proactive_in = "沈黙時の能動判断" in p
    rag_n = list(mode.llm.rag_memories or [])
    print(f"\n=== {label} ===")
    print(f"  llm.rag_memories set: {len(rag_n)} 件  {[m.get('user') for m in rag_n]}")
    print(f"  prompt contains RAG block:      {rag_in}")
    print(f"  prompt contains Screen block:   {screen_in}")
    print(f"  prompt contains proactive block:{proactive_in}")
    print(f"  system prompt length: {len(p)} chars")
    return rag_in, screen_in


async def main():
    print("Silence-nudge injection diagnostic (no API). Does screen + RAG reach the prompt?")
    # Case 1: a recent real user turn exists -> search_similar path
    r1 = await _check("Case 1: recent real turn present (search_similar)",
                      [{"user": "最近ローグライク始めた", "ai": "いいね", "user_timestamp": "", "ai_timestamp": ""}],
                      vlm_on=True)
    # Case 2: cold start, no recent real turn -> get_random_turns fallback
    r2 = await _check("Case 2: no recent real turn (random fallback)", [], vlm_on=True)
    # Case 3: VLM off
    r3 = await _check("Case 3: VLM off (screen absent, RAG should still load)", [], vlm_on=False)

    print("\n--- VERDICT ---")
    print(f"  RAG reaches silence prompt: case1={r1[0]} case2={r2[0]} case3={r3[0]}")
    print(f"  Screen reaches silence prompt: case1={r1[1]} case2={r2[1]} (case3 expected False={r3[1]})")


if __name__ == "__main__":
    asyncio.run(main())
