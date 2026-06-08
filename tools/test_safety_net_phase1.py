"""Phase 1 Tier-1 test: task-completion safety net (NO API / NO hardware).

Drives the real overdue pipeline (process_input -> _run_response_pipeline) with a stubbed
LLM that produces a SUBSTANTIVE answer but does NOT call clear_active_instruction. The
safety net must auto-complete the instruction (-> PROVISIONAL_DONE) so completion does not
depend on the LLM remembering the tool call. A non-substantive response must NOT auto-complete.

Run:  venv\\Scripts\\python.exe tools\\test_safety_net_phase1.py
"""
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from modules.task_manager import TaskManager  # noqa: E402
from modules.task_schema import InstructionStatus  # noqa: E402
from modes.talk_mode import TalkMode  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


def _aret(v):
    async def f():
        return v
    return f()


def iso(dt: datetime) -> str:
    return dt.isoformat()


class StubLLM:
    def __init__(self, response: str):
        self._response = response
        self.history = []
        self.rag_memories = []
        self.recent_turns = []
        self.silence_summary = None
        self.committed_facts = []
        self.nudge_self_memory = []
        self.vlm_context = ""
        self.one_shot_context = ""

    async def generate_stream(self, text, is_internal_nudge=False):
        # Substantive answer, but deliberately does NOT call any clear tool.
        yield self._response


class StubFeedback:
    def signal_turn_done(self):
        pass

    async def audit_provisional_instruction(self, **kwargs):
        # No real AI2 call in Tier-1; the PROVISIONAL_DONE transition is what we assert.
        return None


def _make_mode(tm: TaskManager, response: str) -> TalkMode:
    mode = TalkMode()
    mode.llm = StubLLM(response)
    mode.rag = SimpleNamespace(
        search_similar=lambda q: _aret([]),
        get_random_turns=lambda count=2: _aret([]),
    )
    mode.conversation_cache = SimpleNamespace(
        get_recent_turns=lambda count=5, exclude_ellipsis=False: _aret([]),
        get_silence_summary=lambda: _aret({"silence_seconds": 1, "ellipsis_count": 0, "last_real_user_ts": None}),
        add_turn=lambda u, a: _aret(None),
    )
    mode.tts = SimpleNamespace(generate_audio=lambda s: _aret(b""))
    mode.player = SimpleNamespace(
        add_to_queue=lambda w: None,
        queue=SimpleNamespace(qsize=lambda: 0, empty=lambda: True),
        is_playing=False,
        interrupt_signal=False,
    )
    mode.feedback = StubFeedback()
    mode.task_manager = tm
    mode.vlm_bridge = None
    mode.running = True
    return mode


async def _make_overdue(tm: TaskManager) -> str:
    now = datetime.now()
    # Register with a FUTURE deadline (so the B2 clamp doesn't null it), then force it overdue
    # by setting the field directly + reconcile -> ACTIVE (same pattern as the timing test).
    tm.enqueue_command_nowait({
        "kind": "set_active_instruction", "instruction": "好きな動物を答える",
        "deadline_at": iso(now + timedelta(seconds=30)), "reason": "test", "created_by": "ai1",
    })
    await tm.flush_command_queue()
    iid = next(iter(tm._active_instructions.keys()))
    tm._active_instructions[iid].deadline_at = iso(now - timedelta(seconds=1))
    await tm._reconcile_instruction_status()  # PENDING -> ACTIVE
    return iid


async def _make_tm():
    fd, path = tempfile.mkstemp(suffix="_safety.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()
    tm = TaskManager(tasks_file=path)
    await tm.start()
    return tm, path


async def _teardown(tm, path):
    try:
        await tm.stop()
    except Exception:
        pass
    await asyncio.sleep(0.05)
    try:
        os.remove(path)
    except OSError:
        pass


async def test_substantive_auto_completes() -> None:
    print("\n--- Safety net: substantive overdue answer WITHOUT clear -> auto PROVISIONAL_DONE ---")
    tm, path = await _make_tm()
    try:
        iid = await _make_overdue(tm)
        check("instruction is ACTIVE before turn",
              tm._active_instructions[iid].status == InstructionStatus.ACTIVE)
        mode = _make_mode(tm, "猫だよ、気まぐれなところが好き")
        await mode.process_input(f"[内部: 期限超過 {iid} を履行 — 好きな動物を答える]", is_internal_nudge=True)
        await tm.flush_command_queue()
        inst = tm.get_instruction(iid)
        check("auto-completed to PROVISIONAL_DONE (completion NOT lost)",
              inst and inst["status"] == InstructionStatus.PROVISIONAL_DONE.value,
              f"status={inst['status'] if inst else None}")
    finally:
        await _teardown(tm, path)


async def test_nonsubstantive_stays_active() -> None:
    print("\n--- Safety net: non-substantive '…' must NOT auto-complete ---")
    tm, path = await _make_tm()
    try:
        iid = await _make_overdue(tm)
        mode = _make_mode(tm, "…")
        await mode.process_input(f"[内部: 期限超過 {iid} を履行 — 好きな動物を答える]", is_internal_nudge=True)
        await tm.flush_command_queue()
        inst = tm.get_instruction(iid)
        check("stays ACTIVE (no auto-complete on aizuchi)",
              inst and inst["status"] == InstructionStatus.ACTIVE.value,
              f"status={inst['status'] if inst else None}")
    finally:
        await _teardown(tm, path)


async def main() -> None:
    await test_substantive_auto_completes()
    await test_nonsubstantive_stays_active()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
