"""Phase-3 Tier-1 test: deterministic CODE GATES for Bug-B / Bug-E (NO API).

Prompt rules alone could not stop gpt-5.4-mini from (B) fulfilling a pending reservation
early on silence nudges, or (E) re-greeting — confirmed in Tier-3. These gates enforce it
in code:
  B: while a PENDING reservation's deadline is within IDLE_SUPPRESS_PENDING_WINDOW_SEC,
     silence "…" nudges do not fire at all (wait quietly; no LLM call).
  E: on internal nudges, if a greeting already happened in recent_turns, a LEADING
     greeting sentence is stripped before TTS / recording.

Run:  venv\\Scripts\\python.exe tools\\test_code_gates_phase1.py
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
from modes.talk_mode import TalkMode  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


def iso(dt: datetime) -> str:
    return dt.isoformat()


def _aret(v):
    async def f():
        return v
    return f()


class CountingLLM:
    def __init__(self, sentences):
        self._sentences = list(sentences)
        self.calls = 0
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
        self.calls += 1
        for s in self._sentences:
            yield s


def _make_mode(llm, tm=None, recent=None):
    mode = TalkMode()
    mode.llm = llm
    mode.rag = SimpleNamespace(
        search_similar=lambda q, *a, **k: _aret([]),
        get_random_turns=lambda count=2: _aret([]),
    )
    mode.conversation_cache = SimpleNamespace(
        get_recent_turns=lambda count=5, exclude_ellipsis=False: _aret(list(recent or [])),
        get_silence_summary=lambda: _aret({"silence_seconds": 12, "ellipsis_count": 0, "last_real_user_ts": None}),
        add_turn=lambda u, a: _aret(None),
    )
    mode.tts = SimpleNamespace(generate_audio=lambda s: _aret(b""))
    mode.player = SimpleNamespace(
        add_to_queue=lambda w: None,
        queue=SimpleNamespace(qsize=lambda: 0, empty=lambda: True),
        is_playing=False, interrupt_signal=False,
    )
    mode.feedback = SimpleNamespace(signal_turn_done=lambda: None,
                                    audit_provisional_instruction=lambda **k: _aret(None))
    mode.task_manager = tm
    mode.vlm_bridge = None
    mode.running = True
    return mode


async def _make_tm():
    fd, path = tempfile.mkstemp(suffix="_gates.jsonl")
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


async def _add(tm, instruction, deadline_at):
    tm.enqueue_command_nowait({
        "kind": "set_active_instruction", "instruction": instruction,
        "deadline_at": deadline_at, "reason": "t", "created_by": "ai1",
    })
    await tm.flush_command_queue()


async def test_b_gate() -> None:
    print("\n--- Bug-B code gate: imminent PENDING suppresses silence nudges ---")
    tm, path = await _make_tm()
    try:
        now = datetime.now()
        # 期限 30s 後（window 120s 内）の PENDING
        await _add(tm, "好きな魚を答える", iso(now + timedelta(seconds=30)))
        check("helper: imminent pending detected",
              tm.has_imminent_pending_instruction(window_sec=120))
        check("helper: tight window excludes it",
              not tm.has_imminent_pending_instruction(window_sec=5))

        llm = CountingLLM(["サーモン。"])
        mode = _make_mode(llm, tm=tm)
        streak0 = mode._nudge_streak_count
        await mode._process_idle_input("…", is_silence_nudge=True)
        check("silence nudge SUPPRESSED (no LLM call)", llm.calls == 0, f"calls={llm.calls}")
        check("backoff streak NOT advanced", mode._nudge_streak_count == streak0)

        # VLM nudge は対象外（画面反応は許可）
        await mode._process_idle_input("[内部: 画面に新しい変化があった - 自然にリアクションして]",
                                       is_silence_nudge=False)
        check("VLM nudge still fires", llm.calls == 1, f"calls={llm.calls}")

        # 期限到来（ACTIVE 化）→ PENDING でなくなる → 沈黙 nudge 再開
        iid = next(iter(tm._active_instructions.keys()))
        tm._active_instructions[iid].deadline_at = iso(now - timedelta(seconds=1))
        await tm._reconcile_instruction_status()
        await mode._process_idle_input("…", is_silence_nudge=True)
        check("after ACTIVE, silence nudge resumes", llm.calls == 2, f"calls={llm.calls}")
    finally:
        await _teardown(tm, path)

    # 遠い deadline（window 外）は抑制しない
    tm2, path2 = await _make_tm()
    try:
        await _add(tm2, "明日の話をする", iso(datetime.now() + timedelta(seconds=300)))
        llm2 = CountingLLM(["なにか話す。"])
        mode2 = _make_mode(llm2, tm=tm2)
        await mode2._process_idle_input("…", is_silence_nudge=True)
        check("far-deadline pending does NOT suppress", llm2.calls == 1, f"calls={llm2.calls}")
    finally:
        await _teardown(tm2, path2)


async def test_e_filter() -> None:
    print("\n--- Bug-E code gate: leading greeting stripped on internal nudge ---")
    greeted = [{"user": "こんばんは", "ai": "こんばんは、えへへ。", "user_timestamp": "", "ai_timestamp": ""}]

    # 1) 挨拶済み + 内部 nudge → 先頭の挨拶文だけ剥がれ、本文は残る
    llm = CountingLLM(["こんばんは、えへへ。", "今日はゆるっといこ。"])
    mode = _make_mode(llm, recent=greeted)
    resp = await mode.process_input("…", is_internal_nudge=True)
    check("leading greeting stripped from response",
          resp == "今日はゆるっといこ。", f"resp={resp!r}")

    # 2) 挨拶済みでも実ユーザターンは対象外（ユーザーに挨拶し返すのは正当）
    llm2 = CountingLLM(["こんばんは、えへへ。", "今日はゆるっといこ。"])
    mode2 = _make_mode(llm2, recent=greeted)
    resp2 = await mode2.process_input("こんばんは", is_internal_nudge=False)
    check("real user turn NOT filtered",
          resp2.startswith("こんばんは、えへへ。"), f"resp={resp2!r}")

    # 3) 未挨拶セッションでは内部 nudge でも剥がさない（初回挨拶は正当）
    llm3 = CountingLLM(["こんばんは、えへへ。", "今日はゆるっといこ。"])
    mode3 = _make_mode(llm3, recent=[])
    resp3 = await mode3.process_input("…", is_internal_nudge=True)
    check("first-greeting session NOT filtered",
          resp3.startswith("こんばんは、えへへ。"), f"resp={resp3!r}")

    # 4) 応答全体が挨拶1文のみ → 空応答（黙る）になる
    llm4 = CountingLLM(["こんばんは、えへへ。"])
    mode4 = _make_mode(llm4, recent=greeted)
    resp4 = await mode4.process_input("…", is_internal_nudge=True)
    check("greeting-only response becomes empty (stay quiet)",
          (resp4 or "").strip() == "", f"resp={resp4!r}")

    # 5) 非挨拶の先頭文はそのまま
    llm5 = CountingLLM(["ローグライク、進んだ？"])
    mode5 = _make_mode(llm5, recent=greeted)
    resp5 = await mode5.process_input("…", is_internal_nudge=True)
    check("non-greeting first sentence untouched",
          resp5 == "ローグライク、進んだ？", f"resp={resp5!r}")


async def main() -> None:
    await test_b_gate()
    await test_e_filter()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
