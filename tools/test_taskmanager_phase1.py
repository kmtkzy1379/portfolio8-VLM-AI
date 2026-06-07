"""Phase 1 standalone TaskManager tests (NO audio / NO LLM / NO hardware).

Deterministic checks of the instruction-lifecycle logic that the Phase-1 fixes
depend on. Drives the real TaskManager against a throwaway temp tasks.jsonl.

Covers:
- Fix-B2: a previous-day / past deadline_at is corrected to the future (or nulled),
  so the instruction does NOT become instantly ACTIVE (the +14s early-fire bug class).
- Correct timing: stays PENDING before the deadline (callback does NOT fire early),
  flips to ACTIVE + fires the activation callback exactly once when the deadline passes.

Run:  venv\\Scripts\\python.exe tools\\test_taskmanager_phase1.py
"""
import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta

# repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from modules.task_manager import TaskManager  # noqa: E402
from modules.task_schema import InstructionStatus  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))


def iso(dt: datetime) -> str:
    return dt.isoformat()


async def _make_tm() -> tuple[TaskManager, str]:
    fd, path = tempfile.mkstemp(suffix="_tasks.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()  # start empty
    tm = TaskManager(tasks_file=path)
    await tm.start()
    return tm, path


async def _add_instruction(tm: TaskManager, instruction: str, deadline_at: str) -> None:
    tm.enqueue_command_nowait({
        "kind": "set_active_instruction",
        "instruction": instruction,
        "deadline_at": deadline_at,
        "reason": "test",
        "created_by": "ai1",
    })
    await tm.flush_command_queue()


async def _teardown(tm: TaskManager, path: str) -> None:
    try:
        await tm.stop()
    except Exception:
        pass
    await asyncio.sleep(0.05)
    try:
        os.remove(path)
    except OSError:
        pass


async def test_fix_b2_previous_day() -> None:
    print("\n--- Fix-B2: previous-day deadline must NOT instant-activate ---")
    tm, path = await _make_tm()
    try:
        now = datetime.now()
        # Reproduce the observed bug input: LLM emits "yesterday + 30s" for a "30秒後" promise.
        buggy = now - timedelta(days=1) + timedelta(seconds=30)
        await _add_instruction(tm, "好きな動物を答える", iso(buggy))

        insts = list(tm._active_instructions.values())
        check("registered exactly one instruction", len(insts) == 1, f"n={len(insts)}")
        if not insts:
            return
        inst = insts[0]
        if inst.deadline_at is None:
            check("past deadline handled (nulled -> default expire)", True, "deadline_at=None")
        else:
            dldt = datetime.fromisoformat(inst.deadline_at)
            check("corrected deadline is in the future", dldt > now, f"deadline_at={inst.deadline_at}")
            check("corrected deadline ~ +30s (not +1 day off)",
                  abs((dldt - now).total_seconds()) < 120,
                  f"delta={(dldt - now).total_seconds():.0f}s")

        prompt = tm.get_active_instructions_for_prompt()
        derived = [p.get("derived_status") for p in prompt]
        check("not instant-ACTIVE", all(d != "active" for d in derived), f"derived={derived}")
    finally:
        await _teardown(tm, path)


async def test_correct_timing() -> None:
    print("\n--- Timing: PENDING before deadline, ACTIVE + 1 callback after ---")
    tm, path = await _make_tm()
    fired: list[dict] = []
    tm.register_instruction_activated_callback(lambda d: fired.append(d))
    try:
        now = datetime.now()
        await _add_instruction(tm, "30秒後に好きな動物を答える", iso(now + timedelta(seconds=30)))
        insts = list(tm._active_instructions.values())
        check("registered exactly one instruction", len(insts) == 1, f"n={len(insts)}")
        if not insts:
            return
        inst = insts[0]

        await tm._reconcile_instruction_status()
        check("PENDING before deadline", inst.status == InstructionStatus.PENDING, f"status={inst.status.value}")
        check("callback NOT fired early", len(fired) == 0, f"fired={len(fired)}")

        # Advance time deterministically by moving the deadline into the past.
        inst.deadline_at = iso(now - timedelta(seconds=1))
        await tm._reconcile_instruction_status()
        check("ACTIVE after deadline", inst.status == InstructionStatus.ACTIVE, f"status={inst.status.value}")
        check("callback fired exactly once", len(fired) == 1, f"fired={len(fired)}")
        if fired:
            payload = fired[0]
            check("callback payload carries instruction text",
                  payload.get("instruction") == "30秒後に好きな動物を答える",
                  f"instruction={payload.get('instruction')!r}")

        # Idempotency: another reconcile must not double-fire.
        await tm._reconcile_instruction_status()
        check("no double-fire on re-reconcile", len(fired) == 1, f"fired={len(fired)}")
    finally:
        await _teardown(tm, path)


async def _commit_fact(tm: TaskManager, topic: str, answer: str,
                       iid: str | None = None, scope: str = "eve", source: str = "nudge") -> None:
    tm.enqueue_command_nowait({
        "kind": "commit_fact",
        "scope": scope,
        "topic_norm": topic,
        "answer_text": answer,
        "instruction_id": iid,
        "source": source,
    })
    await tm.flush_command_queue()


async def test_committed_fact_defend() -> None:
    print("\n--- Fix-9b: committed-fact store DEFENDS the first answer (anti-drift) ---")
    tm, path = await _make_tm()
    try:
        await _commit_fact(tm, "好きな動物", "猫だよ", iid="i_x")
        facts = tm.get_committed_facts_for_prompt()
        check("first commit stored", facts == [{"topic": "好きな動物", "answer": "猫だよ"}], f"facts={facts}")

        # Drift attempt: same topic, different answer -> must be DEFENDED (not overwritten).
        await _commit_fact(tm, "好きな動物", "うさぎ", iid="i_x")
        facts = tm.get_committed_facts_for_prompt()
        check("drift defended: still 猫だよ (not うさぎ)",
              facts == [{"topic": "好きな動物", "answer": "猫だよ"}], f"facts={facts}")
        n_active = len([f for f in tm._committed_facts.values() if f.status == "active"])
        check("no duplicate fact created", n_active == 1, f"active={n_active}")

        # A different topic coexists.
        await _commit_fact(tm, "好きな色", "青", iid="i_y")
        topics = {f["topic"]: f["answer"] for f in tm.get_committed_facts_for_prompt()}
        check("second topic coexists, first unchanged",
              topics.get("好きな色") == "青" and topics.get("好きな動物") == "猫だよ", f"topics={topics}")
    finally:
        await _teardown(tm, path)


async def test_committed_fact_recovery() -> None:
    print("\n--- Fix-9b: committed facts persist across restart (persona-durable) ---")
    fd, path = tempfile.mkstemp(suffix="_tasks.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()
    tm1 = TaskManager(tasks_file=path)
    await tm1.start()
    try:
        await _commit_fact(tm1, "好きな動物", "猫だよ", iid="i_x")
        await tm1._write_queue.join()  # ensure persisted to disk before restart
    finally:
        try:
            await tm1.stop()
        except Exception:
            pass
    await asyncio.sleep(0.05)

    tm2 = TaskManager(tasks_file=path)  # new instance, same file
    await tm2.start()
    try:
        facts = tm2.get_committed_facts_for_prompt()
        check("fact restored after restart",
              facts == [{"topic": "好きな動物", "answer": "猫だよ"}], f"facts={facts}")
    finally:
        await _teardown(tm2, path)


async def test_capture_maybe_commit_fact() -> None:
    print("\n--- Fix-9b: capture (_maybe_commit_fact) routes Eve's answer to the store ---")
    from modes.talk_mode import TalkMode
    mode = TalkMode()  # __init__ only -> no hardware
    tm, path = await _make_tm()
    mode.task_manager = tm
    try:
        now = datetime.now()
        await _add_instruction(tm, "好きな動物を答える", iso(now + timedelta(seconds=30)))
        iid = next(iter(tm._active_instructions.keys()))
        topic = tm._normalize_instruction("好きな動物を答える")

        # (a) idle/normal path: a single active candidate -> answer attributed to it.
        mode._maybe_commit_fact("…", "猫だよ", is_internal_nudge=True)
        await tm.flush_command_queue()
        facts = tm.get_committed_facts_for_prompt()
        check("idle capture stored 猫 under the instruction topic",
              facts == [{"topic": topic, "answer": "猫だよ"}], f"facts={facts}")

        # (b) overdue path: id embedded in input_text; a drift answer must be DEFENDED.
        mode._maybe_commit_fact(f"[内部: 期限超過 {iid} を履行 — 好きな動物を答える]", "うさぎ",
                                is_internal_nudge=True)
        await tm.flush_command_queue()
        facts = tm.get_committed_facts_for_prompt()
        check("overdue drift defended via capture (still 猫)",
              facts == [{"topic": topic, "answer": "猫だよ"}], f"facts={facts}")

        # (c) a non-substantive answer ("うん") must NOT be captured.
        mode._maybe_commit_fact("…", "うん", is_internal_nudge=True)
        await tm.flush_command_queue()
        n_active = len([f for f in tm._committed_facts.values() if f.status == "active"])
        check("non-substantive answer not captured", n_active == 1, f"active={n_active}")
    finally:
        await _teardown(tm, path)


async def main() -> None:
    # Surface the Fix-B2 correction WARN so its evidence is visible.
    logging.basicConfig(level=logging.WARNING, format="    [log] %(name)s: %(message)s")
    await test_fix_b2_previous_day()
    await test_correct_timing()
    await test_committed_fact_defend()
    await test_committed_fact_recovery()
    await test_capture_maybe_commit_fact()

    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    if npass != total:
        print("FAILURES:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name} ({detail})")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
