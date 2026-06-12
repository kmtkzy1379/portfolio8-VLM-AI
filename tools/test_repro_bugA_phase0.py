"""Phase-0 repro (Tier-1, NO API): Bug A — acknowledgment captured as committed fact.

Live evidence (tasks.jsonl cf_6b05c799, 2026-06-11): when the user creates a reservation
and Eve merely ACKNOWLEDGES it (「うん、30秒後にちゃんと答えるね。」), `_maybe_commit_fact`
attributes the response to the still-PENDING instruction and commits it as the "answer";
later unrelated turns keep appending to recent_expressions.

These checks assert the DESIRED behavior, so on unfixed main they FAIL — that failure IS
the deterministic reproduction. After Fix-A they must pass (and will be folded into
test_taskmanager_phase1.py).

Run:  venv\\Scripts\\python.exe tools\\test_repro_bugA_phase0.py
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


async def main() -> None:
    fd, path = tempfile.mkstemp(suffix="_reproA.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()
    tm = TaskManager(tasks_file=path)
    await tm.start()
    try:
        mode = TalkMode()
        mode.task_manager = tm

        # 予約を PENDING（期限未到来 +30s）で登録 — ライブの 23:23:26 の状態。
        tm.enqueue_command_nowait({
            "kind": "set_active_instruction", "instruction": "好きな魚を答える",
            "deadline_at": iso(datetime.now() + timedelta(seconds=30)),
            "reason": "30秒後に答える約束", "created_by": "ai1",
        })
        await tm.flush_command_queue()

        print("\n--- Bug-A repro: PENDING 中の了解発話は fact にしない（ライブでは汚染された） ---")
        # 1) 了解発話（ライブで answer_text に化けた発話そのもの）
        mode._maybe_commit_fact("8秒後に好きな魚を教えてくれる?", "うん、30秒後にちゃんと答えるね。",
                                is_internal_nudge=False)
        await tm.flush_command_queue()
        facts = tm.get_committed_facts_for_prompt()
        check("acknowledgment NOT captured while PENDING",
              len(facts) == 0, f"facts={facts}")

        # 2) その後の無関係ターン（ライブで recent_expressions に蓄積された発話）
        mode._maybe_commit_fact("…", "あ、たしかに早かった。まだ待つね。", is_internal_nudge=True)
        await tm.flush_command_queue()
        facts2 = tm.get_committed_facts_for_prompt()
        check("unrelated turn NOT appended while PENDING",
              len(facts2) == 0, f"facts={facts2}")

        # 3) 対照: 期限到来（ACTIVE）後の実回答は捕捉される（正当経路は生きる）
        iid = next(iter(tm._active_instructions.keys()))
        tm._active_instructions[iid].deadline_at = iso(datetime.now() - timedelta(seconds=1))
        await tm._reconcile_instruction_status()
        mode._maybe_commit_fact("…", "さばかな。脂がのってて好き。", is_internal_nudge=True)
        await tm.flush_command_queue()
        facts3 = tm.get_committed_facts_for_prompt()
        check("real answer IS captured once ACTIVE (legit path intact)",
              len(facts3) == 1 and "さば" in facts3[0]["answer"], f"facts={facts3}")
    finally:
        try:
            await tm.stop()
        except Exception:
            pass
        await asyncio.sleep(0.05)
        try:
            os.remove(path)
        except OSError:
            pass

    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
