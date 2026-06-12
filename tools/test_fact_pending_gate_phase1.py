"""Phase-3 Tier-1 test: Bug-B render gate — facts tied to derived-PENDING instructions
must NOT render in the prompt (NO API).

Fix-8 hides the instruction_pending_block on internal nudges, but a committed fact tied
to a PENDING reservation re-leaks the topic via the committed_facts block (live incident:
silence nudge answered the fish early). This gate is the always-on render-side backstop —
it also neutralizes already-polluted facts on disk.

Run:  venv\\Scripts\\python.exe tools\\test_fact_pending_gate_phase1.py
"""
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from modules.task_manager import TaskManager  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


def iso(dt: datetime) -> str:
    return dt.isoformat()


async def _make_tm():
    fd, path = tempfile.mkstemp(suffix="_pgate.jsonl")
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


async def _commit(tm, topic, answer, iid=None):
    tm.enqueue_command_nowait({
        "kind": "commit_fact", "scope": "eve", "topic_norm": topic,
        "answer_text": answer, "instruction_id": iid, "source": "conversation",
    })
    await tm.flush_command_queue()


async def main() -> None:
    tm, path = await _make_tm()
    try:
        now = datetime.now()
        # PENDING（期限未到来 +60s）の予約に紐づく fact（= 汚染 fact のライブ再現）
        tm.enqueue_command_nowait({
            "kind": "set_active_instruction", "instruction": "好きな魚を答える",
            "deadline_at": iso(now + timedelta(seconds=60)), "reason": "t", "created_by": "ai1",
        })
        await tm.flush_command_queue()
        iid = next(iter(tm._active_instructions.keys()))
        await _commit(tm, "好きな魚を答える", "うん、30秒後にちゃんと答えるね。", iid=iid)

        print("\n--- Bug-B gate: PENDING 紐づき fact は描画しない ---")
        check("PENDING-linked fact suppressed from prompt",
              tm.get_committed_facts_for_prompt() == [],
              f"facts={tm.get_committed_facts_for_prompt()}")

        # deadline 過去化 → derived ACTIVE → 描画される（期限超過の履行で drift を守る / Tier-2 C 経路）
        tm._active_instructions[iid].deadline_at = iso(now - timedelta(seconds=1))
        await tm._reconcile_instruction_status()
        facts = tm.get_committed_facts_for_prompt()
        check("derived-ACTIVE -> fact renders (overdue drift-defense intact)",
              len(facts) == 1 and facts[0]["topic"] == "好きな魚を答える", f"facts={facts}")

        # done クリア後 → DONE は fact を残す（ペルソナ持続）→ 描画継続
        tm.enqueue_command_nowait({
            "kind": "clear_active_instruction", "id": iid, "status": "done",
            "eve_response": "さばかな", "reason": "履行",
        })
        await tm.flush_command_queue()
        check("after done -> fact still renders (persona-durable)",
              len(tm.get_committed_facts_for_prompt()) == 1)

        # 迷子 iid / iid=None は描画（ペルソナ持続）
        await _commit(tm, "好きな動物", "猫だよ", iid="i_dangling")
        await _commit(tm, "好きな色", "青", iid=None)
        topics = {f["topic"] for f in tm.get_committed_facts_for_prompt()}
        check("dangling iid renders", "好きな動物" in topics, f"topics={topics}")
        check("iid=None renders", "好きな色" in topics, f"topics={topics}")

        # 関連性ゲートと共存: PENDING 紐づきは relevance パスでも常に除外される
        print("\n--- coexists with relevance gate (>limit) ---")
        tm.enqueue_command_nowait({
            "kind": "set_active_instruction", "instruction": "好きな寿司を答える",
            "deadline_at": iso(datetime.now() + timedelta(seconds=60)), "reason": "t", "created_by": "ai1",
        })
        await tm.flush_command_queue()
        iid2 = [i for i, v in tm._active_instructions.items() if v.instruction == "好きな寿司を答える"][0]
        await _commit(tm, "好きな寿司を答える", "うん、あとで答えるね", iid=iid2)
        for i in range(6):
            await _commit(tm, f"好きな物{i}", f"答{i}")
        gated = tm.get_committed_facts_for_prompt(relevance_text="寿司の話なんだけど")
        gtopics = {f["topic"] for f in gated}
        check("relevance path also excludes PENDING-linked fact",
              "好きな寿司を答える" not in gtopics, f"topics={gtopics}")
        check("relevance path still bounded", len(gated) <= 6, f"n={len(gated)}")
    finally:
        await _teardown(tm, path)

    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
