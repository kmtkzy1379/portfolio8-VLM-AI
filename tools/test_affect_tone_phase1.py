"""Phase 1 Tier-1 test: affect->tone WEAK hint gate (NO API call — only builds the prompt).

The affect tone hint is a *weak, possibly-stale* tone nudge derived from AI2's affect.
It must inject ONLY when fresh (age <= TTL) AND confident (intensity >= MIN), and must be
ABSENT when stale / low-confidence / disabled / missing / clock-skewed, and must be
SUPPRESSED while an overdue ACTIVE instruction is pending (fulfillment takes priority).
It is prompt-text only and never touches task/clear paths.

Run:  venv\\Scripts\\python.exe tools\\test_affect_tone_phase1.py
"""
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import Config  # noqa: E402
Config.GROQ_API_KEY = Config.GROQ_API_KEY or "x"
Config.OPENAI_API_KEY = Config.OPENAI_API_KEY or "x"

from modules.llm import LLMHandler  # noqa: E402
from modules.task_manager import TaskManager  # noqa: E402

_results: list[tuple[str, bool, str]] = []
_MARKER = "補助トーンヒント"  # unique header of the affect-tone block


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))


def _fresh_llm() -> LLMHandler:
    llm = LLMHandler()
    llm.system_prompt = "テスト用システムプロンプト"
    llm.recent_turns = []
    llm.silence_summary = None
    llm.committed_facts = []
    llm.nudge_self_memory = []
    llm.rag_memories = []
    llm.vlm_context = ""
    llm.one_shot_context = ""
    llm._task_manager = None
    return llm


def _ago(sec: float) -> str:
    return (datetime.now() - timedelta(seconds=sec)).isoformat()


def run_gate_tests() -> None:
    orig = (Config.AFFECT_TONE_ENABLE, Config.AFFECT_TONE_TTL_SEC, Config.AFFECT_TONE_MIN_CONFIDENCE)
    Config.AFFECT_TONE_ENABLE = True
    Config.AFFECT_TONE_TTL_SEC = 120
    Config.AFFECT_TONE_MIN_CONFIDENCE = 0.4
    try:
        # 1) fresh + confident -> block present, with the required pieces.
        print("\n--- affect->tone: fresh + confident -> hint injected ---")
        llm = _fresh_llm()
        llm.affect = {"affect": "curiosity", "intensity": 0.8, "valence": "pos"}
        llm._affect_set_at = _ago(30)
        p = llm._build_system_prompt("…")
        check("hint block present", _MARKER in p)
        check("affect label rendered (curiosity)", "curiosity" in p)
        check("elapsed time rendered (秒前)", "秒前" in p)
        check("confidence rendered (0.8)", "0.8" in p)
        check("current-context-wins wording present", "最優先" in p and "矛盾" in p)
        check("isolation wording present (clear に影響させない)",
              "clear 判断には" in p and "一切影響させない" in p)

        # 2) stale (older than TTL) -> absent.
        print("\n--- affect->tone: stale (> TTL) -> suppressed ---")
        llm.affect = {"affect": "boredom", "intensity": 0.8, "valence": "neg"}
        llm._affect_set_at = _ago(Config.AFFECT_TONE_TTL_SEC + 60)
        check("stale affect -> no hint", _MARKER not in llm._build_system_prompt("…"))

        # 3) low-confidence -> absent.
        print("\n--- affect->tone: low confidence (< MIN) -> suppressed ---")
        llm.affect = {"affect": "curiosity", "intensity": 0.2, "valence": "pos"}
        llm._affect_set_at = _ago(10)
        check("low-confidence affect -> no hint", _MARKER not in llm._build_system_prompt("…"))

        # 4) disabled by config -> absent.
        print("\n--- affect->tone: AFFECT_TONE_ENABLE=False -> suppressed ---")
        Config.AFFECT_TONE_ENABLE = False
        llm.affect = {"affect": "calm", "intensity": 0.9, "valence": "neu"}
        llm._affect_set_at = _ago(5)
        check("disabled -> no hint", _MARKER not in llm._build_system_prompt("…"))
        Config.AFFECT_TONE_ENABLE = True

        # 5) affect None / set_at None -> absent, no exception.
        print("\n--- affect->tone: missing affect -> suppressed, no error ---")
        llm.affect = None
        llm._affect_set_at = None
        try:
            check("affect=None -> no hint, no error", _MARKER not in llm._build_system_prompt("…"))
        except Exception as e:  # noqa: BLE001
            check("affect=None -> no hint, no error", False, f"{type(e).__name__}: {e}")

        # 6) clock skew (future timestamp -> negative age) -> absent, no exception.
        print("\n--- affect->tone: future timestamp (clock skew) -> suppressed ---")
        llm.affect = {"affect": "surprise", "intensity": 0.9, "valence": "pos"}
        llm._affect_set_at = (datetime.now() + timedelta(seconds=90)).isoformat()
        try:
            check("future ts -> no hint, no error", _MARKER not in llm._build_system_prompt("…"))
        except Exception as e:  # noqa: BLE001
            check("future ts -> no hint, no error", False, f"{type(e).__name__}: {e}")

        # 7) unparseable timestamp -> absent, no exception.
        print("\n--- affect->tone: unparseable timestamp -> suppressed ---")
        llm._affect_set_at = "not-a-timestamp"
        try:
            check("bad ts -> no hint, no error", _MARKER not in llm._build_system_prompt("…"))
        except Exception as e:  # noqa: BLE001
            check("bad ts -> no hint, no error", False, f"{type(e).__name__}: {e}")
    finally:
        Config.AFFECT_TONE_ENABLE, Config.AFFECT_TONE_TTL_SEC, Config.AFFECT_TONE_MIN_CONFIDENCE = orig


async def test_suppressed_during_overdue_active() -> None:
    """Even a fresh+confident affect must be suppressed while an overdue ACTIVE
    instruction is pending — fulfillment must not be diluted by a tone hint."""
    print("\n--- affect->tone: suppressed while overdue ACTIVE instruction pending ---")
    orig = (Config.AFFECT_TONE_ENABLE, Config.AFFECT_TONE_TTL_SEC, Config.AFFECT_TONE_MIN_CONFIDENCE)
    Config.AFFECT_TONE_ENABLE = True
    Config.AFFECT_TONE_TTL_SEC = 120
    Config.AFFECT_TONE_MIN_CONFIDENCE = 0.4
    fd, path = tempfile.mkstemp(suffix="_afftone.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()
    tm = TaskManager(tasks_file=path)
    await tm.start()
    try:
        now = datetime.now()
        tm.enqueue_command_nowait({
            "kind": "set_active_instruction", "instruction": "好きな動物を答える",
            "deadline_at": (now + timedelta(seconds=30)).isoformat(),
            "reason": "t", "created_by": "ai1",
        })
        await tm.flush_command_queue()
        iid = next(iter(tm._active_instructions.keys()))
        tm._active_instructions[iid].deadline_at = (now - timedelta(seconds=1)).isoformat()
        await tm._reconcile_instruction_status()  # -> ACTIVE (overdue)

        llm = _fresh_llm()
        llm.set_task_manager(tm)
        llm.affect = {"affect": "curiosity", "intensity": 0.9, "valence": "pos"}
        llm._affect_set_at = _ago(5)  # fresh + confident
        p = llm._build_system_prompt("…")
        check("overdue ACTIVE block present (fulfillment context shown)", "好きな動物を答える" in p)
        check("affect hint SUPPRESSED while overdue active", _MARKER not in p)
    finally:
        Config.AFFECT_TONE_ENABLE, Config.AFFECT_TONE_TTL_SEC, Config.AFFECT_TONE_MIN_CONFIDENCE = orig
        try:
            await tm.stop()
        except Exception:
            pass
        await asyncio.sleep(0.02)
        try:
            os.remove(path)
        except OSError:
            pass


async def _amain() -> None:
    run_gate_tests()
    await test_suppressed_during_overdue_active()


if __name__ == "__main__":
    asyncio.run(_amain())
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
