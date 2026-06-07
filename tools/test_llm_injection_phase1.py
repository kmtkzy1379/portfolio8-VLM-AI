"""Phase 1 Tier-1 injection test (NO API call — only builds the prompt string).

Verifies that _build_system_prompt actually renders:
- the Fix-9b committed-facts firm-consistency block, and
- the Phase-0 nudge self-memory block,
when the corresponding llm attributes are populated.

Run:  venv\\Scripts\\python.exe tools\\test_llm_injection_phase1.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Dummy keys so LLMHandler construction never needs real creds (no network is opened).
from config import Config  # noqa: E402
Config.GROQ_API_KEY = Config.GROQ_API_KEY or "x"
Config.OPENAI_API_KEY = Config.OPENAI_API_KEY or "x"

from modules.llm import LLMHandler  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))


def run() -> None:
    llm = LLMHandler()
    llm.system_prompt = "テスト用システムプロンプト"
    # The two contexts under test:
    llm.committed_facts = [{"topic": "好きな動物", "answer": "猫だよ"}]
    llm.nudge_self_memory = [{"category": "early", "text": "猫だよ"}]
    # Defensive defaults for other context inputs.
    llm.recent_turns = []
    llm.silence_summary = None
    llm.rag_memories = []
    llm.vlm_context = ""
    llm.one_shot_context = ""
    llm._task_manager = None

    print("\n--- Fix-9b: committed-facts block renders in the system prompt ---")
    try:
        prompt = llm._build_system_prompt("…")
    except Exception as e:  # noqa: BLE001
        check("_build_system_prompt runs", False, f"{type(e).__name__}: {e}")
        return
    check("_build_system_prompt runs without error", True)
    check("committed-facts block header present",
          "既にコミットした自分の答え" in prompt)
    check("committed fact value rendered (好きな動物 = 猫だよ)",
          "好きな動物 = 猫だよ" in prompt)
    check("anti-drift wording present (猫→うさぎ禁止)",
          "猫と言ったのに後でうさぎ" in prompt)
    check("firm-but-not-robotic wording present",
          "表現は毎回少し変えてよいが、答えの中身は保つ" in prompt)
    check("Phase-0 nudge self-memory block also renders",
          "この沈黙中に自分が既に言ったこと" in prompt)

    # When empty, the block must NOT appear (no noise).
    llm.committed_facts = []
    prompt2 = llm._build_system_prompt("…")
    check("no committed-facts block when store empty",
          "既にコミットした自分の答え" not in prompt2)


if __name__ == "__main__":
    run()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
