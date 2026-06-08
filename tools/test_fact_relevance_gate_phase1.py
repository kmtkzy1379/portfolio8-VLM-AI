"""Phase 1 Tier-1 test: committed-fact relevance gate (NO API / pure logic).

get_committed_facts_for_prompt(limit, relevance_text):
- relevance_text falsy OR active<=limit  -> byte-identical to old behavior (last `limit`).
- active>limit AND relevance_text given   -> keep (recent_floor newest 2) + (token-relevant);
  cap to `limit`. This bounds prompt growth as preferences accumulate WITHOUT dropping the
  drift-defending fact (recency floor + permissive token match).

Run:  venv\\Scripts\\python.exe tools\\test_fact_relevance_gate_phase1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.task_manager import TaskManager  # noqa: E402
from modules.task_schema import CommittedFact  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))


def _tm_with(facts: list[tuple[str, str]]) -> TaskManager:
    """Build a TaskManager (no start/IO) and inject active CommittedFacts directly.
    facts = list of (topic, answer) in chronological order (committed_at increasing)."""
    tm = TaskManager(tasks_file=os.devnull)
    for i, (topic, answer) in enumerate(facts):
        f = CommittedFact(
            fact_id=f"f_{i}", topic_norm=topic, answer_text=answer,
            committed_at=f"2026-01-01T00:00:{i:02d}", status="active",
            recent_expressions=[answer],
        )
        tm._committed_facts[f.fact_id] = f
    return tm


def run() -> None:
    # ---- ≤limit path is byte-identical (this is the case ALL existing tests hit) ----
    print("\n--- relevance gate: <=limit facts -> unchanged (drift-defense intact) ---")
    tm1 = _tm_with([("好きな動物", "猫")])
    no_rel = tm1.get_committed_facts_for_prompt()
    with_rel = tm1.get_committed_facts_for_prompt(relevance_text="うさぎに変えてよ")  # B-style miss
    check("1 fact, no relevance -> returns the fact",
          no_rel == [{"topic": "好きな動物", "answer": "猫", "recent_expressions": ["猫"]}])
    check("1 fact, relevance that does NOT token-match -> STILL returns it (<=limit path)",
          with_rel == no_rel, f"got={with_rel}")

    print("\n--- relevance gate: exactly limit facts -> unchanged ---")
    six = [(f"好きな物{i}", f"答{i}") for i in range(6)]
    tm6 = _tm_with(six)
    check("6 facts (==limit) -> all 6 regardless of relevance",
          len(tm6.get_committed_facts_for_prompt(relevance_text="無関係")) == 6)

    # ---- >limit path: gating activates ----
    print("\n--- relevance gate: >limit facts + relevance -> bounded & focused ---")
    seven = [
        ("好きな飲み物", "コーヒー"),   # f_0 oldest, irrelevant to query
        ("好きな色", "青"),            # f_1 irrelevant
        ("好きな動物", "猫"),          # f_2 RELEVANT (query mentions 動物)
        ("好きな食べ物", "カレー"),     # f_3 irrelevant
        ("好きなゲーム", "ポケモン"),    # f_4 irrelevant
        ("好きな季節", "秋"),          # f_5 newest-2 (recency floor)
        ("好きな映画", "SF"),          # f_6 newest-1 (recency floor)
    ]
    tm7 = _tm_with(seven)

    # No relevance -> old behavior: last `limit` (drops the 2 oldest by recency).
    old = tm7.get_committed_facts_for_prompt()
    check(">limit, no relevance -> last 6 (recency, unchanged behavior)",
          len(old) == 6 and old[0]["topic"] == "好きな色")  # f_0 dropped

    gated = tm7.get_committed_facts_for_prompt(relevance_text="ねえ、動物の話なんだけど")
    topics = [g["topic"] for g in gated]
    check(">limit + relevance -> bounded to <=limit", len(gated) <= 6, f"n={len(gated)}")
    check("token-relevant fact kept (好きな動物=猫)", "好きな動物" in topics, f"topics={topics}")
    check("recency floor kept (newest 2: 季節 & 映画)",
          "好きな季節" in topics and "好きな映画" in topics, f"topics={topics}")
    # Zero-overlap old facts are dropped (好きな色/青 and 好きなゲーム/ポケモン share no token
    # with 「動物の話」). Note: 飲み物/食べ物 legitimately stay — they share 「物」 with 動物
    # (weakly related "favorite-thing" topics); over-inclusion within `limit` is safe.
    check("zero-overlap old facts dropped (色 & ゲーム)",
          "好きな色" not in topics and "好きなゲーム" not in topics, f"topics={topics}")
    check("gate actually trimmed (7 -> fewer)", len(gated) < 7, f"n={len(gated)}")

    # Even with a relevance miss on the queried topic, the recency floor guarantees the
    # newest facts are present (so a just-committed preference is never silently dropped).
    print("\n--- relevance gate: total miss -> recency floor still guarantees newest ---")
    gated_miss = tm7.get_committed_facts_for_prompt(relevance_text="まったく無関係xyz")
    tmiss = [g["topic"] for g in gated_miss]
    check("total miss -> still returns newest 2 at least",
          "好きな季節" in tmiss and "好きな映画" in tmiss, f"topics={tmiss}")
    check("total miss -> bounded (<= limit)", len(gated_miss) <= 6, f"n={len(gated_miss)}")


if __name__ == "__main__":
    run()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
