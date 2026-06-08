"""Phase 1 Tier-1 test: affect plumbing feedback -> update_memory (NO API).

Verifies the CRITICAL aging contract: FeedbackHandler._inject_to_llm forwards the
*freshly-computed* affect (with a timestamp) only when one was produced this cycle,
and forwards affect=None on silent / restore cycles so the AI1-side timestamp is NOT
re-stamped (age keeps growing -> TTL expires -> stale moods don't look fresh forever).

Run:  venv\\Scripts\\python.exe tools\\test_affect_plumbing_phase1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.feedback import FeedbackHandler  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))


class _RecordingLLM:
    def __init__(self):
        self.calls: list[dict] = []

    def update_memory(self, payload, *, affect=None, affect_set_at=None):
        self.calls.append({"payload": payload, "affect": affect, "affect_set_at": affect_set_at})


def run() -> None:
    print("\n--- affect plumbing: _inject_to_llm forwards fresh affect / None on silent ---")
    # Build a bare FeedbackHandler without its heavy __init__ (we only exercise _inject_to_llm,
    # which depends solely on the static _extract_ai1_note + self.llm_handler).
    fh = object.__new__(FeedbackHandler)
    stub = _RecordingLLM()
    fh.llm_handler = stub

    # 1) Fresh affect this cycle -> forwarded WITH a timestamp.
    aff = {"affect": "curiosity", "intensity": 0.8, "valence": "pos"}
    fh._inject_to_llm("ただのフィードバック本文", new_affect=aff)
    check("fresh cycle calls update_memory", len(stub.calls) == 1)
    check("fresh cycle forwards the affect dict", stub.calls[-1]["affect"] is aff)
    check("fresh cycle stamps a timestamp", bool(stub.calls[-1]["affect_set_at"]))

    # 2) Silent / restore cycle (no new affect) -> affect=None, NO timestamp re-stamp.
    fh._inject_to_llm("ただのフィードバック本文")
    check("silent cycle still calls update_memory", len(stub.calls) == 2)
    check("silent cycle forwards affect=None (no re-stamp)", stub.calls[-1]["affect"] is None)
    check("silent cycle has no timestamp", stub.calls[-1]["affect_set_at"] is None)

    # 3) Explicit None passthrough behaves the same (restore path: _inject_to_llm(last_fb)).
    fh._inject_to_llm("ただのフィードバック本文", new_affect=None)
    check("explicit None -> affect None", stub.calls[-1]["affect"] is None)
    check("explicit None -> no timestamp", stub.calls[-1]["affect_set_at"] is None)


if __name__ == "__main__":
    run()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
