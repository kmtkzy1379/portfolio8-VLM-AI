"""Phase-3 Tier-1 test: Bug-C1 — scene-change flag makes narration describe the CURRENT
screen (NO API; prompt construction + flag plumbing only).

Root cause (live): NarrationRequest carries deltas accumulated SINCE THE LAST narration,
i.e. mostly the PRE-change screen; only the screenshot is current. On MAJOR changes the
narration described the old screen. Fix: is_scene_change flag -> prompt instruction to
prioritize the screenshot and treat old deltas / previous observations as pre-change.

Run:  venv\\Scripts\\python.exe tools\\test_vlm_scene_change_phase1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.common.datatypes import NarrationRequest  # noqa: E402
from vlm.narration.llm_client import NarrationEngine  # noqa: E402
from vlm.narration.prompt_builder import PromptBuilder  # noqa: E402

_results: list[tuple[str, bool, str]] = []
MARK = "【画面切替】"


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


class StubEncoder:
    def to_compact_text(self, delta):
        return "E1 person moved"

    def to_temporal_text(self, deltas):
        return "E1 person moved (temporal)"


def _user_text(messages: list) -> str:
    """最後の user メッセージの text パートを連結。"""
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if p.get("type") == "text":
                    parts.append(p["text"])
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def run() -> None:
    pb = PromptBuilder(delta_encoder=StubEncoder())

    print("\n--- Bug-C1: prompt builder scene-change instruction ---")
    msgs_on = pb.build([object()], "前の画面の話", is_scene_change=True)
    check("is_scene_change=True -> 【画面切替】 instruction present", MARK in _user_text(msgs_on))
    check("instruction prioritizes the CURRENT screenshot",
          "今この瞬間に映っているもの" in _user_text(msgs_on))
    msgs_off = pb.build([object()], "前の画面の話", is_scene_change=False)
    check("is_scene_change=False -> byte-stable (no instruction)", MARK not in _user_text(msgs_off))
    check("default (omitted) -> no instruction", MARK not in _user_text(pb.build([object()], "")))

    print("\n--- Bug-C1: flag forwarded through NarrationEngine.narrate ---")
    eng = NarrationEngine(delta_encoder=StubEncoder())
    captured = {}
    orig_build = eng._prompt_builder.build

    def rec_build(*a, **k):
        captured["is_scene_change"] = k.get("is_scene_change")
        return orig_build(*a, **k)

    eng._prompt_builder.build = rec_build
    eng._call_llm = lambda messages: "ナレーション"
    out = eng.narrate([object()], is_scene_change=True)
    check("narrate(is_scene_change=True) forwards flag to builder",
          captured.get("is_scene_change") is True, f"captured={captured}")
    check("narrate returns narration", out == "ナレーション")

    print("\n--- Bug-C1: NarrationRequest carries the flag; reset request defaults False ---")
    req = NarrationRequest(deltas=[], key_crops=None, relations_text="", memory_text="",
                           screenshot=None, frame_id=1, is_scene_change=True)
    check("NarrationRequest.is_scene_change settable", req.is_scene_change is True)
    reset_req = NarrationRequest(deltas=[], key_crops=None, relations_text="",
                                 memory_text="", screenshot=None, frame_id=0, is_reset=True)
    check("reset request defaults is_scene_change=False", reset_req.is_scene_change is False)


if __name__ == "__main__":
    run()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
