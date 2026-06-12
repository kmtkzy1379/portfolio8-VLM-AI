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
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
Config.GROQ_API_KEY = Config.GROQ_API_KEY or "x"
Config.OPENAI_API_KEY = Config.OPENAI_API_KEY or "x"

from vlm.common.datatypes import NarrationRequest  # noqa: E402
from vlm.narration.llm_client import NarrationEngine  # noqa: E402
from vlm.narration.prompt_builder import PromptBuilder  # noqa: E402
from modes.talk_mode import TalkMode  # noqa: E402
from modules.llm import LLMHandler  # noqa: E402

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


def run_c2() -> None:
    print("\n--- Bug-C2: VLM nudge text carries the newest alert narration ---")
    t = TalkMode._build_vlm_nudge_text("画面がコードエディタに切り替わり、Pythonのコードが表示")
    check("snippet embedded", "コードエディタ" in t and "最新の画面:" in t)
    check("starts with the internal-nudge prefix", t.startswith("[内部: 画面に新しい変化があった"))
    check("no em-dash (RAG-query/督促 regex と衝突しない)", "—" not in t)
    t2 = TalkMode._build_vlm_nudge_text("ダッシュ—入り")
    check("em-dash in snippet sanitized", "—" not in t2)
    check("optional-reaction wording present (Fix-H)",
          "必須ではない" in t and "実況・報告はしない" in t)
    check("empty snippet -> still optional-reaction text",
          TalkMode._build_vlm_nudge_text("").startswith("[内部: 画面に新しい変化があった")
          and "必須ではない" in TalkMode._build_vlm_nudge_text(""))

    print("\n--- Bug-C2: alert freshness gate + newest accessor ---")
    llm = LLMHandler()
    now = time.time()
    check("no alerts -> newest is None", llm.newest_vlm_alert() is None)
    llm._vlm_alerts.append((now - 45, "古い変化", "major"))
    check("stale alert (45s) -> no nudge trigger", not llm.has_unseen_vlm_alerts(0.0))
    llm._vlm_alerts.append((now - 2, "新しい変化", "major"))
    check("fresh alert (2s) -> triggers", llm.has_unseen_vlm_alerts(0.0))
    check("newest returns the latest", llm.newest_vlm_alert()[1] == "新しい変化")
    check("consumed (since_ts=now) -> no retrigger", not llm.has_unseen_vlm_alerts(now))

    print("\n--- Bug-C2: proactive block stays gated to '…' (not the new nudge format) ---")
    llm2 = LLMHandler()
    llm2.system_prompt = "テスト"
    llm2.recent_turns = []
    llm2.silence_summary = None
    llm2.committed_facts = []
    llm2.nudge_self_memory = []
    llm2.rag_memories = []
    llm2.vlm_context = ""
    llm2.one_shot_context = ""
    llm2._task_manager = None
    p = llm2._build_system_prompt(TalkMode._build_vlm_nudge_text("コードエディタに切替"))
    check("proactive block absent on new VLM-nudge format", "沈黙時の能動判断" not in p)


if __name__ == "__main__":
    run()
    run_c2()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
