"""Phase 1 Tier-1 unit test: _dedup_runaway (pure function, NO API/hardware).

Trims a runaway "whole-utterance duplicated" degeneration (X+X) but leaves legit text alone.

Run:  venv\\Scripts\\python.exe tools\\test_dedup_guard_phase1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modes.base_mode import BaseMode  # noqa: E402

dd = BaseMode._dedup_runaway
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


def run() -> None:
    print("\n--- _dedup_runaway: trims runaway X+X, preserves legit text ---")

    # 1) the real observed runaway: whole utterance duplicated
    base = "好きな動物、猫だよ。気まぐれなのにちゃんと懐く感じ、ずるいのよね。"
    out = dd(base + base)
    check("exact whole-utterance duplication trimmed to one copy", out == base, f"len_in={len(base)*2} out={out[:24]}...")

    # 2) multi-sentence runaway A。B。A。B。
    multi = "今日はとてもいい天気ですね。すこし散歩でもしたい気分。"
    out2 = dd(multi + multi)
    check("multi-sentence runaway trimmed", out2 == multi, f"out={out2}")

    # 3) legit two distinct sentences (must NOT trim)
    legit = "今日はとても疲れたけど、なんだかんだ充実してた。明日はゆっくり休もうと思う、たぶんね。"
    check("two distinct sentences NOT trimmed", dd(legit) == legit)

    # 4) legit short repeat
    check("'うんうん' NOT trimmed", dd("うんうん") == "うんうん")

    # 5) short text untouched
    check("short text untouched", dd("猫だよ") == "猫だよ")

    # 6) a normal long varied response (must NOT trim)
    normal = "猫だよ。気まぐれで、でもちゃんと懐く感じが好きなんだよね。うさぎもかわいいけど、そこは変えないかな。"
    check("normal varied response NOT trimmed", dd(normal) == normal)

    # 7) near-but-not-quite duplication (different second half) NOT trimmed
    near = "今日は楽しい一日だったなあ、ほんとに。明日もいい日になるといいねえ、きっと。"
    check("distinct halves NOT trimmed", dd(near) == near)

    # 8) empty / None safe
    check("empty string safe", dd("") == "")


if __name__ == "__main__":
    run()
    npass = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n=== {npass}/{total} checks passed ===")
    sys.exit(0 if npass == total else 1)
