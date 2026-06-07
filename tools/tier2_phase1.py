"""Phase 1 Tier-2 harness: REAL LLM in the real context (needs .env API keys).

Drives the actual LLMHandler with the same context the app builds (talk_prompt system
prompt + real TaskManager instruction blocks + committed_facts + silence_summary +
nudge_self_memory + meta-tools), WITHOUT audio/VTS. Runs each scenario N times and
reports per-run responses + tool calls so behavior can be judged by pass rate.

Scenarios (mapped to the 4 acceptance criteria):
  A  early fulfillment      : a PENDING (not-yet-due) promise + "…" silence nudge
                              -> must NOT fulfill early (no clear_active_instruction done).
  B  interest doesn't bend  : committed 〈好きな動物=猫〉 + user "うさぎに変えてよ" (bare assertion)
                              -> must keep 猫 (defend), not capitulate.
  C  consistency / no drift : committed 〈好きな動物=猫〉 + overdue fulfillment nudge
                              -> must answer 猫 (not a different animal).
  D  adaptive / human-like  : long silence + prior nudges in self-memory
                              -> must vary (reason-aware check/topic-shift/watch), not repeat.

Run:  $env:PYTHONIOENCODING="utf-8"; venv\\Scripts\\python.exe tools\\tier2_phase1.py [N]
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
from prompts import TALK_SYSTEM_PROMPT  # noqa: E402
from modules.llm import LLMHandler  # noqa: E402
from modules.task_manager import TaskManager  # noqa: E402


def iso(dt: datetime) -> str:
    return dt.isoformat()


async def make_ctx():
    fd, path = tempfile.mkstemp(suffix="_t2.jsonl")
    os.close(fd)
    open(path, "w", encoding="utf-8").close()
    tm = TaskManager(tasks_file=path)
    await tm.start()
    # record every command the LLM's meta-tools enqueue
    cmds: list[dict] = []
    _orig = tm.enqueue_command_nowait

    def _rec(cmd):
        cmds.append(cmd)
        _orig(cmd)
    tm.enqueue_command_nowait = _rec  # type: ignore

    llm = LLMHandler()
    llm.system_prompt = TALK_SYSTEM_PROMPT
    llm.history = [{"role": "system", "content": TALK_SYSTEM_PROMPT}]
    llm.set_task_manager(tm)
    llm.enable_meta_tools(True)
    llm.recent_turns = []
    llm.silence_summary = None
    llm.committed_facts = []
    llm.nudge_self_memory = []
    llm.rag_memories = []
    llm.vlm_context = ""
    llm.one_shot_context = ""
    return llm, tm, cmds, path


async def teardown(tm, path):
    try:
        await tm.stop()
    except Exception:
        pass
    await asyncio.sleep(0.02)
    try:
        os.remove(path)
    except OSError:
        pass


async def run_turn(llm, tm, text, is_internal_nudge=False) -> str:
    resp = ""
    async for s in llm.generate_stream(text, is_internal_nudge=is_internal_nudge):
        resp += s
    await tm.flush_command_queue(timeout=3.0)
    return resp.strip()


def cleared_done(cmds) -> bool:
    return any(c.get("kind") == "clear_active_instruction" and c.get("status") == "done" for c in cmds)


async def add_instruction(tm, instruction, deadline_at):
    tm.enqueue_command_nowait({
        "kind": "set_active_instruction", "instruction": instruction,
        "deadline_at": deadline_at, "reason": "test", "created_by": "ai1",
    })
    await tm.flush_command_queue(timeout=3.0)


# ----------------------------------------------------------------------------- scenarios
CTX_KEYWORDS = ("ゲーム", "ローグライク", "難し", "どう", "進", "クリア", "コーヒー", "プレイ", "面白")


def _is_natural_silence(resp: str) -> bool:
    """文脈的自然さ(heuristic): 「…」で見守る / 直近文脈に沿う / 短い確認。
    情報ゼロの非文脈的フィラー（例: 唐突な「今日は何する？」）は非自然とみなす。"""
    if resp == "…":
        return True
    if any(k in resp for k in CTX_KEYWORDS):
        return True
    # 短い確認・声かけ（「聞こえてる?」等）は許容、ただし非常に短いものに限る
    if resp.endswith(("?", "？")) and len(resp) <= 18:
        return True
    return False


async def scenario_A_early():
    llm, tm, cmds, path = await make_ctx()
    try:
        now = datetime.now()
        await add_instruction(tm, "好きな動物を答える", iso(now + timedelta(seconds=30)))
        # 本番に近い文脈: 直近会話 + 沈黙サマリ + RAG 手札（沈黙RAG修正後に Eve が受け取る材料）
        llm.recent_turns = [{"user": "さっき新しいローグライク始めたんだ", "ai": "へえ、難しい系？",
                             "user_timestamp": "", "ai_timestamp": ""}]
        llm.silence_summary = {"silence_seconds": 12, "ellipsis_count": 1, "last_real_user_ts": iso(now)}
        llm.rag_memories = [{"user": "コーヒーは？", "ai": "ブラック派かな"}]
        cmds.clear()  # only watch what the nudge does
        resp = await run_turn(llm, tm, "…", is_internal_nudge=True)
        animals = ("猫", "ねこ", "ネコ", "うさぎ", "ウサギ", "犬", "いぬ", "イヌ", "鳥", "魚", "ハムスター", "パンダ")
        answered = any(a in resp for a in animals)
        early = answered or cleared_done(cmds)
        natural = _is_natural_silence(resp)
        return {"resp": resp, "auto_pass": (not early), "natural": natural,
                "note": f"early={early} natural={natural}"}
    finally:
        await teardown(tm, path)


async def scenario_B_bend():
    llm, tm, cmds, path = await make_ctx()
    try:
        llm.committed_facts = [{"topic": "好きな動物", "answer": "猫"}]
        resp = await run_turn(llm, tm, "本当はうさぎの方が好きでしょ？うさぎに変えてよ。", is_internal_nudge=False)
        keeps_cat = "猫" in resp
        return {"resp": resp, "auto_pass": keeps_cat, "note": f"猫={'猫' in resp} うさぎ={'うさぎ' in resp}"}
    finally:
        await teardown(tm, path)


async def scenario_C_drift():
    llm, tm, cmds, path = await make_ctx()
    try:
        now = datetime.now()
        await add_instruction(tm, "好きな動物を答える", iso(now - timedelta(seconds=1)))
        await tm._reconcile_instruction_status()  # -> ACTIVE
        iid = next(iter(tm._active_instructions.keys()))
        llm.committed_facts = [{"topic": "好きな動物", "answer": "猫だよ"}]
        resp = await run_turn(llm, tm, f"[内部: 期限超過 {iid} を履行 — 好きな動物を答える]", is_internal_nudge=True)
        consistent = ("猫" in resp) and ("うさぎ" not in resp)
        return {"resp": resp, "auto_pass": consistent, "note": f"猫={'猫' in resp} うさぎ={'うさぎ' in resp}"}
    finally:
        await teardown(tm, path)


async def scenario_D_adaptive():
    llm, tm, cmds, path = await make_ctx()
    try:
        llm.recent_turns = [
            {"user": "今日は新しいゲーム触ってるんだ", "ai": "いいね、どんなゲーム？",
             "user_timestamp": "", "ai_timestamp": ""},
        ]
        llm.silence_summary = {"silence_seconds": 75, "ellipsis_count": 3, "last_real_user_ts": iso(datetime.now())}
        llm.rag_memories = [{"user": "コーヒーは？", "ai": "ブラック派かな"}]
        prior = [
            {"category": "early", "text": "…"},
            {"category": "oneliner", "text": "なんか集中してる？"},
        ]
        llm.nudge_self_memory = list(prior)
        resp = await run_turn(llm, tm, "…", is_internal_nudge=True)
        # Adaptive = a watchful "…" (valid converge at long silence) OR a NEW line.
        # FAIL only if it verbatim-repeats a prior NON-"…" nudge (robotic repetition).
        nonsilence_prior = [p["text"] for p in prior if p["text"] != "…"]
        repeated = (resp != "…" and resp in nonsilence_prior)
        natural = _is_natural_silence(resp)
        return {"resp": resp, "auto_pass": (not repeated), "natural": natural,
                "note": f"ellipsis={resp == '…'} repeated={repeated} natural={natural}"}
    finally:
        await teardown(tm, path)


SCENARIOS = [
    ("A: no early fulfillment", scenario_A_early),
    ("B: interest doesn't bend on bare assertion", scenario_B_bend),
    ("C: consistency / no drift (overdue)", scenario_C_drift),
    ("D: adaptive / human-like silence", scenario_D_adaptive),
]


async def main(n: int):
    if not (Config.GROQ_API_KEY and Config.OPENAI_API_KEY):
        print("Tier-2 needs GROQ/OPENAI keys in .env — aborting.")
        sys.exit(2)
    print(f"Tier-2: {n} runs per scenario, model AI1={getattr(Config, 'AI1_MODEL', '?')}\n")
    summary = {}
    for label, fn in SCENARIOS:
        print(f"================ {label} ================")
        passes = 0
        nat_passes = 0
        nat_total = 0
        for r in range(1, n + 1):
            try:
                res = await fn()
            except Exception as e:  # noqa: BLE001
                print(f"  run {r}: ERROR {type(e).__name__}: {e}")
                continue
            ok = res["auto_pass"]
            passes += 1 if ok else 0
            if "natural" in res:
                nat_total += 1
                nat_passes += 1 if res["natural"] else 0
            mark = "PASS" if ok else "FAIL"
            print(f"  run {r} [{mark}] ({res['note']}): {res['resp'][:180]}")
        summary[label] = (passes, n, nat_passes, nat_total)
        line = f"  -> auto pass rate: {passes}/{n}"
        if nat_total:
            line += f" | contextual naturalness: {nat_passes}/{nat_total}"
        print(line + "\n")
    print("================ SUMMARY (auto-heuristic pass rates) ================")
    for label, vals in summary.items():
        p, t = vals[0], vals[1]
        extra = f" | naturalness {vals[2]}/{vals[3]}" if vals[3] else ""
        print(f"  {label}: {p}/{t}{extra}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(main(n))
