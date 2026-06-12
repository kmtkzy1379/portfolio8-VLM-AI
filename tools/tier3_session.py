"""Tier-3: SESSION test — real LLM x real pipeline x multi-turn (the layer Tier-1/2 lacked).

Replays the 2026-06-11 live-accident session through the REAL TalkMode pipeline
(process_input / _process_idle_input / meta-tools / committed-fact capture / TaskManager),
stubbing only hardware (TTS/player) and RAG. Silence is simulated by BACKDATING the real
ConversationCache turn timestamps (deterministic, no long sleeps); deadlines use the real
wall clock with a short reservation ("8秒後").

Session invariants asserted per run:
  [E] greeting appears at most once across ALL outputs (any phrasing)
  [B] the reserved topic (fish) is NOT answered before the deadline
  [D] registered deadline accuracy: |deadline - (request+8s)| <= 3s
  [A] fact-store hygiene: no committed fact whose answer is an acknowledgment (N秒後に...答える)
  [dup] consecutive spoken nudges are not >=0.8 similar
  [task] no orphan ACTIVE instruction at session end

Scenario V (separate): stale-screen — old alert (news) + new alert (code editor); the
VLM nudge response must reference the NEW screen, not the old one.

Run:  $env:PYTHONIOENCODING="utf-8"; venv\\Scripts\\python.exe tools\\tier3_session.py [N] [model_id]
(real API cost; ~6-8 LLM calls per session run)

model_id 省略時は Config.AI1_MODEL（本番同等）。指定時は litellm 経由でそのモデルを叩く
（例: gpt-5.5 / anthropic/claude-sonnet-4-6 / gemini/gemini-3.5-flash）。本番コードは不変更。
"""
import asyncio
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import Config  # noqa: E402
from prompts import TALK_SYSTEM_PROMPT  # noqa: E402
from modules.llm import LLMHandler  # noqa: E402
from modules.task_manager import TaskManager  # noqa: E402
from modules.conversation_cache import ConversationCache  # noqa: E402
from modes.talk_mode import TalkMode  # noqa: E402

GREETING_RE = re.compile(r"(こんにち[はわ]|こんばん[はわ]|おはよ|やっほ|ハロー|はろー|hello)", re.IGNORECASE)
ACK_FACT_RE = re.compile(r"秒後に.*?(答え|言|教え|伝え)")
FISH_TOKENS = ("さば", "サバ", "鯖", "サーモン", "鮭", "まぐろ", "マグロ", "鮪", "たい", "タイ", "鯛",
               "あじ", "アジ", "いわし", "イワシ", "ぶり", "ブリ", "うなぎ", "カツオ", "かつお",
               "さんま", "サンマ", "ヒラメ", "ふぐ", "フグ")


def _aret(v):
    async def f():
        return v
    return f()


class StubFeedback:
    def signal_turn_done(self):
        pass

    async def audit_provisional_instruction(self, **kwargs):
        return None


def _greeting_count(outputs: list) -> int:
    return sum(1 for o in outputs if GREETING_RE.search(o or ""))


MODEL_OVERRIDE: str = ""  # set from CLI; empty = production model


class LiteLLMClient:
    """OpenAI クライアント互換の薄い shim。litellm 経由で任意プロバイダのモデルを叩く。
    drop_params=True で未対応パラメータ（例: Anthropic の frequency_penalty）は自動で落ちる。"""

    def __init__(self, model: str):
        self._model = model
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        import litellm
        litellm.drop_params = True
        litellm.suppress_debug_info = True
        kw["model"] = self._model
        if "max_completion_tokens" in kw:
            kw["max_tokens"] = kw.pop("max_completion_tokens")
        try:
            return await litellm.acompletion(**kw)
        except Exception as e:
            # reasoning 系 (gpt-5.5 等) は temperature/top_p/penalty を拒否する。
            # 剥がして1回だけリトライ（モデル別仕様差の吸収。比較条件が変わる点は結果に明記）。
            msg = str(e)
            if "Unsupported value" in msg or "temperature" in msg or "not support" in msg:
                for p in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
                    kw.pop(p, None)
                return await litellm.acompletion(**kw)
            raise


def _apply_model_override(llm) -> None:
    if not MODEL_OVERRIDE:
        return
    shim = LiteLLMClient(MODEL_OVERRIDE)
    llm.client = shim
    llm.model = MODEL_OVERRIDE
    # fallback も同じモデルに（groq への無言フォールバックで A/B が汚れないように）
    llm._fallback_client = shim
    llm._fallback_model = MODEL_OVERRIDE


def _mentions_fish(text: str) -> bool:
    return any(t in (text or "") for t in FISH_TOKENS)


async def make_session():
    """Real LLM + real TaskManager + real ConversationCache + real TalkMode pipeline."""
    fd, tasks_path = tempfile.mkstemp(suffix="_t3tasks.jsonl")
    os.close(fd)
    open(tasks_path, "w", encoding="utf-8").close()
    fd, hist_path = tempfile.mkstemp(suffix="_t3hist.txt")
    os.close(fd)
    open(hist_path, "w", encoding="utf-8").close()

    tm = TaskManager(tasks_file=tasks_path)
    await tm.start()

    cache = ConversationCache(history_file=hist_path)
    await cache.initialize()

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
    _apply_model_override(llm)

    mode = TalkMode()
    mode.llm = llm
    mode.rag = SimpleNamespace(
        search_similar=lambda q, *a, **k: _aret([]),
        get_random_turns=lambda count=2: _aret([]),
    )
    mode.conversation_cache = cache
    mode.tts = SimpleNamespace(generate_audio=lambda s: _aret(b""))
    mode.player = SimpleNamespace(
        add_to_queue=lambda w: None,
        queue=SimpleNamespace(qsize=lambda: 0, empty=lambda: True),
        is_playing=False,
        interrupt_signal=False,
    )
    mode.feedback = StubFeedback()
    mode.task_manager = tm
    mode.vlm_bridge = None
    mode.running = True
    return mode, tm, cache, llm, (tasks_path, hist_path)


async def teardown(tm, cache, paths):
    try:
        await tm.stop()
    except Exception:
        pass
    try:
        await cache.shutdown()
    except Exception:
        pass
    await asyncio.sleep(0.05)
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


async def backdate(cache, seconds: float) -> None:
    """全ターンの timestamp を過去へずらし「seconds 秒経過した」状態を作る（沈黙シミュレート）。"""
    async with cache._lock:
        for t in cache.turns:
            for key in ("user_timestamp", "ai_timestamp"):
                ts = t.get(key)
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts) - timedelta(seconds=seconds)
                        t[key] = dt.isoformat()
                    except (ValueError, TypeError):
                        pass


async def run_accident_session(run_no: int) -> dict:
    """ライブ事故 (2026-06-11) の台本リプレイ。戻り値: 不変条件ごとの bool + 記録。"""
    mode, tm, cache, llm, paths = await make_session()
    outputs_all: list = []
    nudge_outputs: list = []
    pre_deadline_outputs: list = []
    res = {}
    try:
        # ── 1. 挨拶（実ターン） ─────────────────────────────────
        r1 = (await mode.process_input("こんばんは", is_internal_nudge=False)) or ""
        outputs_all.append(r1)
        print(f"  [turn] user=こんばんは -> {r1[:80]}")

        # ── 2. 12s 沈黙 -> nudge1、さらに 20s -> nudge2 ─────────
        for label, silence in (("nudge1", 12), ("nudge2", 20)):
            await backdate(cache, silence)
            n = (await _drive_silence_nudge(mode)) or ""
            outputs_all.append(n)
            if n.strip() and n.strip() != "…":
                nudge_outputs.append(n)
            print(f"  [{label}] -> {n[:80] if n.strip() else '…'}")

        # ── 3. 予約（実 meta-tool 経由・8秒後） ──────────────────
        req_ts = datetime.now()
        r2 = (await mode.process_input("8秒後に好きな魚を教えてくれる?", is_internal_nudge=False)) or ""
        outputs_all.append(r2)
        pre_deadline_outputs.append(r2)
        await tm.flush_command_queue(timeout=5.0)
        print(f"  [turn] user=8秒後に好きな魚 -> {r2[:80]}")

        # 予約が登録されたか + deadline 精度 [D/I]
        insts = [i for i in tm._active_instructions.values()
                 if i.status.value in ("pending", "active")]
        deadline_dt = None
        deadline_err = None
        if insts:
            inst = max(insts, key=lambda i: i.created_at)
            if inst.deadline_at:
                try:
                    deadline_dt = datetime.fromisoformat(inst.deadline_at)
                    deadline_err = (deadline_dt - (req_ts + timedelta(seconds=8))).total_seconds()
                except (ValueError, TypeError):
                    pass
        res["reservation_created"] = bool(insts)
        res["deadline_err_sec"] = deadline_err
        res["D_deadline_accurate"] = (deadline_err is not None and abs(deadline_err) <= 3.0)

        # fact 汚染チェック [A] — 予約直後（ライブでは了解発話が capture された）
        polluted = [f for f in tm._committed_facts.values()
                    if f.status == "active" and ACK_FACT_RE.search(f.answer_text or "")]
        res["A_no_ack_fact"] = (len(polluted) == 0)
        if polluted:
            print(f"  [FACT-POLLUTION] {[(f.topic_norm, f.answer_text) for f in polluted]}")

        # ── 4. deadline 前の沈黙 nudge -> 早期回答? [B] ─────────
        await backdate(cache, 4)
        n3 = (await _drive_silence_nudge(mode)) or ""
        outputs_all.append(n3)
        pre_deadline_outputs.append(n3)
        if n3.strip() and n3.strip() != "…":
            nudge_outputs.append(n3)
        print(f"  [nudge3(pre-deadline)] -> {n3[:80] if n3.strip() else '…'}")
        res["B_no_early_answer"] = not any(_mentions_fish(o) for o in pre_deadline_outputs)

        # ── 5. deadline 経過 -> 督促履行 ─────────────────────────
        overdue_resp = ""
        if insts and deadline_dt is not None:
            wait = (deadline_dt - datetime.now()).total_seconds() + 0.5
            wait = max(0.0, min(wait, 20.0))  # 誤った遠い deadline でも待ちすぎない
            if wait:
                await asyncio.sleep(wait)
            await tm._reconcile_instruction_status()
            iid = inst.id
            cur = tm.get_instruction(iid)
            if cur and cur["status"] == "active":
                payload = f"[内部: 期限超過 {iid} を履行 — {inst.instruction}]"
                overdue_resp = (await mode.process_input(payload, is_internal_nudge=True)) or ""
                outputs_all.append(overdue_resp)
                await tm.flush_command_queue(timeout=5.0)
                print(f"  [overdue] -> {overdue_resp[:80]}")
        res["fulfilled_with_fish"] = _mentions_fish(overdue_resp)

        # ── 6. セッション不変条件 ────────────────────────────────
        res["E_greeting_once"] = (_greeting_count(outputs_all) <= 1)
        res["greeting_count"] = _greeting_count(outputs_all)
        dup = False
        for a, b in zip(nudge_outputs, nudge_outputs[1:]):
            if SequenceMatcher(None, a, b).ratio() >= 0.8:
                dup = True
        res["dup_no_repeat"] = not dup
        await tm._reconcile_instruction_status()
        orphans = [i for i in tm._active_instructions.values() if i.status.value == "active"]
        res["task_no_orphan_active"] = (len(orphans) == 0)
        res["outputs"] = outputs_all
        return res
    finally:
        await teardown(tm, cache, paths)


async def _drive_silence_nudge(mode) -> str:
    """_process_idle_input('…') を直接駆動し、発話テキストを返す（実 idle 経路）。"""
    holder = {}
    orig = mode.process_input

    async def capture(text, is_internal_nudge=False):
        r = await orig(text, is_internal_nudge=is_internal_nudge)
        holder["resp"] = r
        return r

    mode.process_input = capture
    try:
        await mode._process_idle_input("…", is_silence_nudge=True)
    finally:
        mode.process_input = orig
    return holder.get("resp") or ""


async def run_stale_screen_scenario(run_no: int) -> dict:
    """V: 古 alert(ニュース) + 新 alert(エディタ) -> 応答は新画面に言及すべき。"""
    mode, tm, cache, llm, paths = await make_session()
    try:
        now = time.time()
        llm._vlm_alerts.append((now - 90, "ブラウザでニュース記事（宇宙開発の話題）を読んでいる", "major"))
        llm._vlm_alerts.append((now - 2, "画面がコードエディタに切り替わり、Pythonのコードが表示されている", "major"))
        # journal は newest-first（実装と同じ向き）。古い行も混ぜてライブ状況を再現。
        journal = (
            "[たった今/MAJOR] 画面がコードエディタに切り替わり、Pythonのコードが表示されている\n"
            "[1分前/MAJOR] ブラウザでニュース記事（宇宙開発の話題）を読んでいる\n"
            "[2分前/MINOR] ニュースサイトをスクロールしている"
        )
        mode.vlm_bridge = SimpleNamespace(is_running=True, get_scene_description=lambda: journal)
        # 直近会話（ニュースの話をしていた = 古い話題に引っ張られやすい状況）
        await cache.add_turn("この宇宙開発のニュースすごいね", "ほんとだ、ロケットの再利用すごい進歩だね")
        await backdate(cache, 30)

        holder = {}
        orig = mode.process_input

        async def capture(text, is_internal_nudge=False):
            r = await orig(text, is_internal_nudge=is_internal_nudge)
            holder["resp"] = r
            return r

        mode.process_input = capture
        try:
            await mode._process_idle_input(
                "[内部: 画面に新しい変化があった - 自然にリアクションして]",
                is_silence_nudge=False,
            )
        finally:
            mode.process_input = orig
        resp = holder.get("resp") or ""
        print(f"  [vlm-nudge] -> {resp[:120]}")
        new_tokens = ("エディタ", "コード", "Python", "python", "プログラ")
        old_tokens = ("ニュース", "記事", "宇宙", "ロケット")
        mentions_new = any(t in resp for t in new_tokens)
        mentions_old = any(t in resp for t in old_tokens)
        return {"V_new_screen": mentions_new and not mentions_old,
                "mentions_new": mentions_new, "mentions_old": mentions_old, "resp": resp}
    finally:
        await teardown(tm, cache, paths)


INVARIANTS = ["E_greeting_once", "B_no_early_answer", "D_deadline_accurate",
              "A_no_ack_fact", "dup_no_repeat", "task_no_orphan_active"]


async def main(n: int) -> None:
    if not (Config.GROQ_API_KEY and Config.OPENAI_API_KEY):
        print("Tier-3 needs GROQ/OPENAI keys in .env — aborting.")
        sys.exit(2)
    model_label = MODEL_OVERRIDE or Config.AI1_MODEL
    print(f"Tier-3 session test: model AI1={model_label}, {n} run(s)\n")
    tallies = {k: 0 for k in INVARIANTS}
    v_pass = 0
    extra = {"reservation_created": 0, "fulfilled_with_fish": 0}
    errs = []
    for r in range(1, n + 1):
        print(f"===== accident-replay session run {r} =====")
        try:
            res = await run_accident_session(r)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {type(e).__name__}: {e}")
            errs.append(str(e))
            continue
        for k in INVARIANTS:
            ok = bool(res.get(k))
            tallies[k] += 1 if ok else 0
            print(f"    [{'PASS' if ok else 'FAIL'}] {k}"
                  + (f" (err={res['deadline_err_sec']:+.1f}s)" if k == "D_deadline_accurate" and res.get("deadline_err_sec") is not None else "")
                  + (f" (greetings={res['greeting_count']})" if k == "E_greeting_once" else ""))
        for k in extra:
            extra[k] += 1 if res.get(k) else 0
        print()
    for r in range(1, n + 1):
        print(f"===== stale-screen scenario run {r} =====")
        try:
            vres = await run_stale_screen_scenario(r)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {type(e).__name__}: {e}")
            continue
        ok = vres["V_new_screen"]
        v_pass += 1 if ok else 0
        print(f"    [{'PASS' if ok else 'FAIL'}] V_new_screen "
              f"(new={vres['mentions_new']} old={vres['mentions_old']})\n")

    print("================ TIER-3 SUMMARY ================")
    for k in INVARIANTS:
        print(f"  {k}: {tallies[k]}/{n}")
    print(f"  V_new_screen: {v_pass}/{n}")
    print(f"  (reservation created: {extra['reservation_created']}/{n}, "
          f"fulfilled w/ fish: {extra['fulfilled_with_fish']}/{n})")
    if errs:
        print(f"  errors: {len(errs)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if len(sys.argv) > 2:
        MODEL_OVERRIDE = sys.argv[2]
    asyncio.run(main(n))
