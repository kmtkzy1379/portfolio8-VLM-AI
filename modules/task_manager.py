"""TaskManager: Plan-and-Execute 用の Task Queue 状態管理 + 永続化。

責務:
- in-memory の active Plan / Task 状態保持
- tasks.jsonl への append-only 永続化（既存 rag.py の write queue パターン踏襲）
- 状態遷移の中央管理（user_action sticky / Validator 自動 DONE / 楽観 IN_PROGRESS）
- worker の自己復旧（3 回まで再起動）
- 起動時の SoT 整合（goal.txt との reconcile）

スレッドモデル:
- すべての書き換えは event loop 上の `_command_worker_body` 内で行われる
- sync 入口は `enqueue_command_nowait` だけ（同一 loop 前提）
- 読み取り (`is_empty`, `get_active_tasks`, `format_for_prompt`) は lock を取らない
  eventually-consistent fast path
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import statistics
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Optional

from config import Config
from modules.task_schema import (
    ActiveInstruction,
    CommittedFact,
    InstructionStatus,
    Plan,
    Task,
    TaskPriority,
    TaskStatus,
    new_fact_id,
    new_instruction_id,
    new_plan_id,
    now_iso,
    priority_rank,
)

logger = logging.getLogger(__name__)
# UI / debug 表示用ロガー（modules/llm.py:18 と同型）
task_logger = logging.getLogger("eve.task")
task_logger.propagate = False


# Validator 自動 DONE 判定用の閾値（プラン本文に合わせる）
_AUTO_DONE_PROGRESS_MEDIAN = 0.75
_AUTO_DONE_ALIGNMENT_LATEST = 0.6
_AUTO_DONE_WINDOW = 3
# Validator 自動 FAILED 判定
_AUTO_FAIL_FACTUALITY_MEDIAN = 0.4
_AUTO_FAIL_WINDOW = 3
# 完了済み task を in-memory から evict するまでの Feedback サイクル数
_EVICT_AFTER_CYCLES = 3
# 24 時間以上前の plan は起動時に invalidate
_PLAN_STALENESS_HOURS = 24


class TaskManager:
    """Task Queue 状態管理 + 永続化。"""

    def __init__(self, tasks_file: Optional[str] = None):
        self.tasks_file: str = tasks_file or getattr(Config, "TASK_QUEUE_FILE", "tasks.jsonl")
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 状態
        self._active_plan: Optional[Plan] = None
        # task_id -> Task のフラット index（active_plan.tasks と同じ Task インスタンスを指す）
        self._tasks_by_id: dict[str, Task] = {}
        # Validator スコア履歴（task_id -> deque[{turn_index, score_dict, evaluated_at}]）
        self._task_scores: dict[str, deque] = {}
        # replan 判定用の全体スコア window
        self._overall_score_window: deque = deque(maxlen=5)
        # in_progress 中の task id（mark_in_progress_on_turn が書く）
        self._in_progress_top_id: Optional[str] = None
        # 完了 / 失敗 / skipped 後のサイクル経過カウント（key=task_id, val=cycle count）
        self._evict_counters: dict[str, int] = {}

        # Planner 走行中フラグ（prompt race 防止）
        self._installing_plan: bool = False
        # Validator の多重発火防止
        self._last_validator_run_cycle_id: int = -1

        # ファイル書き込み worker
        self._command_queue: asyncio.Queue = asyncio.Queue()
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._command_worker_task: Optional[asyncio.Task] = None
        self._write_worker_task: Optional[asyncio.Task] = None
        self._stopped: bool = False

        # Step 1.5: active_instruction の保持と callback
        self._active_instructions: dict[str, ActiveInstruction] = {}
        self._activated_callback: Optional[Callable[[dict], None]] = None
        # 同一 instruction を二重発火させないための済み id set
        self._activated_callback_fired: set[str] = set()
        # Step 1.5 Fix Layer E: 同一 PROVISIONAL_DONE に対する audit 二重発火防止 set
        self._audit_spawned_ids: set[str] = set()
        # Fix-9b: コミット事実 / 一貫性ストア（fact_id -> CommittedFact）。
        # 回答ドリフト（猫→うさぎ）防止。active なものだけ system prompt に注入（ペルソナ持続）。
        self._committed_facts: dict[str, CommittedFact] = {}

    # ==================================================================
    # Public read API（sync fast path、lock を取らない）
    # ==================================================================
    @property
    def active_plan(self) -> Optional[Plan]:
        return self._active_plan

    @property
    def installing_plan(self) -> bool:
        return self._installing_plan

    def is_empty(self) -> bool:
        """active な Task が 1 つもないか。"""
        if self._active_plan is None:
            return True
        for t in self._active_plan.tasks:
            if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                return False
        return True

    def get_active_tasks(self, n: int = 3) -> list[Task]:
        """優先度＋依存解決でトップ N を返す。
        DAG 解決は最小限（PENDING/IN_PROGRESS のみ抽出、依存 task が DONE か skipped 以外なら除外）。
        """
        if self._active_plan is None:
            return []
        candidates: list[Task] = []
        for t in self._active_plan.tasks:
            if t.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                continue
            # 依存の解決確認
            blocked = False
            for dep_id in t.depends_on:
                dep = self._tasks_by_id.get(dep_id)
                if dep is None:
                    continue
                if dep.status not in (TaskStatus.DONE, TaskStatus.SKIPPED):
                    blocked = True
                    break
            if not blocked:
                candidates.append(t)
        candidates.sort(key=lambda t: priority_rank(t.priority))
        return candidates[:n]

    def format_for_prompt(self) -> str:
        """system prompt 注入用テキスト。size cap を内蔵。
        installing_plan 中は空文字列を返して race を避ける。
        """
        if self._installing_plan:
            return ""
        max_active = int(getattr(Config, "TASK_MAX_ACTIVE", 3))
        tasks = self.get_active_tasks(n=max_active)
        if not tasks:
            return ""
        lines = ["[Active Tasks — 上から順に意識（参考情報）]:"]
        for t in tasks:
            desc = t.description or ""
            if len(desc) > 40:
                desc = desc[:39] + "…"
            prio_label = (t.priority.value if isinstance(t.priority, TaskPriority) else str(t.priority)).upper()
            lines.append(f"  {t.id} [{prio_label}] {desc}")
        block = "\n".join(lines) + "\n"
        # 全体 300 字 cap
        if len(block) > 300:
            block = block[:299] + "…\n"
        return block

    def should_replan(self) -> tuple[bool, str]:
        """中央値ベースの replan 判定（Step 6 で詳細化、Step 1-2 では False 固定）。"""
        threshold = float(getattr(Config, "VALIDATOR_REPLAN_THRESHOLD", 0.5))
        scores = [s for s in self._overall_score_window if isinstance(s, (int, float))]
        if len(scores) < 3:
            return False, ""
        med = statistics.median(scores)
        if med < threshold:
            return True, f"overall median {med:.2f} < {threshold:.2f}"
        return False, ""

    # ==================================================================
    # Sync 入口（_execute_tool 等から呼ばれる）
    # ==================================================================
    def enqueue_command_nowait(self, cmd: dict) -> None:
        """同期コードからコマンドを enqueue する。
        同一 event loop 内であれば put_nowait は安全。"""
        try:
            self._command_queue.put_nowait(cmd)
        except asyncio.QueueFull:
            logger.warning("[TaskManager] command queue full, dropping: %s", cmd.get("kind", "?"))

    def mark_in_progress_on_turn(self, top_task_id: Optional[str]) -> None:
        """応答開始時にトップ task を擬似 IN_PROGRESS にする（sync, fast path）。
        installing_plan 中はスキップ。"""
        if self._installing_plan or top_task_id is None:
            return
        t = self._tasks_by_id.get(top_task_id)
        if t is None:
            return
        if t.status == TaskStatus.PENDING:
            t.status = TaskStatus.IN_PROGRESS
            t.updated_at = now_iso()
            self._in_progress_top_id = top_task_id
            # 状態遷移ログは Step 2 以降で詳細出力。今は記録のみ。

    # ==================================================================
    # Async 書き換え API（worker から呼ばれる or 直接 await される）
    # ==================================================================
    async def start(self) -> None:
        """起動: event loop 保存 + ファイル復元 + worker 起動。"""
        self._loop = asyncio.get_running_loop()
        await self._restore_from_file()
        self._reconcile_with_goal_file(getattr(Config, "GOAL_FILE", "goal.txt"))
        self._command_worker_task = asyncio.create_task(self._command_worker_wrapper())
        self._write_worker_task = asyncio.create_task(self._write_worker())
        # Step 1.5: 起動直後に reconcile を 1 回走らせ、再起動中に deadline 到達していた
        # instruction を ACTIVE / EXPIRED に正しく昇格させる（callback は未登録なので発火しない）
        try:
            await self._reconcile_instruction_status()
        except Exception as e:  # noqa: BLE001
            logger.warning("[TaskManager] initial reconcile failed: %s", e)
        task_logger.info("[TaskManager] started, active_plan=%s, n_tasks=%d, n_instructions=%d",
                         self._active_plan.plan_id if self._active_plan else None,
                         len(self._tasks_by_id),
                         len(self._active_instructions))

    async def stop(self) -> None:
        self._stopped = True
        # 終了シグナル
        await self._command_queue.put(None)
        await self._write_queue.put(None)
        if self._command_worker_task is not None:
            self._command_worker_task.cancel()
        if self._write_worker_task is not None:
            self._write_worker_task.cancel()

    async def install_plan(self, plan: Optional[Plan]) -> None:
        """新 plan を active 化。`plan is None` は no-op（fail-soft）。
        旧 plan の Active task はすべて SKIPPED に遷移し、in-memory deque から evict。"""
        if plan is None:
            return
        self._installing_plan = True
        try:
            # 旧 plan の active task を SKIPPED
            if self._active_plan is not None:
                old_plan = self._active_plan
                for t in old_plan.tasks:
                    if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                        t.status = TaskStatus.SKIPPED
                        t.updated_at = now_iso()
                old_plan.superseded_by = plan.plan_id
                await self._write_queue.put({
                    "_kind": "plan_superseded",
                    "plan_id": old_plan.plan_id,
                    "superseded_by": plan.plan_id,
                    "at": now_iso(),
                })

            # 新 plan を採用
            self._active_plan = plan
            self._tasks_by_id = {t.id: t for t in plan.tasks}
            self._task_scores = {t.id: deque(maxlen=10) for t in plan.tasks}
            self._evict_counters = {}

            # 永続化（plan 行 + 各 task 行）
            await self._write_queue.put({"_kind": "plan", **plan.to_jsonable(), "tasks": [t.id for t in plan.tasks]})
            for t in plan.tasks:
                await self._write_queue.put({"_kind": "task", "plan_id": plan.plan_id, **t.to_jsonable()})
            task_logger.info("[TaskManager] installed plan %s with %d tasks", plan.plan_id, len(plan.tasks))
        finally:
            self._installing_plan = False

    async def update_status(
        self,
        task_id: str,
        status: str,
        reason: str = "",
        source: str = "user_action",
    ) -> None:
        """状態遷移の中央 API。user_action sticky ルールはここで適用される。
        source: "user_action" | "validator" | "rollback"
        """
        t = self._tasks_by_id.get(task_id)
        if t is None:
            task_logger.warning("[TaskManager] update_status: unknown task_id=%s", task_id)
            return

        # user_action sticky: 既に user_action で確定した task は validator では上書きしない
        if source == "validator" and t.last_authoritative_kind == "user_action":
            task_logger.debug("[TaskManager] skip validator update on user-sticky task=%s", task_id)
            return

        try:
            new_status = TaskStatus(status)
        except ValueError:
            task_logger.warning("[TaskManager] update_status: invalid status=%s", status)
            return

        old_status = t.status
        t.status = new_status
        t.updated_at = now_iso()
        if reason:
            t.failure_reason = reason if new_status == TaskStatus.FAILED else None
        if source in ("user_action", "validator"):
            t.last_authoritative_kind = source

        await self._write_queue.put({
            "_kind": "status",
            "task_id": task_id,
            "status": new_status.value,
            "from": old_status.value if isinstance(old_status, TaskStatus) else str(old_status),
            "source": source,
            "reason": reason,
            "at": now_iso(),
        })
        task_logger.info("[TaskManager] %s: %s -> %s (%s)", task_id, old_status, new_status.value, source)

    async def record_validation(
        self,
        task_id: str,
        scores: dict,
        turn_index: Optional[int] = None,
    ) -> None:
        """Validator のスコア結果を記録。自動 DONE / FAILED 判定もここで実施。"""
        t = self._tasks_by_id.get(task_id)
        if t is None:
            return
        dq = self._task_scores.get(task_id)
        if dq is None:
            dq = deque(maxlen=10)
            self._task_scores[task_id] = dq
        dq.append({"turn_index": turn_index, "scores": scores, "at": now_iso()})

        # overall window（replan 判定用）
        ov = scores.get("score") or scores.get("goal_alignment")
        if isinstance(ov, (int, float)):
            self._overall_score_window.append(float(ov))

        # 自動 DONE
        if t.status == TaskStatus.IN_PROGRESS and t.last_authoritative_kind != "user_action":
            recent = list(dq)[-_AUTO_DONE_WINDOW:]
            progs = [r["scores"].get("task_progress") for r in recent
                     if isinstance(r["scores"].get("task_progress"), (int, float))]
            if len(progs) >= _AUTO_DONE_WINDOW:
                med = statistics.median(progs)
                latest_align = recent[-1]["scores"].get("goal_alignment")
                if (
                    med >= _AUTO_DONE_PROGRESS_MEDIAN
                    and isinstance(latest_align, (int, float))
                    and latest_align >= _AUTO_DONE_ALIGNMENT_LATEST
                ):
                    await self.update_status(
                        task_id,
                        TaskStatus.DONE.value,
                        reason=f"auto-DONE (median progress {med:.2f})",
                        source="validator",
                    )
                    return
        # 自動 FAILED
        if t.status == TaskStatus.IN_PROGRESS and t.last_authoritative_kind != "user_action":
            recent = list(dq)[-_AUTO_FAIL_WINDOW:]
            facts = [r["scores"].get("factuality") for r in recent
                     if isinstance(r["scores"].get("factuality"), (int, float))]
            if len(facts) >= _AUTO_FAIL_WINDOW:
                med = statistics.median(facts)
                if med < _AUTO_FAIL_FACTUALITY_MEDIAN:
                    await self.update_status(
                        task_id,
                        TaskStatus.FAILED.value,
                        reason=f"auto-FAILED (median factuality {med:.2f})",
                        source="validator",
                    )
                    return
        # 履歴のみの記録
        await self._write_queue.put({
            "_kind": "validation",
            "task_id": task_id,
            "scores": scores,
            "turn_index": turn_index,
            "at": now_iso(),
        })

    async def rollback_in_progress(self) -> None:
        """CancelledError 経路で IN_PROGRESS を PENDING に戻す。"""
        if self._active_plan is None:
            return
        for t in self._active_plan.tasks:
            if t.status == TaskStatus.IN_PROGRESS and t.last_authoritative_kind != "user_action":
                t.status = TaskStatus.PENDING
                t.updated_at = now_iso()
                await self._write_queue.put({
                    "_kind": "status",
                    "task_id": t.id,
                    "status": TaskStatus.PENDING.value,
                    "from": TaskStatus.IN_PROGRESS.value,
                    "source": "rollback",
                    "at": now_iso(),
                })
        self._in_progress_top_id = None

    async def evict_finished_after_cycle(self) -> None:
        """Feedback サイクル毎に呼んで、完了 task を 3 サイクル後に in-memory から evict。
        ファイルからは消さない（append-only）。"""
        if self._active_plan is None:
            return
        keep: list[Task] = []
        for t in self._active_plan.tasks:
            if t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED):
                cnt = self._evict_counters.get(t.id, 0) + 1
                if cnt >= _EVICT_AFTER_CYCLES:
                    # evict
                    self._evict_counters.pop(t.id, None)
                    self._tasks_by_id.pop(t.id, None)
                    self._task_scores.pop(t.id, None)
                    continue
                self._evict_counters[t.id] = cnt
            keep.append(t)
        self._active_plan.tasks = keep

    # ==================================================================
    # 内部: worker + 永続化
    # ==================================================================
    async def _command_worker_wrapper(self) -> None:
        """worker 本体を例外時に 3 回まで自動再起動するラッパ。"""
        restart_count = 0
        while restart_count < 4 and not self._stopped:
            try:
                await self._command_worker_body()
                break
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                restart_count += 1
                logger.exception("[TaskManager] worker died (%d/3), restarting: %s", restart_count, e)
                await asyncio.sleep(1.0)
        else:
            if not self._stopped:
                logger.critical("[TaskManager] worker permanently dead after 4 attempts")
                task_logger.critical("[TaskManager] command worker permanently dead")

    async def _command_worker_body(self) -> None:
        """コマンドを取り出してディスパッチする本体。"""
        while True:
            cmd = await self._command_queue.get()
            try:
                if cmd is None:
                    return
                kind = cmd.get("kind")
                if kind == "update_status":
                    await self.update_status(
                        cmd.get("task_id", ""),
                        cmd.get("status", TaskStatus.IN_PROGRESS.value),
                        cmd.get("reason", ""),
                        cmd.get("source", "user_action"),
                    )
                elif kind == "record_validation":
                    await self.record_validation(
                        cmd.get("task_id", ""),
                        cmd.get("scores", {}),
                        cmd.get("turn_index"),
                    )
                elif kind == "set_active_instruction":
                    await self._handle_set_active_instruction(cmd)
                elif kind == "clear_active_instruction":
                    await self._handle_clear_active_instruction(cmd)
                elif kind == "notify_clear_rejected":
                    await self._handle_notify_clear_rejected(cmd)
                elif kind == "confirm_provisional_done":
                    await self._handle_confirm_provisional_done(cmd)
                elif kind == "revert_provisional_done":
                    await self._handle_revert_provisional_done(cmd)
                elif kind == "commit_fact":
                    await self._handle_commit_fact(cmd)
                else:
                    logger.debug("[TaskManager] unknown command kind=%s", kind)
            finally:
                self._command_queue.task_done()

    async def _write_worker(self) -> None:
        """ファイル追記 worker（rag.py:_write_worker と同型）。"""
        while True:
            try:
                entry = await self._write_queue.get()
                if entry is None:
                    break
                await asyncio.to_thread(self._append_to_file, entry)
                self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("[TaskManager] write worker error: %s", e)
                try:
                    self._write_queue.task_done()
                except ValueError:
                    pass

    def _append_to_file(self, entry: dict) -> None:
        try:
            with open(self.tasks_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("[TaskManager] file write error: %s", e)

    # ==================================================================
    # 起動時復元 + SoT 整合
    # ==================================================================
    async def _restore_from_file(self) -> None:
        """tasks.jsonl の末尾から最新 active plan を tail 復元。
        IN_PROGRESS の task は起動時に PENDING へダウングレード（crash 安全）。
        24h 以上前の plan は自動 invalidate。"""
        if not os.path.exists(self.tasks_file):
            return
        try:
            lines = await asyncio.to_thread(self._read_all_lines)
        except Exception as e:  # noqa: BLE001
            logger.warning("[TaskManager] restore read error: %s", e)
            return
        if not lines:
            return

        # 末尾から走査して、最新の plan_id を見つける（superseded されていないもの）
        plans: dict[str, dict] = {}  # plan_id -> meta
        tasks: dict[str, list[dict]] = {}  # plan_id -> task dicts
        status_updates: list[dict] = []  # 全 status_update
        # Step 1.5: active_instruction の復元
        instruction_records: dict[str, dict] = {}  # id -> 最新の active_instruction record
        instruction_status_updates: list[dict] = []  # instruction_status の時系列
        # Fix-9b: committed_fact の復元
        committed_fact_records: dict[str, dict] = {}  # fact_id -> 最新の committed_fact record
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = entry.get("_kind")
            if kind == "plan":
                plans[entry.get("plan_id", "")] = entry
            elif kind == "task":
                pid = entry.get("plan_id", "")
                tasks.setdefault(pid, []).append(entry)
            elif kind == "plan_superseded":
                pid = entry.get("plan_id", "")
                if pid in plans:
                    plans[pid]["_superseded"] = True
            elif kind == "status":
                status_updates.append(entry)
            elif kind == "active_instruction":
                iid = entry.get("id", "")
                if iid:
                    instruction_records[iid] = entry
            elif kind == "instruction_status":
                instruction_status_updates.append(entry)
            elif kind == "committed_fact":
                fid = entry.get("fact_id", "")
                if fid:
                    committed_fact_records[fid] = entry

        # Fix-9b: committed_fact を復元（plan の有無に関係なく load = ペルソナ持続）。
        # 同 fact_id の最新レコードが status を確定（active のみ in-memory に載せる）。
        for fid, rec in committed_fact_records.items():
            try:
                fact = CommittedFact.from_jsonable(rec)
            except Exception:  # noqa: BLE001
                continue
            if fact.status == "active":
                self._committed_facts[fact.fact_id] = fact
        if self._committed_facts:
            task_logger.info("[TaskManager] restored %d committed_facts", len(self._committed_facts))

        # 最新の非 superseded plan を選ぶ
        active_plan_meta = None
        for pid, p in reversed(list(plans.items())):
            if not p.get("_superseded"):
                active_plan_meta = p
                break
        if active_plan_meta is None:
            return

        plan_id = active_plan_meta.get("plan_id", "")
        created_at_str = active_plan_meta.get("created_at", "")

        # Staleness check: 24h 以上前なら invalidate
        try:
            if created_at_str:
                created_dt = datetime.fromisoformat(created_at_str)
                if datetime.now() - created_dt > timedelta(hours=_PLAN_STALENESS_HOURS):
                    task_logger.info("[TaskManager] plan %s is stale (> %dh), invalidating",
                                     plan_id, _PLAN_STALENESS_HOURS)
                    return
        except Exception:  # noqa: BLE001
            pass

        # Task 復元（最新の status_update を反映）
        latest_status: dict[str, dict] = {}
        for su in status_updates:
            tid = su.get("task_id", "")
            if not tid:
                continue
            latest_status[tid] = su

        task_objs: list[Task] = []
        for td in tasks.get(plan_id, []):
            tid = td.get("id", "")
            su = latest_status.get(tid)
            if su:
                td["status"] = su.get("status", td.get("status"))
            try:
                t = Task.from_jsonable(td)
            except Exception:  # noqa: BLE001
                continue
            # crash 安全: IN_PROGRESS → PENDING
            if t.status == TaskStatus.IN_PROGRESS:
                t.status = TaskStatus.PENDING
            task_objs.append(t)

        self._active_plan = Plan(
            plan_id=plan_id,
            goal_short_snapshot=active_plan_meta.get("goal_short_snapshot", ""),
            goal_long_snapshot=active_plan_meta.get("goal_long_snapshot", ""),
            tasks=task_objs,
            created_at=created_at_str,
        )
        self._tasks_by_id = {t.id: t for t in task_objs}
        self._task_scores = {t.id: deque(maxlen=10) for t in task_objs}
        task_logger.info("[TaskManager] restored plan %s with %d tasks", plan_id, len(task_objs))

        # Step 1.5: active_instruction の復元
        # 最新の instruction_status を集約（id -> 最後の status update）
        latest_inst_status: dict[str, dict] = {}
        for su in instruction_status_updates:
            iid = su.get("id", "")
            if iid:
                latest_inst_status[iid] = su
        for iid, rec in instruction_records.items():
            try:
                inst = ActiveInstruction.from_jsonable(rec)
            except Exception:  # noqa: BLE001
                continue
            su = latest_inst_status.get(iid)
            if su:
                try:
                    inst.status = InstructionStatus(su.get("status", inst.status.value))
                except Exception:  # noqa: BLE001
                    pass
                if su.get("activated_at"):
                    inst.activated_at = su["activated_at"]
                if su.get("cleared_at"):
                    inst.cleared_at = su["cleared_at"]
                if su.get("cleared_reason"):
                    inst.cleared_reason = su["cleared_reason"]
            # Fix-6 P2-f: 再起動時 PROVISIONAL_DONE → DONE 確定。
            # 旧 Fix-3 の ACTIVE 降格は撤廃 (タスク復活問題の再発防止)。
            # 新 Fix-6 では audit は state を戻す権限なし (advisory) なので、
            # 再起動で audit 結果ロストしても DONE 確定が一貫した結末。
            if inst.status == InstructionStatus.PROVISIONAL_DONE:
                inst.status = InstructionStatus.DONE
                inst.cleared_at = now_iso()
                inst.cleared_reason = "auto-confirmed on restart (Fix-6 advisory policy)"
                inst.audit_pending_at = None
                task_logger.info(
                    "[Instruction] %s PROVISIONAL_DONE -> DONE (restart auto-confirm, Fix-6 advisory)",
                    iid,
                )
            # DONE / EXPIRED / SUPERSEDED な instruction も in-memory には残す（evict は別経路）
            self._active_instructions[iid] = inst
        if self._active_instructions:
            task_logger.info("[TaskManager] restored %d active_instructions", len(self._active_instructions))

    def _read_all_lines(self) -> list[str]:
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            return f.readlines()

    def _reconcile_with_goal_file(self, goal_path: str) -> None:
        """goal.txt と active_plan.goal_short_snapshot を比較し、
        goal.txt が新しいなら plan を invalidate。"""
        if self._active_plan is None:
            return
        if not os.path.exists(goal_path):
            return
        try:
            with open(goal_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
        except Exception:  # noqa: BLE001
            return
        goal_short = gj.get("goal_short", "")
        extracted_at_str = gj.get("extracted_at", "")
        if not goal_short:
            return
        if goal_short == self._active_plan.goal_short_snapshot:
            return
        # 不一致: timestamp で判定
        try:
            goal_dt = datetime.fromisoformat(extracted_at_str) if extracted_at_str else None
            plan_dt = (
                datetime.fromisoformat(self._active_plan.created_at)
                if self._active_plan.created_at else None
            )
        except Exception:  # noqa: BLE001
            goal_dt = plan_dt = None

        if goal_dt is not None and plan_dt is not None and goal_dt > plan_dt:
            task_logger.info(
                "[TaskManager] goal.txt is newer than plan, invalidating plan %s",
                self._active_plan.plan_id,
            )
            self._active_plan = None
            self._tasks_by_id = {}
            self._task_scores = {}
        # timestamp 不在 / plan の方が新しい場合はそのまま維持

    # ==================================================================
    # Step 1.5: active_instruction API
    # ==================================================================
    def register_instruction_activated_callback(
        self, cb: Callable[[dict], None]
    ) -> None:
        """PENDING → ACTIVE 遷移時に呼ばれる callback を登録（mode-agnostic）。
        callback はメインループ上で同期呼び出しされる前提。"""
        self._activated_callback = cb

    def pop_pending_audits(self) -> list[dict]:
        """Step 1.5 Fix Layer E E2: PROVISIONAL_DONE 状態で未だ audit 未発火の
        instructions を返す。 sync fast path。 同一 instruction の二重 audit を防ぐため、
        返した id は `_audit_spawned_ids` に追加される（confirm/revert で discard）。

        戻り値の各 dict: {id, instruction, eve_response, audit_reason (前回未達成があれば)}
        """
        result: list[dict] = []
        for inst in self._active_instructions.values():
            if inst.status != InstructionStatus.PROVISIONAL_DONE:
                continue
            if inst.id in self._audit_spawned_ids:
                continue
            self._audit_spawned_ids.add(inst.id)
            result.append({
                "id": inst.id,
                "instruction": inst.instruction,
                "eve_response": inst.provisional_response or "",
                "previous_audit_reason": inst.audit_reason,  # 前回 revert があれば
            })
        return result

    async def flush_command_queue(self, timeout: float = 2.0) -> None:
        """Step 1.5 Fix Layer E E0: command_queue が drain されるまで await。
        LLMHandler が enqueue した clear_active_instruction が
        _handle_clear_active_instruction で処理され、 PROVISIONAL_DONE に
        遷移完了するのを保証する。 worker が死んでる場合の永久 hang を防ぐため timeout 付き。"""
        try:
            await asyncio.wait_for(self._command_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[TaskManager] flush_command_queue timed out after %.1fs (qsize=%d)",
                timeout, self._command_queue.qsize(),
            )

    def has_active_instruction(self) -> bool:
        """Step 1.5 Fix C2: 期限超過 (ACTIVE) instruction が現在 1 件以上あるかを返す
        (sync fast path)。 RAG 側で capability_denial 動的フィルタの判定に使う。
        PROVISIONAL_DONE / PENDING / EXPIRING は含めない（履行催促状態のみ）。"""
        for inst in self._active_instructions.values():
            if inst.status == InstructionStatus.ACTIVE:
                return True
        return False

    def get_active_instructions_for_prompt(self) -> list[dict]:
        """system prompt 注入用に instruction の dict 配列を返す。
        derived_status は datetime.now() / deadline_at / activated_at から動的に計算。
        DONE / EXPIRED / SUPERSEDED の instruction、および計算上 EXPIRED 扱いになるものは
        表示しない。sync fast path。"""
        now = datetime.now()
        default_expire = float(getattr(Config, "ACTIVE_INSTRUCTION_DEFAULT_EXPIRE_SEC", 300))
        result: list[dict] = []
        for inst in self._active_instructions.values():
            if inst.status in (
                InstructionStatus.DONE,
                InstructionStatus.EXPIRED,
                InstructionStatus.SUPERSEDED,
                # Step 1.5 Fix Layer E E7: PROVISIONAL_DONE は AI2 audit 待ちの内部状態。
                # ユーザー / Eve には DONE 同等として隠蔽（system prompt に表示しない）。
                InstructionStatus.PROVISIONAL_DONE,
            ):
                continue
            derived = self._compute_derived_status(inst, now)
            if derived == InstructionStatus.EXPIRED:
                continue
            item: dict = {
                "id": inst.id,
                "instruction": inst.instruction,
                "reason": inst.reason,
                "created_at": inst.created_at,
                "deadline_at": inst.deadline_at,
                "activated_at": inst.activated_at,
                "derived_status": derived.value,
                "created_by": inst.created_by,
            }
            # 補助情報（表示時に使う）
            if inst.deadline_at:
                try:
                    dl = datetime.fromisoformat(inst.deadline_at)
                    delta = (now - dl).total_seconds()
                    if delta < 0:
                        item["seconds_until_deadline"] = int(-delta)
                    else:
                        item["seconds_overdue"] = int(delta)
                except (ValueError, TypeError):
                    pass
            if inst.created_at and not inst.deadline_at:
                try:
                    created = datetime.fromisoformat(inst.created_at)
                    expire_at = created + timedelta(seconds=default_expire)
                    item["seconds_until_default_expire"] = int((expire_at - now).total_seconds())
                except (ValueError, TypeError):
                    pass
            result.append(item)
        return result

    # ==================================================================
    # Fix-9b: committed-fact / 一貫性ストア
    # ==================================================================
    def get_instruction(self, iid: str) -> Optional[dict]:
        """instruction id から軽い dict を返す（capture 用 sync fast path）。None if 不在。"""
        inst = self._active_instructions.get(iid)
        if inst is None:
            return None
        return {"id": inst.id, "instruction": inst.instruction, "status": inst.status.value}

    async def _handle_commit_fact(self, cmd: dict) -> None:
        """Fix-9b: Eve がコミットした回答を一貫性ストアに記録する。
        既に同 (scope, topic_norm) の active fact があれば **最初の答えを守る**（上書きしない）。
        これが回答ドリフト防止の核心。"""
        scope = cmd.get("scope", "eve")
        topic_norm = (cmd.get("topic_norm") or "").strip()
        answer_text = (cmd.get("answer_text") or "").strip()
        if not topic_norm or not answer_text:
            return
        instruction_id = cmd.get("instruction_id")
        ans = answer_text[:120]
        # Defend: 同じ話題に active な事実が既にあれば中身(answer_text)は保持。
        # ① Facet rotation: 新しい言い回しは recent_expressions に蓄積（表現の多様化用）。
        for f in self._committed_facts.values():
            if f.status == "active" and f.scope == scope and f.topic_norm == topic_norm:
                if f.instruction_id is None and instruction_id:
                    f.instruction_id = instruction_id  # link 補完（中身は保持）
                if ans and ans not in f.recent_expressions:
                    f.recent_expressions.append(ans)
                    if len(f.recent_expressions) > 4:
                        f.recent_expressions = f.recent_expressions[-4:]
                    await self._write_queue.put({"_kind": "committed_fact", **f.to_jsonable()})
                task_logger.debug(
                    "[CommitFact] defended %s 〈%s〉, recorded expression variant 〈%s〉",
                    f.fact_id, f.answer_text[:24], ans[:24],
                )
                return
        fact = CommittedFact(
            fact_id=new_fact_id(),
            topic_norm=topic_norm,
            answer_text=ans,
            scope=scope,
            instruction_id=instruction_id,
            committed_at=now_iso(),
            source=cmd.get("source", "nudge"),
            status="active",
            recent_expressions=[ans],
        )
        self._committed_facts[fact.fact_id] = fact
        self._evict_committed_facts_over_cap(32)
        await self._write_queue.put({"_kind": "committed_fact", **fact.to_jsonable()})
        task_logger.info("[CommitFact] committed %s 〈%s = %s〉", fact.fact_id, topic_norm, fact.answer_text[:30])

    def _evict_committed_facts_over_cap(self, cap: int) -> None:
        """active なコミット事実が cap を超えたら最古から落とす（メモリ/プロンプト保護）。"""
        active = [f for f in self._committed_facts.values() if f.status == "active"]
        if len(active) <= cap:
            return
        active.sort(key=lambda f: f.committed_at)
        for f in active[: len(active) - cap]:
            self._committed_facts.pop(f.fact_id, None)

    @staticmethod
    def _relevance_tokens(text: str) -> set:
        """関連性判定用の軽量トークン化（依存なし・再現率重視）。
        日本語/英数の連続(len>=2)を語として拾い、漢字は1文字も語として拾う。"""
        text = (text or "").lower()
        tokens = set()
        # 日本語(ひらがな・カタカナ・漢字)/英数の連続(len>=2)を語として拾う。
        for run in re.findall(r"[0-9a-z぀-ヿ一-鿿]{2,}", text):
            tokens.add(run)
        # 漢字は単体でも語になりやすいので1文字も拾う（再現率重視）。
        for kanji in re.findall(r"[一-鿿]", text):
            tokens.add(kanji)
        return tokens

    @classmethod
    def _fact_is_relevant(cls, f, rel_tokens: set) -> bool:
        """fact の topic/answer のトークンが relevance トークンと重なるか。"""
        ftoks = cls._relevance_tokens(f.topic_norm) | cls._relevance_tokens(f.answer_text)
        return bool(ftoks & rel_tokens)

    def _committed_fact_to_prompt(self, f) -> dict:
        return {"topic": f.topic_norm, "answer": f.answer_text,
                "recent_expressions": list(f.recent_expressions)}

    def has_imminent_pending_instruction(self, window_sec: float = 120.0) -> bool:
        """期限が window_sec 以内に迫った derived-PENDING 予約があるか（sync fast path）。

        Bug-B(code gate): この間は沈黙 "…" nudge を発火しない（早期履行をコードで遮断）。
        deadline 無し（既定 expire 待ち）の PENDING は対象外 — 長時間 Eve を黙らせない。
        """
        now = datetime.now()
        for item in self.get_active_instructions_for_prompt():
            if item.get("derived_status") != "pending" or not item.get("deadline_at"):
                continue
            try:
                dl = datetime.fromisoformat(item["deadline_at"])
            except (ValueError, TypeError):
                continue
            if 0 <= (dl - now).total_seconds() <= window_sec:
                return True
        return False

    def _fact_blocked_by_pending_instruction(self, f) -> bool:
        """Bug-B gate: 期限未到来 (derived PENDING) の予約に紐づく fact は描画しない。

        Fix-8 が instruction_pending_block を内部 nudge で隠しても、committed_facts 経由で
        〈予約トピック = …〉がリークすると沈黙 nudge が早期履行する（ライブで観測）。
        render 側の常時バックストップ（ディスク上の汚染済み fact にも効く）。
        期限到来 (ACTIVE) は描画する — 期限超過の履行で drift を守るのに必要。
        終端/迷子/instruction_id=None は描画（ペルソナ持続）。
        """
        iid = getattr(f, "instruction_id", None)
        if not iid:
            return False
        inst = self._active_instructions.get(iid)
        if inst is None:
            return False
        return self._compute_derived_status(inst, datetime.now()) == InstructionStatus.PENDING

    def get_committed_facts_for_prompt(
        self, limit: int = 6, relevance_text: Optional[str] = None
    ) -> list[dict]:
        """system prompt 注入用に active なコミット事実を返す（sync fast path）。

        relevance_text 無し or active が limit 以内 → 従来どおり最新 limit 件（byte-identical）。
        relevance_text あり & active が limit 超 → 「いまの話題に関連する fact」+「最新 recent_floor 件」を
        選んで最新 limit 件に絞る（嗜好が増えてもプロンプトが無限に伸びないようにする）。
        recent_floor が「直近コミットした事実」を必ず残すので、トークン照合の取りこぼしでもドリフト防御は保たれる。
        """
        active = [f for f in self._committed_facts.values()
                  if f.status == "active" and not self._fact_blocked_by_pending_instruction(f)]
        active.sort(key=lambda f: f.committed_at)

        if not relevance_text or len(active) <= limit:
            return [self._committed_fact_to_prompt(f) for f in active[-limit:]]

        # gating path: limit を超える数の fact があり、関連文脈が与えられた時のみ。
        recent_floor = 2  # 最新 2 件は無条件に残す（直近の嗜好を落とさない）。
        rel_tokens = self._relevance_tokens(relevance_text)
        keep_ids = {id(f) for f in active[-recent_floor:]}
        for f in active:
            if id(f) not in keep_ids and self._fact_is_relevant(f, rel_tokens):
                keep_ids.add(id(f))
        chosen = [f for f in active if id(f) in keep_ids]  # active 順(committed_at)を維持
        return [self._committed_fact_to_prompt(f) for f in chosen[-limit:]]

    def _release_facts_for_instruction(self, iid: str, new_status: str) -> None:
        """instruction の終端遷移に伴い紐づく fact の status を更新。
        DONE では eve-fact を active のまま残す（ペルソナ持続）。SUPERSEDED で superseded。"""
        if not iid:
            return
        for f in self._committed_facts.values():
            if f.instruction_id == iid and f.status == "active":
                f.status = new_status
                try:
                    self._write_queue.put_nowait({"_kind": "committed_fact", **f.to_jsonable()})
                except Exception:  # noqa: BLE001
                    pass

    def _compute_derived_status(
        self, inst: ActiveInstruction, now: datetime
    ) -> InstructionStatus:
        """current time と instruction state から derived status を計算（純粋関数）。"""
        # 終了状態はそのまま
        if inst.status in (
            InstructionStatus.DONE,
            InstructionStatus.EXPIRED,
            InstructionStatus.SUPERSEDED,
        ):
            return inst.status

        # 既に ACTIVE: grace 超過なら EXPIRED
        if inst.status == InstructionStatus.ACTIVE:
            grace = float(getattr(Config, "INSTRUCTION_OVERDUE_GRACE_SEC", 60))
            if inst.activated_at:
                try:
                    act = datetime.fromisoformat(inst.activated_at)
                    if (now - act).total_seconds() >= grace:
                        return InstructionStatus.EXPIRED
                except (ValueError, TypeError):
                    pass
            return InstructionStatus.ACTIVE

        # PENDING / EXPIRING: deadline_at をチェック
        if inst.deadline_at:
            try:
                dl = datetime.fromisoformat(inst.deadline_at)
                if now >= dl:
                    return InstructionStatus.ACTIVE
                return InstructionStatus.PENDING
            except (ValueError, TypeError):
                # parse 失敗時は deadline 未指定扱い（次の分岐へ）
                pass

        # deadline_at 未指定: default_expire を見る
        default_expire = float(getattr(Config, "ACTIVE_INSTRUCTION_DEFAULT_EXPIRE_SEC", 300))
        if inst.created_at:
            try:
                created = datetime.fromisoformat(inst.created_at)
                elapsed = (now - created).total_seconds()
                if elapsed >= default_expire:
                    return InstructionStatus.EXPIRED
                if elapsed >= default_expire - 60:
                    return InstructionStatus.EXPIRING
            except (ValueError, TypeError):
                pass
        return InstructionStatus.PENDING

    async def _reconcile_instruction_status(self) -> None:
        """全 instruction の derived_status を計算し、状態遷移を確定 + 永続化 + callback 発火。

        遷移パターン:
        - PENDING → ACTIVE: deadline_at 到達 → activated_at セット + 永続化 + callback
        - ACTIVE → EXPIRED: grace 超過 → cleared_reason="overdue_unfulfilled, overdue_by=N秒" + WARN
        - PENDING → EXPIRED: default_expire 超過 → cleared_reason="default_expired"
        - PENDING ↔ EXPIRING: 表示用のみ、永続化しない
        """
        now = datetime.now()
        to_fire: list[dict] = []
        for inst in list(self._active_instructions.values()):
            if inst.status in (
                InstructionStatus.DONE,
                InstructionStatus.EXPIRED,
                InstructionStatus.SUPERSEDED,
                # Fix-7 F1: PROVISIONAL_DONE は audit handler のみが遷移権限を持つ。
                # reconcile は触らない (旧バグ: _compute_derived_status で ACTIVE に
                # フォールスルー → 強制 ACTIVE 復帰ループ発生)。
                InstructionStatus.PROVISIONAL_DONE,
            ):
                continue
            derived = self._compute_derived_status(inst, now)
            old = inst.status

            # Step 1.5 Fix A2-4: ACTIVE のまま再 nudge を要請されているケース。
            # notify_clear_rejected で _activated_callback_fired から id が外されていれば
            # 「ACTIVE のままだが callback fired 状態が外れている」 → 再 callback 発火する。
            if (
                derived == InstructionStatus.ACTIVE
                and old == InstructionStatus.ACTIVE
                and inst.id not in self._activated_callback_fired
            ):
                self._activated_callback_fired.add(inst.id)
                if self._activated_callback is not None:
                    to_fire.append({
                        "id": inst.id,
                        "instruction": inst.instruction,
                        "reason": inst.reason,
                        "deadline_at": inst.deadline_at,
                        "activated_at": inst.activated_at,
                        # Step 1.5 Fix Layer E E6: 前回 audit 失敗の理由を nudge に注入
                        "audit_reason": inst.audit_reason,
                    })
                task_logger.info(
                    "[Instruction] %s re-nudge fired (after clear rejection)",
                    inst.id,
                )
                continue  # この iteration の他の遷移処理はスキップ

            if derived == InstructionStatus.EXPIRING:
                # 表示用ラベル、persistence しない
                continue
            if derived == old:
                continue
            inst.status = derived
            if derived == InstructionStatus.ACTIVE:
                inst.activated_at = now.isoformat()
                await self._write_queue.put({
                    "_kind": "instruction_status",
                    "id": inst.id,
                    "status": derived.value,
                    "activated_at": inst.activated_at,
                    "from": old.value,
                    "at": now.isoformat(),
                })
                task_logger.info(
                    "[Instruction] %s PENDING -> ACTIVE (deadline %s)",
                    inst.id, inst.deadline_at,
                )
                if inst.id not in self._activated_callback_fired:
                    self._activated_callback_fired.add(inst.id)
                    if self._activated_callback is not None:
                        to_fire.append({
                            "id": inst.id,
                            "instruction": inst.instruction,
                            "reason": inst.reason,
                            "deadline_at": inst.deadline_at,
                            "activated_at": inst.activated_at,
                            # Step 1.5 Fix Layer E E6: 通常 PENDING → ACTIVE では audit_reason は None
                            "audit_reason": inst.audit_reason,
                        })
            elif derived == InstructionStatus.EXPIRED:
                if old == InstructionStatus.ACTIVE and inst.deadline_at:
                    try:
                        dl = datetime.fromisoformat(inst.deadline_at)
                        overdue = int((now - dl).total_seconds())
                    except (ValueError, TypeError):
                        overdue = -1
                    cleared_reason = f"overdue_unfulfilled, overdue_by={overdue}s"
                else:
                    cleared_reason = "default_expired"
                inst.cleared_at = now.isoformat()
                inst.cleared_reason = cleared_reason
                self._activated_callback_fired.discard(inst.id)
                await self._write_queue.put({
                    "_kind": "instruction_status",
                    "id": inst.id,
                    "status": derived.value,
                    "cleared_at": inst.cleared_at,
                    "cleared_reason": cleared_reason,
                    "from": old.value,
                    "at": now.isoformat(),
                })
                task_logger.warning(
                    "[Instruction] %s EXPIRED (%s) — 履行失敗シグナル",
                    inst.id, cleared_reason,
                )
        # callback 発火は iteration 後（mutation 防止）
        for payload in to_fire:
            try:
                if self._activated_callback is not None:
                    self._activated_callback(payload)
            except Exception as e:  # noqa: BLE001
                logger.warning("[TaskManager] activation callback error: %s", e)

    async def _handle_set_active_instruction(self, cmd: dict) -> None:
        """set_active_instruction コマンドのハンドラ（command worker 内で呼ばれる）。"""
        instruction = (cmd.get("instruction") or "").strip()
        reason = cmd.get("reason", "")
        deadline_at = cmd.get("deadline_at")
        created_by = cmd.get("created_by", "ai1")

        if not instruction:
            task_logger.warning("[Instruction] set: empty instruction, skip")
            return

        # Bug-D: 相対指定 (delay_seconds) はサーバ側で now+delay を計算して優先する。
        # LLM の絶対時刻計算ミス（「30秒後」→ +58s や過去時刻）をクラスごと排除する。
        # 不正値は無視して absolute fallback（既存の parse 防衛 / B2 クランプはそのまま生きる）。
        delay = cmd.get("delay_seconds")
        if delay is not None:
            try:
                d = float(delay)
                if 0.0 < d <= 86400.0:
                    deadline_at = (datetime.now() + timedelta(seconds=d)).isoformat()
                    task_logger.info(
                        "[Instruction] delay_seconds=%.0f -> deadline_at=%s", d, deadline_at,
                    )
            except (ValueError, TypeError):
                pass

        # deadline_at の parse 防衛
        if deadline_at:
            try:
                datetime.fromisoformat(deadline_at)
            except (ValueError, TypeError) as e:
                task_logger.warning(
                    "[Instruction] deadline_at parse failed: %r (%s) — instruction 登録を拒否",
                    deadline_at, e,
                )
                return

        # Fix-B2: 過去日付の deadline_at を補正する。
        # LLM が日付を誤ると前日 deadline を emit し、登録直後に ACTIVE 化して
        # 督促 nudge が即発火・反復する（tasks.jsonl:63,66 で観測）。
        # 12h 以上過去 = 日付ズレとみなし日数ぶん未来へ送る。救済不能なら None にして
        # 既定 expire（ACTIVE_INSTRUCTION_DEFAULT_EXPIRE_SEC）に委ね、即 ACTIVE 爆発を防ぐ。
        if deadline_at:
            try:
                dl = datetime.fromisoformat(deadline_at)
                created = datetime.now()
                if dl <= created:
                    gap = (created - dl).total_seconds()
                    if gap >= 43200:  # 12h
                        corrected = dl + timedelta(days=round(gap / 86400))
                        if corrected > created:
                            task_logger.warning(
                                "[Instruction] deadline_at が %.0fs 過去 — 日付誤りとみなし補正: %s -> %s",
                                gap, deadline_at, corrected.isoformat(),
                            )
                            deadline_at = corrected.isoformat()
                    if datetime.fromisoformat(deadline_at) <= created:
                        task_logger.warning(
                            "[Instruction] deadline_at が補正後も過去 (%s) — 既定 expire に委ねる",
                            deadline_at,
                        )
                        deadline_at = None
            except (ValueError, TypeError):
                pass

        # Dedup
        norm_new = self._normalize_instruction(instruction)
        for existing in self._active_instructions.values():
            if existing.status in (
                InstructionStatus.DONE,
                InstructionStatus.EXPIRED,
                InstructionStatus.SUPERSEDED,
            ):
                continue
            norm_old = self._normalize_instruction(existing.instruction)
            ratio = difflib.SequenceMatcher(None, norm_new, norm_old).ratio()
            if ratio >= 0.85 and self._deadline_similar(deadline_at, existing.deadline_at):
                task_logger.info(
                    "[Instruction] duplicate registration, reusing existing id=%s (ratio=%.2f)",
                    existing.id, ratio,
                )
                return

        # 新規登録
        inst = ActiveInstruction(
            id=new_instruction_id(),
            instruction=instruction,
            created_at=now_iso(),
            deadline_at=deadline_at,
            status=InstructionStatus.PENDING,
            created_by=created_by,
            reason=reason,
        )
        self._active_instructions[inst.id] = inst
        await self._write_queue.put({
            "_kind": "active_instruction",
            **inst.to_jsonable(),
        })
        task_logger.info(
            "[Instruction] %s registered (deadline=%s): %s",
            inst.id, deadline_at or "(default 5min)", instruction[:40],
        )

    async def _handle_clear_active_instruction(self, cmd: dict) -> None:
        """clear_active_instruction コマンドのハンドラ。

        Step 1.5 Fix Layer E: status=done で `eve_response` 同梱なら、
        即 DONE 化せず PROVISIONAL_DONE に遷移して AI2 audit を待つ。
        eve_response 無し（audit 無効環境の fallback）は従来通り即 DONE。
        status=superseded は発話不要で即時クリア（変更なし）。
        """
        iid = cmd.get("id", "")
        status_str = cmd.get("status", "done")
        reason = cmd.get("reason", "")
        eve_response = cmd.get("eve_response")  # Layer E: A2 通過後に LLMHandler が同梱
        inst = self._active_instructions.get(iid)
        if inst is None:
            task_logger.warning("[Instruction] clear: unknown id=%s", iid)
            return

        # Fix-3.5 F3: 終了状態 + PROVISIONAL_DONE への二重 clear を蹴る。
        # 状態遷移は単調進行（terminal state は不可逆）。 PROVISIONAL_DONE も含めるのは
        # 二重 AI2 audit 発火防止のため。 既存 _handle_notify_clear_rejected /
        # _handle_confirm_provisional_done / _handle_revert_provisional_done と同型 guard。
        if inst.status in (
            InstructionStatus.DONE,
            InstructionStatus.EXPIRED,
            InstructionStatus.SUPERSEDED,
            InstructionStatus.PROVISIONAL_DONE,
        ):
            task_logger.warning(
                "[Instruction] %s clear ignored (already %s, illegal re-clear): requested=%s, reason=%r",
                iid, inst.status.value, status_str, (reason or "")[:60],
            )
            return

        try:
            new_status = InstructionStatus(status_str)
        except ValueError:
            new_status = InstructionStatus.DONE
        # AI1 経由の clear は DONE / SUPERSEDED のみ受理（EXPIRED は内部判定のみ）
        if new_status not in (InstructionStatus.DONE, InstructionStatus.SUPERSEDED):
            new_status = InstructionStatus.DONE

        # Step 1.5 Fix Layer E E0: status=done + eve_response → PROVISIONAL_DONE で audit 待ち
        if new_status == InstructionStatus.DONE and eve_response:
            inst.status = InstructionStatus.PROVISIONAL_DONE
            inst.provisional_response = eve_response
            inst.audit_pending_at = now_iso()
            await self._write_queue.put({
                "_kind": "instruction_status",
                "id": iid,
                "status": InstructionStatus.PROVISIONAL_DONE.value,
                "audit_pending_at": inst.audit_pending_at,
                "reason": reason,
                "at": now_iso(),
            })
            task_logger.info(
                "[Instruction] %s -> PROVISIONAL_DONE (awaiting AI2 audit): %s",
                iid, (eve_response or "")[:60],
            )
            return

        # eve_response 無し（audit 無効環境）or status=superseded → 従来通り即時クリア
        inst.status = new_status
        # Fix-9b: ユーザーが予約を明示取消（superseded）したら、紐づくコミット事実も superseded に。
        # DONE（履行完了）では eve-fact を active のまま残す（ペルソナ持続 = 一度言った好みは消えない）。
        if new_status == InstructionStatus.SUPERSEDED:
            self._release_facts_for_instruction(iid, "superseded")
        inst.cleared_at = now_iso()
        inst.cleared_reason = reason
        self._activated_callback_fired.discard(iid)
        await self._write_queue.put({
            "_kind": "instruction_status",
            "id": iid,
            "status": new_status.value,
            "cleared_at": inst.cleared_at,
            "cleared_reason": reason,
            "at": now_iso(),
        })
        task_logger.info(
            "[Instruction] %s cleared as %s: %s",
            iid, new_status.value, reason[:40] if reason else "",
        )

    async def _handle_notify_clear_rejected(self, cmd: dict) -> None:
        """Step 1.5 Fix A2-4: LLM が空発話で done を呼んだのを A2 ゲートが弾いた時の処理。

        - `_activated_callback_fired` から id を外す → 次の reconcile で再 callback 発火
        - `activated_at` をリセット（grace 期間を再カウント、 即時 EXPIRED を防ぐ）
        - `rejection_count` を increment、 3 回累積で強制 EXPIRED 化（無限ループ防止）
        """
        iid = cmd.get("id", "")
        reason = cmd.get("reason", "")
        inst = self._active_instructions.get(iid)
        if inst is None:
            task_logger.warning("[Instruction] notify_clear_rejected: unknown id=%s", iid)
            return
        # 既に終端状態なら何もしない
        if inst.status in (
            InstructionStatus.DONE,
            InstructionStatus.EXPIRED,
            InstructionStatus.SUPERSEDED,
        ):
            task_logger.debug(
                "[Instruction] %s notify_clear_rejected ignored (status=%s)",
                iid, inst.status.value,
            )
            return

        inst.rejection_count += 1
        now = datetime.now()

        # 3 回累積で強制 EXPIRED 化
        if inst.rejection_count >= 3:
            inst.status = InstructionStatus.EXPIRED
            inst.cleared_at = now.isoformat()
            inst.cleared_reason = f"rejected_3_times: {reason[:60]}"
            self._activated_callback_fired.discard(iid)
            await self._write_queue.put({
                "_kind": "instruction_status",
                "id": iid,
                "status": InstructionStatus.EXPIRED.value,
                "cleared_at": inst.cleared_at,
                "cleared_reason": inst.cleared_reason,
                "from": "rejection_count>=3",
                "at": now.isoformat(),
            })
            task_logger.warning(
                "[Instruction] %s force-EXPIRED after 3 rejections: %s",
                iid, reason[:60],
            )
            return

        # 再 nudge 準備: callback fired を外し、 activated_at をリセット
        self._activated_callback_fired.discard(iid)
        if inst.status == InstructionStatus.ACTIVE:
            # ACTIVE 中の reject → grace 期間を再カウント
            inst.activated_at = now.isoformat()
        await self._write_queue.put({
            "_kind": "instruction_rejected",
            "id": iid,
            "rejection_count": inst.rejection_count,
            "reason": reason,
            "activated_at": inst.activated_at,
            "at": now.isoformat(),
        })
        task_logger.info(
            "[Instruction] %s clear rejected (count=%d): %s — 次 reconcile で再 nudge",
            iid, inst.rejection_count, reason[:60],
        )

    async def _handle_confirm_provisional_done(self, cmd: dict) -> None:
        """Step 1.5 Fix Layer E E5: AI2 audit が fulfilled=true を返した時のハンドラ。
        PROVISIONAL_DONE → DONE 確定。"""
        iid = cmd.get("id", "")
        reason = cmd.get("reason", "audit confirmed")
        inst = self._active_instructions.get(iid)
        if inst is None:
            task_logger.warning("[Audit] confirm: unknown id=%s", iid)
            return
        # race: 既に grace 超過で EXPIRED 化、 or 別経路で SUPERSEDED など
        if inst.status != InstructionStatus.PROVISIONAL_DONE:
            task_logger.info(
                "[Audit] %s confirm ignored (status=%s, late result)",
                iid, inst.status.value,
            )
            return
        inst.status = InstructionStatus.DONE
        inst.cleared_at = now_iso()
        inst.cleared_reason = reason
        inst.audit_pending_at = None
        self._activated_callback_fired.discard(iid)
        self._audit_spawned_ids.discard(iid)
        await self._write_queue.put({
            "_kind": "instruction_status",
            "id": iid,
            "status": InstructionStatus.DONE.value,
            "cleared_at": inst.cleared_at,
            "cleared_reason": reason,
            "source": "ai2_audit",
            "at": now_iso(),
        })
        task_logger.info(
            "[Audit] %s confirmed DONE: %s",
            iid, (reason or "")[:60],
        )

    async def _handle_revert_provisional_done(self, cmd: dict) -> None:
        """Fix-6 P2-c: AI2 audit が fulfilled=false を返した時のハンドラ。

        **【Fix-6 で advisory 化】** 旧設計 (Fix-3) の ACTIVE 復帰 + 連続 3 回強制 EXPIRED は **撤廃**。
        新方針: PROVISIONAL_DONE → **DONE 強制遷移** + WARN ログ + 反省記録のみ。
        AI2 audit は state を戻す権限を持たない (advisory)。
        """
        iid = cmd.get("id", "")
        ai2_reason = cmd.get("ai2_reason", "未達成")
        inst = self._active_instructions.get(iid)
        if inst is None:
            task_logger.warning("[Audit advisory] revert: unknown id=%s", iid)
            return
        # race: 既に終端状態 → 結果を無視
        if inst.status != InstructionStatus.PROVISIONAL_DONE:
            task_logger.info(
                "[Audit advisory] %s ignored (status=%s, late result)",
                iid, inst.status.value,
            )
            return

        # Fix-6 P2-c: rejection_count は反省データとして累積するが state 遷移には使わない
        inst.rejection_count += 1
        now = datetime.now()

        # Fix-6 P2-c: revert 廃止、 DONE 強制遷移 + advisory WARN ログ
        inst.status = InstructionStatus.DONE
        inst.cleared_at = now.isoformat()
        inst.cleared_reason = f"audit_advisory: kept as DONE despite fulfilled=false ({ai2_reason[:60]})"
        inst.audit_reason = ai2_reason  # 反省データとして保持
        inst.audit_pending_at = None
        self._activated_callback_fired.discard(iid)
        self._audit_spawned_ids.discard(iid)
        await self._write_queue.put({
            "_kind": "instruction_status",
            "id": iid,
            "status": InstructionStatus.DONE.value,
            "cleared_at": inst.cleared_at,
            "cleared_reason": inst.cleared_reason,
            "from": "provisional_done",
            "audit_reason": ai2_reason,
            "rejection_count": inst.rejection_count,
            "source": "ai2_audit_advisory",
            "at": now.isoformat(),
        })
        task_logger.warning(
            "[Audit advisory] %s fulfilled=false but kept as DONE: %s (rejection_count=%d, reflection data only)",
            iid, ai2_reason[:60], inst.rejection_count,
        )

    @staticmethod
    def _normalize_instruction(s: str) -> str:
        """dedup 用の正規化: 末尾句読点除去 + 連続空白圧縮。"""
        s = (s or "").strip()
        s = s.rstrip("。.！!？?")
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _deadline_similar(a: Optional[str], b: Optional[str]) -> bool:
        """deadline_at が類似（10 秒以内）なら True。両方 None でも True（dedup 対象）。"""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            da = datetime.fromisoformat(a)
            db = datetime.fromisoformat(b)
            return abs((da - db).total_seconds()) <= 10
        except (ValueError, TypeError):
            return False
