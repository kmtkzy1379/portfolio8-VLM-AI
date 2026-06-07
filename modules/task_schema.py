"""Task / Plan dataclass 定義。

Plan-and-Execute タスク管理機構の型定義のみ。状態管理ロジックは task_manager.py に持つ。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    CRITICAL = "critical"  # ユーザーの明示指示・約束
    HIGH = "high"          # 現在の goal_short 直結
    NORMAL = "normal"      # goal_long サブステップ
    LOW = "low"            # 雑談・繋ぎ


_PRIORITY_RANK = {
    TaskPriority.CRITICAL: 0,
    TaskPriority.HIGH: 1,
    TaskPriority.NORMAL: 2,
    TaskPriority.LOW: 3,
}


def priority_rank(p: TaskPriority) -> int:
    """ソート用の優先度ランク (小さいほど高優先)。"""
    return _PRIORITY_RANK.get(p, 99)


def new_task_id() -> str:
    """uuid4 の先頭 8 hex を返す。"""
    return "t_" + uuid.uuid4().hex[:8]


def new_plan_id() -> str:
    return "p_" + uuid.uuid4().hex[:8]


def now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class Task:
    id: str
    description: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    parent_goal_short: str = ""
    validation_score: Optional[float] = None
    failure_reason: Optional[str] = None
    # user_action sticky ルール用: 最後に状態を確定したのが誰か
    last_authoritative_kind: Optional[str] = None  # "user_action" | "validator" | None

    def to_jsonable(self) -> dict:
        d = asdict(self)
        # enum を value に展開
        d["priority"] = self.priority.value if isinstance(self.priority, TaskPriority) else self.priority
        d["status"] = self.status.value if isinstance(self.status, TaskStatus) else self.status
        return d

    @classmethod
    def from_jsonable(cls, d: dict) -> "Task":
        prio = d.get("priority", TaskPriority.NORMAL.value)
        stat = d.get("status", TaskStatus.PENDING.value)
        return cls(
            id=d["id"],
            description=d.get("description", ""),
            priority=TaskPriority(prio) if not isinstance(prio, TaskPriority) else prio,
            status=TaskStatus(stat) if not isinstance(stat, TaskStatus) else stat,
            depends_on=list(d.get("depends_on", [])),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            parent_goal_short=d.get("parent_goal_short", ""),
            validation_score=d.get("validation_score"),
            failure_reason=d.get("failure_reason"),
            last_authoritative_kind=d.get("last_authoritative_kind"),
        )


@dataclass
class Plan:
    plan_id: str
    goal_short_snapshot: str
    goal_long_snapshot: str
    tasks: list[Task]
    created_at: str
    superseded_by: Optional[str] = None

    def to_jsonable(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "goal_short_snapshot": self.goal_short_snapshot,
            "goal_long_snapshot": self.goal_long_snapshot,
            "tasks": [t.to_jsonable() for t in self.tasks],
            "created_at": self.created_at,
            "superseded_by": self.superseded_by,
        }


class InstructionStatus(str, Enum):
    PENDING = "pending"
    EXPIRING = "expiring"
    ACTIVE = "active"
    PROVISIONAL_DONE = "provisional_done"  # Step 1.5 Fix Layer E: A2 通過後、 AI2 audit 待ち
    DONE = "done"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


def new_instruction_id() -> str:
    return "i_" + uuid.uuid4().hex[:8]


@dataclass
class ActiveInstruction:
    id: str
    instruction: str
    created_at: str
    deadline_at: Optional[str] = None
    activated_at: Optional[str] = None
    status: InstructionStatus = InstructionStatus.PENDING
    created_by: str = "ai1"
    reason: str = ""
    cleared_at: Optional[str] = None
    cleared_reason: Optional[str] = None
    # Step 1.5 Fix A2-4: 連続却下カウンター（syntactic + semantic 共通）。
    # notify_clear_rejected を受けるたびに increment、 3 で強制 EXPIRED 化。
    rejection_count: int = 0
    # Step 1.5 Fix Layer E: AI2 audit 関連フィールド（PROVISIONAL_DONE 時のみ意味を持つ）。
    audit_pending_at: Optional[str] = None     # PROVISIONAL_DONE 化した時刻
    audit_reason: Optional[str] = None         # AI2 が返した未達成理由（次 nudge に注入）
    provisional_response: Optional[str] = None  # PROVISIONAL_DONE 化時の Eve 応答 snapshot

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, InstructionStatus) else self.status
        return d

    @classmethod
    def from_jsonable(cls, d: dict) -> "ActiveInstruction":
        stat = d.get("status", InstructionStatus.PENDING.value)
        return cls(
            id=d["id"],
            instruction=d.get("instruction", ""),
            created_at=d.get("created_at", ""),
            deadline_at=d.get("deadline_at"),
            activated_at=d.get("activated_at"),
            status=InstructionStatus(stat) if not isinstance(stat, InstructionStatus) else stat,
            created_by=d.get("created_by", "ai1"),
            reason=d.get("reason", ""),
            cleared_at=d.get("cleared_at"),
            cleared_reason=d.get("cleared_reason"),
            rejection_count=int(d.get("rejection_count", 0)),
            audit_pending_at=d.get("audit_pending_at"),
            audit_reason=d.get("audit_reason"),
            provisional_response=d.get("provisional_response"),
        )
