"""trakt_a2a.tasks — A2A task state, as the specification defines it.

A readiness assessment takes minutes and produces a structured deliverable, so
it is a **Task**, not a message-and-reply. This module holds the state machine
and the store, and nothing else — no execution, no authorisation, no Trakt.

Task states are the A2A v1.0 set. We use the four a synchronous specialist can
legitimately reach, plus ``rejected`` for a request outside the advertised
skill and ``auth-required`` for one we cannot authorise. States we do not
implement are absent rather than stubbed: ``input-required`` would need a
clarification round-trip the specialist never asks for, and pretending to
support it would advertise a lifecycle a caller could wait on forever.

Idempotency
-----------
A task is keyed by the caller's own ``taskId`` when it supplies one. Re-sending
the same request returns the existing task rather than starting a second
assessment, because an assessment costs real money and a retried HTTP request is
not a second instruction. This is deliberately *not* distributed exactly-once
delivery — it is a store keyed by an identifier the caller controls, which is
what a retry actually needs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: A2A v1.0 TaskState values, in their JSON spelling.
SUBMITTED = "submitted"
WORKING = "working"
COMPLETED = "completed"
FAILED = "failed"
CANCELED = "canceled"
REJECTED = "rejected"
AUTH_REQUIRED = "auth-required"

TERMINAL_STATES = frozenset({COMPLETED, FAILED, CANCELED, REJECTED})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Task:
    """One delegated unit of work, in the shape A2A serialises."""

    id: str
    context_id: str
    state: str = SUBMITTED
    #: Every transition, so a caller that polled late can still see the path
    #: taken. The card advertises ``stateTransitionHistory``; this is it.
    history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append({"state": self.state, "timestamp": _now()})

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def transition(self, state: str, *, message: Optional[str] = None) -> None:
        """Move to a new state, recording when and why.

        Refuses to leave a terminal state. A task that reported ``completed``
        and later reports ``failed`` would make every consumer's record of it
        wrong, and there is no legitimate reason for a synchronous specialist to
        need it.
        """
        if self.is_terminal:
            raise ValueError(
                f"task {self.id} is already {self.state}; a terminal task "
                "cannot transition again")
        self.state = state
        entry: Dict[str, Any] = {"state": state, "timestamp": _now()}
        if message:
            entry["message"] = message
        self.history.append(entry)

    def add_artifact(self, artifact: Dict[str, Any]) -> None:
        self.artifacts.append(artifact)

    def to_dict(self) -> Dict[str, Any]:
        """The A2A wire shape — camelCase, as the JSON binding requires."""
        payload: Dict[str, Any] = {
            "id": self.id,
            "contextId": self.context_id,
            "status": {"state": self.state,
                       "timestamp": self.history[-1]["timestamp"]},
            "artifacts": list(self.artifacts),
            "history": list(self.messages),
            "metadata": dict(self.metadata),
        }
        if self.error:
            payload["status"]["message"] = self.error
        return payload


class TaskStore:
    """An in-process task store, safe for concurrent access.

    Not a database. A production deployment needs durability across restarts,
    and this does not provide it — which is stated here rather than discovered
    later, and listed in the report's production-readiness section.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._by_context: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def create(self, task_id: str, context_id: str,
               metadata: Optional[Dict[str, Any]] = None) -> Task:
        with self._lock:
            task = Task(id=task_id, context_id=context_id,
                        metadata=dict(metadata or {}))
            self._tasks[task_id] = task
            self._by_context.setdefault(context_id, []).append(task_id)
            return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def for_context(self, context_id: str) -> List[Task]:
        """Every task in a conversation, oldest first.

        This is what makes a follow-up cheap: the specialist answers from the
        assessment already produced in this context rather than re-running the
        portfolio.
        """
        with self._lock:
            return [self._tasks[t] for t in self._by_context.get(context_id, [])
                    if t in self._tasks]

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)
