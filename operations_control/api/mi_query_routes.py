"""operations_control.api.mi_query_routes — reviewing live MI Query usage.

The Day-1 calibration surface: who asked, what they asked, what Trakt
understood, which capability ran, what the user saw, whether it answered,
refused or errored — and, after review, whether the response was any good.

Deliberately NOT the OCC system dashboard. This module reads one governed
record type and lets an operator classify it; it knows nothing about runs,
gates, publications or service health, so it can later become one module of the
wider operations console without carrying anything back out of it.

Every route is client-scoped through the same ``require_client`` guard that
governs the rest of the OCC API: an operator sees the telemetry of the clients
they are entitled to and no others.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel

from ..mi_query_telemetry import (
    ANSWERED,
    ERROR,
    PROBLEMATIC,
    REFUSED,
    REVIEW_CLASSIFICATIONS,
    UNREVIEWED,
)

from .auth import Principal, authenticate, require_client

#: The windows an operator actually asks for on Day 1.
WINDOWS = {"24h": 24, "72h": 72, "7d": 168}


def _days_for(window: str) -> Optional[List[str]]:
    """The UTC day partitions a window touches, or None for 'everything'."""
    hours = WINDOWS.get(window)
    if hours is None:
        return None
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    days, cursor = [], start.date()
    while cursor <= now.date():
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def _within(rec: Dict[str, Any], window: str) -> bool:
    hours = WINDOWS.get(window)
    if hours is None:
        return True
    try:
        asked = datetime.fromisoformat(str(rec.get("asked_at")).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001 — an unparseable stamp is not filtered out
        return True
    if asked.tzinfo is None:
        asked = asked.replace(tzinfo=timezone.utc)
    return asked >= datetime.now(timezone.utc) - timedelta(hours=hours)


def _load(eng, principal: Principal, client: Optional[str],
          window: str) -> List[Dict[str, Any]]:
    clients = ([client] if client
               else principal.visible_clients(eng.store.known_clients()))
    for c in clients:
        require_client(principal, c)
    days = _days_for(window)
    out: List[Dict[str, Any]] = []
    for c in clients:
        out.extend(r for r in eng.store.list_mi_queries(c, days=days)
                   if _within(r, window))
    return sorted(out, key=lambda r: r.get("asked_at", ""), reverse=True)


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counters for the selected window.

    The reviewed correctness rate is computed over REVIEWED responses only, and
    is reported beside its denominator — a rate over a subset presented as an
    accuracy figure would be a claim the evidence does not support.
    """
    reviewed = [r for r in rows
                if (r.get("review") or {}).get("classification") not in
                (None, UNREVIEWED)]
    correct = [r for r in reviewed
               if (r["review"]["classification"] in ("CORRECT",
                                                     "APPROPRIATE_REFUSAL"))]
    problematic = [r for r in reviewed
                   if r["review"]["classification"] in PROBLEMATIC]
    latencies = sorted(int(r["duration_ms"]) for r in rows
                       if isinstance(r.get("duration_ms"), (int, float)))
    by_class: Dict[str, int] = {}
    for r in reviewed:
        c = r["review"]["classification"]
        by_class[c] = by_class.get(c, 0) + 1

    def pct(n: int, d: int) -> Optional[float]:
        return round(100.0 * n / d, 1) if d else None

    return {
        "total_questions": len(rows),
        "unique_users": len({r.get("user_id") for r in rows if r.get("user_id")}),
        "answered": sum(1 for r in rows if r.get("outcome") == ANSWERED),
        "refused": sum(1 for r in rows if r.get("outcome") == REFUSED),
        "errors": sum(1 for r in rows if r.get("outcome") == ERROR),
        "answered_pct": pct(sum(1 for r in rows if r.get("outcome") == ANSWERED),
                            len(rows)),
        "refused_pct": pct(sum(1 for r in rows if r.get("outcome") == REFUSED),
                           len(rows)),
        "error_pct": pct(sum(1 for r in rows if r.get("outcome") == ERROR),
                         len(rows)),
        "unreviewed": len(rows) - len(reviewed),
        "reviewed": len(reviewed),
        "reviewed_correct": len(correct),
        "reviewed_problematic": len(problematic),
        # Over the reviewed subset ONLY, and the denominator travels with it.
        "reviewed_correctness_pct": pct(len(correct), len(reviewed)),
        "review_breakdown": by_class,
        "median_latency_ms": (int(statistics.median(latencies))
                              if latencies else None),
        "p95_latency_ms": (latencies[min(len(latencies) - 1,
                                         int(0.95 * len(latencies)))]
                           if latencies else None),
    }


def _row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The query-log row — enough to scan, not the whole record."""
    return {
        "query_id": rec.get("query_id"),
        "asked_at": rec.get("asked_at"),
        "client_id": rec.get("client_id"),
        "portfolio_id": rec.get("portfolio_id"),
        "user_id": rec.get("user_id"),
        "channel": rec.get("channel"),
        "question": rec.get("question"),
        "outcome": rec.get("outcome"),
        "route": rec.get("route"),
        "refusal_reason": rec.get("refusal_reason"),
        "error_code": rec.get("error_code"),
        "duration_ms": rec.get("duration_ms"),
        "review": (rec.get("review") or {}).get("classification", UNREVIEWED),
    }


class ReviewBody(BaseModel):
    classification: str
    note: Optional[str] = None
    client: Optional[str] = None


def register(app, get_engine) -> None:
    """Mount the MI Query telemetry routes on the OCC API."""

    @app.get("/ops/mi-queries/summary")
    def mi_query_summary(window: str = "72h", client: Optional[str] = None,
                         principal: Principal = Depends(authenticate)
                         ) -> Dict[str, Any]:
        rows = _load(get_engine(), principal, client, window)
        return {"ok": True, "window": window, "summary": _summary(rows)}

    @app.get("/ops/mi-queries")
    def mi_query_log(window: str = "72h", client: Optional[str] = None,
                     outcome: Optional[str] = None, user: Optional[str] = None,
                     portfolio: Optional[str] = None,
                     review: Optional[str] = None, q: Optional[str] = None,
                     limit: int = 200,
                     principal: Principal = Depends(authenticate)
                     ) -> Dict[str, Any]:
        rows = _load(get_engine(), principal, client, window)
        if outcome:
            rows = [r for r in rows if r.get("outcome") == outcome.upper()]
        if user:
            rows = [r for r in rows if r.get("user_id") == user]
        if portfolio:
            rows = [r for r in rows if r.get("portfolio_id") == portfolio]
        if review:
            wanted = review.upper()
            if wanted == "PROBLEMATIC":
                rows = [r for r in rows
                        if (r.get("review") or {}).get("classification")
                        in PROBLEMATIC]
            else:
                rows = [r for r in rows
                        if (r.get("review") or {}).get("classification",
                                                       UNREVIEWED) == wanted]
        if q:
            needle = q.lower()
            rows = [r for r in rows
                    if needle in str(r.get("question", "")).lower()]
        return {"ok": True, "window": window, "count": len(rows),
                "queries": [_row(r) for r in rows[:limit]]}

    @app.get("/ops/mi-queries/{query_id}")
    def mi_query_detail(query_id: str, client: Optional[str] = None,
                        principal: Principal = Depends(authenticate)
                        ) -> Dict[str, Any]:
        eng = get_engine()
        clients = ([client] if client
                   else principal.visible_clients(eng.store.known_clients()))
        for c in clients:
            require_client(principal, c)
            rec = eng.store.load_mi_query(c, query_id)
            if rec:
                return {"ok": True, "query": rec}
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})

    @app.post("/ops/mi-queries/{query_id}/review")
    def mi_query_review(query_id: str, body: ReviewBody,
                        principal: Principal = Depends(authenticate)
                        ) -> Dict[str, Any]:
        """Record an operator's judgement of a response.

        Calibration evidence only: it never touches the answer the client was
        given, which has already been served and is immutable here.
        """
        classification = (body.classification or "").upper()
        if classification not in REVIEW_CLASSIFICATIONS:
            raise HTTPException(status_code=400, detail={
                "errorCode": "OPS_BAD_CLASSIFICATION",
                "message": "Choose one of: "
                           + ", ".join(REVIEW_CLASSIFICATIONS) + "."})
        eng = get_engine()
        clients = ([body.client] if body.client
                   else principal.visible_clients(eng.store.known_clients()))
        for c in clients:
            require_client(principal, c)
            rec = eng.store.load_mi_query(c, query_id)
            if not rec:
                continue
            rec["review"] = {
                "classification": classification,
                "reviewer": principal.name,
                "reviewed_at": datetime.now(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "note": (body.note or "").strip() or None}
            eng.store.save_mi_query(rec)
            eng.store.append_audit(
                c, "mi_query_reviewed", actor=principal.name,
                detail={"query_id": query_id, "classification": classification})
            return {"ok": True, "query_id": query_id, "review": rec["review"]}
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})

    @app.get("/ops/mi-queries/export/calibration")
    def mi_query_export(window: str = "72h", client: Optional[str] = None,
                        reviewed_only: bool = True,
                        principal: Principal = Depends(authenticate)
                        ) -> Dict[str, Any]:
        """The EXTERNAL-MODEL-SAFE calibration export.

        Carries the question, what Trakt understood, which capability ran, what
        happened and the human verdict — and NO portfolio content. The answer
        text, the artefacts, the snapshot content hash and every figure the
        engine computed are excluded by construction: this export exists to
        improve recognition and routing, and none of that requires a client's
        numbers to leave the governed environment.

        The governed record still holds the answer for operator review inside
        OCC; it is this export, and only this export, that is safe to hand to an
        external model.
        """
        rows = _load(get_engine(), principal, client, window)
        if reviewed_only:
            rows = [r for r in rows
                    if (r.get("review") or {}).get("classification")
                    not in (None, UNREVIEWED)]
        return {
            "ok": True, "window": window, "count": len(rows),
            "export_kind": "external_model_safe",
            "excludes": ["answer", "artifacts", "aggregate values",
                         "deterministic payload values", "loan rows",
                         "portfolio values", "content_hash"],
            "queries": [{
                "query_id": r.get("query_id"),
                "asked_at": r.get("asked_at"),
                "question": r.get("question"),
                "interpretation": r.get("interpretation") or {},
                "parser": r.get("parser") or {},
                "route": r.get("route"),
                "capability": r.get("capability"),
                "outcome": r.get("outcome"),
                "refusal_reason": r.get("refusal_reason"),
                "error_code": r.get("error_code"),
                "dataset_view": r.get("dataset_view"),
                "reporting_period": r.get("reporting_period"),
                "snapshot_id": r.get("snapshot_id"),
                "quality_classification":
                    (r.get("review") or {}).get("classification", UNREVIEWED),
                "reviewer_note": (r.get("review") or {}).get("note"),
            } for r in rows],
        }
