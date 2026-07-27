"""mi_agent_api/presenters.py — governed result → channel payload.

The only place that knows what a channel's wire format looks like. A presenter
re-keys and formats; it never calculates, never re-resolves a dataset, never
re-checks a policy and never decides what the answer is.

Backward compatibility is the design constraint here. React's payload is the
capability's analytical dictionary returned **verbatim**, with one additive
``governance`` key. Nothing was renamed, reordered or removed, so the existing
React client, its tests and the React↔Copilot parity suite are unaffected.
"""

from __future__ import annotations

from typing import Any, Dict

from trakt_core.envelope import GovernedResult


def to_react_payload(result: GovernedResult[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``POST /mi/query`` body.

    The pre-existing analytical envelope plus an additive ``governance`` block
    (capability id, schema version, status, request id, snapshot identity, policy
    state, typed error). Existing consumers that ignore unknown keys are
    unaffected; new consumers get machine-readable governance without a second
    round trip.
    """
    payload: Dict[str, Any] = dict(result.result or {})
    payload["governance"] = result.governance_block()

    # Surface the snapshot the answer came from in the place React already reads
    # dataset metadata, so provenance is visible without parsing the new block.
    meta = payload.get("metadata")
    if isinstance(meta, dict) and result.snapshot is not None:
        meta.setdefault("snapshotId", result.snapshot.snapshot_id)
        meta.setdefault("contentHash", result.snapshot.content_hash)
        meta.setdefault("requestId", result.request_id)
    return payload


def to_copilot_error(result: GovernedResult[Dict[str, Any]]) -> Dict[str, Any]:
    """The Copilot body for a governed refusal or failure.

    Keeps the ``{"ok": false, "error": "<message>"}`` shape the Copilot action
    contract already documents, and adds the machine-readable code alongside.
    """
    body: Dict[str, Any] = {
        "ok": False,
        "error": (result.error.message if result.error
                  else "Trakt could not answer this question."),
    }
    if result.error is not None:
        body["errorCode"] = result.error.code
        body["retryable"] = result.error.retryable
        body["category"] = result.error.category
    body["requestId"] = result.request_id
    return body
