#!/usr/bin/env python3
"""Enumerate global/asset-scoped governed rules and where they came from.

READ ONLY. This script opens storage, reads, and writes nothing — not a file,
not a blob, not an audit event. Run it against the LIVE store BEFORE deleting
any client, because the evidence it reports is destroyed by that deletion.

WHY THIS IS NEEDED
------------------
A governed rule approved at ``global`` or ``asset`` scope outlives the client
whose delivery produced it. That is the point of those scopes — a decision about
what a column MEANS for an asset class is not one client's property. But it
means deleting a client can silently orphan rules that client originated, and
leave rules in force that nobody can now explain.

``RuleRecord.client_id`` is ``""`` for global and asset scope BY DESIGN, so
origin cannot be read off the rule. It has to be reconstructed:

    operations-control/_global/rules/<rule_id>/current.json   the rule
    operations-control/<client>/audit/*.json                  rule_persisted

``rule_persisted`` audit events are appended under the client whose workflow
approved the rule, and carry ``rule_id``. Joining the two recovers the origin
the rule itself does not record. A rule with no matching audit event is
reported as UNATTRIBUTED rather than guessed at.

USAGE
-----
Against the live Azure store, with the same settings the OCC API uses::

    export TRAKT_STORAGE_BACKEND=blob
    export TRAKT_BLOB_CONNECTION='<the ops storage connection string>'
    export TRAKT_OPS_CONTAINER=operations-control      # if yours differs
    python scripts/enumerate_global_rules.py --client ERE

    # machine-readable, to keep as pre-wipe evidence:
    python scripts/enumerate_global_rules.py --client ERE --json > global_rules_pre_wipe.json

Against a local file-backed store::

    export TRAKT_STORAGE_BACKEND=file
    export TRAKT_LOCAL_BLOB_ROOT=/path/to/blob
    python scripts/enumerate_global_rules.py --client ERE

EXIT CODES
----------
0  enumeration completed (whatever it found)
2  the store could not be reached — the result is NOT "no rules"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UNATTRIBUTED = "UNATTRIBUTED"


def _load_store():
    from apps.blob_trigger_app.storage import open_storage
    from operations_control.stores import OpsLayout, OpsStore
    return OpsStore(open_storage(), OpsLayout.from_env())


def _read_json(storage, uri: str) -> Optional[Dict[str, Any]]:
    try:
        if not storage.exists(uri):
            return None
        return json.loads(storage.read_text(uri))
    except Exception:  # noqa: BLE001 — one unreadable rule must not stop the sweep
        return None


def global_rules(store) -> List[Dict[str, Any]]:
    """Every rule at global/asset scope, current version only."""
    prefix = store.layout.rules_prefix(None)
    out: List[Dict[str, Any]] = []
    for uri in store.storage.list(prefix):
        if not uri.endswith("/current.json"):
            continue
        doc = _read_json(store.storage, uri)
        if doc:
            doc["_uri"] = uri
            out.append(doc)
    return out


def rule_origins(store, clients: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """``{rule_id: [{client, workflow_id, at, actor}, ...]}`` from the audit.

    The audit is the only place the link survives: ``rule_persisted`` is
    appended under the client whose workflow approved the rule.
    """
    origins: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for client in clients:
        try:
            events = store.list_audit(client)
        except Exception:  # noqa: BLE001
            continue
        for event in events:
            if event.get("action") != "rule_persisted":
                continue
            rule_id = str(event.get("rule_id") or "")
            if not rule_id:
                continue
            origins[rule_id].append({
                "client": client,
                "workflow_id": event.get("workflow_id") or "",
                "decision_id": event.get("decision_id") or "",
                "at": event.get("at") or event.get("timestamp") or "",
                "actor": event.get("actor") or "",
                "detail": event.get("detail") or {},
            })
    return dict(origins)


def enumerate_rules(store, focus_client: str = "") -> Dict[str, Any]:
    clients = store.known_clients()
    rules = global_rules(store)
    origins = rule_origins(store, clients)

    rows: List[Dict[str, Any]] = []
    for rule in rules:
        rule_id = str(rule.get("rule_id") or "")
        found = origins.get(rule_id, [])
        attributed = sorted({o["client"] for o in found})
        rows.append({
            "rule_id": rule_id,
            "version": rule.get("version"),
            "kind": rule.get("kind"),
            "scope": rule.get("scope"),
            "status": rule.get("status"),
            "description": rule.get("description") or "",
            "approved_by": rule.get("approved_by") or "",
            "approved_at": rule.get("approved_at") or "",
            # On the rule itself these are the only surviving provenance;
            # reported even when the audit join finds nothing.
            "decision_id": rule.get("decision_id") or "",
            "workflow_id": rule.get("workflow_id") or "",
            "origin_clients": attributed or [UNATTRIBUTED],
            "origin_events": found,
            "from_focus_client": bool(focus_client and focus_client in attributed),
            "uri": rule.get("_uri"),
        })

    rows.sort(key=lambda r: (r["scope"] or "", r["kind"] or "", r["rule_id"]))
    from_focus = [r for r in rows if r["from_focus_client"]]
    unattributed = [r for r in rows if r["origin_clients"] == [UNATTRIBUTED]]
    return {
        "container": store.layout.container,
        "clients_scanned": clients,
        "focus_client": focus_client,
        "total_global_rules": len(rows),
        "active_global_rules": sum(1 for r in rows if r["status"] == "active"),
        "from_focus_client": len(from_focus),
        "unattributed": len(unattributed),
        "rules": rows,
    }


def _print_human(report: Dict[str, Any]) -> None:
    focus = report["focus_client"]
    print(f"container            : {report['container']}")
    print(f"clients scanned      : {', '.join(report['clients_scanned']) or '(none)'}")
    print(f"global/asset rules   : {report['total_global_rules']} "
          f"({report['active_global_rules']} active)")
    if focus:
        print(f"originating from {focus:<4}: {report['from_focus_client']}")
    print(f"unattributed         : {report['unattributed']}")
    print()
    if not report["rules"]:
        print("No global or asset-scoped rules found.")
        print()
        print("IMPORTANT: if this ran against a development environment with no")
        print("Azure access, that is NOT evidence that production has none.")
        return
    header = f"{'scope':<9} {'kind':<14} {'rule_id':<22} {'origin':<22} status"
    print(header)
    print("-" * len(header))
    for r in report["rules"]:
        origin = ",".join(r["origin_clients"])[:21]
        mark = " *" if r["from_focus_client"] else ""
        print(f"{str(r['scope'] or ''):<9} {str(r['kind'] or ''):<14} "
              f"{r['rule_id']:<22} {origin:<22} {r['status'] or ''}{mark}")
    if focus:
        print()
        print(f"* originated from {focus} — these survive its deletion and "
              "should be reviewed before the wipe.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client", default="",
                    help="Client to highlight as the wipe candidate, e.g. ERE")
    ap.add_argument("--json", action="store_true",
                    help="Emit the full machine-readable report")
    args = ap.parse_args()

    try:
        store = _load_store()
        report = enumerate_rules(store, focus_client=args.client)
    except Exception as exc:  # noqa: BLE001
        print(f"STORE UNREACHABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("This is NOT 'no rules found'. Fix storage configuration and "
              "re-run before deleting anything.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
