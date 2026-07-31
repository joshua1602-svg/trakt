"""Cancel workflows left behind by testing, in bulk.

Cancelling one delivery at a time from the OCC is right for real work. It is
not right for clearing forty test runs, which is what this is for.

It goes through ``OpsEngine.cancel`` rather than deleting anything, so every
cancellation is a governed transition with a reason, an event on the delivery
and an entry in the client's hash-chained audit — the same record the button
in the UI produces. Cancelled deliveries stay readable; they simply leave the
working list, and their open questions stop being asked.

Dry run by default. Nothing changes until ``--apply`` is passed.

    python scripts/cancel_test_workflows.py --client ERE
    python scripts/cancel_test_workflows.py --client ERE --status needs_review
    python scripts/cancel_test_workflows.py --client ERE --before 2026-07-01 --apply

Published deliveries are never touched: cancelling one would imply the report
that went out came back, and it did not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operations_control.contracts import RUN_TERMINAL  # noqa: E402
from operations_control.engine import OpsEngine  # noqa: E402
from operations_control.stores import OpsStore  # noqa: E402

DEFAULT_REASON = "Cancelled in bulk: left behind by testing."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", action="append", dest="clients",
                        help="Client identifier. Repeatable. Defaults to all.")
    parser.add_argument("--status", action="append", dest="statuses",
                        help="Only these statuses. Repeatable.")
    parser.add_argument("--workflow", action="append", dest="workflows",
                        help="Only these workflow identifiers. Repeatable.")
    parser.add_argument("--before",
                        help="Only deliveries created before this date "
                             "(YYYY-MM-DD).")
    parser.add_argument("--reason", default=DEFAULT_REASON,
                        help="Recorded against every cancellation.")
    parser.add_argument("--actor", default="bulk-cleanup",
                        help="Recorded as who did it.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually cancel. Without this, nothing changes.")
    args = parser.parse_args(argv)

    store = OpsStore.from_env()
    engine = OpsEngine(store)
    clients = args.clients or store.known_clients()

    chosen: list[tuple[str, str, str]] = []
    for client_id in clients:
        for row in store.list_workflows(client_id):
            status = str(row.get("status") or "")
            if status in RUN_TERMINAL:
                continue                      # published or already cancelled
            if args.statuses and status not in args.statuses:
                continue
            if args.workflows and row.get("workflow_id") not in args.workflows:
                continue
            created = str(row.get("created_at") or "")
            if args.before and created[:10] >= args.before:
                continue
            chosen.append((client_id, str(row["workflow_id"]), status))

    if not chosen:
        print("Nothing matched. Nothing to do.")
        return 0

    for client_id, workflow_id, status in chosen:
        print(f"  {client_id:20} {workflow_id:24} {status}")
    print(f"\n{len(chosen)} deliverie{'s' if len(chosen) != 1 else ''} matched.")

    if not args.apply:
        print("Dry run — nothing changed. Pass --apply to cancel these.")
        return 0

    cancelled = 0
    for client_id, workflow_id, _ in chosen:
        run = store.load_workflow(client_id, workflow_id)
        if run is None:
            continue
        try:
            engine.cancel(run, actor=args.actor, reason=args.reason)
        except Exception as exc:              # noqa: BLE001 — report, continue
            print(f"  ! {workflow_id}: {exc}")
            continue
        cancelled += 1
    print(f"Cancelled {cancelled}. Each one is kept and audited; "
          f"find them with the Cancelled filter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
