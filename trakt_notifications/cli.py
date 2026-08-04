#!/usr/bin/env python3
"""Operator commands for the notification pathway.

Follows the convention already established by
``apps.blob_trigger_app.approvals`` and ``…ops``: a small argparse CLI over the
same storage the runtime uses, so an operator inspects and acts on real state
rather than a reporting copy of it. No new administration UI — the OCC does not
have a natural home for this yet, and inventing one would be a larger change
than the capability warrants.

    python -m trakt_notifications.cli recipients ERE
    python -m trakt_notifications.cli authorise ERE <recipient_id> --contexts total --by jo
    python -m trakt_notifications.cli outbox ERE --failures
    python -m trakt_notifications.cli show ERE <batch_id>
    python -m trakt_notifications.cli deliver ERE
    python -m trakt_notifications.cli diagnose ERE
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from apps.blob_trigger_app.storage import open_storage

from .layout import NotificationLayout
from .outbox import Outbox, STATE_PERMANENT, STATE_RETRYABLE
from .recipients import RecipientStore
from .store import BatchStore


def _context(args):
    storage = open_storage(local_root=getattr(args, "local_root", None))
    return storage, NotificationLayout.from_env()


def cmd_recipients(args) -> int:
    storage, layout = _context(args)
    rows = RecipientStore(storage, layout).list(args.tenant)
    if not rows:
        print(f"No recipients registered for {args.tenant}.")
        return 0
    for r in rows:
        state = ("enabled" if r.notifications_enabled else "DISABLED")
        addressable = "addressable" if r.addressable else "NOT ADDRESSABLE"
        print(f"{r.recipient_id}  {state}  {addressable}  "
              f"install={r.installation_status}  "
              f"contexts={','.join(r.portfolio_contexts) or '-'}  "
              f"{r.display_name or ''}")
    return 0


def cmd_authorise(args) -> int:
    storage, layout = _context(args)
    store = RecipientStore(storage, layout)
    try:
        recipient = store.authorise(
            args.tenant, args.recipient_id,
            portfolio_contexts=args.contexts, actor=args.by,
            enable=not args.disable)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{recipient.recipient_id}: "
          f"{'enabled' if recipient.notifications_enabled else 'disabled'} for "
          f"{','.join(recipient.portfolio_contexts) or 'no contexts'}")
    if recipient.notifications_enabled and not recipient.addressable:
        # Worth saying plainly: an authorised recipient with no captured
        # conversation still receives nothing, and the reason is not obvious.
        print("NOTE: no Teams conversation reference has been captured yet — "
              "the user must install the Trakt app before anything is "
              "delivered.")
    return 0


def cmd_outbox(args) -> int:
    storage, layout = _context(args)
    outbox = Outbox(storage, layout)
    rows = (outbox.failures(args.tenant) if args.failures
            else outbox.list(args.tenant))
    if not rows:
        print(f"Nothing in the {args.tenant} outbox"
              f"{' needing attention' if args.failures else ''}.")
        return 0
    for i in rows:
        print(f"{i.item_id}  {i.state:<18} {i.message_type:<18} "
              f"attempts={i.attempt_count}  batch={i.notification_batch_id}  "
              f"recipient={i.recipient_id}"
              + (f"  next={i.next_attempt_at}" if i.next_attempt_at else "")
              + (f"\n    last error: {i.last_error}" if i.last_error else ""))
    return 0


def cmd_show(args) -> int:
    storage, layout = _context(args)
    batch = BatchStore(storage, layout).load(args.tenant, args.batch_id)
    if batch is None:
        print(f"No such batch: {args.batch_id}", file=sys.stderr)
        return 1
    print(batch.to_json())
    return 0


def cmd_deliver(args) -> int:
    from .delivery import run_once
    storage, _layout = _context(args)
    report = run_once(args.tenant, storage=storage, limit=args.limit)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if not report.errors else 1


def cmd_diagnose(args) -> int:
    """The operator diagnostic view for failed deliveries.

    Structured rather than pretty, so it can be pasted into an incident record
    or piped into a report. Groups by failure state, because the two need
    different actions: a retryable item will be picked up again on its own, a
    permanent one will not.
    """
    storage, layout = _context(args)
    outbox = Outbox(storage, layout)
    recipients = {r.recipient_id: r
                  for r in RecipientStore(storage, layout).list(args.tenant)}

    report = {"tenant_id": args.tenant, "retrying": [], "failed": [],
              "recipients_without_conversation": [],
              "recipients_unauthorised": []}
    for item in outbox.failures(args.tenant):
        entry = {
            "item_id": item.item_id,
            "message_type": item.message_type,
            "batch_id": item.notification_batch_id,
            "recipient_id": item.recipient_id,
            "attempts": item.attempt_count,
            "last_error": item.last_error,
            "next_attempt_at": item.next_attempt_at,
        }
        if item.state == STATE_RETRYABLE:
            report["retrying"].append(entry)
        elif item.state == STATE_PERMANENT:
            report["failed"].append(entry)

    for rid, recipient in recipients.items():
        if recipient.notifications_enabled and not recipient.addressable:
            report["recipients_without_conversation"].append(rid)
        if recipient.addressable and not recipient.portfolio_contexts:
            report["recipients_unauthorised"].append(rid)

    print(json.dumps(report, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trakt_notifications.cli",
        description="Operator commands for proactive Teams notifications.")
    parser.add_argument("--local-root", default=None,
                        help="Local root emulating blob containers (local/dev).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("recipients", help="List registered recipients.")
    p.add_argument("tenant")
    p.set_defaults(func=cmd_recipients)

    p = sub.add_parser("authorise",
                       help="Map a recipient to permitted portfolio contexts.")
    p.add_argument("tenant")
    p.add_argument("recipient_id")
    p.add_argument("--contexts", nargs="+", required=True)
    p.add_argument("--by", required=True, help="Operator making the decision.")
    p.add_argument("--disable", action="store_true",
                   help="Record the mapping but leave delivery off.")
    p.set_defaults(func=cmd_authorise)

    p = sub.add_parser("outbox", help="List outbox items.")
    p.add_argument("tenant")
    p.add_argument("--failures", action="store_true",
                   help="Only items needing attention.")
    p.set_defaults(func=cmd_outbox)

    p = sub.add_parser("show", help="Print a generated notification payload.")
    p.add_argument("tenant")
    p.add_argument("batch_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("deliver", help="Run one delivery pass.")
    p.add_argument("tenant")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_deliver)

    p = sub.add_parser("diagnose",
                       help="Structured report on failed deliveries.")
    p.add_argument("tenant")
    p.set_defaults(func=cmd_diagnose)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
