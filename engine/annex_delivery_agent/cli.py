"""
engine.annex_delivery_agent.cli
===============================

Command-line access to the Annex Delivery Agent.

Prepare a report::

    python -m engine.annex_delivery_agent.cli prepare \\
        --tenant ERE --annex ESMA_Annex2 \\
        --reporting-date 2025-11-30 \\
        --projected-input out/ere_ESMA_Annex2_projected.csv

List what this deployment can prepare::

    python -m engine.annex_delivery_agent.cli list-annexes

Approve and publish (two separate, deliberate steps)::

    python -m engine.annex_delivery_agent.cli approve --run-id run_… …
    python -m engine.annex_delivery_agent.cli publish --run-id run_… …

The CLI is a trusted in-process caller: it builds an internal
:class:`~trakt_core.context.ExecutionContext` for the named tenant. It is not a
way to bypass tenancy — the same authorisation runs, and a portfolio outside the
tenant's namespace is refused here exactly as it is over HTTP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import shim
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.context import ExecutionContext  # noqa: E402
from trakt_core.errors import TraktError  # noqa: E402

from engine.annex_delivery_agent.agent import (  # noqa: E402
    DEFAULT_STORE_ROOT, AnnexDeliveryAgent,
)
from engine.annex_delivery_agent.contracts import DeliveryRequest  # noqa: E402
from engine.annex_delivery_agent.occ_adapter import operator_view  # noqa: E402
from engine.annex_delivery_agent.persistence import DeliveryRunStore  # noqa: E402
from engine.annex_delivery_agent.publication import FilesystemPublicationSink  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annex-delivery",
        description="Prepare, approve and publish a regulatory annex report.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list-annexes", help="Show the reports this deployment supports.")

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--tenant", required=True, help="Authenticated client (tenant) id.")
        sp.add_argument("--actor", default="cli", help="Identity recorded in the audit trail.")
        sp.add_argument("--annex", default="ESMA_Annex2", help="Report to prepare.")
        sp.add_argument("--annex-version", default=None, help="Report version (default: latest).")
        sp.add_argument("--portfolio", default=None, help="Portfolio selector within the tenant.")
        sp.add_argument("--reporting-date", required=True, help="Reporting reference date.")
        sp.add_argument("--store-root", default=DEFAULT_STORE_ROOT,
                        help="Operational store root for delivery runs.")

    prep = sub.add_parser("prepare", help="Run the delivery lifecycle (never publishes).")
    common(prep)
    prep.add_argument("--projected-input", required=True, help="Projected annex CSV.")
    prep.add_argument("--client-config", default=None, help="Client annex configuration YAML.")
    prep.add_argument("--currency", default=None, help="Currency for amount elements.")
    prep.add_argument("--performance-mode", choices=["PRF", "NPRF"], default=None)
    prep.add_argument("--force-rerun", action="store_true",
                      help="Rebuild even if a completed artefact exists.")
    prep.add_argument("--force-reason", default=None, help="Recorded reason for the rerun.")
    prep.add_argument("--json", action="store_true", help="Emit the governed result as JSON.")

    for name, help_text in (("request-approval", "Send a prepared report for approval."),
                            ("approve", "Approve a prepared report for publication."),
                            ("reject", "Reject a prepared report."),
                            ("publish", "Publish an approved report.")):
        sp = sub.add_parser(name, help=help_text)
        common(sp)
        sp.add_argument("--run-id", required=True)
        sp.add_argument("--projected-input", default="", help="Only used to rebuild the request.")
        if name == "reject":
            sp.add_argument("--reason", required=True)
        if name == "approve":
            sp.add_argument("--note", default="")
        if name == "publish":
            sp.add_argument("--publish-root", default=None,
                            help="Destination root for the published package.")
    return p


def _context(args: argparse.Namespace) -> ExecutionContext:
    return ExecutionContext.for_internal(args.tenant, actor_id=args.actor)


def _request(args: argparse.Namespace) -> DeliveryRequest:
    options = {}
    for key in ("currency", "performance_mode"):
        value = getattr(args, key, None)
        if value:
            options[key] = value
    return DeliveryRequest(
        annex_id=args.annex,
        annex_version=getattr(args, "annex_version", None),
        reporting_date=args.reporting_date,
        projected_input=getattr(args, "projected_input", "") or "",
        output_root=args.store_root,
        portfolio_id=args.portfolio,
        client_config_reference=getattr(args, "client_config", None),
        force_rerun=bool(getattr(args, "force_rerun", False)),
        force_reason=getattr(args, "force_reason", None),
        options=options,
    )


def _agent(args: argparse.Namespace) -> AnnexDeliveryAgent:
    sink = None
    publish_root = getattr(args, "publish_root", None)
    if publish_root:
        sink = FilesystemPublicationSink(publish_root)
    return AnnexDeliveryAgent(store=DeliveryRunStore(args.store_root),
                              publication_sink=sink)


def _print_view(view: dict) -> None:
    print(f"\n{view.get('headline', '')}")
    print(f"  status ............ {view.get('status')}")
    metrics = view.get("metrics") or {}
    if metrics.get("records"):
        print(f"  records ........... {metrics['records']:,}")
        print(f"  fields ............ {metrics.get('delivered_fields')} delivered "
              f"of {metrics.get('projected_fields')} projected")
    if metrics.get("artefact_label"):
        print(f"  artefact .......... {metrics['artefact_label']} "
              f"({metrics.get('artefact_size_bytes', 0) / (1024 * 1024):.1f} MB)")
    if metrics.get("total_seconds"):
        print(f"  total time ........ {metrics['total_seconds']}s")
    for stage in view.get("stages") or []:
        print(f"  [{stage['state']:>16}] {stage['label']}")
        if stage.get("detail"):
            print(f"                     {stage['detail']}")
    automatic = view.get("automatic_values") or {}
    if automatic.get("lines"):
        print("\n  Automatic values requiring review:")
        for line in automatic["lines"][:20]:
            print(f"    - {line}")
        if len(automatic["lines"]) > 20:
            print(f"    … and {len(automatic['lines']) - 20} more (see the evidence file)")
    for err in view.get("blocking_errors") or []:
        print(f"\n  BLOCKED: {err.get('message')}")
        if err.get("remedy"):
            print(f"           {err['remedy']}")
    if view.get("disclaimer"):
        print(f"\n  {view['disclaimer']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list-annexes":
        for definition in AnnexDeliveryAgent().supported_annexes():
            print(f"{definition['annex_id']}  {definition['report_version']}  "
                  f"{definition['annex_name']}")
        return 0

    agent = _agent(args)
    context = _context(args)
    request = _request(args)

    try:
        if args.command == "prepare":
            governed = agent.prepare(context, request)
            if getattr(args, "json", False):
                print(json.dumps(governed.to_dict(), indent=2, default=str))
                return 0 if governed.ok else 1
            payload = governed.result
            if not payload:
                print(f"BLOCKED: {governed.error.message if governed.error else 'unknown'}",
                      file=sys.stderr)
                return 2
            from engine.annex_delivery_agent.contracts import DeliveryResult
            _print_view(operator_view(DeliveryResult.from_dict(payload)))
            return 0 if governed.ok else 1

        gate = agent.gate_for(context, request, run_id=args.run_id)
        if args.command == "request-approval":
            result = gate.request_approval(context)
        elif args.command == "approve":
            result = gate.approve(context, note=args.note)
        elif args.command == "reject":
            result = gate.reject(context, reason=args.reason)
        else:
            result = gate.publish(context)
        _print_view(operator_view(result))
        return 0

    except TraktError as exc:
        print(f"REFUSED [{exc.code}]: {exc.message}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
