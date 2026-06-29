"""engine.orchestrator_agent.cli — operator entrypoint for the orchestration.

Examples
--------
    # Multi-portfolio MI pipeline (per-portfolio onboard→transform→validate,
    # then assemble → MI):
    python -m engine.orchestrator_agent \
      --client ERE --target mi --out-dir orchestration_out \
      --portfolio direct_001=inputs/direct_book \
      --portfolio acquired_001=inputs/acquired_book_1 \
      --acquisition-date acquired_001=2026-08-15 --seller acquired_001="Seller A"

    # Resume after resolving a halted gate:
    python -m engine.orchestrator_agent --resume orchestration_out/orun_ere_*/run_state.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .adapters import PortfolioSpec, RealAgentAdapters
from .orchestrator import VALID_TARGETS, run_orchestration
from .state import RunState, STEP_DONE, STEP_HALTED


def _kv_map(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected id=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _build_specs(args) -> List[PortfolioSpec]:
    labels = _kv_map(args.label)
    acq = _kv_map(args.acquisition_date)
    sellers = _kv_map(args.seller)
    types = _kv_map(args.type)
    specs: List[PortfolioSpec] = []
    for item in args.portfolio:
        if "=" not in item:
            raise SystemExit(f"--portfolio expects id=input_path, got {item!r}")
        pid, path = item.split("=", 1)
        pid = pid.strip()
        specs.append(PortfolioSpec(
            source_portfolio_id=pid,
            input=path.strip(),
            source_portfolio_type=types.get(pid),
            source_portfolio_label=labels.get(pid),
            acquisition_date=acq.get(pid),
            seller_name=sellers.get(pid),
            allow_unknown_acquisition_date=args.allow_unknown_acquisition_date,
        ))
    return specs


def _print_summary(state: RunState) -> None:
    print("=" * 64)
    print(f"Orchestration {state.run_id} — status: {state.status}")
    print(f"  client: {state.client_id}  target: {state.target}")
    for p in state.portfolios:
        steps = " ".join(f"{n}:{p.step(n).status}" for n in ("onboard", "transform", "validate", "stamp"))
        print(f"  - {p.source_portfolio_id:14} [{p.status}]  {steps}")
    print(f"  assemble: {state.assemble.status}  route: {state.route.status}")
    if state.central_canonical_path:
        print(f"  central canonical: {state.central_canonical_path}")
    if state.blockers:
        print("  BLOCKERS (resolve, then re-run with --resume):")
        for b in state.blockers:
            print(f"    · {b}")
    print("=" * 64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m engine.orchestrator_agent",
        description="Governed Agentic Orchestration — drive Onboarding → "
                    "Transformation → Validation per portfolio, then Assembler → "
                    "MI. Reuses existing agents via their manifest handoffs.")
    ap.add_argument("--resume", default="", help="Path to a run_state.json to resume.")
    ap.add_argument("--client", default="", help="Client id (recorded on the run).")
    ap.add_argument("--target", choices=list(VALID_TARGETS), default="mi")
    ap.add_argument("--regime", default=None, help="Regime for target=regime/all (e.g. ESMA_Annex2).")
    ap.add_argument("--out-dir", default="orchestration_out", help="Run output root.")
    ap.add_argument("--portfolio", action="append", default=[],
                    help="Repeatable: source_portfolio_id=input_dir (e.g. direct_001=inputs/direct).")
    ap.add_argument("--type", action="append", default=[],
                    help="Repeatable id=direct|acquired (else derived from the id prefix).")
    ap.add_argument("--label", action="append", default=[], help="Repeatable id=\"Label\".")
    ap.add_argument("--acquisition-date", action="append", default=[], help="Repeatable id=YYYY-MM-DD.")
    ap.add_argument("--seller", action="append", default=[], help="Repeatable id=\"Seller\".")
    ap.add_argument("--allow-unknown-acquisition-date", action="store_true")
    ap.add_argument("--client-name", default="", help="Lender display name for onboarding.")
    ap.add_argument("--registry", default="config/system/fields_registry.yaml")
    ap.add_argument("--onboarding-mode", default="mi_only")
    args = ap.parse_args(argv)

    adapters = RealAgentAdapters(
        registry=args.registry, client_name=args.client_name or None,
        onboarding_mode=args.onboarding_mode)
    created_at = datetime.now(timezone.utc).isoformat()

    if args.resume:
        state = RunState.load(args.resume)
        state = run_orchestration(
            state.client_id, [], target=state.target, out_root=state.out_root,
            adapters=adapters, created_at=created_at, regime=args.regime,
            resume_state=state)
    else:
        if not args.client or not args.portfolio:
            ap.error("--client and at least one --portfolio are required (or use --resume).")
        specs = _build_specs(args)
        state = run_orchestration(
            args.client, specs, target=args.target, out_root=args.out_dir,
            adapters=adapters, created_at=created_at, regime=args.regime)

    _print_summary(state)
    return 0 if state.status == STEP_DONE else (3 if state.status == STEP_HALTED else 2)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
