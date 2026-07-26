"""demo_platform.run_demo — the one repeatable command that builds the demo.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

Usage::

    python -m demo_platform.run_demo --all          # everything, from a clean reset
    python -m demo_platform.run_demo --generate     # source extracts only
    python -m demo_platform.run_demo --orchestrate  # pipeline + assembler + publish
    python -m demo_platform.run_demo --artefacts    # deck, validation, risk, audit
    python -m demo_platform.run_demo --metrics      # export the film's fixtures
    python -m demo_platform.run_demo --assert       # reconciliation gate only
    python -m demo_platform.run_demo --safety       # data-safety scan only

Stages, in order:

  1. reset the synthetic demo environment;
  2. generate both portfolios' source extracts for all reporting periods;
  3. establish the approved onboarding contract per portfolio (Stage 1);
  4. run the recurring orchestration per portfolio per period (Stage 2);
  5. assemble each period into the consolidated platform canonical;
  6. publish the current period as the active MI Agent data source;
  7. produce the governed outputs (investor deck, validation, risk, audit);
  8. export the demo manifest and the film's metric + surface fixtures;
  9. run the reconciliation gate — the run FAILS if the narrative is unsupported;
 10. run the data-safety scan — the run FAILS on any prohibited content;
 11. copy the fixtures into the Remotion project.

Every stage is idempotent, so any subset can be re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import config as cfg
from . import schemas

warnings.simplefilter("ignore")

# The pipeline components log at INFO, which is right for an operator tailing an
# Azure log stream and wrong for a demo build transcript: the storage-backend and
# HTTP lines would bury the stage output. Warnings and errors still surface.
logging.basicConfig(level=logging.WARNING)
for _noisy in ("trakt.blob_trigger.storage", "httpx", "httpcore", "mi_agent_api",
               "trakt.mi_agent.decks", "root"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_RULE = "─" * 78


def _banner(step: str, title: str) -> None:
    print(f"\n{_RULE}\n {step}  {title}\n{_RULE}")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_reset(*, verbose: bool = True) -> None:
    from . import orchestration
    _banner("1/11", "Reset the synthetic demonstration environment")
    orchestration.reset(verbose=verbose)


def stage_generate(*, verbose: bool = True) -> Dict[str, Any]:
    from . import generator
    _banner("2/11", "Generate the synthetic source extracts")
    generated = generator.generate(verbose=verbose)
    written = generator.write_source_files(generated)
    print(f"  [generate] {len(written)} source extracts, "
          f"calibrated in {generated.iterations} iterations")
    for f in written:
        print(f"    {f['portfolio']:16} {f['period']}  {f['rows']:>6,} rows  "
              f"£{f['total_balance'] / 1e6:>9,.2f}m  {Path(f['path']).name}")
    return {
        "calibration": generated.calibration.to_dict(),
        "iterations": generated.iterations,
        "modelMetrics": {k: v for k, v in generated.model_metrics.items()
                         if k != "calibration_history"},
        "calibrationHistory": generated.model_metrics.get("calibration_history"),
        "injectedExceptions": generated.injected_exceptions,
        "files": written,
    }


def stage_onboard(*, verbose: bool = True) -> Dict[str, Any]:
    from . import onboarding
    _banner("3/11", "Stage 1 — establish the approved onboarding contracts")
    return onboarding.run(verbose=verbose)


def stage_orchestrate(*, verbose: bool = True) -> Dict[str, Any]:
    from . import orchestration
    _banner("4/11", "Stage 2 — recurring orchestration, assembly and publication")
    return orchestration.run(verbose=verbose)


def stage_artefacts(orchestration_result: Dict[str, Any], *,
                    verbose: bool = True) -> Dict[str, Any]:
    from . import artefacts
    _banner("5/11", "Produce the governed outputs")
    return artefacts.build(orchestration_result, verbose=verbose)


def stage_metrics(*, verbose: bool = True) -> Dict[str, Any]:
    from . import metrics
    _banner("6/11", "Export the metric and surface fixtures")
    payload = metrics.export(verbose=verbose)
    payload["schemas"] = schemas.as_dict()
    (cfg.export_dir() / "demo_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def stage_manifest(parts: Dict[str, Any]) -> Dict[str, Any]:
    _banner("7/11", "Write the machine-readable demo manifest")
    manifest = {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "generatedFor": "Trakt product demonstration",
        "narrative": cfg.as_dict(),
        "sourceSchemas": schemas.as_dict(),
        "schemaResolution": schemas.resolution_summary(),
        "generation": parts.get("generation"),
        "onboarding": {
            pid: {
                "portfolio": contract["portfolio"],
                "source": contract["source"],
                "mapping": {k: v for k, v in contract["mapping"].items()
                            if k != "decisions"},
                "mappingDecisions": contract["mapping"]["decisions"],
                "transformations": contract["transformations"],
                "validation": contract["validation"],
            }
            for pid, contract in (parts.get("onboarding") or {}).items()
        },
        "orchestration": {
            "runs": [
                {k: v for k, v in run.items()
                 if k not in ("run_manifest", "validation_dashboard")}
                for run in (parts.get("orchestration") or {}).get("runs", [])
            ],
            "assemblies": (parts.get("orchestration") or {}).get("assemblies"),
            "latestPointer": (parts.get("orchestration") or {}).get("latest_pointer"),
            "reconciliation": (parts.get("orchestration") or {}).get("reconciliation"),
        },
        "artefacts": {
            key: {k: v for k, v in art.items() if k not in ("payload", "listing")}
            for key, art in (parts.get("artefacts") or {}).items()
            if isinstance(art, dict)
        },
        "paths": {
            "source": str(cfg.source_dir()),
            "runs": str(cfg.runs_dir()),
            "platformStore": str(cfg.local_blob_root()),
            "platformPrefix": cfg.platform_prefix(),
            "decks": str(cfg.deck_root()),
            "artefacts": str(cfg.artefact_dir()),
            "export": str(cfg.export_dir()),
            "videoFixtures": str(cfg.video_fixture_dir()),
        },
    }
    out = cfg.export_dir() / "demo_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"  [manifest] {out}")
    return manifest


def stage_assert(export_payload: Optional[Dict[str, Any]] = None, *,
                 verbose: bool = True) -> Dict[str, Any]:
    from . import assertions
    _banner("8/11", "Reconciliation gate")
    return assertions.enforce(export_payload, verbose=verbose)


def stage_safety(*, verbose: bool = True) -> Dict[str, Any]:
    from . import safety
    _banner("9/11", "Data-safety scan")
    return safety.enforce(verbose=verbose)


def stage_publish_fixtures(*, verbose: bool = True) -> List[str]:
    """Copy the exported fixtures into the Remotion project's public folder."""
    _banner("10/11", "Publish the fixtures to the Remotion project")
    dest = cfg.video_fixture_dir()
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in ("demo_metrics.json", "demo_manifest.json",
                 "assertion_report.json", "safety_report.json"):
        src = cfg.export_dir() / name
        if src.exists():
            shutil.copyfile(src, dest / name)
            copied.append(name)
    catalogue = cfg.artefact_dir() / "artefact_catalogue.json"
    if catalogue.exists():
        shutil.copyfile(catalogue, dest / "artefact_catalogue.json")
        copied.append("artefact_catalogue.json")
    if verbose:
        for name in copied:
            print(f"  [fixtures] {dest / name}")
    return copied


def stage_summary(assertion_report: Dict[str, Any]) -> None:
    _banner("11/11", "Summary")
    m = assertion_report.get("metrics", {})
    rows = [
        ("Prior funded balance", f"£{(m.get('priorFundedBalance') or 0) / 1e9:,.4f}bn"),
        ("Current funded balance", f"£{(m.get('currentFundedBalance') or 0) / 1e9:,.4f}bn"),
        ("Movement", f"£{(m.get('movement') or 0) / 1e6:,.3f}m"),
        ("  Portfolio A", f"£{(m.get('portfolioAMovement') or 0) / 1e6:,.3f}m"),
        ("  Portfolio B", f"£{(m.get('portfolioBMovement') or 0) / 1e6:,.3f}m"),
        (f"  {m.get('primaryRegion')} movement",
         f"£{(m.get('primaryRegionMovement') or 0) / 1e6:,.3f}m "
         f"({(m.get('primaryRegionShare') or 0) * 100:.1f}% of the increase)"),
        ("WA current LTV (current)", f"{m.get('currentWaCurrentLtvPoints'):.3f}%"),
        ("WA current LTV (prior)", f"{m.get('priorWaCurrentLtvPoints'):.3f}%"),
        ("Avg borrower age (current)", f"{m.get('currentAvgBorrowerAge'):.3f} years"),
        ("Avg borrower age (prior)", f"{m.get('priorAvgBorrowerAge'):.3f} years"),
        ("Consolidated loan count", f"{m.get('consolidatedLoanCount'):,}"),
    ]
    for label, value in rows:
        print(f"  {label:<32} {value}")
    print(f"\n  Reconciliation: {assertion_report['checksPassed']}"
          f"/{assertion_report['checksRun']} checks passed")
    print(f"  Fixtures:       {cfg.video_fixture_dir()}")
    print(f"\n  Next: cd demo-video && npm install && npm run render")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m demo_platform.run_demo",
        description="Build the Trakt synthetic product demonstration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="Run every stage from a clean reset (the default).")
    ap.add_argument("--generate", action="store_true", help="Source extracts only.")
    ap.add_argument("--onboard", action="store_true", help="Onboarding contracts only.")
    ap.add_argument("--orchestrate", action="store_true",
                    help="Pipeline + assembler + publish only.")
    ap.add_argument("--artefacts", action="store_true", help="Governed outputs only.")
    ap.add_argument("--metrics", action="store_true", help="Export fixtures only.")
    ap.add_argument("--assert", dest="assert_only", action="store_true",
                    help="Reconciliation gate only.")
    ap.add_argument("--safety", action="store_true", help="Data-safety scan only.")
    ap.add_argument("--no-reset", action="store_true",
                    help="With --all, keep the existing workspace.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    selective = any((args.generate, args.onboard, args.orchestrate, args.artefacts,
                     args.metrics, args.assert_only, args.safety))
    run_all = args.all or not selective

    started = time.time()
    parts: Dict[str, Any] = {}

    try:
        if run_all and not args.no_reset:
            stage_reset(verbose=verbose)
        if run_all or args.generate:
            parts["generation"] = stage_generate(verbose=verbose)
        if run_all or args.onboard:
            parts["onboarding"] = stage_onboard(verbose=verbose)
        if run_all or args.orchestrate:
            parts["orchestration"] = stage_orchestrate(verbose=verbose)
        if run_all or args.artefacts:
            parts["artefacts"] = stage_artefacts(
                parts.get("orchestration") or {"runs": [], "assemblies": []},
                verbose=verbose)
        export_payload = None
        if run_all or args.metrics:
            export_payload = stage_metrics(verbose=verbose)
        if run_all:
            stage_manifest(parts)
        report = None
        if run_all or args.assert_only:
            report = stage_assert(export_payload, verbose=verbose)
        if run_all or args.safety:
            stage_safety(verbose=verbose)
        if run_all:
            stage_publish_fixtures(verbose=verbose)
            if report:
                stage_summary(report)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not traceback
        print(f"\nDEMO RUN FAILED\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"\nCompleted in {time.time() - started:,.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
