"""mi_agent_pptx.cli — generate the investor/funder PPTX pack from a run.

Usage
-----
    python -m mi_agent_pptx.cli \
        --run-dir out/runs/<run_id> \
        --deck-config configs/pptx/investor_pack.yaml \
        --client-name "Client Name" \
        --as-of-date "YYYY-MM-DD" \
        --output reports/client_investor_pack_YYYYMMDD.pptx

The generator consumes MI Agent run artifacts only — it never imports Streamlit
or the legacy ``streamlit_app_erm.py``. Missing artifacts degrade to branded
placeholders and appendix coverage notes rather than failing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .artifact_loader import load_run_artifacts
from .chart_resolver import ChartResolver
from .data_resolver import resolve_data
from .deck_config import load_deck_config
from .insight_resolver import StraplineResolver
from .metric_resolver import MetricResolver
from .placeholders import AppendixNotes
from .pptx_builder import BuildContext, DeckBuilder
from .pptx_theme import THEME
from .registry_loader import RegistryLoader
from .validation import validate_build, validate_deck_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mi_agent_pptx.cli",
        description="Generate an institutional investor/funder PPTX pack from a "
                    "completed MI Agent pipeline run.",
    )
    p.add_argument("--run-dir", required=True,
                   help="MI Agent run directory (e.g. out/runs/<run_id>).")
    p.add_argument("--deck-config",
                   default="configs/pptx/investor_pack.yaml",
                   help="Path to the deck config YAML.")
    p.add_argument("--client-name", default="Client",
                   help="Client / portfolio name shown on the cover.")
    p.add_argument("--as-of-date", default=None,
                   help="Data cut-off date (YYYY-MM-DD). Inferred from the tape "
                        "when omitted.")
    p.add_argument("--output", default=None,
                   help="Output .pptx path (default: reports/<client>_investor_pack.pptx).")
    p.add_argument("--lens", default=None,
                   help="Portfolio lens (total|direct|acquired|cohort). "
                        "Defaults to the deck config's default_lens.")
    p.add_argument("--consolidated", action="store_true",
                   help="Consolidated funded lens — suppress broker channel where "
                        "acquired portfolios carry no broker data.")
    p.add_argument("--work-dir", default=None,
                   help="Directory for intermediate chart PNGs (default: a "
                        "'_charts' folder beside the output).")
    p.add_argument("--repo-root", default=None,
                   help="Override repo root for registry resolution.")
    return p


def _default_output(client_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in client_name.lower()).strip("_")
    return f"reports/{slug or 'client'}_investor_pack.pptx"


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # --- config + registries -------------------------------------------
    config = load_deck_config(args.deck_config)
    cfg_report = validate_deck_config(config)
    for w in cfg_report.warnings:
        print(f"[config][warn] {w}")
    for e in cfg_report.errors:
        print(f"[config][error] {e}")

    registries = RegistryLoader(Path(args.repo_root) if args.repo_root else None)

    # --- artifacts + data ----------------------------------------------
    artifacts = load_run_artifacts(args.run_dir)
    for note in artifacts.coverage_notes:
        print(f"[artifacts] {note}")

    appendix = AppendixNotes()
    appendix.extend(artifacts.coverage_notes)

    if artifacts.has_tape:
        resolved = resolve_data(artifacts.tape, registries,
                                as_of_date=args.as_of_date)
    else:
        # Empty analytical frame — deck renders fully with placeholders.
        import pandas as pd
        resolved = resolve_data(pd.DataFrame(), registries,
                                as_of_date=args.as_of_date)
        appendix.add("No canonical typed tape resolved — deck rendered with "
                     "branded placeholders throughout.")

    as_of = args.as_of_date or resolved.as_of_date or ""

    # --- resolvers ------------------------------------------------------
    analytics = {
        "analytics": artifacts.artifact("analytics"),
        "metrics": artifacts.artifact("metrics"),
        "validation": artifacts.artifact("validation"),
        "risk_monitor": artifacts.artifact("risk_monitor"),
    }
    metric_resolver = MetricResolver(resolved, registries, analytics=analytics)
    strapline_metrics = {
        k: metric_resolver.resolve(config.metric_spec(k))
        for k in config.metrics.keys()
    }
    strapline_resolver = StraplineResolver(
        metrics=strapline_metrics,
        llm_artifact=artifacts.artifact("straplines"),
    )

    output = args.output or _default_output(args.client_name)
    work_dir = args.work_dir or str(Path(output).with_suffix("")) + "_charts"
    chart_resolver = ChartResolver(resolved, registries, work_dir, theme=THEME)

    # --- context --------------------------------------------------------
    source_artifacts = []
    if artifacts.tape_path:
        source_artifacts.append(str(artifacts.tape_path.name))
    if artifacts.pipeline_tape_path:
        source_artifacts.append(str(artifacts.pipeline_tape_path.name))
    source_artifacts.extend(sorted(
        p.name for k, p in artifacts.json_paths.items() if k != "run_state"))

    ctx = BuildContext(
        client_name=args.client_name,
        as_of_date=as_of,
        run_dir=str(args.run_dir),
        lens=args.lens or config.default_lens,
        consolidated=args.consolidated,
        logo_path=config.logo_path,
        source_artifacts=source_artifacts,
    )

    # --- build ----------------------------------------------------------
    builder = DeckBuilder(config, ctx, metric_resolver, chart_resolver,
                          strapline_resolver, appendix, theme=THEME)
    build_report = builder.build(output)

    # --- validate -------------------------------------------------------
    build_validation = validate_build(build_report)
    print(f"\nDeck: {build_report['output']}")
    print(f"Slides: {len(build_report['slides'])}")
    print(f"Validation: {build_validation.summary()}")
    for e in build_validation.errors:
        print(f"[build][error] {e}")
    for w in build_validation.warnings:
        print(f"[build][warn] {w}")
    if build_report["coverage_notes"]:
        print(f"Coverage notes: {len(build_report['coverage_notes'])} "
              f"(see appendix slide).")

    return 0 if build_validation.ok else 2


def main() -> None:  # pragma: no cover - thin entrypoint
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
