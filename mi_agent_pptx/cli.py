"""mi_agent_pptx.cli — generate the investor/funder PPTX pack from a run.

Usage
-----
    python -m mi_agent_pptx.cli \
        --run-dir out/runs/<run_id> \
        --deck-config configs/pptx/investor_pack.yaml \
        --client-name "Client Name" \
        --as-of-date "YYYY-MM-DD" \
        --output out/runs/<run_id>/reports/investor_pack.pptx

The deck is a faithful export of the React MI dashboard: it consumes the SAME
MI Agent API computations behind the ``/mi/*`` endpoints (see :mod:`mi_api`) and
renders those payloads verbatim, so every number and chart equals the dashboard
for the same portfolio. Missing payloads degrade to branded placeholders — a
slide is a placeholder only when the dashboard would also have no data for it.

Optional ``--prior-run-dir`` supplies the previous reporting period so the
funded KPI tiles carry month-on-month deltas. ``--pipeline-root`` (or
``MI_AGENT_PIPELINE_ROOT``) points at the container of the client's runs so the
client-level, cross-run pipeline source resolves.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import yaml

from .artifact_loader import RunArtifacts, load_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mi_agent_pptx.cli",
        description="Generate an institutional investor/funder PPTX pack from a "
                    "completed MI Agent pipeline run, aligned to the React "
                    "dashboard.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--deck-config", default="configs/pptx/investor_pack.yaml")
    p.add_argument("--client-name", default="Client")
    p.add_argument("--client-id", default=None,
                   help="Portfolio/client id (defaults to run_state.json client_id).")
    p.add_argument("--run-id", default=None,
                   help="Run id (defaults to run_state.json run_id / run dir name).")
    p.add_argument("--as-of-date", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--output-root", default=None,
                   help="Root of the client's runs for the multi-run (evolution / "
                        "risk) endpoints. Defaults to MI_AGENT_ONBOARDING_OUTPUT_ROOT "
                        "then the run dir's parent.")
    p.add_argument("--pipeline-root", default=None,
                   help="Root to discover the governed pipeline source (M2L KFI "
                        "extracts). Pipeline is a client-level, cross-run source "
                        "and does NOT live in the funded run dir; point this at "
                        "the container of the client's runs (or set "
                        "MI_AGENT_PIPELINE_ROOT). Defaults to the run dir's parent.")
    p.add_argument("--prior-run-dir", default=None,
                   help="Previous reporting-period run directory (enables MoM deltas).")
    p.add_argument("--lens", default=None, help="(accepted for compatibility)")
    p.add_argument("--consolidated", action="store_true",
                   help="(accepted for compatibility)")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--repo-root", default=None)
    return p


def _default_output(client_name: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in client_name.lower()).strip("_")
    return f"reports/{slug or 'client'}_investor_pack.pptx"


def _load_slides(deck_config: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load the deck metadata + slide sequence from the YAML config."""
    try:
        cfg = yaml.safe_load(Path(deck_config).read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[config][error] could not read {deck_config}: {exc}")
        cfg = {}
    deck_meta = cfg.get("deck", {}) if isinstance(cfg, dict) else {}
    slides = cfg.get("slides", []) if isinstance(cfg, dict) else []
    if not slides:
        print("[config][warn] no slides in deck config — using the default sequence.")
        slides = _DEFAULT_SLIDES
    return deck_meta or {}, slides


_DEFAULT_SLIDES: List[Dict[str, Any]] = [
    {"id": "cover", "type": "cover", "title": "Investor & Funder MI Pack"},
    {"id": "executive_summary", "type": "kpi_summary", "title": "Executive Summary"},
    {"id": "stratification_1", "type": "strat_barlists",
     "title": "Funded Stratifications — LTV & Rate", "keys": ["ltv", "rate"]},
    {"id": "stratification_2", "type": "strat_barlists",
     "title": "Funded Stratifications — Borrower Age & Product", "keys": ["age", "product"]},
    {"id": "geography", "type": "geo", "title": "Geographic Exposure"},
    {"id": "funded_evolution", "type": "funded_evolution", "title": "Funded Balance Evolution"},
    {"id": "cohorts", "type": "cohorts", "title": "Vintage Cohorts"},
    {"id": "pipeline", "type": "pipeline_summary", "title": "Pipeline Overview"},
    {"id": "pipeline_evolution", "type": "pipeline_evolution", "title": "Pipeline Evolution"},
    {"id": "funnel", "type": "funnel", "title": "Origination Funnel"},
    {"id": "forecast_bridge", "type": "forecast_bridge",
     "title": "Forecast Bridge — Funded to Forecast"},
    {"id": "forecast_projection", "type": "forecast_projection",
     "title": "Forecast Projection — Run-Rate Scale-Up"},
    {"id": "risk", "type": "risk", "title": "Risk Limits"},
    {"id": "methodology", "type": "methodology", "title": "Methodology & Notes"},
    {"id": "appendix", "type": "appendix", "title": "Appendix — Data Coverage"},
]


# --------------------------------------------------------------------------- #
# Funded / pipeline prep + discovery helpers (the layer the dashboard uses).
# These are imported by mi_api and by the regression tests, so they live here.
# --------------------------------------------------------------------------- #

def _prep_funded(tape: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise a funded tape with the MI Agent's own funded prep.

    Reuses ``mi_agent_api.funded_prep`` (the exact layer the dashboard uses) so
    the deck sees the same derived fields/dimensions (original LTV, vintage,
    youngest age, borrower type, buckets). Falls back to the raw tape if the
    prep is unavailable, so the deck still renders.
    """
    try:
        from mi_agent_api.funded_prep import prepare_funded_mi_dataset
        out, _report = prepare_funded_mi_dataset(tape)
        return out
    except Exception:  # noqa: BLE001 — never block the deck on prep
        return tape


def _prep_pipeline(tape: pd.DataFrame, as_of: Optional[str]) -> pd.DataFrame:
    """Canonicalise a raw pipeline tape (18a / M2L) with the MI Agent's own
    pipeline prep (``mi_agent_api.pipeline_prep``) — the same layer the dashboard
    uses — so pipeline/forecast charts resolve against real canonical fields.
    Falls back to the local alias canonicaliser, then the raw tape."""
    try:
        from mi_agent_api.pipeline_prep import prepare_pipeline_mi_dataset
        out, _report = prepare_pipeline_mi_dataset(tape, as_of_date=as_of)
        if out is not None and not out.empty:
            return out
    except Exception:  # noqa: BLE001
        pass
    try:
        from .pipeline_prep import canonicalise_pipeline
        return canonicalise_pipeline(tape, as_of=as_of)
    except Exception:  # noqa: BLE001
        return tape


def _read_source(path: str):
    """Read a governed pipeline source (CSV/XLSX), returning ``None`` on failure."""
    p = Path(path)
    try:
        if p.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(p)
        return pd.read_csv(p, low_memory=False)
    except Exception:  # noqa: BLE001
        return None


def _uri_derived_root(uri: str) -> Optional[str]:
    """A discovery root derived from an ``MI_AGENT_PIPELINE_URI`` snapshot pointer
    (strip the filename and a trailing ``{date}``/``latest`` folder)."""
    p = uri.rstrip("/")
    if p.endswith(".csv") or p.endswith(".json"):
        p = p.rsplit("/", 1)[0]
    tail = p.rsplit("/", 1)[-1]
    if tail == "latest" or __import__("re").match(r"^\d{4}-\d{2}-\d{2}$", tail):
        p = p.rsplit("/", 1)[0]
    return p or None


def _pipeline_roots(artifacts: RunArtifacts, explicit: Optional[str]) -> List[str]:
    """Ordered candidate roots to discover the governed pipeline source.

    Pipeline is a client-level, cross-run source — it does NOT live in the funded
    run dir. Precedence mirrors the MI API (``mi_agent_api.app._pipeline_root``):
    explicit flag → ``MI_AGENT_PIPELINE_ROOT`` → ``MI_AGENT_PIPELINE_URI`` root →
    the run dir (normal runs materialise the M2L under it) → the run dir's parent
    (the container of sibling runs, for split funded/pipeline backfills)."""
    roots: List[str] = []
    if explicit:
        roots.append(explicit)
    env = os.environ.get("MI_AGENT_PIPELINE_ROOT")
    if env:
        roots.append(env)
    uri = os.environ.get("MI_AGENT_PIPELINE_URI")
    if uri:
        derived = _uri_derived_root(uri)
        if derived:
            roots.append(derived)
    run_dir = Path(artifacts.run_dir)
    roots.append(str(run_dir))
    roots.append(str(run_dir.parent))
    seen: set = set()
    out: List[str] = []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _filter_client(sources: list, client_id: Optional[str]) -> list:
    """Keep only sources whose path carries the client token, when known — so a
    broad parent root does not pull a different client's pipeline. If nothing
    matches by path, don't over-filter (the tree may be single-client)."""
    if not client_id or not sources:
        return sources
    tok = str(client_id).strip().lower()
    if not tok:
        return sources
    matched = [s for s in sources
               if tok in str(s.get("source_file", "")).lower()]
    return matched or sources


def _resolve_pipeline_tape(artifacts: RunArtifacts, as_of: Optional[str],
                           pipeline_root: Optional[str] = None):
    """Resolve the pipeline frame the way the MI dashboard does.

    Discovers the RICH governed weekly source (``M2L*KFI*Pipeline*.csv/.xlsx``)
    across the pipeline roots (client-level, cross-run — see :func:`_pipeline_roots`),
    preps it with the MI Agent's own prep, and only falls back to the thin
    ``18a_central_pipeline_tape.csv`` when no richer source exists anywhere.
    """
    client_id = (artifacts.run_state.get("client_id")
                 if isinstance(artifacts.run_state, dict) else None)
    try:
        from mi_agent_api.pipeline_contract import discover_pipeline_sources
        for root in _pipeline_roots(artifacts, pipeline_root):
            try:
                sources = discover_pipeline_sources(root)
            except Exception:  # noqa: BLE001 — a bad root must not abort discovery
                continue
            sources = _filter_client(sources, client_id)
            if not sources:
                continue
            newest = sources[-1]  # discovery returns oldest -> newest
            raw = _read_source(newest.get("source_file", ""))
            if raw is not None and not raw.empty:
                return _prep_pipeline(raw, as_of or newest.get("pipeline_as_of_date"))
    except Exception:  # noqa: BLE001 — discovery is best effort
        pass
    # Fallback: the thin 18a tape the artifact loader already found.
    if artifacts.has_pipeline:
        return _prep_pipeline(
            artifacts.pipeline_tape, as_of or artifacts.run_state.get("reporting_date"))
    return None


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def run(argv: Optional[List[str]] = None) -> int:
    from .deck import DeckBuilder, DeckContext
    from .mi_api import build_dashboard_data
    from .pptx_theme import THEME

    args = build_parser().parse_args(argv)

    if args.repo_root:
        os.environ.setdefault("MI_AGENT_ONBOARDING_OUTPUT_ROOT",
                              str(Path(args.repo_root)))

    deck_meta, slides = _load_slides(args.deck_config)

    # Compute the dashboard payloads (identical to the /mi/* endpoints).
    data = build_dashboard_data(
        args.run_dir,
        client_id=args.client_id,
        run_id=args.run_id,
        as_of=args.as_of_date,
        output_root=args.output_root,
        pipeline_root=args.pipeline_root,
        prior_run_dir=args.prior_run_dir,
    )
    for note in data.notes:
        print(f"[mi_api] {note}")

    as_of = args.as_of_date or data.reporting_date or ""
    output = args.output or _default_output(args.client_name)
    work_dir = args.work_dir or (str(Path(output).with_suffix("")) + "_charts")

    ctx = DeckContext(
        client_name=args.client_name,
        as_of_date=as_of,
        run_dir=str(args.run_dir),
        work_dir=work_dir,
        footer=deck_meta.get("footer", DeckContext.footer),
        deck_name=deck_meta.get("name", DeckContext.deck_name),
        logo_path=deck_meta.get("logo_path"),
    )

    builder = DeckBuilder(data, ctx, theme=THEME)
    report = builder.build(slides, output)

    placeholders = [r for r in report["slides"] if r.get("placeholder")]
    print(f"\nDeck: {report['output']}")
    print(f"Slides: {len(report['slides'])} "
          f"({len(placeholders)} placeholder{'s' if len(placeholders) != 1 else ''})")
    for r in placeholders:
        print(f"  [placeholder] {r['id']}: {r.get('title')}")
    if report["coverage_notes"]:
        print(f"Coverage notes: {len(report['coverage_notes'])} (see appendix).")
    # A deck always renders (placeholders never fail the build).
    return 0


def main() -> None:  # pragma: no cover
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
