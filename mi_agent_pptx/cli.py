"""mi_agent_pptx.cli — generate the investor/funder PPTX pack from a run.

Usage
-----
    python -m mi_agent_pptx.cli \
        --run-dir out/runs/<run_id> \
        --deck-config configs/pptx/investor_pack.yaml \
        --client-name "Client Name" \
        --as-of-date "YYYY-MM-DD" \
        --output out/runs/<run_id>/reports/investor_pack.pptx

Optional ``--prior-run-dir`` supplies the previous reporting period so KPI tiles
render prior-period deltas ("+£0.7MM vs prior"). The generator consumes MI Agent
run artifacts only — never Streamlit — and resolves each lens (funded / pipeline
/ forecast) from its own frame so the pipeline total is never the funded total.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .artifact_loader import RunArtifacts, load_run_artifacts
from .chart_resolver import ChartResolver
from .data_resolver import ResolvedData, resolve_data
from .deck_config import load_deck_config
from .insight_resolver import StraplineResolver
from .metric_resolver import MetricResolver
from .placeholders import AppendixNotes
from .pptx_builder import BuildContext, DeckBuilder
from .pptx_theme import THEME
from .registry_loader import RegistryLoader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mi_agent_pptx.cli",
        description="Generate an institutional investor/funder PPTX pack from a "
                    "completed MI Agent pipeline run.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--deck-config", default="configs/pptx/investor_pack.yaml")
    p.add_argument("--client-name", default="Client")
    p.add_argument("--as-of-date", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--pipeline-root", default=None,
                   help="Root to discover the governed pipeline source (M2L KFI "
                        "extracts). Pipeline is a client-level, cross-run source "
                        "and does NOT live in the funded run dir; point this at "
                        "the container of the client's runs (or set "
                        "MI_AGENT_PIPELINE_ROOT). Defaults to the run dir, then "
                        "its parent (sibling runs).")
    p.add_argument("--prior-run-dir", default=None,
                   help="Previous reporting-period run directory (enables MoM deltas).")
    p.add_argument("--lens", default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--repo-root", default=None)
    return p


def _default_output(client_name: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in client_name.lower()).strip("_")
    return f"reports/{slug or 'client'}_investor_pack.pptx"


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


def _lens_bundle(artifacts: RunArtifacts, registries: RegistryLoader,
                 as_of: Optional[str],
                 pipeline_root: Optional[str] = None) -> Dict[str, Optional[ResolvedData]]:
    """Resolve the funded / pipeline / forecast frames for a run."""
    funded_tape = (_prep_funded(artifacts.tape) if artifacts.has_tape
                   else pd.DataFrame())
    funded = resolve_data(funded_tape, registries, as_of_date=as_of)
    pipeline = None
    pipe_df = _resolve_pipeline_tape(artifacts, as_of, pipeline_root)
    if pipe_df is not None and not pipe_df.empty:
        pipeline = resolve_data(pipe_df, registries, as_of_date=as_of)
    # Forecast charts (run-rate / cumulative) draw from the pipeline frame's
    # expected-completion data; the forecast KPI uses the registry bridge.
    forecast = pipeline
    return {"funded": funded, "pipeline": pipeline, "forecast": forecast}


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_deck_config(args.deck_config)
    from .validation import validate_build, validate_deck_config
    cfg_report = validate_deck_config(config)
    for w in cfg_report.warnings:
        print(f"[config][warn] {w}")
    for e in cfg_report.errors:
        print(f"[config][error] {e}")

    registries = RegistryLoader(Path(args.repo_root) if args.repo_root else None)

    artifacts = load_run_artifacts(args.run_dir)
    for note in artifacts.coverage_notes:
        print(f"[artifacts] {note}")

    appendix = AppendixNotes()

    lenses = _lens_bundle(artifacts, registries, args.as_of_date,
                          pipeline_root=args.pipeline_root)
    # Carry forward the loader's coverage notes, but drop its thin-18a "no
    # pipeline tape" note when the pipeline lens actually resolved (via the
    # cross-run rich-source discovery) so the appendix isn't self-contradictory.
    for note in artifacts.coverage_notes:
        if lenses.get("pipeline") is not None and "no pipeline tape" in note.lower():
            continue
        appendix.add(note)

    as_of = args.as_of_date or (lenses["funded"].as_of_date if lenses["funded"] else "") or ""
    if not artifacts.has_tape:
        appendix.add("No canonical typed tape resolved — deck rendered with "
                     "branded placeholders throughout.")
    # Key the pipeline note on whether the pipeline LENS actually resolved (via
    # the rich source discovery), not on whether the thin 18a artifact was found.
    if lenses.get("pipeline") is None:
        appendix.add("No pipeline source resolved for this run — pipeline & "
                     "forecast lenses render as branded placeholders.")

    # Prior-period lenses (for MoM deltas), optional.
    prior_lenses: Dict[str, Optional[ResolvedData]] = {}
    prior_label = ""
    if args.prior_run_dir:
        prior_art = load_run_artifacts(args.prior_run_dir)
        prior_lenses = _lens_bundle(prior_art, registries, None,
                                    pipeline_root=args.pipeline_root)
        pdate = prior_lenses["funded"].as_of_date if prior_lenses["funded"] else ""
        prior_label = f"prior run {Path(args.prior_run_dir).name}" + (
            f" (as-of {pdate})" if pdate else "")

    analytics = {k: artifacts.artifact(k) for k in
                 ("analytics", "metrics", "validation", "risk_monitor")}
    metric_resolver = MetricResolver(lenses, registries, analytics=analytics,
                                     prior_lenses=prior_lenses, default_lens="funded")
    strapline_metrics = {k: metric_resolver.resolve(config.metric_spec(k))
                         for k in config.metrics.keys()}
    strapline_resolver = StraplineResolver(
        metrics=strapline_metrics, llm_artifact=artifacts.artifact("straplines"))

    output = args.output or _default_output(args.client_name)
    work_dir = args.work_dir or str(Path(output).with_suffix("")) + "_charts"
    chart_resolvers: Dict[str, Optional[ChartResolver]] = {}
    for lens in ("funded", "pipeline", "forecast"):
        data = lenses.get(lens)
        chart_resolvers[lens] = (
            ChartResolver(data, registries, work_dir, theme=THEME, lens=lens)
            if (data is not None and data.df is not None and not data.df.empty)
            else None)

    source_artifacts: List[str] = []
    if artifacts.tape_path:
        source_artifacts.append(artifacts.tape_path.name)
    if artifacts.pipeline_tape_path:
        source_artifacts.append(artifacts.pipeline_tape_path.name)
    source_artifacts += sorted(p.name for k, p in artifacts.json_paths.items()
                               if k != "run_state")

    ctx = BuildContext(
        client_name=args.client_name, as_of_date=as_of, run_dir=str(args.run_dir),
        lens=args.lens or config.default_lens, consolidated=args.consolidated,
        logo_path=config.logo_path, prior_label=prior_label,
        source_artifacts=source_artifacts)

    builder = DeckBuilder(config, ctx, metric_resolver, chart_resolvers,
                          strapline_resolver, appendix, theme=THEME)
    build_report = builder.build(output)

    build_validation = validate_build(build_report)
    print(f"\nDeck: {build_report['output']}")
    print(f"Slides: {len(build_report['slides'])}")
    print(f"Validation: {build_validation.summary()}")
    for e in build_validation.errors:
        print(f"[build][error] {e}")
    for w in build_validation.warnings:
        print(f"[build][warn] {w}")
    if build_report["coverage_notes"]:
        print(f"Coverage notes: {len(build_report['coverage_notes'])} (see appendix).")
    return 0 if build_validation.ok else 2


def main() -> None:  # pragma: no cover
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
