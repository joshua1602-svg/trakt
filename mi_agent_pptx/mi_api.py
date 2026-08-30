"""mi_agent_pptx.mi_api — headless bridge to the MI Agent API computations.

Produces the SAME payloads the React dashboard renders, by calling the exact
compute functions behind the ``/mi/*`` endpoints in-process (no HTTP server, no
LLM, and — deliberately — no FastAPI import, so the deck runs anywhere the
compute modules ship, including the Azure Functions PPTX stage).

Resolution parity is the point: rather than a PPTX-only guesser, the deck resolves
a run exactly as the dashboard does and then calls the identical compute functions.

Historical (multi-period) resolution is covered for both deployments:

* **Azure / blob roots** — ``MI_AGENT_ONBOARDING_OUTPUT_ROOT`` = a ``blob://``
  platform root: funded evolution loads the dated platform canonicals
  (``evolution.funded_frames`` blob branch); ``MI_AGENT_PIPELINE_ROOT`` = a
  ``blob://`` pipeline root is mirrored locally (``_materialise_pipeline_root``, the
  same mirror ``app._pipeline_discovery_root`` performs) so pipeline evolution /
  funnel / run-rate projection discover every dated weekly snapshot.
* **Local downloaded history** — a filesystem root carrying dated cuts
  (``…/{YYYY-MM-DD}/platform_canonical_typed.csv`` for funded,
  ``…/{YYYY-MM-DD}/…pipeline…`` for weekly extracts) is discovered directly; the
  historical cuts do NOT need to live inside the current run directory.
* **Single local run** — one funded cut / one weekly extract: the time-series
  surfaces report ``singlePeriod`` and the deck renders an *insufficient history*
  placeholder (not "data unavailable").

Every resolution is recorded in :attr:`DashboardData.diagnostics` for the deck's
data-coverage appendix (current sources, history roots checked, dated-cut counts,
and the placeholder reason per time-series slide).
"""

from __future__ import annotations

import json
import os
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .deck_context import DeckPortfolioContext

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"


@dataclass
class DashboardData:
    """The dashboard payloads for one run, plus resolution provenance."""

    client_id: str
    run_id: str
    reporting_date: Optional[str] = None
    funded: Dict[str, Any] = field(default_factory=dict)          # /mi/snapshot
    pipeline: Dict[str, Any] = field(default_factory=dict)        # /mi/pipeline/snapshot
    forecast: Dict[str, Any] = field(default_factory=dict)        # /mi/forecast/snapshot
    funded_evolution: Dict[str, Any] = field(default_factory=dict)
    pipeline_evolution: Dict[str, Any] = field(default_factory=dict)
    funnel: Dict[str, Any] = field(default_factory=dict)
    forecast_evolution: Dict[str, Any] = field(default_factory=dict)
    cohorts: Dict[str, Any] = field(default_factory=dict)
    geo: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    #: Governed concentration-test envelope (/mi/concentration-tests): current,
    #: expected-forecast and full-pipeline-stress states per approved test.
    concentration: Dict[str, Any] = field(default_factory=dict)
    extrapolation: Dict[str, Any] = field(default_factory=dict)
    multidim: Dict[str, Any] = field(default_factory=dict)
    cohort_progression: Dict[str, Any] = field(default_factory=dict)
    #: Per-vintage static-pool series, keyed by vintage — one governed
    #: ``funded_cohort_progression`` call each, exactly as the React
    #: Cohorts tab issues when a user selects a vintage.
    cohort_series: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    #: The governed portfolio context this deck describes (scope, constituent
    #: books, per-book reporting dates, per-type funded snapshots). ``None`` only
    #: when no funded dataset resolved at all.
    portfolio: Optional["DeckPortfolioContext"] = None
    #: The deterministic executive summary (see :mod:`mi_agent_pptx.insights`).
    insights: Dict[str, Any] = field(default_factory=dict)
    #: Governed movement attribution per dimension (evolution.funded_bridge),
    #: computed ONCE per scope and shared by every slide that narrates movement.
    movement: Dict[str, Any] = field(default_factory=dict)
    #: Deterministic watch items (see :mod:`mi_agent_pptx.watchlist`).
    watchlist: Dict[str, Any] = field(default_factory=dict)
    #: The ECONOMIC opening-to-closing bridge (evolution.funded_balance_movement):
    #: opening + new - exits + movement on continuing = closing, with the exit
    #: leg split on evidence. Distinct from ``movement``, which attributes the
    #: same net change across dimensions.
    balance_movement: Dict[str, Any] = field(default_factory=dict)
    #: Per-constituent-book forward view (forecast_bridge.portfolio_projections):
    #: current balance, expected originations, governed run-off retention where
    #: the client supplied a curve, and the disclosure where they did not.
    portfolio_projections: Dict[str, Any] = field(default_factory=dict)
    #: Utilisation history per approved concentration test across governed
    #: snapshots (``/mi/concentration-tests/history``). Empty when no approved
    #: configuration exists or no history resolves.
    concentration_history: Dict[str, Any] = field(default_factory=dict)
    #: What Trakt can and cannot report for THIS portfolio, from the published
    #: capability registry (``trakt_core.capability``) — ``metric id ->
    #: Availability``. This is how the pack stays asset-agnostic: it asks
    #: whether a capability resolves for this book's canonical shape, never
    #: whether the book is a particular asset class. Discovery reads columns
    #: and two enum mixes; it computes none of the metrics it describes.
    capabilities: Dict[str, Any] = field(default_factory=dict)
    #: The GOVERNED reporting currency for this book, resolved through
    #: ``mi_agent_api.currency`` exactly as the dashboard resolves it for a
    #: request. The deck never picks a currency of its own.
    currency_code: str = "GBP"

    def note(self, msg: str) -> None:
        if msg and msg not in self.notes:
            self.notes.append(msg)


def _guard_ctx(data: DashboardData, registry, scope, cid, tenant_id,
               snapshot, type_snaps, reporting_date):
    """Assemble the deck's governed portfolio context, never raising."""
    if registry is None or scope is None:
        return None
    try:
        from .deck_context import build_context
        return build_context(
            tenant_id=tenant_id or cid, client_id=cid, registry=registry,
            scope=scope, snapshot=snapshot, type_snapshots=type_snaps,
            reporting_date=reporting_date)
    except Exception as exc:  # noqa: BLE001
        data.note(f"portfolio_context: {type(exc).__name__}: {exc}")
        return None


def _guard(note_target: DashboardData, label: str,
           fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Run *fn*, returning ``{}`` and recording a note on failure."""
    try:
        out = fn()
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001 — a missing payload must not fail the deck
        note_target.note(f"{label}: {type(exc).__name__}: {exc}")
        return {}


def _run_ids(run_dir: Path, client_id: Optional[str], run_id: Optional[str]):
    """Resolve (client_id, run_id) from run_state.json / the run dir name."""
    rs: Dict[str, Any] = {}
    p = run_dir / "run_state.json"
    if p.exists():
        try:
            rs = json.loads(p.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            rs = {}
    cid = client_id or rs.get("client_id") or "client"
    rid = run_id or rs.get("run_id") or run_dir.name
    return str(cid), str(rid), rs


def _funded_canonical(run_path: Path) -> Optional[str]:
    """The run's funded platform canonical (the funded book the dashboard serves)."""
    try:
        from .artifact_loader import load_run_artifacts
        art = load_run_artifacts(run_path)
        if art.has_tape and art.tape_path is not None:
            return str(art.tape_path)
    except Exception:  # noqa: BLE001
        pass
    conventional = run_path / "out_platform" / _PLATFORM_CANONICAL_NAME
    return str(conventional) if conventional.exists() else None


@contextmanager
def _api_env(overrides: Dict[str, Optional[str]]):
    """Temporarily set the env the MI Agent data layer reads, then restore it — so
    the deck resolves a run exactly as the dashboard does, without leaking config.

    Scope note: this is a BATCH stage (one deck per pipeline run), not a
    per-request path. The governed MI capability never mutates the environment to
    select a dataset. Removing this last use requires threading an explicit
    dataset selector through ``mi_agent_api.data_source``, which is tracked as
    follow-up work rather than done here because it would change resolution for
    every caller at once.
    """
    saved = {k: os.environ.get(k) for k in overrides}

    def _reset_cache():
        try:
            from mi_agent_api import data_source
            data_source.reset_cache()
        except Exception:  # noqa: BLE001
            pass

    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _reset_cache()
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _reset_cache()


# --------------------------------------------------------------------------- #
# Resolution helpers.
#
# These used to be hand-copied from the like-named private helpers in
# mi_agent_api.app, because that module pulls in FastAPI and could not be
# imported here. Dataset resolution now lives in the interface-neutral
# mi_agent_api.datasets, so the deck calls the SAME implementation the API does
# instead of a drifting copy.
# --------------------------------------------------------------------------- #

def _materialise_pipeline_root(root: Optional[str]) -> Optional[str]:
    """A LOCAL discovery root for ``root`` (blob roots are mirrored to scratch)."""
    from mi_agent_api import datasets as _ds
    return _ds._materialise_pipeline_root(root)


def _pipeline_discovery_root(out_root: str) -> str:
    """The pipeline discovery root, blob-mirrored to local where needed.

    Falls back to the deck's own run root when no pipeline root is configured —
    the one behavioural difference from the API helper, which has no run root.
    """
    root = os.environ.get("MI_AGENT_PIPELINE_ROOT") or out_root
    return _materialise_pipeline_root(root) or root


def _pipeline_history(root: str, client_id: str) -> Optional[Dict[str, Any]]:
    """The multi-week historical completion model: None unless ≥2 weekly
    extracts exist (a single extract is not a history)."""
    from mi_agent_api import pipeline_contract as pc
    try:
        model = pc.build_pipeline_history(root, client_id)
    except Exception:  # noqa: BLE001
        return None
    if int((model or {}).get("uniqueWeeklyExtractsUsed", 0)) < 2:
        return None
    return model


def _kfi_lag_weeks(model: Optional[Dict[str, Any]]) -> Optional[int]:
    """Median KFI→completion lag in whole weeks (``app._kfi_lag_weeks_from_model``)."""
    timing = ((model or {}).get("historicalCompletionTimingByStage") or {}).get("KFI") or {}
    median_days = timing.get("medianDays")
    return max(1, round(float(median_days) / 7.0)) if median_days else None


def _resolve_scope(df: Optional[pd.DataFrame], context_id: Optional[str],
                   client_id: Optional[str]):
    """``(registry, scope)`` for a frame and a requested portfolio context.

    Uses the SAME governed resolver React and Copilot use, so an investor pack can
    never disagree with the workspace about what a book contains.
    """
    from mi_agent.portfolio_scope import resolve
    return resolve(df, context_id, client_id=client_id)


def _scoped_to(df: Optional[pd.DataFrame], cid: str,
               context_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Narrow a canonical frame to a governed portfolio context.

    ``context_id`` is the explicit governed scope (``total`` / ``direct`` /
    ``acquired`` / a ``source_portfolio_id``). When it is ``None`` the legacy
    behaviour applies and ``cid`` itself is used as the context — a client id
    resolves to the consolidated platform, while ``direct``/``acquired`` passed
    as ``--client-id`` continue to narrow exactly as before.
    """
    requested = context_id if context_id is not None else cid
    if df is None or getattr(df, "empty", True) or not requested:
        return df
    try:
        from mi_agent.portfolio_scope import apply_scope
        _registry, scope = _resolve_scope(df, requested, cid)
        if scope.is_total:
            return df
        scoped = apply_scope(df, scope)
        return scoped if scoped is not None and len(scoped) else df
    except Exception:  # noqa: BLE001 - deck generation must never break on scope
        return df


def _funded_frame(cid: str, context_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """The prepared funded frame for the active run (the platform canonical set via
    ``MI_AGENT_PLATFORM_CANONICAL``), scoped to the governed portfolio context.

    The frame goes through the SAME ``prepare_funded_mi_dataset`` the dashboard
    applies (and that ``_local_funded_frames`` already applies to the prior
    periods). Without it the CURRENT period lacked derivations the prior periods
    had — ``months_on_book``, ``vintage_year``, ``time_on_book_bucket`` — so the
    deck silently dropped the weighted-months-on-book measure the dashboard
    shows, and the two channels disagreed on how many KPIs the book even has.
    The prep only fills columns that are absent, so a frame that arrives already
    prepared is unchanged.
    """
    from mi_agent_api import data_source
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset
    df = data_source.get_dataframe()
    if df is None or df.empty:
        return None
    try:
        df, _report = prepare_funded_mi_dataset(df)
    except Exception:  # noqa: BLE001 — an un-preppable tape still renders a deck
        pass
    return _scoped_to(df, cid, context_id)


def _unscoped_funded_frame() -> Optional[pd.DataFrame]:
    """The active funded frame BEFORE portfolio narrowing (for registry build)."""
    from mi_agent_api import data_source
    df = data_source.get_dataframe()
    return None if df is None or df.empty else df


def _type_snapshots(funded_df, registry, scope, semantics, *, cid: str, rid: str,
                    reporting_date, prior_df, prior_rid, prior_rd,
                    data: DashboardData) -> Dict[str, Dict[str, Any]]:
    """A governed funded snapshot per portfolio TYPE inside the scope.

    Computed with the SAME ``compute_funded_snapshot`` the total uses, over the
    frame narrowed to each type — so a type slice and the total can never be
    computed differently. Only produced when the scope spans more than one type;
    a single-book deck costs nothing extra.
    """
    from mi_agent.portfolio_scope import apply_scope
    from mi_agent_api import snapshots as snap
    from trakt_core.portfolio import resolve_scope

    types = sorted({str(t).lower() for t in (getattr(scope, "portfolio_types", ()) or ())
                    if t})
    if len(types) < 2:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for ptype in types:
        try:
            tscope = resolve_scope(registry, ptype)
            tdf = apply_scope(funded_df, tscope)
            if tdf is None or tdf.empty:
                continue
            tprior = apply_scope(prior_df, tscope) if prior_df is not None else None
            out[ptype] = snap.compute_funded_snapshot(
                tdf, semantics, client_id=cid, run_id=rid,
                reporting_date=reporting_date, prior_df=tprior,
                prior_run_id=prior_rid, prior_reporting_date=prior_rd,
                scope=tscope)
        except Exception as exc:  # noqa: BLE001 — one type must not break the deck
            data.note(f"type_snapshot[{ptype}]: {type(exc).__name__}: {exc}")
    return out


def _pipeline_client(prow, cid: str) -> str:
    """The client the governed pipeline sources actually live under.

    Prefer the funded client (``app._resolve_pipeline_source`` passes it, and in
    production the pipeline root carries the client in its path). When strict
    path-inferred matching finds nothing under that client — a local run layout
    where the pipeline tree is keyed by ``direct_001`` / the run folder rather than
    the funded client — fall back to the client discovery infers from the tree."""
    from mi_agent_api import pipeline_contract as pc
    try:
        if pc.resolve_pipeline_source(prow, cid, None):
            return cid
    except Exception:  # noqa: BLE001
        pass
    try:
        srcs = pc.discover_pipeline_sources(prow)  # client_id=None → all, inferred
        if srcs:
            return srcs[-1].get("client_id") or cid
    except Exception:  # noqa: BLE001
        pass
    return cid


# --------------------------------------------------------------------------- #
# Dated-cut discovery for the diagnostics + local (downloaded) history support.
# --------------------------------------------------------------------------- #

def _dated_funded_cuts(out_root: str, cid: str) -> List[Tuple[str, str]]:
    """``[(date, uri_or_path)]`` for every dated funded platform canonical under
    *out_root* — the dashboard's blob cuts, or a local downloaded history tree
    (``…/{YYYY-MM-DD}/platform_canonical_typed.csv``). Oldest → newest."""
    root = str(out_root)
    if str(root).startswith("blob://"):
        try:
            from apps.blob_trigger_app.storage import open_storage
            from mi_agent_api import platform_snapshots_blob as pb
            dated = pb.list_dated_platform_canonicals(root, open_storage())
            return [(d["date"], d["uri"]) for d in dated]
        except Exception:  # noqa: BLE001
            return []
    cuts: Dict[str, str] = {}
    base = Path(root)
    if base.exists():
        for p in base.glob(f"**/{_PLATFORM_CANONICAL_NAME}"):
            date = p.parent.name if _DATE_RE.fullmatch(p.parent.name) else None
            if not date:
                m = _DATE_RE.search(str(p))
                date = m.group(1) if m else None
            if date:
                cuts.setdefault(date, str(p))
    return sorted(cuts.items())


def _local_funded_frames(cuts: List[Tuple[str, str]], cid: str,
                         context_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Prepared funded frames from LOCAL dated platform canonicals, scoped to the
    client — the local analogue of ``platform_snapshots_blob.build_funded_evolution_frames``."""
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset
    frames: List[Dict[str, Any]] = []
    for date, path in cuts:
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        raw = _scoped_to(raw, cid, context_id)
        try:
            df, _rep = prepare_funded_mi_dataset(raw)
        except Exception:  # noqa: BLE001
            continue
        frames.append({"run_id": date, "reporting_date": date, "df": df, "source": path})
    return frames


def _prior_funded(cuts: List[Tuple[str, str]], cid: str, reporting_date: Optional[str],
                  context_id: Optional[str] = None):
    """The prepared funded frame for the reporting period BEFORE *reporting_date*
    (the most recent dated cut strictly earlier), for month-on-month KPI deltas."""
    if not cuts:
        return None, None, None
    earlier = [(d, p) for d, p in cuts if not reporting_date or d < str(reporting_date)]
    if not earlier:
        return None, None, None
    frames = _local_funded_frames([earlier[-1]], cid, context_id)
    if not frames:
        return None, None, None
    d, _p = earlier[-1]
    return frames[0]["df"], d, d


# --------------------------------------------------------------------------- #
# Deck-specific funded enrichments (same stratify / banding engine as the API).
# --------------------------------------------------------------------------- #

_BALANCE = "current_outstanding_balance"


# NOTE: ``_ltv_series`` / ``_age_series`` / ``_broker_series`` /
# ``_borrower_type_series`` / ``_ticket_series`` / ``_stratify_dim`` /
# ``_extra_stratifications`` USED TO LIVE HERE. They gave the
# deck three stratifications the dashboard did not have, picked their own source
# columns, and — for ticket size — carried bin edges that contradicted
# ``config/mi/buckets.yaml``. That made the renderer a second owner of an
# economic definition.
#
# All three dimensions are now declared in ``mi_agent_api.snapshots._STRAT_DIMS``
# and computed by the same engine as every other stratification, so they arrive
# on the governed snapshot payload like the rest and the deck simply draws them.
# Nothing was relocated into a parallel PPTX helper: the code is gone, and the
# capability is in the layer that already owned the other eight dimensions.


def _multidim(df: pd.DataFrame, scope=None) -> Dict[str, Any]:
    """The governed multi-dimensional cross-tabs for this book.

    ``_matrix`` and the pair list used to live here, which made the deck the only
    owner of a grouping the React product could not reach. Both now live in
    ``mi_agent_api.snapshots`` (``cross_tab`` / ``multidimensional``, served at
    ``/mi/multidim``), so the dashboard and the pack consume one analytical
    result with one set of axis orders.
    """
    from mi_agent_api import snapshots as snap
    return snap.multidimensional(df, scope)


def _cohort_progression(out_root: str, cid: str, *,
                        vintage: Optional[str] = None,
                        lens_filters: Optional[Dict[str, str]] = None,
                        lens_label: str = "Total") -> Dict[str, Any]:
    """Static-pool cohort progression across reporting periods — the dashboard's
    ``/mi/cohorts/progression``, called with the same arguments the React Cohorts
    tab sends when a user picks a vintage and a scope."""
    from mi_agent_api import evolution
    return evolution.funded_cohort_progression(
        out_root, cid, grain="Y", vintage=vintage,
        lens_filters=lens_filters, lens_label=lens_label)


def _pipeline_extract_count(root: str, cid: str) -> int:
    """Number of dated weekly pipeline extracts discoverable under *root* for the
    client (the dashboard's ``weekly_extract_inventory``)."""
    from mi_agent_api import pipeline_contract as pc
    try:
        return int(len(pc.weekly_extract_inventory(root, cid).get("extracts", [])))
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def build_dashboard_data(
    run_dir: str | Path,
    *,
    client_id: Optional[str] = None,
    run_id: Optional[str] = None,
    as_of: Optional[str] = None,
    output_root: Optional[str] = None,
    pipeline_root: Optional[str] = None,
    prior_run_dir: Optional[str] = None,  # accepted for CLI compatibility (unused)
    portfolio_context: Optional[str] = None,
    tenant_id: Optional[str] = None,
    scale_targets: Optional[List[float]] = None,
) -> DashboardData:
    """Compute the full set of dashboard payloads for *run_dir*, headless.

    ``portfolio_context`` is the governed scope the deck reports on (``total`` /
    ``direct`` / ``acquired`` / a ``source_portfolio_id``). ``None`` preserves the
    pre-existing behaviour exactly: the scope is derived from ``client_id``.
    """
    run_path = Path(run_dir)
    cid, rid, rs = _run_ids(run_path, client_id, run_id)
    data = DashboardData(client_id=cid, run_id=rid)
    pid = f"{cid}/{rid}"

    funded_uri = _funded_canonical(run_path)
    out_root = (output_root or os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
                or str(run_path.parent))
    pipe_root_env = pipeline_root or os.environ.get("MI_AGENT_PIPELINE_ROOT")
    overrides: Dict[str, Optional[str]] = {
        "MI_AGENT_ONBOARDING_OUTPUT_ROOT": out_root,
        "MI_AGENT_PIPELINE_ROOT": pipe_root_env or str(run_path.parent),
        "MI_AGENT_CLIENT_ID": cid,
    }
    if funded_uri:
        overrides["MI_AGENT_PLATFORM_CANONICAL"] = funded_uri

    with ExitStack() as stack, _api_env(overrides):
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        from mi_agent_api import snapshots as snap

        try:
            semantics = load_mi_semantics(semantics_path())
        except Exception as exc:  # noqa: BLE001
            data.note(f"semantics: {exc}")
            semantics = {}

        prow = _pipeline_discovery_root(out_root)
        pipe_cid = _pipeline_client(prow, cid)
        history = _pipeline_history(prow, pipe_cid)

        funded_df = _funded_frame(cid, portfolio_context)
        reporting_date = as_of or rs.get("reporting_date")
        if funded_df is not None and not funded_df.empty:
            try:
                reporting_date = reporting_date or snap.infer_reporting_date(rid, funded_df)
            except Exception:  # noqa: BLE001
                pass
        data.reporting_date = reporting_date

        # -- Governed reporting currency -------------------------------
        # The same resolution the API performs per request
        # (``datasets._apply_request_currency``): approved client configuration
        # outranks the tape, the tape outranks the platform default.
        #
        # It is ENTERED here, not merely recorded, because the figures are
        # formatted downstream of this point: ``compute_funded_snapshot`` writes
        # each KPI tile's display string through ``currency.format_money``, and
        # the insight/watchlist generators write prose money. Both must run with
        # the book's currency in force or the deck says GBP while the dashboard
        # says EUR. ``stack`` closes at the end of the function, so the process
        # is left exactly as it was found.
        try:
            from mi_agent_api import currency as _currency
            data.currency_code = _currency.resolve_currency_code(
                funded_df, client_id=cid)
            stack.enter_context(_currency.use_currency(data.currency_code))
        except Exception as exc:  # noqa: BLE001 - never break a deck on currency
            data.note(f"currency: {type(exc).__name__}: {exc}")

        funded_cuts = _dated_funded_cuts(out_root, cid)
        prior_df, prior_rid, prior_rd = _prior_funded(funded_cuts, cid, reporting_date,
                                                      portfolio_context)

        # -- Governed portfolio context (scope + constituent books) -------
        # Resolved from the UNSCOPED frame so the registry sees every book, then
        # narrowed to the requested context — the same two-step the API performs.
        registry = scope = None
        try:
            registry, scope = _resolve_scope(
                _unscoped_funded_frame(),
                portfolio_context if portfolio_context is not None else cid, cid)
            # A plain client id is not a *requested* scope: it means Total. Don't
            # report a fallback the caller never asked for.
            if portfolio_context is None and getattr(scope, "fell_back_to_total", False):
                scope = _resolve_scope(_unscoped_funded_frame(), None, cid)[1]
        except Exception as exc:  # noqa: BLE001 — scope must never break the deck
            data.note(f"portfolio_scope: {type(exc).__name__}: {exc}")

        # -- FUNDED snapshot (KPIs + stratifications) --------------------
        if funded_df is not None and not funded_df.empty:
            data.funded = _guard(data, "funded_snapshot", lambda: snap.compute_funded_snapshot(
                funded_df, semantics, client_id=cid, run_id=rid,
                reporting_date=reporting_date, prior_df=prior_df,
                prior_run_id=prior_rid, prior_reporting_date=prior_rd,
                scope=scope))
            data.multidim = _guard(data, "multidim",
                                   lambda: _multidim(funded_df, scope))
            data.cohorts = _guard(data, "cohorts",
                                  lambda: _cohorts(funded_df, cid, pid, reporting_date))
            # Static-pool seasoning, ONE governed call per cohort the deck will
            # draw — the same call the React Cohorts tab makes when a user picks
            # a vintage. The Total series is kept for the appendix; the per-
            # vintage series are what the seasoning slide plots.
            data.cohort_progression = _guard(data, "cohort_progression",
                                             lambda: _cohort_progression(out_root, cid))
            data.cohort_series = _guard(data, "cohort_series",
                                        lambda: _cohort_series(out_root, cid, scope,
                                                               data))
            data.geo = _guard(data, "geo", lambda: _geo(funded_df, cid, rid))
            # Per-type governed snapshots (only when the scope spans >1 type).
            type_snaps = _type_snapshots(
                funded_df, registry, scope, semantics, cid=cid, rid=rid,
                reporting_date=reporting_date, prior_df=prior_df,
                prior_rid=prior_rid, prior_rd=prior_rd, data=data)
            data.portfolio = _guard_ctx(data, registry, scope, cid, tenant_id,
                                        data.funded, type_snaps, reporting_date)
        else:
            data.note("No funded dataset resolved for this run — funded sections "
                      "are omitted from the deck.")

        # -- PIPELINE snapshot (latest governed weekly extract) ----------
        pipe_df, pipe_report, source = _pipeline(prow, pipe_cid, rid, semantics, history, data)

        # -- FORECAST snapshot (funded + weighted pipeline bridge) -------
        data.forecast = _guard(data, "forecast", lambda: _forecast(
            cid, rid, reporting_date, funded_df, pipe_df, pipe_report,
            data.pipeline, source))

        # -- Multi-run EVOLUTION / FUNNEL / FORECAST ---------------------
        data.funded_evolution = _guard(
            data, "funded_evolution",
            lambda: _funded_evo(out_root, cid, rid, funded_cuts, scope,
                                portfolio_context))
        data.pipeline_evolution = _guard(data, "pipeline_evolution",
                                         lambda: _pipeline_evo(prow, pipe_cid, history))
        data.funnel = _guard(data, "funnel", lambda: _funnel(prow, pipe_cid, history))
        data.forecast_evolution = _guard(data, "forecast_evolution",
                                         lambda: _forecast_evo(out_root, prow, cid, rid, history))

        # -- RISK limits / FORECAST extrapolation (multi-run) ------------
        # Concentration first: when no operator-approved configuration exists it
        # falls back to the extracted limit monitor INTERNALLY and returns it in
        # the same envelope. Calling risk_limits again here would run the
        # identical governed computation a second time for the same scope and
        # date, so it is only resolved when concentration produced nothing.
        data.concentration = _guard(data, "concentration",
                                    lambda: _concentration(out_root, cid, rid, scope))
        # Movement: how each approved test's utilisation has travelled. Only
        # asked for when there ARE approved tests — the history service would
        # otherwise repeat the same "no approved configuration" answer.
        if data.concentration.get("tests"):
            data.concentration_history = _guard(
                data, "concentration_history",
                lambda: _concentration_history(out_root, cid, rid, scope))
        # Only when the concentration service itself could not run — it consults
        # the extracted monitor internally, so any other path would repeat it.
        if not data.concentration:
            data.risk = _guard(data, "risk", lambda: _risk(out_root, cid, rid))
        data.extrapolation = _guard(
            data, "extrapolation",
            lambda: _extrapolation(out_root, prow, cid, rid, history,
                                   scale_targets))

        # -- Movement attribution (governed bridge, once per dimension) ------
        data.movement = _guard(data, "movement",
                               lambda: _movement(out_root, cid, rid, scope, data,
                                                 prior_reporting_date=prior_rd))

        # -- ECONOMIC movement: what happened to the LOANS -------------------
        # The same governed composition ``/mi/evolution/funded-movement`` serves.
        data.balance_movement = _guard(
            data, "balance_movement",
            lambda: _balance_movement(out_root, cid, rid, scope, prior_rd))

        # -- Per-book forward view -------------------------------------------
        # Already served at /mi/forecast/snapshot and rendered by nothing.
        data.portfolio_projections = _guard(
            data, "portfolio_projections",
            lambda: _portfolio_projections(funded_df, registry, scope, data))

        # -- WHAT THIS BOOK SUPPORTS -----------------------------------------
        # Resolved AFTER the funded history, because several capabilities turn
        # on how many governed snapshots exist rather than on any column.
        data.capabilities = _guard(
            data, "capabilities",
            lambda: _capabilities(funded_df, data.funded_evolution)) or {}

        pipe_snapshots = _pipeline_extract_count(prow, pipe_cid)

    # -- Deterministic executive summary (no LLM) ------------------------
    # Built last: every generator reads an already-resolved governed payload.
    # Re-entered under the book's currency because these generators write prose
    # money (``insight_generators.money``) and the scope above has closed.
    from mi_agent_api import currency as _currency_tail
    with _currency_tail.use_currency(data.currency_code):
        data.insights = _guard(data, "insights", lambda: _insights(data))
        data.watchlist = _guard(data, "watchlist", lambda: _watchlist(data))

    # Provenance + diagnostics ------------------------------------------
    if source and source.get("source_file"):
        data.source_files.append(Path(source["source_file"]).name)
    data.diagnostics = _diagnostics(data, out_root, prow, funded_uri, source,
                                    funded_cuts, pipe_snapshots)

    if not data.pipeline:
        data.note("No pipeline source resolved — pipeline & forecast sections "
                  "are omitted from the deck.")
    return data


def _diagnostics(data, out_root, prow, funded_uri, source, funded_cuts, pipe_snapshots):
    """The data-coverage provenance the appendix renders (requirement #4)."""
    def _pph(payload, min_periods=2):
        periods = len(payload.get("periods", []))
        single = bool(payload.get("singlePeriod")) or periods < min_periods
        return periods, single

    f_periods, f_single = _pph(data.funded_evolution)
    p_periods, p_single = _pph(data.pipeline_evolution)
    proj = ((data.extrapolation.get("completionRunRateForecast") or {}).get("available")
            or (data.extrapolation.get("kfiConversionForecast") or {}).get("available"))
    risk_ok = bool(data.risk.get("available", False)) or bool(data.risk.get("tests"))
    pipeline_ok = bool(data.pipeline)
    return {
        "fundedCurrentSource": Path(funded_uri).name if funded_uri else None,
        "pipelineCurrentSource": (Path(source["source_file"]).name
                                  if source and source.get("source_file") else None),
        "fundedHistoryRoot": out_root,
        "fundedCutsFound": len(funded_cuts),
        "pipelineHistoryRoot": prow,
        "pipelineSnapshotsFound": pipe_snapshots,
        "timeSeries": {
            "funded_evolution": {
                "placeholder": f_single,
                "reason": (f"insufficient history — {len(funded_cuts)} funded cut(s) "
                           f"found, need ≥2" if f_single else None),
                "periods": f_periods},
            "pipeline_evolution": {
                "placeholder": p_single,
                "reason": (f"insufficient history — {pipe_snapshots} weekly extract(s) "
                           f"found, need ≥2" if p_single else None),
                "periods": p_periods},
            "funnel": {
                "placeholder": not pipeline_ok,
                "reason": ("current-week funnel shown (single weekly extract)"
                           if pipeline_ok and pipe_snapshots < 2 else
                           (None if pipeline_ok else "no pipeline source resolved"))},
            "forecast_projection": {
                "placeholder": not proj,
                "reason": (None if proj else
                           f"insufficient run-rate history — {pipe_snapshots} weekly "
                           f"extract(s) found")},
            "risk": {
                "placeholder": not risk_ok,
                "reason": (None if risk_ok else "no Schedule 8 risk-limit extract")},
        },
    }


# --------------------------------------------------------------------------- #
# Per-endpoint compute wrappers (call the SAME functions app.py's handlers call).
# --------------------------------------------------------------------------- #

def _resolve_pipeline_source(prow, cid, rid, data: DashboardData):
    """The governed pipeline source, resolved EXACTLY as the dashboard resolves it.

    The deck used to call ``pipeline_contract.resolve_pipeline_source(root, …)``
    directly, which only ever performs discovery under a root. The API resolves
    through ``datasets._resolve_pipeline_source``, which first honours the
    durable weekly snapshot pointer (``MI_AGENT_PIPELINE_URI``) and an explicit
    ``MI_AGENT_PIPELINE_SOURCE`` before falling back to that same discovery.

    In a deployment where the pipeline is published as a durable snapshot rather
    than a discoverable dated tree, the old path found nothing and the whole
    pipeline section silently vanished from the investor pack while the
    dashboard showed it. Using the shared resolver removes that divergence by
    construction; the root-based call remains as the fallback so an explicitly
    supplied ``--pipeline-root`` still wins where discovery is the deployment.
    """
    from mi_agent_api import pipeline_contract as pc
    try:
        from mi_agent_api import datasets as _ds
        source = _ds._resolve_pipeline_source(cid, rid)
        if source:
            return source
    except Exception as exc:  # noqa: BLE001 — fall through to root discovery
        data.note(f"pipeline_source(shared): {type(exc).__name__}: {exc}")
    try:
        return pc.resolve_pipeline_source(prow, cid, rid)
    except Exception as exc:  # noqa: BLE001
        data.note(f"pipeline_source: {exc}")
        return None


def _pipeline(prow, cid, rid, semantics, history, data: DashboardData):
    """Resolve + snapshot the latest governed weekly pipeline extract."""
    from mi_agent_api import pipeline_contract as pc
    source = _resolve_pipeline_source(prow, cid, rid, data)
    if not source:
        return None, None, None
    try:
        pdf, report = pc.load_prepared_pipeline(source, historical_model=history)
        try:
            prior_week = pc.compute_prior_week_aggregates(source, historical_model=history)
        except Exception:  # noqa: BLE001
            prior_week = None
        snap_out = pc.compute_pipeline_snapshot(
            pdf, report, semantics, client_id=source.get("client_id", cid),
            run_id=rid, source=source, prior_week=prior_week)
        if snap_out.get("pipelineRowCount"):
            data.pipeline = snap_out
        return pdf, report, source
    except Exception as exc:  # noqa: BLE001
        data.note(f"pipeline_snapshot: {exc}")
        return None, None, source


def _forecast(cid, rid, reporting_date, funded_df, pipe_df, pipe_report, pipe_snap, source):
    from mi_agent_api import forecast_bridge as fb
    from mi_agent_api import workspace
    env = fb.compute_forecast_bridge(
        client_id=cid, run_id=rid, funded_reporting_date=reporting_date,
        funded_df=funded_df, pipeline_df=pipe_df, pipeline_report=pipe_report,
        pipeline_snapshot=(pipe_snap or None), pipeline_source=source)
    try:
        env["forecastBreakdowns"] = workspace.forecast_breakdowns(funded_df, pipe_df)
    except Exception:  # noqa: BLE001
        env.setdefault("forecastBreakdowns", {})
    return env


def _movement(out_root, cid, rid, scope, data: DashboardData,
              prior_reporting_date: Optional[str] = None) -> Dict[str, Any]:
    """Governed movement attribution for the deck's scope.

    ``funded_bridge`` scopes through ``lens_filters``, which filters on the
    provenance columns. A Total scope passes no filter; a type scope filters to
    that type — the same narrowing the rest of the deck uses.

    The bridge opens at ``prior_reporting_date``: the SAME period the funded
    snapshot compares against. Without it the governed bridge opens at the
    earliest period available, and a deck with a long history would attribute
    the whole series beside a one-period headline — the two halves of the
    movement slide measuring different windows.
    """
    from . import movement as _mv

    lens_filters, lens_label = _scope_lens(scope)
    return _mv.build_bridges(out_root, cid, rid,
                             start_period=prior_reporting_date,
                             lens_filters=lens_filters,
                             lens_label=lens_label, note=data.note)


def _balance_movement(out_root, cid, rid, scope, prior_reporting_date):
    """The governed economic bridge, opened at the SAME period the funded
    snapshot compares against so the pack measures one window throughout."""
    from mi_agent_api import evolution
    start = str(prior_reporting_date)[:7] if prior_reporting_date else None
    return evolution.funded_balance_movement(out_root, cid, rid, scope=scope,
                                             start_period=start)


def _capabilities(funded_df, funded_evolution) -> Dict[str, Any]:
    """Every published capability resolved against this portfolio's shape.

    ``metric id -> Availability``. The registry is asset-agnostic by
    construction — a capability declares the economic conditions it needs, and
    any book meeting them gets it — which is precisely the property the pack
    needs: conditional reporting driven by what the tape supports, never by a
    branch on what the book is called.
    """
    from trakt_core import capability as cap

    periods = len((funded_evolution or {}).get("periods") or ()) or 1
    shape = cap.describe_portfolio(funded_df, history_periods=periods)
    return {a.metric: a for a in cap.resolve_all(shape)}


def _portfolio_projections(funded_df, registry, scope, data: DashboardData):
    """The per-constituent-book forward view.

    ``forecast_bridge.portfolio_projections`` is the same function
    ``/mi/forecast/snapshot`` attaches as ``portfolioProjections``. It applies a
    run-off curve ONLY where the client has supplied an approved one and
    discloses every book where it has not; nothing is modelled here.
    """
    from mi_agent_api import forecast_bridge as fb
    bridge = (data.forecast or {}).get("forecastBridge") or {}
    weighted = bridge.get("weightedExpectedFundedAmount")
    return fb.portfolio_projections(
        funded_df, registry, scope,
        weighted_pipeline=(float(weighted) if weighted else 0.0))


def _watchlist(data: DashboardData) -> Dict[str, Any]:
    from . import watchlist as _wl
    return _wl.build(data)


def _insights(data: DashboardData) -> Dict[str, Any]:
    from . import insights as _ins
    return _ins.build(data)


def _cohorts(funded_df, cid, pid, reporting_date):
    from mi_agent_api import cohorts
    return cohorts.cohort_analysis(
        funded_df, client_id=cid, portfolio_id=pid,
        reporting_date=reporting_date, grain="Y", dimension="vintage")


def _cohort_series(out_root, cid, scope, data: DashboardData) -> Dict[str, Any]:
    """Per-vintage static-pool series for the cohorts the deck will show.

    The vintages come from the governed cohort table already resolved into
    ``data.cohorts``, so the slide can never plot a cohort the composition table
    does not contain. Each series is one call to the SAME progression service the
    dashboard uses; nothing is grouped or aggregated here.
    """
    from . import cohorts as _co

    formation = _co.adapt_formation(data.cohorts)
    if not _co.formation_is_meaningful(formation):
        return {"available": False, "reason": formation.reason
                or "no governed cohort composition for this book", "series": {}}

    chosen, overflow = _co.select_cohorts(formation)
    if not chosen:
        return {"available": False,
                "reason": "no vintage holds a material share of the book",
                "series": {}}

    lens_filters, lens_label = _scope_lens(scope)
    series: Dict[str, Any] = {}
    for vintage in chosen:
        series[vintage] = _guard(
            data, f"cohort_series[{vintage}]",
            lambda v=vintage: _cohort_progression(
                out_root, cid, vintage=v, lens_filters=lens_filters,
                lens_label=lens_label))
    return {"available": True, "series": series, "overflow": overflow,
            "lens": lens_label}


def _scope_lens(scope):
    """(lens_filters, lens_label) for the governed scope — the same narrowing the
    movement bridge applies, so every multi-period surface reports one book."""
    if scope is None or getattr(scope, "is_total", True):
        return None, "Total"
    from trakt_core.portfolio import FIELD_PORTFOLIO_TYPE
    types = [t for t in (getattr(scope, "portfolio_types", ()) or ()) if t]
    if len(types) == 1:
        return {FIELD_PORTFOLIO_TYPE: types[0]}, str(getattr(scope, "label", types[0]))
    return None, str(getattr(scope, "label", "Total"))


def _geo(funded_df, cid, rid):
    from mi_agent_api import geo
    out = dict(geo.exposure_by_itl3(funded_df))
    out.update({"dataset": "geo_itl3", "portfolioId": f"{cid}/{rid}"})
    return out


def _funded_evo(out_root, cid, rid, funded_cuts, scope=None, context_id=None):
    """Funded evolution: the dashboard's resolver first (blob dated cuts / local
    central-tape cuts); when that yields <2 periods, supplement from LOCAL dated
    platform canonicals so downloaded history renders too (requirement #3).

    ``scope`` narrows the series to the governed portfolio context, so a scoped
    deck's trend describes the same book as its KPIs rather than the whole
    platform."""
    from mi_agent_api import evolution
    result = evolution.funded_evolution(out_root, cid, rid, scope=scope)
    if len(result.get("periods", [])) >= 2:
        return result
    frames = _local_funded_frames(funded_cuts, cid, context_id)
    if len(frames) >= 2:
        return evolution.assemble_funded_evolution(frames, cid, rid, lineage={
            "source": "dated platform canonicals (platform_canonical_typed.csv)",
            "metric": "funded book actuals per reporting cut",
            "note": "One period per dated funded cut under the onboarding output root."})
    return result


def _pipeline_evo(prow, cid, history):
    from mi_agent_api import evolution
    return evolution.pipeline_evolution(prow, cid, None, historical_model=history)


def _funnel(prow, cid, history):
    from mi_agent_api import evolution
    return evolution.pipeline_funnel_evolution(
        prow, cid, None, lag_weeks=_kfi_lag_weeks(history))


def _forecast_evo(out_root, prow, cid, rid, history):
    from mi_agent_api import evolution
    return evolution.forecast_evolution(out_root, prow or out_root, cid, rid,
                                        historical_model=history)


def _risk(out_root, cid, rid):
    from mi_agent_api import risk_limits
    return risk_limits.compute_risk_limits(out_root, cid, rid)


def _concentration_history(out_root, cid, rid, scope):
    """Utilisation of each approved test across REAL governed snapshots.

    ``concentration_tests_api.compute_history`` — the same service behind
    ``/mi/concentration-tests/history``, which the React Risk Limits workspace
    already reads. It evaluates today's approved configuration against each
    historical frame, so the series is comparable period to period. The deck
    renders the direction; it computes none of it.
    """
    from mi_agent_api import concentration_tests_api as ct
    return ct.compute_history(out_root, cid, rid, scope=scope)


def _concentration(out_root, cid, rid, scope):
    """The governed concentration-test envelope — the SAME service the Risk Limits
    workspace, MI Query and Copilot use (``/mi/concentration-tests``).

    This is the operator-approved capability. It supersedes the deck's previous
    use of ``risk_limits`` alone, which is the *legacy extracted* monitor the
    concentration service itself presents as explicitly NOT operator-approved.
    It additionally carries the forward-looking states the investor section
    needs — Expected (forecast model) and Full Pipeline (maximum-exposure
    stress) — which the legacy monitor has no concept of.
    """
    from mi_agent_api import concentration_tests_api as ct
    return ct.compute_concentration_tests(out_root, cid, rid, scope=scope)


def _extrapolation(out_root, prow, cid, rid, history, scale_targets=()):
    """The governed scale-up projection.

    ``scale_targets`` are the funding / securitisation thresholds the DECK
    CONFIG names (``deck.scale_targets`` in the pack definition). They are
    passed to the governed ladder through the ``extra_thresholds`` parameter it
    already exposes, so naming a target is a configuration decision and not a
    new forecast primitive — the projection itself is unchanged.
    """
    from mi_agent_api import forecast_extrapolation as fx
    return fx.build_extrapolation(out_root, prow or out_root, cid, rid,
                                  history_model=history,
                                  extra_thresholds=tuple(scale_targets or ()))
