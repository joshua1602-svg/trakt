"""MI workspace orchestration — Funded / Pipeline / Forecast views.

Composes the existing funded snapshot + pipeline snapshot + forecast bridge into
one workspace view-model and supports the tab-aware MI Agent query. It never
merges the funded and pipeline SPINE datasets; the Forecast view frame is a
DERIVED, in-memory projection (funded balance + probability-weighted pipeline,
the deterministic bridge) used only for view breakdowns / queries.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric
from mi_agent import portfolio_lens as lens_mod

VIEWS = ("funded", "pipeline", "forecast")
DEFAULT_VIEW = "funded"

#: The DATA ROOT a caller read, mapped to the dataset that root holds. The map
#: is the derivation: a route names the root it consumed, not the answer.
_ROOT_DATASET = (("output_root", "funded"), ("pipeline_root", "pipeline"))


def datasets_read(**roots: Any) -> Tuple[str, ...]:
    """The datasets a caller actually read, from the ROOTS it actually consumed.

    THE POINT IS THAT IT IS DERIVED. Five routes in `chat_routing` used to write
    ``{"dataset": "funded", ...}`` as a literal at their own return site, and
    three of them were wrong: `_route_portfolio_summary`, `_route_risk` and
    `_route_bridge` answered "Summarise the current pipeline" and its siblings
    from the funded book while `metadata.datasetContext` said `pipeline`. A
    constant cannot be wrong about itself in any detectable way, and a route that
    cannot say what it read cannot be checked against what it was asked for.

    Pass the root you read and the name follows. A route that later reads the
    pipeline extract passes `pipeline_root` and says so without anyone
    remembering to edit a string — which is the whole difference between a
    derivation and a re-typed constant.

    Order follows `_ROOT_DATASET`, so a composition reads `funded+pipeline`
    rather than depending on keyword order at the call site.
    """
    out: List[str] = []
    for key, dataset in _ROOT_DATASET:
        if roots.get(key) is not None and dataset not in out:
            out.append(dataset)
    return tuple(out)


def reconciliation_for(datasets: Iterable[str], **extra: Any) -> Dict[str, Any]:
    """The reconciliation block naming the datasets an answer was computed from.

    ONE IMPLEMENTATION, ADOPTED — not a sixth. `mi_workflows.analytical.route.
    _reconciliation` already derived this correctly, from the capabilities that
    ran, and its docstring already said why: *"an offer-stage question reads the
    pipeline and nothing else, and reporting it as a full-coverage funded answer
    would misdescribe what was measured."* That was right, and it was the only
    site doing it. This is that logic, lifted so both callers share it; the
    alternative — five sites each deriving their own way — is the disease rather
    than the cure.

    Full coverage is claimed only where the funded book is among the datasets,
    which is the contract the analytical route already applied.
    """
    names = [str(d) for d in datasets if d]
    funded = any(n.startswith(DEFAULT_VIEW) for n in names)
    out: Dict[str, Any] = {
        "dataset": "+".join(names) or DEFAULT_VIEW,
        "coverage_by_balance_pct": 100.0 if funded else None,
    }
    # Extras pass through UNFILTERED, including None. Dropping a None-valued
    # `reporting_date` would change the envelope shape for the routes adopting
    # this, and a consolidation that quietly alters its callers' output is not a
    # consolidation.
    out.update(extra)
    return out

#: The PRE-FUNDING ARTEFACTS. A question naming one of these is about the
#: pipeline tape even though it names no view: nobody asks how many "pipeline"
#: they have, they ask how many APPLICATIONS.
#:
#: Not a new vocabulary. This is `chat_routing._PIPELINE_WORDS` moved to the
#: owner, minus `pipeline` itself (which :data:`VIEWS` already covers) and minus
#: `case`. It lived in a route for as long as it did because that route was the
#: only consumer that needed it — which is precisely what made it a second
#: owner.
#:
#: WHY `case` IS NOT HERE, measured rather than assumed. The retired second
#: owner listed it, but it was reachable only from the compare and evolution
#: routes, so its ambiguity never surfaced. Read by the ONE owner it decides
#: every question, and in this estate a bare `case` means a FUNDED LOAN at
#: least as often as a pipeline case:
#:
#:     "Which region gained the most cases since last month?"
#:
#: is filed in the P1C golden bank under `# -- loan count --`, beside "Which
#: region added the most loans month-on-month?", and expects a ranked FUNDED
#: movement. Classifying it as pipeline turns that answer into a refusal.
#:
#: The evidence for dropping it rather than keeping it: across the 882 distinct
#: corpus questions, NOT ONE reaches the pipeline through `case`. Every
#: artefact-driven movement comes from `application`, `kfi` or `offer`, all of
#: which are unambiguous. `case` bought nothing and cost a golden-bank answer.
#:
#: A question that means the pipeline case still says so — "how many PIPELINE
#: cases are there?" resolves through the view name, unaffected.
PIPELINE_ARTEFACTS = ("kfi", "application", "offer")

# Unqualified "amount"/"balance" resolves to this column per view. The pipeline
# prepared dataset and the forecast frame both carry the view's primary metric
# under ``current_outstanding_balance``, so the SAME query shape works per view.
PRIMARY_METRIC = {
    "funded": "current_outstanding_balance",
    "pipeline": "current_outstanding_balance",
    "forecast": "current_outstanding_balance",
}

# Dimension columns carried onto the derived forecast frame (intersection of the
# funded + pipeline canonical columns that MI queries stratify on).
_SHARED_DIMS = [
    "geographic_region_obligor", "collateral_geography", "ltv_bucket",
    "original_ltv_bucket", "broker_channel", "origination_channel",
    "current_loan_to_value", "current_interest_rate", "interest_rate_bucket",
    "age_bucket", "ticket_bucket", "expected_completion_month",
]


def view_named_by_question(question: str) -> Optional[str]:
    """The view the QUESTION itself names, or ``None`` if it names none.

    The VIEW-NAME half of :func:`resolve_dataset`, which is the owner. Exposed
    separately because a caller sometimes needs to know whether the question
    named a view OUTRIGHT or fell through to a governed step — and because a
    second copy of this vocabulary is the defect B21 fixed.

    Historically this was extracted from `resolve_active_view`, which folded the
    question and the workspace tab together so that a caller could not recover
    which of the two had decided. The tab is no longer an input to anything
    here, so there is nothing left to disentangle. `funded` is returned like any other view: it is the
    default AND a thing a question can name outright, and collapsing those two
    is how an explicit "the funded book" becomes indistinguishable from silence.
    """
    q = (question or "").lower()
    for view in ("forecast", "pipeline", "funded"):
        if lens_mod.undisclaimed_mention(q, view):
            return view
    return None


#: GOVERNED SPAN OWNERSHIP is applied to the QUESTION BEFORE it reaches this
#: module, never by adding an input here.
#:
#: This module's vocabulary — forecast / pipeline / funded, and the pre-funding
#: artefacts — belongs to no book field, so any of those words found inside a
#: span the book has already claimed as one value of one field belongs to the
#: value. Measured: a broker called "Pipeline Mortgage Club" served every
#: question about it from the pipeline extract — 8 cases in place of its 63
#: funded loans, with the broker narrowing gone and nothing said.
#:
#: The masking is `mi_agent.categorical_spans`' and the caller's to apply:
#: `resolve_dataset` takes ONE argument and `test_dataset_ownership::
#: test_the_resolver_cannot_be_handed_a_tab` exists to keep it that way — "not
#: 'it ignores the tab', it has nowhere to put one". A second parameter here
#: would be a place to put one.


def resolve_dataset(question: Optional[str]) -> str:
    """**THE** dataset a natural-language MI question is about.

    ``FUNDED`` | ``PIPELINE`` | ``FORECAST``, decided by the QUESTION and by
    nothing else. This is the single semantic owner. There is no second one, and
    the caller's workspace tab is deliberately not a parameter — a parameter it
    does not have cannot quietly become an input again.

    Why the tab is gone
    -------------------
    Natural-language MI is self-contained: the user should not have to know
    which tab they are on to ask a correct question. Measured before this
    change, six of fourteen probe questions were TAB-SENSITIVE — the same
    sentence was served from a different dataset depending on the tab. Two of
    them are worth naming:

        "How many cases are there?"
            funded tab -> the funded book, pipeline tab -> the pipeline
        "the balance by seasoning segment excluding pipeline cases"
            pipeline tab -> the pipeline. The question rules the pipeline out
            in words and the tab put it back.

    The tab still selects what the UI DISPLAYS. It no longer decides what a
    question MEANS. Those are different responsibilities.

    Precedence
    ----------
    ``forecast`` > ``pipeline`` > ``funded`` > pre-funding artefact > default.

    The first three are :func:`view_named_by_question` unchanged, so nothing it
    already decided can move. The artefact step fires ONLY where it returned
    ``None``, which is exactly the gap `chat_routing._dataset_for` was covering
    alone — and covering with the opposite precedence, testing its tape
    vocabulary BEFORE any forecast reading, so "Forecast application volumes
    next quarter" was `pipeline` to it. Forecast wins here.

    Population is a different axis
    ------------------------------
    Direct / Acquired / a named SPV never appear above. They select a POPULATION
    WITHIN a dataset and are `portfolio_lens`'s to resolve. "The acquired funded
    balance" is the funded dataset, acquired population, and conflating the two
    axes is how a scope word could have chosen a tape.

    Vocabulary choice, measured
    ---------------------------
    `mi_workflows.analytical.intent` expresses these concepts structurally, as
    ``REQ_PIPELINE_DATASET`` / ``REQ_FORECAST``, and was the obvious candidate
    to own this. It was censused rather than assumed and it is the wrong tool:
    those requirements decide whether a question is REFUSABLE and are checked
    AGAINST a dataset, not used to select one, so their vocabularies are much
    wider. Over the 882 distinct corpus questions they move **59 (6.7%)**,
    including "top brokers by expected funded amount" to forecast. The rule
    below moves **5 (0.6%)**, and all five name a pre-funding artefact.
    """
    named = view_named_by_question(question)
    if named is not None:
        return named
    low = (question or "").lower()
    if any(lens_mod.undisclaimed_mention(low, w) for w in PIPELINE_ARTEFACTS):
        return "pipeline"
    return DEFAULT_VIEW


def resolve_active_view(question: str, dataset_context: Optional[str] = None) -> str:
    """Retained caller-facing name for :func:`resolve_dataset`.

    ``dataset_context`` is ACCEPTED AND IGNORED. It used to be the fallback when
    the question named no view, and that fallback is the tab dependence this
    change removes. The parameter survives only so existing callers keep
    working; it has no semantic effect and
    `test_dataset_ownership.py::test_the_tab_argument_is_inert` pins that.

    Prefer :func:`resolve_dataset` in new code, which does not offer the
    parameter at all.
    """
    return resolve_dataset(question)


def build_forecast_view_frame(funded_df: Optional[pd.DataFrame],
                              pipeline_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Derived forecast frame: one row per funded loan (contribution = funded
    balance) and per pipeline case (contribution = weighted expected funded
    amount), carrying shared dimensions. ``current_outstanding_balance`` holds the
    forecast contribution so any "X by dimension" query yields forecast X by
    dimension. NOT persisted; never written to the spine.
    """
    parts: List[pd.DataFrame] = []
    if funded_df is not None and len(funded_df):
        f = pd.DataFrame(index=funded_df.index)
        f["current_outstanding_balance"] = coerce_numeric(
            funded_df.get("current_outstanding_balance", pd.Series(dtype=float)))
        for d in _SHARED_DIMS:
            if d in funded_df.columns:
                f[d] = funded_df[d].values
        f["state_component"] = "funded"
        parts.append(f)
    if pipeline_df is not None and len(pipeline_df):
        p = pd.DataFrame(index=pipeline_df.index)
        # The forecast CONTRIBUTION of a pipeline case is its weighted expected
        # funded amount (amount x completion probability) — not its gross amount.
        p["current_outstanding_balance"] = coerce_numeric(
            pipeline_df.get("weighted_expected_funded_amount", pd.Series(dtype=float)))
        for d in _SHARED_DIMS:
            if d in pipeline_df.columns:
                p[d] = pipeline_df[d].values
        p["state_component"] = "forecast_pipeline"
        # Drop pipeline rows with no weightable contribution (withdrawn/unknown).
        p = p[coerce_numeric(p["current_outstanding_balance"]).notna()]
        parts.append(p)
    if not parts:
        return pd.DataFrame(columns=["current_outstanding_balance", "state_component"])
    return pd.concat(parts, ignore_index=True)


def _dim_sum(df: Optional[pd.DataFrame], dim: str, col: str) -> Dict[str, float]:
    if df is None or dim not in df.columns or col not in df.columns:
        return {}
    amt = coerce_numeric(df[col])
    grp = amt.groupby(df[dim].astype(str)).sum()
    return {str(k): float(v) for k, v in grp.items()
            if str(k).strip() and str(k) not in ("nan", "NaT", "None")}


def forecast_dimension_breakdown(funded_df: Optional[pd.DataFrame],
                                 pipeline_df: Optional[pd.DataFrame],
                                 dim: str) -> List[Dict[str, Any]]:
    """``[{key, fundedAmount, weightedPipelineAmount, forecastAmount}]`` for one
    dimension — funded exposure + weighted expected pipeline = forecast. Derived
    by aggregate composition (no row merge), ordered by forecast amount desc."""
    funded = _dim_sum(funded_df, dim, "current_outstanding_balance")
    pipe = _dim_sum(pipeline_df, dim, "weighted_expected_funded_amount")
    keys = set(funded) | set(pipe)
    rows = []
    for k in keys:
        fa = round(funded.get(k, 0.0), 2)
        wp = round(pipe.get(k, 0.0), 2)
        rows.append({"key": k, "fundedAmount": fa, "weightedPipelineAmount": wp,
                     "forecastAmount": round(fa + wp, 2)})
    rows.sort(key=lambda r: r["forecastAmount"], reverse=True)
    return rows


def forecast_breakdowns(funded_df: Optional[pd.DataFrame],
                        pipeline_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Forecast-by-dimension breakdowns for the Forecast view (region / LTV /
    completion month), capped where long."""
    from .pipeline_contract import cap_breakdown
    region = forecast_dimension_breakdown(funded_df, pipeline_df, "geographic_region_obligor")
    ltv = forecast_dimension_breakdown(funded_df, pipeline_df, "ltv_bucket")
    # Completion-month: pipeline contributes weighted by month; funded is "now".
    month = _dim_sum(pipeline_df, "expected_completion_month", "weighted_expected_funded_amount")
    by_month = [{"month": k, "weightedExpectedFundedAmount": round(v, 2)}
                for k, v in sorted(month.items())]
    # Re-cap region/ltv to top 10 for the visual, keyed on forecastAmount.
    def _cap(rows):
        capped = cap_breakdown(
            [{"key": r["key"], "caseCount": 0, "pipelineAmount": r["forecastAmount"],
              "weightedExpectedFundedAmount": r["weightedPipelineAmount"]} for r in rows], 10)
        return capped
    return {
        "byRegion": region,
        "byLtvBucket": ltv,
        "byCompletionMonth": by_month,
        "byRegionCapped": _cap(region),
        "byLtvBucketCapped": _cap(ltv),
    }


# --------------------------------------------------------------------------- #
# Lineage ("How calculated") per view — from existing metadata.
# --------------------------------------------------------------------------- #
def lineage_for(view: str, *, funded_reporting_date: Optional[str] = None,
                pipeline_as_of_date: Optional[str] = None,
                completion_probability_basis: Optional[str] = None,
                source_file: Optional[str] = None,
                pipeline_source_folder_date: Optional[str] = None,
                current_pipeline_snapshot_date: Optional[str] = None,
                current_pipeline_source_file: Optional[str] = None,
                historical_model_evidence: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """Per-view "How calculated" lineage. The pipeline / forecast views carry the
    historical completion-model evidence and keep distinct concepts separate: the
    funded reporting date, the CURRENT pipeline snapshot (latest weekly extract +
    its file), and the historical probability observation window (start/end)."""
    if view == "funded":
        return {
            "view": "funded",
            "source": "18_central_lender_tape.csv",
            "metric": "current_outstanding_balance",
            "fundedReportingDate": funded_reporting_date,
            "explanation": "Funded book actuals from the governed central lender tape.",
        }
    evidence = historical_model_evidence or {}
    # The current snapshot date is the latest weekly extract; fall back to the as-of.
    snapshot_date = current_pipeline_snapshot_date or pipeline_as_of_date
    # Observation window: prefer the dedup inventory window, else the model evidence.
    window_start = evidence.get("observationWindowStart")
    window_end = evidence.get("observationWindowEnd")
    common = {
        "completionProbabilityBasis": completion_probability_basis,
        # Current pipeline snapshot (latest weekly extract) — distinct from window.
        "currentPipelineSnapshotDate": snapshot_date,
        "currentPipelineSourceFile": current_pipeline_source_file
            or (source_file.split("/")[-1] if source_file else None),
        "pipelineSourceFolderDate": pipeline_source_folder_date,
        # Historical probability observation window — distinct from the as-of date.
        "historicalObservationWindowStart": window_start,
        "historicalObservationWindowEnd": window_end,
        "observationWindowStart": window_start,
        "observationWindowEnd": window_end,
        "uniqueWeeklyExtractsUsed": evidence.get("uniqueWeeklyExtractsUsed"),
        "sourceFilesScanned": evidence.get("sourceFilesScanned"),
        "historicalModelEvidence": evidence,
    }
    if view == "pipeline":
        return {
            "view": "pipeline",
            "source": source_file or "governed weekly pipeline files",
            "metric": "expected_funded_amount",
            "weightedMetric": "expected_funded_amount × completion_probability",
            "pipelineAsOfDate": snapshot_date,
            **common,
            "explanation": "Origination pipeline (pre-funded), governed weekly extract; "
                           "completion probabilities from the historical weekly-snapshot model.",
        }
    return {
        "view": "forecast",
        "metric": "forecast_funded_balance",
        "formula": "forecast funded balance = funded balance + Σ(expected_funded_amount × completion_probability)",
        "fundedReportingDate": funded_reporting_date,
        "pipelineAsOfDate": snapshot_date,
        **common,
        "explanation": "Deterministic bridge: funded actuals + probability-weighted pipeline.",
    }
