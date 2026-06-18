"""mi_agent.semantic_resolver — Step 0 governed term resolution (Phase 6).

Resolves MI dimension *terms* (e.g. "portfolio", "stage") to canonical semantic
fields with Trakt's rules, returning structured issues instead of guessing:

  * "portfolio" -> Trakt portfolio reference (portfolio_id), requires config;
  * "acquired portfolio" -> acquired_portfolio_id;
  * "SPV" -> spv_id;
  * "stage" / "pipeline stage" -> pipeline_stage, ONLY in a pipeline context;
  * "IFRS stage" / "IFRS 9 stage" -> ifrs9_stage;
  * "risk stage" / "internal risk stage" -> internal_risk_stage;
  * balance / interest-rate / time-on-book -> quantile bucket dimensions.

This is governed, deterministic, and LLM-free.
"""

from __future__ import annotations

import re
import dataclasses as _dc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mi_agent.portfolio_reference import PortfolioReferenceConfig
from mi_agent.quantile_buckets import QUANTILE_DIMENSIONS
from mi_agent.states.models import WARNING, make_issue
from mi_agent.states.route_contracts import canonical_state

# Phase 6 issue codes.
MISSING_PORTFOLIO_REFERENCE_CONFIG = "missing_portfolio_reference_config"
INVALID_STAGE_CONTEXT = "invalid_stage_context"
AMBIGUOUS_DIMENSION = "ambiguous_dimension"
UNRESOLVED_ROUTE_DIMENSION = "unresolved_route_dimension"

# States / contexts in which pipeline_stage is meaningful.
_PIPELINE_STATES = {"total_pipeline", "total_forecast_funded"}
_PIPELINE_CONTEXTS = {"pipeline", "origination"}


@dataclass
class DimensionResolution:
    term: str
    field: Optional[str] = None
    kind: str = "unresolved"          # registry|portfolio_ref|bucket_quantile|unresolved
    source_field: Optional[str] = None  # for bucket_quantile
    issues: List[Dict[str, Any]] = _dc.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.field is not None and self.kind != "unresolved"


def _norm(term: str) -> str:
    return re.sub(r"[_\s]+", " ", str(term).strip().lower())


def _pipeline_context_ok(context: Optional[str], state: Optional[str]) -> bool:
    if context and _norm(context) in _PIPELINE_CONTEXTS:
        return True
    if state and canonical_state(state) in _PIPELINE_STATES:
        return True
    return False


def resolve_dimension(term: str, *, context: Optional[str] = "pipeline",
                      state: Optional[str] = None,
                      portfolio_config: Optional[PortfolioReferenceConfig] = None,
                      semantics: Optional[dict] = None) -> DimensionResolution:
    """Resolve a single dimension *term* to a canonical field per Trakt rules."""
    t = _norm(term)
    res = DimensionResolution(term=term)

    # 1. Acquired portfolio (check BEFORE bare "portfolio").
    if "acquired portfolio" in t or t == "acquired_portfolio_id":
        res.field, res.kind = "acquired_portfolio_id", "registry"
        return res

    # 2. SPV.
    if t == "spv" or "spv" in t.split() or t == "spv_id":
        res.field, res.kind = "spv_id", "registry"
        return res

    # 3. Stage family (specific before bare "stage").
    if "ifrs" in t:
        res.field, res.kind = "ifrs9_stage", "registry"
        return res
    if "risk stage" in t or "internal risk stage" in t or t == "internal_risk_stage":
        res.field, res.kind = "internal_risk_stage", "registry"
        return res
    if t in ("stage", "pipeline stage") or t == "pipeline_stage":
        if not _pipeline_context_ok(context, state):
            res.issues.append(make_issue(
                INVALID_STAGE_CONTEXT, WARNING,
                f"'stage' resolves to pipeline_stage, which is not valid in "
                f"context={context!r} state={state!r} (pipeline states only)",
                field="pipeline_stage"))
            return res
        res.field, res.kind = "pipeline_stage", "registry"
        return res

    # 4. Portfolio -> Trakt portfolio reference (NEVER acquired_portfolio_id).
    if "portfolio" in t:
        if portfolio_config is not None and portfolio_config.has_portfolio_references:
            res.field, res.kind = "portfolio_id", "portfolio_ref"
            return res
        res.issues.append(make_issue(
            MISSING_PORTFOLIO_REFERENCE_CONFIG, WARNING,
            "'portfolio' requires a Trakt portfolio reference config; none "
            "configured for this client (not resolving to acquired_portfolio_id)",
            field="portfolio_id"))
        return res

    # 5. Quantile bucket dimensions (asset-agnostic).
    bucket_terms = {
        "balance band": "balance_band", "balance bucket": "balance_band",
        "interest rate bucket": "interest_rate_bucket",
        "rate bucket": "interest_rate_bucket",
        "time on book": "time_on_book_bucket",
        "time on book bucket": "time_on_book_bucket",
        "months on book": "time_on_book_bucket",
    }
    bdim = bucket_terms.get(t) or (t if t in QUANTILE_DIMENSIONS else None)
    if bdim:
        res.field, res.kind = bdim, "bucket_quantile"
        res.source_field = QUANTILE_DIMENSIONS.get(bdim)
        return res

    # 6. Direct registry key hit.
    if semantics and t.replace(" ", "_") in (semantics.get("fields") or {}):
        res.field, res.kind = t.replace(" ", "_"), "registry"
        return res

    # 7. Keyword fallback against the registry.
    if semantics:
        from mi_agent.llm_query_parser import find_field
        hit = find_field(semantics, keywords=(t,), strict=True)
        if hit:
            res.field, res.kind = hit, "registry"
            return res

    res.issues.append(make_issue(
        UNRESOLVED_ROUTE_DIMENSION, WARNING,
        f"could not resolve dimension term {term!r}", field=term))
    return res


def resolve_route_dimensions(route: str, *, semantics: dict,
                             portfolio_config: Optional[PortfolioReferenceConfig] = None,
                             routes_dir=None) -> Dict[str, DimensionResolution]:
    """Resolve every ``allowed_dimensions`` entry for a route. A registry key, a
    portfolio reference, or a recognised quantile-bucket dimension all count as
    'resolved'."""
    from mi_agent.states.route_contracts import load_route_contract
    contract = load_route_contract(route, routes_dir=routes_dir)
    dims = contract.get("allowed_dimensions") or []
    semkeys = set((semantics.get("fields") or {}))
    out: Dict[str, DimensionResolution] = {}
    for d in dims:
        # A route dimension may already be a registry key (most are).
        if d in semkeys:
            out[d] = DimensionResolution(term=d, field=d, kind="registry")
            continue
        if d in QUANTILE_DIMENSIONS:
            out[d] = DimensionResolution(term=d, field=d, kind="bucket_quantile",
                                         source_field=QUANTILE_DIMENSIONS[d])
            continue
        out[d] = resolve_dimension(d, context="pipeline", semantics=semantics,
                                   portfolio_config=portfolio_config)
    return out
