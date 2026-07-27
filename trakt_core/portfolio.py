"""trakt_core.portfolio — the governed portfolio contract.

ONE place defines what a client's portfolio hierarchy is, what each portfolio can
support analytically, which portfolios a requested context resolves to, and which
of them actually contributed to an answer. Every client-facing channel — React,
Copilot, PPTX, drill-through, future enterprise agents — consumes these
structures; none of them may re-derive any of it.

    canonical provenance + governed portfolio metadata
            │
            ├─►  PortfolioRegistry     (who exists, and what can they support)
            ├─►  resolve_scope()       (which portfolios does this context mean)
            ├─►  resolve_capabilities() (what is analytically valid for that scope)
            └─►  ScopeCoverage         (who actually answered, and who did not)

Design rules this module encodes:

* The hierarchy is **dynamic**. ``direct`` means "every governed portfolio whose
  ``source_portfolio_type`` is direct" — not ``direct_001``. ``total`` is the sum
  of every eligible portfolio. Adding ``direct_002`` changes the answer with no
  code change.
* Business applicability is **metadata-driven**. Origination, pipeline data,
  forecast treatment and runoff profiles are portfolio attributes, defaulted from
  a single governed table (:data:`DEFAULT_ORIGINATION_BY_TYPE`) and overridable
  per portfolio. Nothing keys off a portfolio *name*.
* Exclusions are **disclosed, never silent**. Every portfolio dropped from an
  answer carries a machine-readable reason code.

This module is deliberately dependency-free (no pandas, no web framework): it
takes plain records and returns plain dataclasses, so it can be imported by the
onboarding engine, the MI query path, the HTTP adapters and the deck generator
alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: Governed source-portfolio types (mirrors ``engine.provenance``).
PORTFOLIO_TYPE_DIRECT = "direct"
PORTFOLIO_TYPE_ACQUIRED = "acquired"
PORTFOLIO_TYPES: Tuple[str, ...] = (PORTFOLIO_TYPE_DIRECT, PORTFOLIO_TYPE_ACQUIRED)

#: The three context *kinds* a workspace selection can have.
CONTEXT_KIND_TOTAL = "total"
CONTEXT_KIND_TYPE = "type"
CONTEXT_KIND_PORTFOLIO = "portfolio"

#: The reserved group context ids. Everything else is a ``source_portfolio_id``.
CONTEXT_TOTAL = "total"
CONTEXT_DIRECT = PORTFOLIO_TYPE_DIRECT
CONTEXT_ACQUIRED = PORTFOLIO_TYPE_ACQUIRED
GROUP_CONTEXT_IDS: Tuple[str, ...] = (CONTEXT_TOTAL, CONTEXT_DIRECT, CONTEXT_ACQUIRED)

#: Canonical provenance columns the registry is derived from.
FIELD_PORTFOLIO_ID = "source_portfolio_id"
FIELD_PORTFOLIO_TYPE = "source_portfolio_type"
FIELD_PORTFOLIO_LABEL = "source_portfolio_label"

#: Governed capability ids. A channel may render these; it may not invent them.
CAP_FUNDED = "funded"
CAP_RISK = "risk"
CAP_EVOLUTION = "evolution"
CAP_COHORTS = "cohorts"
CAP_PIPELINE = "pipeline"
CAP_ORIGINATION_FORECAST = "origination_forecast"
CAP_RUNOFF_FORECAST = "runoff_forecast"
CAP_CONSOLIDATED_FORECAST = "consolidated_forecast"
CAP_DRILL_THROUGH = "drill_through"
CAP_EXPORT = "export"

CAPABILITIES: Tuple[str, ...] = (
    CAP_FUNDED, CAP_RISK, CAP_EVOLUTION, CAP_COHORTS, CAP_PIPELINE,
    CAP_ORIGINATION_FORECAST, CAP_RUNOFF_FORECAST, CAP_CONSOLIDATED_FORECAST,
    CAP_DRILL_THROUGH, CAP_EXPORT,
)

#: Machine-readable exclusion / disablement reasons. Prose is for humans; these
#: are for callers that must branch without parsing English.
REASON_FIELD_NOT_AVAILABLE = "FIELD_NOT_AVAILABLE"
REASON_NON_ORIGINATING = "NON_ORIGINATING"
REASON_NO_PIPELINE_DATA = "NO_PIPELINE_DATA"
REASON_NO_RUNOFF_PROFILE = "NO_RUNOFF_PROFILE"
REASON_NO_ROWS_IN_SCOPE = "NO_ROWS_IN_SCOPE"
REASON_NO_PORTFOLIOS_IN_SCOPE = "NO_PORTFOLIOS_IN_SCOPE"
REASON_NOT_APPLICABLE = "NOT_APPLICABLE"

#: Forecast treatments. ``origination`` books project new lending; ``runoff``
#: books amortise down a SUPPLIED curve; ``hold_constant`` books are carried flat
#: because no governed runoff assumption exists. Trakt never invents a curve.
FORECAST_ORIGINATION = "origination"
FORECAST_RUNOFF = "runoff"
FORECAST_HOLD_CONSTANT = "hold_constant"

#: The ONE governed default table mapping a portfolio type onto whether it
#: originates new lending. It is a default, not a law: per-portfolio governed
#: metadata overrides it, so a future originating acquired vehicle needs a
#: metadata change, not a code change.
DEFAULT_ORIGINATION_BY_TYPE: Dict[str, bool] = {
    PORTFOLIO_TYPE_DIRECT: True,
    PORTFOLIO_TYPE_ACQUIRED: False,
}


def _clean(value: Any) -> Optional[str]:
    """Trim a scalar to ``None`` when blank or a pandas null sentinel."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "nat", "<na>", "null"):
        return None
    return text


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioRecord:
    """One governed source portfolio and everything the platform may assume of it."""

    portfolio_id: str
    portfolio_type: Optional[str] = None
    label: Optional[str] = None
    #: Does this book write new business? Drives Pipeline + origination forecast.
    originates: bool = False
    #: Is governed pipeline (pre-funded) data actually supplied for it?
    pipeline_data_available: bool = False
    #: ``origination`` | ``runoff`` | ``hold_constant`` — see the constants above.
    forecast_treatment: str = FORECAST_HOLD_CONSTANT
    #: Identifier of a SUPPLIED governed runoff/balance-retention curve. Trakt
    #: consumes it; it never generates one (no internal mortality assumptions).
    runoff_profile_id: Optional[str] = None
    #: Monthly balance-retention factors from the supplied curve, if loaded.
    runoff_profile: Tuple[float, ...] = ()
    #: Reporting dates this portfolio has data for (ascending).
    reporting_dates: Tuple[str, ...] = ()
    #: Rows present in the active dataset for this portfolio.
    row_count: Optional[int] = None
    #: True when the portfolio is present in the canonical data (vs metadata-only).
    present_in_data: bool = True

    @property
    def display_label(self) -> str:
        return self.label or self.portfolio_id

    @property
    def has_runoff_profile(self) -> bool:
        return bool(self.runoff_profile_id) or bool(self.runoff_profile)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio_type": self.portfolio_type,
            "label": self.display_label,
            "originates": self.originates,
            "pipeline_data_available": self.pipeline_data_available,
            "forecast_treatment": self.forecast_treatment,
            "runoff_profile_id": self.runoff_profile_id,
            "has_runoff_profile": self.has_runoff_profile,
            "reporting_dates": list(self.reporting_dates),
            "row_count": self.row_count,
            "present_in_data": self.present_in_data,
        }


@dataclass(frozen=True)
class PortfolioRegistry:
    """Every governed source portfolio for one client, in id order."""

    portfolios: Tuple[PortfolioRecord, ...] = ()
    client_id: Optional[str] = None

    # -- lookups ------------------------------------------------------------ #
    def __bool__(self) -> bool:
        return bool(self.portfolios)

    def __len__(self) -> int:
        return len(self.portfolios)

    def __iter__(self):
        return iter(self.portfolios)

    def get(self, portfolio_id: Optional[str]) -> Optional[PortfolioRecord]:
        pid = _clean(portfolio_id)
        if not pid:
            return None
        low = pid.lower()
        for p in self.portfolios:
            if p.portfolio_id.lower() == low:
                return p
        return None

    def ids(self) -> List[str]:
        return [p.portfolio_id for p in self.portfolios]

    def types(self) -> List[str]:
        """Distinct governed types present, in canonical order then any others."""
        present = {p.portfolio_type for p in self.portfolios if p.portfolio_type}
        ordered = [t for t in PORTFOLIO_TYPES if t in present]
        ordered.extend(sorted(t for t in present if t not in PORTFOLIO_TYPES))
        return ordered

    def of_type(self, portfolio_type: str) -> List[PortfolioRecord]:
        t = (portfolio_type or "").strip().lower()
        return [p for p in self.portfolios if (p.portfolio_type or "") == t]

    def reporting_dates(self) -> List[str]:
        dates: set = set()
        for p in self.portfolios:
            dates.update(p.reporting_dates)
        return sorted(dates)

    # -- the selectable hierarchy ------------------------------------------ #
    def contexts(self) -> List[Dict[str, Any]]:
        """The hierarchical selector model: Total → type groups → portfolios.

        Built entirely from governed metadata, so a new portfolio appears with no
        UI or code change. Type groups appear only when that type is present.
        """
        out: List[Dict[str, Any]] = []
        if self.portfolios:
            out.append({
                "context_id": CONTEXT_TOTAL,
                "context_kind": CONTEXT_KIND_TOTAL,
                "label": "Total",
                "parent_id": None,
                "depth": 0,
                "portfolio_ids": self.ids(),
                "portfolio_types": self.types(),
            })
        for ptype in self.types():
            members = self.of_type(ptype)
            if not members:
                continue
            out.append({
                "context_id": ptype,
                "context_kind": CONTEXT_KIND_TYPE,
                "label": ptype.capitalize(),
                "parent_id": CONTEXT_TOTAL,
                "depth": 1,
                "portfolio_ids": [p.portfolio_id for p in members],
                "portfolio_types": [ptype],
            })
            for p in members:
                out.append({
                    "context_id": p.portfolio_id,
                    "context_kind": CONTEXT_KIND_PORTFOLIO,
                    "label": p.display_label,
                    "parent_id": ptype,
                    "depth": 2,
                    "portfolio_ids": [p.portfolio_id],
                    "portfolio_types": [ptype],
                })
        # Portfolios with no governed type still have to be selectable — they are
        # listed under Total rather than silently dropped.
        untyped = [p for p in self.portfolios if not p.portfolio_type]
        for p in untyped:
            out.append({
                "context_id": p.portfolio_id,
                "context_kind": CONTEXT_KIND_PORTFOLIO,
                "label": p.display_label,
                "parent_id": CONTEXT_TOTAL,
                "depth": 1,
                "portfolio_ids": [p.portfolio_id],
                "portfolio_types": [],
            })
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "portfolios": [p.to_dict() for p in self.portfolios],
            "portfolio_types": self.types(),
        }


def _registry_sort_key(record: "PortfolioRecord") -> Tuple[int, str]:
    """Order the registry by governed type (direct, then acquired, then any
    other, then untyped) and then by id — so the hierarchy reads the same way
    everywhere without any channel re-sorting it."""
    ptype = (record.portfolio_type or "").lower()
    rank = PORTFOLIO_TYPES.index(ptype) if ptype in PORTFOLIO_TYPES else (
        len(PORTFOLIO_TYPES) if ptype else len(PORTFOLIO_TYPES) + 1)
    return rank, record.portfolio_id.lower()


def build_registry(
    records: Iterable[Mapping[str, Any]],
    *,
    metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    client_id: Optional[str] = None,
    type_resolver=None,
) -> PortfolioRegistry:
    """Assemble the registry from canonical provenance records + governed metadata.

    ``records`` are distinct provenance rows (``source_portfolio_id`` /
    ``source_portfolio_type`` / ``source_portfolio_label``, optionally
    ``reporting_date`` and ``row_count``) taken from the active canonical data.
    ``metadata`` is the governed per-portfolio overlay keyed by portfolio id —
    origination capability, pipeline availability, forecast treatment, runoff
    profile. It may add portfolios that are configured but not yet in the data.

    ``type_resolver`` is an optional last-resort callable used ONLY when a
    portfolio carries no governed type in the data and none in the metadata. The
    repository supplies ``engine.provenance.derive_portfolio_type`` (the same
    prefix contract onboarding fails closed on), which is why the fallback is
    governed rather than an ad-hoc string match. Without a resolver the portfolio
    stays untyped rather than being guessed into a group.
    """
    meta = {str(k).strip().lower(): dict(v or {}) for k, v in (metadata or {}).items()}

    merged: Dict[str, Dict[str, Any]] = {}
    for rec in records or ():
        pid = _clean(rec.get(FIELD_PORTFOLIO_ID))
        if not pid:
            continue
        slot = merged.setdefault(pid.lower(), {
            "portfolio_id": pid, "dates": set(), "row_count": 0, "seen": False})
        slot["seen"] = True
        ptype = _clean(rec.get(FIELD_PORTFOLIO_TYPE))
        if ptype:
            slot["portfolio_type"] = ptype.lower()
        label = _clean(rec.get(FIELD_PORTFOLIO_LABEL))
        if label:
            slot["label"] = label
        rdate = _clean(rec.get("reporting_date"))
        if rdate:
            slot["dates"].add(rdate)
        rows = rec.get("row_count")
        if rows is not None:
            try:
                slot["row_count"] = int(slot["row_count"]) + int(rows)
            except (TypeError, ValueError):
                pass

    # Governed metadata may declare a portfolio the current data cut does not
    # contain (e.g. onboarded but not yet in this snapshot). It is registered so
    # scope resolution can disclose it rather than pretend it does not exist.
    for key, cfg in meta.items():
        slot = merged.setdefault(key, {
            "portfolio_id": cfg.get("portfolio_id") or cfg.get(FIELD_PORTFOLIO_ID) or key,
            "dates": set(), "row_count": None, "seen": False})
        slot.setdefault("portfolio_type", None)

    out: List[PortfolioRecord] = []
    for key in sorted(merged):
        slot = merged[key]
        cfg = meta.get(key, {})
        pid = str(slot["portfolio_id"])

        ptype = (_clean(cfg.get(FIELD_PORTFOLIO_TYPE)) or _clean(cfg.get("portfolio_type"))
                 or slot.get("portfolio_type"))
        if ptype:
            ptype = ptype.lower()
        elif type_resolver is not None:
            try:
                ptype = type_resolver(pid)
            except Exception:  # noqa: BLE001 - a fallback must never break discovery
                ptype = None

        label = (_clean(cfg.get(FIELD_PORTFOLIO_LABEL)) or _clean(cfg.get("label"))
                 or slot.get("label"))

        if "originates" in cfg:
            originates = bool(cfg.get("originates"))
        else:
            originates = DEFAULT_ORIGINATION_BY_TYPE.get(ptype or "", False)

        if "pipeline_data_available" in cfg:
            pipeline_available = bool(cfg.get("pipeline_data_available"))
        else:
            # Absent explicit metadata, an originating book is assumed to supply
            # pipeline data; the capability resolver re-checks against the actual
            # discovered pipeline sources before enabling the tab.
            pipeline_available = originates

        runoff_id = _clean(cfg.get("runoff_profile_id")) or _clean(cfg.get("runoff_profile"))
        curve = cfg.get("runoff_curve") or cfg.get("monthly_retention") or ()
        try:
            curve = tuple(float(x) for x in curve)
        except (TypeError, ValueError):
            curve = ()

        treatment = _clean(cfg.get("forecast_treatment"))
        if not treatment:
            if originates:
                treatment = FORECAST_ORIGINATION
            elif runoff_id or curve:
                treatment = FORECAST_RUNOFF
            else:
                treatment = FORECAST_HOLD_CONSTANT

        dates = cfg.get("reporting_dates")
        if dates:
            reporting_dates = tuple(sorted({str(d) for d in dates}))
        else:
            reporting_dates = tuple(sorted(slot.get("dates") or ()))

        out.append(PortfolioRecord(
            portfolio_id=pid,
            portfolio_type=ptype,
            label=label,
            originates=originates,
            pipeline_data_available=pipeline_available,
            forecast_treatment=treatment,
            runoff_profile_id=runoff_id,
            runoff_profile=curve,
            reporting_dates=reporting_dates,
            row_count=slot.get("row_count"),
            present_in_data=bool(slot.get("seen")),
        ))
    out.sort(key=_registry_sort_key)
    return PortfolioRegistry(portfolios=tuple(out), client_id=client_id)


# --------------------------------------------------------------------------- #
# Scope resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioScope:
    """A resolved portfolio context: which books this request actually covers."""

    context_id: str
    context_kind: str
    label: str
    portfolio_ids: Tuple[str, ...] = ()
    portfolio_types: Tuple[str, ...] = ()
    #: True when the requested context could not be matched and Total was used.
    fell_back_to_total: bool = False
    #: The context the caller asked for, when it differed from ``context_id``.
    requested_context_id: Optional[str] = None

    @property
    def is_total(self) -> bool:
        return self.context_kind == CONTEXT_KIND_TOTAL

    @property
    def filters(self) -> Dict[str, Any]:
        """The deterministic provenance filter that realises this scope.

        Total carries no filter (every row). A type or portfolio context filters
        on the explicit resolved id list — never on the type string — so a group
        answers as the sum of its members and a portfolio missing from the data
        cannot silently widen the result.
        """
        if self.context_kind == CONTEXT_KIND_TOTAL:
            return {}
        return {FIELD_PORTFOLIO_ID: list(self.portfolio_ids)}

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "context_id": self.context_id,
            "context_kind": self.context_kind,
            "label": self.label,
            "portfolio_ids": list(self.portfolio_ids),
            "portfolio_types": list(self.portfolio_types),
        }
        if self.fell_back_to_total:
            out["fell_back_to_total"] = True
            out["requested_context_id"] = self.requested_context_id
        return out


def resolve_scope(registry: PortfolioRegistry,
                  context_id: Optional[Any] = None) -> PortfolioScope:
    """Resolve any workspace context onto the portfolios it means.

    Accepts ``None`` / ``"total"``, a type group (``"direct"`` / ``"acquired"`` /
    any governed type present), or an individual ``source_portfolio_id``. A
    mapping such as ``{"id": "direct_002"}`` is accepted for convenience.

    Group membership is computed from the registry every time. There is no fixed
    id list anywhere: ``direct`` is whatever is currently typed ``direct``.
    """
    if isinstance(context_id, Mapping):
        context_id = (context_id.get("context_id") or context_id.get("id")
                      or context_id.get("lens") or context_id.get("value"))
    requested = _clean(context_id)
    ids = tuple(registry.ids())
    types = tuple(registry.types())

    def _total(fell_back: bool = False) -> PortfolioScope:
        return PortfolioScope(
            context_id=CONTEXT_TOTAL, context_kind=CONTEXT_KIND_TOTAL, label="Total",
            portfolio_ids=ids, portfolio_types=types,
            fell_back_to_total=fell_back,
            requested_context_id=requested if fell_back else None)

    if requested is None:
        return _total()
    low = requested.lower()
    if low in ("total", "all", "whole_book", "whole-book", ""):
        return _total()

    if low in types:
        members = registry.of_type(low)
        return PortfolioScope(
            context_id=low, context_kind=CONTEXT_KIND_TYPE, label=low.capitalize(),
            portfolio_ids=tuple(p.portfolio_id for p in members),
            portfolio_types=(low,))

    record = registry.get(low)
    if record is not None:
        return PortfolioScope(
            context_id=record.portfolio_id, context_kind=CONTEXT_KIND_PORTFOLIO,
            label=record.display_label,
            portfolio_ids=(record.portfolio_id,),
            portfolio_types=(record.portfolio_type,) if record.portfolio_type else ())

    # An unrecognised context is never silently narrowed to one book: it widens
    # to Total and says so, so a stale UI selection cannot mislabel a result.
    return _total(fell_back=True)


def scope_records(registry: PortfolioRegistry,
                  scope: PortfolioScope) -> List[PortfolioRecord]:
    """The registry records inside a scope, in scope order."""
    out = []
    for pid in scope.portfolio_ids:
        rec = registry.get(pid)
        if rec is not None:
            out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Capability resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapabilityState:
    """Whether one analysis is valid for a scope, and exactly who contributes."""

    capability: str
    enabled: bool
    reason_code: Optional[str] = None
    detail: Optional[str] = None
    contributing_portfolios: Tuple[str, ...] = ()
    excluded_portfolios: Tuple[str, ...] = ()
    #: True when only some of the scope's portfolios contribute.
    partial: bool = False
    #: False when the underlying data is reported for the contributing GROUP
    #: rather than attributed to each portfolio individually (an extract with no
    #: source-portfolio provenance). Never silently split across books.
    attributed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "enabled": self.enabled,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "contributing_portfolios": list(self.contributing_portfolios),
            "excluded_portfolios": list(self.excluded_portfolios),
            "partial": self.partial,
            "attributed": self.attributed,
        }


def _join(ids: Sequence[str]) -> str:
    return ", ".join(ids)


def _funded_capability(cap: str, scope: PortfolioScope,
                       records: Sequence[PortfolioRecord]) -> CapabilityState:
    """Funded-book analyses: valid for every portfolio that has data in scope."""
    if not records:
        return CapabilityState(cap, False, REASON_NO_PORTFOLIOS_IN_SCOPE,
                               "No governed portfolios resolve for this selection.")
    present = [r.portfolio_id for r in records if r.present_in_data]
    absent = [r.portfolio_id for r in records if not r.present_in_data]
    if not present:
        return CapabilityState(
            cap, False, REASON_NO_ROWS_IN_SCOPE,
            "No canonical data is loaded for the selected portfolios.",
            excluded_portfolios=tuple(absent))
    return CapabilityState(
        cap, True,
        detail=None if not absent else
        f"No canonical data loaded for: {_join(absent)}.",
        contributing_portfolios=tuple(present),
        excluded_portfolios=tuple(absent), partial=bool(absent))


def resolve_capabilities(
    registry: PortfolioRegistry,
    scope: PortfolioScope,
    *,
    pipeline_portfolios: Optional[Iterable[str]] = None,
    pipeline_attributed: bool = True,
) -> Dict[str, CapabilityState]:
    """Resolve every governed capability for one scope.

    ``pipeline_portfolios`` — when supplied — is the set of portfolios for which
    governed pipeline data was actually discovered. It intersects with the
    registry's origination metadata, so a book that is configured to originate
    but has no pipeline extract yet is disclosed rather than shown as an empty
    Pipeline tab.

    ``pipeline_attributed=False`` says the discovered extract carries no
    per-portfolio provenance, so the portfolios listed are the originating GROUP
    the pipeline belongs to rather than books it was individually attributed to.
    The capability says so in as many words. Trakt does not split an unattributed
    pipeline across several originating books: there is no basis for an
    allocation, so none is invented.

    Group behaviour follows from the members, never from the group name: Direct
    Pipeline is enabled if *any* direct portfolio originates and has data, and
    Total Pipeline draws on originating portfolios only, disclosing the rest.
    """
    records = scope_records(registry, scope)
    ids_in_scope = tuple(r.portfolio_id for r in records)

    states: Dict[str, CapabilityState] = {}
    for cap in (CAP_FUNDED, CAP_RISK, CAP_EVOLUTION, CAP_COHORTS,
                CAP_DRILL_THROUGH, CAP_EXPORT):
        states[cap] = _funded_capability(cap, scope, records)

    # ---- pipeline: originating portfolios that actually supply data -------- #
    discovered = None
    if pipeline_portfolios is not None:
        discovered = {str(p).strip().lower() for p in pipeline_portfolios if _clean(p)}

    originating = [r for r in records if r.originates]
    if discovered is None:
        with_data = [r for r in originating if r.pipeline_data_available]
    else:
        with_data = [r for r in originating if r.portfolio_id.lower() in discovered]

    pipeline_ids = tuple(r.portfolio_id for r in with_data)
    pipeline_excluded = tuple(pid for pid in ids_in_scope if pid not in pipeline_ids)
    non_originating = tuple(r.portfolio_id for r in records if not r.originates)

    if not records:
        states[CAP_PIPELINE] = CapabilityState(
            CAP_PIPELINE, False, REASON_NO_PORTFOLIOS_IN_SCOPE,
            "No governed portfolios resolve for this selection.")
    elif pipeline_ids:
        parts = []
        if pipeline_excluded:
            reasons = []
            if non_originating:
                reasons.append(f"non-originating: {_join(non_originating)}")
            awaiting = [pid for pid in pipeline_excluded if pid not in non_originating]
            if awaiting:
                reasons.append(f"no pipeline data supplied: {_join(awaiting)}")
            parts.append("Pipeline is sourced from originating portfolios only. "
                         f"Included: {_join(pipeline_ids)}. Excluded — "
                         + "; ".join(reasons) + ".")
        if not pipeline_attributed:
            parts.append(
                "The governed pipeline extract carries no source-portfolio "
                f"provenance, so it is reported for the originating group "
                f"({_join(pipeline_ids)}) rather than attributed to an individual "
                "portfolio.")
        detail = " ".join(parts) or None
        states[CAP_PIPELINE] = CapabilityState(
            CAP_PIPELINE, True, detail=detail,
            contributing_portfolios=pipeline_ids,
            excluded_portfolios=pipeline_excluded,
            partial=bool(pipeline_excluded) or not pipeline_attributed,
            attributed=pipeline_attributed)
    elif originating:
        states[CAP_PIPELINE] = CapabilityState(
            CAP_PIPELINE, False, REASON_NO_PIPELINE_DATA,
            "No governed pipeline data has been supplied for the originating "
            f"portfolios in this scope ({_join([r.portfolio_id for r in originating])}).",
            excluded_portfolios=ids_in_scope)
    else:
        states[CAP_PIPELINE] = CapabilityState(
            CAP_PIPELINE, False, REASON_NON_ORIGINATING,
            "No portfolio in this scope originates new lending, so there is no "
            "origination pipeline to report.",
            excluded_portfolios=ids_in_scope)

    # ---- origination forecast: mirrors pipeline eligibility ---------------- #
    origination_ids = tuple(r.portfolio_id for r in originating)
    if not records:
        states[CAP_ORIGINATION_FORECAST] = CapabilityState(
            CAP_ORIGINATION_FORECAST, False, REASON_NO_PORTFOLIOS_IN_SCOPE,
            "No governed portfolios resolve for this selection.")
    elif origination_ids:
        excluded = tuple(pid for pid in ids_in_scope if pid not in origination_ids)
        states[CAP_ORIGINATION_FORECAST] = CapabilityState(
            CAP_ORIGINATION_FORECAST, True,
            detail=(f"Future originations are projected for: {_join(origination_ids)}."
                    + (f" Excluded — non-originating: {_join(excluded)}."
                       if excluded else "")),
            contributing_portfolios=origination_ids,
            excluded_portfolios=excluded, partial=bool(excluded))
    else:
        states[CAP_ORIGINATION_FORECAST] = CapabilityState(
            CAP_ORIGINATION_FORECAST, False, REASON_NON_ORIGINATING,
            "No portfolio in this scope originates new lending, so no origination "
            "forecast applies.",
            excluded_portfolios=ids_in_scope)

    # ---- runoff forecast: only where a governed curve was SUPPLIED --------- #
    with_curve = tuple(r.portfolio_id for r in records if r.has_runoff_profile)
    without_curve = tuple(r.portfolio_id for r in records if not r.has_runoff_profile)
    if not records:
        states[CAP_RUNOFF_FORECAST] = CapabilityState(
            CAP_RUNOFF_FORECAST, False, REASON_NO_PORTFOLIOS_IN_SCOPE,
            "No governed portfolios resolve for this selection.")
    elif with_curve:
        states[CAP_RUNOFF_FORECAST] = CapabilityState(
            CAP_RUNOFF_FORECAST, True,
            detail=(f"Runoff modelled from a supplied governed curve for: "
                    f"{_join(with_curve)}."
                    + (f" Runoff not modelled for: {_join(without_curve)}."
                       if without_curve else "")),
            contributing_portfolios=with_curve,
            excluded_portfolios=without_curve, partial=bool(without_curve))
    else:
        states[CAP_RUNOFF_FORECAST] = CapabilityState(
            CAP_RUNOFF_FORECAST, False, REASON_NO_RUNOFF_PROFILE,
            "No governed runoff curve has been supplied for the portfolios in "
            "this scope; their balances are held constant and disclosed as such.",
            excluded_portfolios=ids_in_scope)

    # ---- consolidated forecast: every portfolio, per its own treatment ----- #
    if records:
        states[CAP_CONSOLIDATED_FORECAST] = CapabilityState(
            CAP_CONSOLIDATED_FORECAST, True,
            detail=("Projected balance is the sum of every portfolio in scope "
                    "under its governed forecast treatment."),
            contributing_portfolios=ids_in_scope)
    else:
        states[CAP_CONSOLIDATED_FORECAST] = CapabilityState(
            CAP_CONSOLIDATED_FORECAST, False, REASON_NO_PORTFOLIOS_IN_SCOPE,
            "No governed portfolios resolve for this selection.")
    return states


def capabilities_to_dict(states: Mapping[str, CapabilityState]) -> Dict[str, Any]:
    return {k: v.to_dict() for k, v in states.items()}


# --------------------------------------------------------------------------- #
# Field coverage + the disclosure facts a channel renders
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldCoverage:
    """Which portfolios in scope can answer for one requested field."""

    field_name: str
    available_in: Tuple[str, ...] = ()
    unavailable_in: Tuple[str, ...] = ()

    @property
    def fully_available(self) -> bool:
        return bool(self.available_in) and not self.unavailable_in

    @property
    def unavailable_everywhere(self) -> bool:
        return not self.available_in

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_in": list(self.available_in),
            "unavailable_in": list(self.unavailable_in),
        }


@dataclass(frozen=True)
class ScopeCoverage:
    """The governed disclosure facts for one answer.

    The backend produces this; a channel formats it. React may not derive any of
    it, which is what stops a partial answer from being presented as a Total.
    """

    scope: PortfolioScope
    portfolios_used: Tuple[str, ...] = ()
    portfolios_excluded: Dict[str, Dict[str, str]] = field(default_factory=dict)
    field_coverage: Dict[str, FieldCoverage] = field(default_factory=dict)

    @property
    def portfolios_in_scope(self) -> Tuple[str, ...]:
        return self.scope.portfolio_ids

    @property
    def is_fully_consolidated(self) -> bool:
        """True only when every portfolio in scope contributed to the answer."""
        return (bool(self.portfolios_used)
                and len(self.portfolios_used) == len(self.scope.portfolio_ids)
                and not self.portfolios_excluded)

    @property
    def has_data(self) -> bool:
        return bool(self.portfolios_used)

    def disclosure(self) -> Optional[str]:
        """One plain-English sentence a channel may show verbatim, or ``None``.

        Generated here — never assembled by a client — so React, Copilot and a
        PPTX note all disclose the same facts in the same words.
        """
        in_scope = self.scope.portfolio_ids
        if not in_scope:
            return None
        if not self.portfolios_used:
            fields = sorted(self.field_coverage)
            if fields and all(c.unavailable_everywhere
                              for c in self.field_coverage.values()):
                return (f"{', '.join(fields)} is not available in the canonical "
                        f"platform data for {self.scope.label}.")
            return (f"No data contributed for {self.scope.label} "
                    f"({_join(in_scope)}).")
        if self.is_fully_consolidated:
            if len(in_scope) == 1:
                return None
            return (f"Fully consolidated across all {len(in_scope)} portfolios in "
                    f"{self.scope.label}: {_join(in_scope)}.")
        parts = [f"Answered from {_join(self.portfolios_used)}."]
        for pid, info in sorted(self.portfolios_excluded.items()):
            parts.append(
                f"Excluded {pid}: {info.get('detail') or info.get('reason_code')}.")
        parts.append(f"This is NOT a fully consolidated {self.scope.label} answer.")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """The governed envelope block, in the contract's field names."""
        return {
            "scope_requested": {
                "context_id": self.scope.context_id,
                "context_kind": self.scope.context_kind,
                "label": self.scope.label,
            },
            "portfolios_in_scope": list(self.scope.portfolio_ids),
            "portfolios_used": list(self.portfolios_used),
            "portfolios_excluded": {k: dict(v) for k, v in self.portfolios_excluded.items()},
            "is_fully_consolidated": self.is_fully_consolidated,
            "field_coverage": {k: v.to_dict() for k, v in self.field_coverage.items()},
            "disclosure": self.disclosure(),
        }


def build_coverage(
    scope: PortfolioScope,
    *,
    portfolios_with_rows: Iterable[str],
    field_availability: Optional[Mapping[str, Iterable[str]]] = None,
    field_labels: Optional[Mapping[str, str]] = None,
) -> ScopeCoverage:
    """Derive the disclosure facts for one answer.

    ``portfolios_with_rows`` are the portfolios in scope that carried rows after
    filtering. ``field_availability`` maps each requested field onto the
    portfolios that can answer for it. A portfolio contributes only when it has
    rows AND can answer for every requested field; anything else is excluded with
    a reason. Missing values are never treated as zero — an unavailable field
    removes that portfolio from the answer and says so.
    """
    in_scope = list(scope.portfolio_ids)
    with_rows = {str(p) for p in portfolios_with_rows}

    coverage: Dict[str, FieldCoverage] = {}
    for fname, available in (field_availability or {}).items():
        avail = {str(p) for p in available}
        coverage[fname] = FieldCoverage(
            field_name=fname,
            available_in=tuple(p for p in in_scope if p in avail),
            unavailable_in=tuple(p for p in in_scope if p not in avail))

    used: List[str] = []
    excluded: Dict[str, Dict[str, str]] = {}
    for pid in in_scope:
        if pid not in with_rows:
            excluded[pid] = {
                "reason_code": REASON_NO_ROWS_IN_SCOPE,
                "detail": "no rows for this portfolio in the selected data",
            }
            continue
        missing = [f for f, c in coverage.items() if pid in c.unavailable_in]
        if missing:
            labels = [(field_labels or {}).get(f, f) for f in sorted(missing)]
            excluded[pid] = {
                "reason_code": REASON_FIELD_NOT_AVAILABLE,
                "detail": f"{', '.join(labels)} is not available",
            }
            continue
        used.append(pid)
    return ScopeCoverage(scope=scope, portfolios_used=tuple(used),
                         portfolios_excluded=excluded, field_coverage=coverage)


__all__ = [
    "CAPABILITIES", "CAP_COHORTS", "CAP_CONSOLIDATED_FORECAST", "CAP_DRILL_THROUGH",
    "CAP_EVOLUTION", "CAP_EXPORT", "CAP_FUNDED", "CAP_ORIGINATION_FORECAST",
    "CAP_PIPELINE", "CAP_RISK", "CAP_RUNOFF_FORECAST",
    "CONTEXT_ACQUIRED", "CONTEXT_DIRECT", "CONTEXT_KIND_PORTFOLIO",
    "CONTEXT_KIND_TOTAL", "CONTEXT_KIND_TYPE", "CONTEXT_TOTAL",
    "DEFAULT_ORIGINATION_BY_TYPE", "FIELD_PORTFOLIO_ID", "FIELD_PORTFOLIO_LABEL",
    "FIELD_PORTFOLIO_TYPE", "FORECAST_HOLD_CONSTANT", "FORECAST_ORIGINATION",
    "FORECAST_RUNOFF", "GROUP_CONTEXT_IDS", "PORTFOLIO_TYPES",
    "PORTFOLIO_TYPE_ACQUIRED", "PORTFOLIO_TYPE_DIRECT",
    "REASON_FIELD_NOT_AVAILABLE", "REASON_NON_ORIGINATING",
    "REASON_NOT_APPLICABLE", "REASON_NO_PIPELINE_DATA",
    "REASON_NO_PORTFOLIOS_IN_SCOPE", "REASON_NO_ROWS_IN_SCOPE",
    "REASON_NO_RUNOFF_PROFILE",
    "CapabilityState", "FieldCoverage", "PortfolioRecord", "PortfolioRegistry",
    "PortfolioScope", "ScopeCoverage",
    "build_coverage", "build_registry", "capabilities_to_dict", "resolve_capabilities",
    "resolve_scope", "scope_records",
]
