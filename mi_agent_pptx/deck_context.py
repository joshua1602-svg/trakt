"""mi_agent_pptx.deck_context — the governed portfolio context one deck describes.

A deck must be able to say, without ambiguity, *what it is a report about*: which
tenant, which scope, which constituent books, and as of when. Before this module
that answer lived only in an operator-typed ``--client-name`` on the cover, so a
deck covering the acquired book alone was indistinguishable from a total-portfolio
deck.

This is a **thin composition over the existing governed contract**, not a second
model of a portfolio:

    mi_agent_api.portfolio_context.resolve_context()   →  registry + scope + capabilities
    trakt_core.portfolio.PortfolioRecord               →  per-book type, label, dates
    mi_agent_api.snapshots.compute_funded_snapshot()   →  every economic figure

Nothing here computes an economic value. Balances, counts and weighted averages
are read from the governed snapshot payloads (``kpis[].raw``) that the dashboard
renders from — so a deck can never disagree with the workspace, and there is no
second source of economic truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Display order for known portfolio types. Any further governed type is
#: appended alphabetically, so a new book type needs no change here.
_TYPE_ORDER: Tuple[str, ...] = ("direct", "acquired")

#: Investor-facing labels for the known types.
_TYPE_LABELS: Dict[str, str] = {
    "direct": "Direct originations",
    "acquired": "Acquired portfolio",
}

#: Scope ids that mean "the whole book".
CONTEXT_TOTAL = "total"


def type_label(portfolio_type: Optional[str]) -> str:
    """Investor-facing label for a governed portfolio type."""
    if not portfolio_type:
        return "Unclassified"
    return _TYPE_LABELS.get(str(portfolio_type).lower(),
                            str(portfolio_type).replace("_", " ").capitalize())


def _type_sort_key(portfolio_type: str) -> Tuple[int, str]:
    low = str(portfolio_type).lower()
    return (_TYPE_ORDER.index(low), "") if low in _TYPE_ORDER else (len(_TYPE_ORDER), low)


def order_types(types: Sequence[str]) -> List[str]:
    """Governed types in display order (known first, then alphabetical)."""
    return sorted({str(t).lower() for t in types if t}, key=_type_sort_key)


# --------------------------------------------------------------------------- #
# Snapshot accessors — read-only views over the governed funded snapshot.
# --------------------------------------------------------------------------- #

def kpi_of(snapshot: Optional[Mapping[str, Any]], kpi_id: str) -> Optional[Dict[str, Any]]:
    """One KPI tile from a governed funded snapshot, by id."""
    if not isinstance(snapshot, Mapping):
        return None
    for tile in snapshot.get("kpis") or ():
        if isinstance(tile, Mapping) and tile.get("id") == kpi_id:
            return dict(tile)
    return None


def raw_of(snapshot: Optional[Mapping[str, Any]], kpi_id: str) -> Optional[float]:
    """The NUMERIC value behind a KPI tile — the governed figure, not its format.

    Returns ``None`` when the tile is absent or explicitly unavailable, so a
    caller can distinguish "zero" from "this book cannot answer that".
    """
    tile = kpi_of(snapshot, kpi_id)
    if tile is None or tile.get("available") is False:
        return None
    value = tile.get("raw")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def display_of(snapshot: Optional[Mapping[str, Any]], kpi_id: str) -> Optional[str]:
    """The pre-formatted governed display string for a KPI (e.g. ``£25.3MM``)."""
    tile = kpi_of(snapshot, kpi_id)
    if tile is None or tile.get("available") is False:
        return None
    value = tile.get("value")
    return str(value) if value not in (None, "") else None


# --------------------------------------------------------------------------- #
# Value objects.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PortfolioSummary:
    """One constituent book, as the deck needs to describe it."""

    portfolio_id: str
    label: str
    portfolio_type: Optional[str] = None
    reporting_date: Optional[str] = None
    reporting_dates: Tuple[str, ...] = ()
    row_count: Optional[int] = None
    present_in_data: bool = True
    originates: bool = False

    @property
    def type_label(self) -> str:
        return type_label(self.portfolio_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "label": self.label,
            "portfolio_type": self.portfolio_type,
            "reporting_date": self.reporting_date,
            "reporting_dates": list(self.reporting_dates),
            "row_count": self.row_count,
            "present_in_data": self.present_in_data,
            "originates": self.originates,
        }


@dataclass(frozen=True)
class TypeSlice:
    """One portfolio type's governed funded snapshot.

    ``snapshot`` is the payload from ``compute_funded_snapshot`` computed over the
    frame narrowed to this type — the SAME function the total uses, so a type
    slice and the total are guaranteed to be computed identically.
    """

    portfolio_type: str
    portfolio_ids: Tuple[str, ...] = ()
    snapshot: Dict[str, Any] = field(default_factory=dict)
    reporting_date: Optional[str] = None
    reporting_dates: Tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return type_label(self.portfolio_type)

    def raw(self, kpi_id: str) -> Optional[float]:
        return raw_of(self.snapshot, kpi_id)

    def display(self, kpi_id: str) -> Optional[str]:
        return display_of(self.snapshot, kpi_id)

    # -- the figures the composition and comparison slides bind to ----------
    @property
    def balance(self) -> Optional[float]:
        return self.raw("balance")

    @property
    def loan_count(self) -> Optional[float]:
        return self.raw("loans")

    @property
    def wa_ltv(self) -> Optional[float]:
        return self.raw("wa_current_ltv")

    @property
    def wa_rate(self) -> Optional[float]:
        return self.raw("wa_rate")

    @property
    def avg_balance(self) -> Optional[float]:
        return self.raw("avg_balance")

    @property
    def balance_movement(self) -> Optional[float]:
        """Month-on-month balance change, governed (``mom_balance``)."""
        return self.raw("mom_balance")

    @property
    def loan_movement(self) -> Optional[float]:
        return self.raw("mom_loans")

    @property
    def has_movement(self) -> bool:
        return self.balance_movement is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_type": self.portfolio_type,
            "label": self.label,
            "portfolio_ids": list(self.portfolio_ids),
            "reporting_date": self.reporting_date,
            "balance": self.balance,
            "loan_count": self.loan_count,
            "wa_ltv": self.wa_ltv,
            "wa_rate": self.wa_rate,
            "avg_balance": self.avg_balance,
            "balance_movement": self.balance_movement,
            "loan_movement": self.loan_movement,
        }


@dataclass(frozen=True)
class DeckPortfolioContext:
    """What this deck is a report about. Passed to every slide."""

    tenant_id: str
    client_id: str
    report_scope: str = CONTEXT_TOTAL
    scope_label: str = "Total"
    scope_kind: str = "total"
    portfolios: Tuple[PortfolioSummary, ...] = ()
    type_slices: Tuple[TypeSlice, ...] = ()
    reporting_date: Optional[str] = None
    #: Set when the requested scope could not be matched and Total was used.
    fell_back_to_total: bool = False
    requested_scope: Optional[str] = None
    #: The governed total snapshot for the resolved scope.
    snapshot: Dict[str, Any] = field(default_factory=dict)

    # -- identity ----------------------------------------------------------
    @property
    def portfolio_ids(self) -> Tuple[str, ...]:
        return tuple(p.portfolio_id for p in self.portfolios)

    @property
    def portfolio_names(self) -> Tuple[str, ...]:
        return tuple(p.label for p in self.portfolios)

    @property
    def portfolio_types(self) -> Tuple[str, ...]:
        return tuple(order_types([p.portfolio_type for p in self.portfolios
                                  if p.portfolio_type]))

    @property
    def portfolio_count(self) -> int:
        return len(self.portfolios)

    # -- economics (read from the governed snapshot, never computed here) ---
    @property
    def total_balance(self) -> Optional[float]:
        return raw_of(self.snapshot, "balance")

    @property
    def loan_count(self) -> Optional[float]:
        return raw_of(self.snapshot, "loans")

    @property
    def total_balance_display(self) -> Optional[str]:
        return display_of(self.snapshot, "balance")

    # -- composition -------------------------------------------------------
    def slice_for(self, portfolio_type: str) -> Optional[TypeSlice]:
        low = str(portfolio_type or "").lower()
        for sl in self.type_slices:
            if sl.portfolio_type == low:
                return sl
        return None

    @property
    def has_direct(self) -> bool:
        return self.slice_for("direct") is not None

    @property
    def has_acquired(self) -> bool:
        return self.slice_for("acquired") is not None

    @property
    def is_mixed(self) -> bool:
        """True when more than one governed portfolio type is in scope.

        This is the condition that makes a blended average misleading and a
        split view mandatory.
        """
        return len(self.type_slices) > 1

    @property
    def is_total(self) -> bool:
        return self.scope_kind == "total"

    # -- reporting dates ---------------------------------------------------
    @property
    def reporting_dates(self) -> Dict[str, str]:
        """``portfolio_id -> reporting date`` for every book in scope."""
        return {p.portfolio_id: p.reporting_date for p in self.portfolios
                if p.reporting_date}

    @property
    def type_reporting_dates(self) -> Dict[str, str]:
        """``portfolio_type -> latest reporting date`` for every type in scope."""
        out: Dict[str, str] = {}
        for p in self.portfolios:
            if not (p.portfolio_type and p.reporting_date):
                continue
            key = str(p.portfolio_type).lower()
            if p.reporting_date > out.get(key, ""):
                out[key] = p.reporting_date
        return out

    @property
    def distinct_reporting_dates(self) -> Tuple[str, ...]:
        return tuple(sorted({d for d in self.reporting_dates.values() if d}))

    @property
    def has_mixed_reporting_dates(self) -> bool:
        """True when constituent books do not share one reporting date.

        When this is true the total is a combination across dates and must say
        so — an investor must never read a mixed-date total as a clean cut.
        """
        return len(self.distinct_reporting_dates) > 1

    # -- reconciliation ----------------------------------------------------
    def type_balance_total(self) -> Optional[float]:
        """Sum of the per-type governed balances (for the reconciliation gate)."""
        vals = [sl.balance for sl in self.type_slices if sl.balance is not None]
        return sum(vals) if vals else None

    def type_loan_total(self) -> Optional[float]:
        vals = [sl.loan_count for sl in self.type_slices if sl.loan_count is not None]
        return sum(vals) if vals else None

    def to_dict(self) -> Dict[str, Any]:
        """The artefact-metadata view of this context."""
        return {
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "report_scope": self.report_scope,
            "scope_label": self.scope_label,
            "scope_kind": self.scope_kind,
            "portfolio_ids": list(self.portfolio_ids),
            "portfolio_names": list(self.portfolio_names),
            "portfolio_types": list(self.portfolio_types),
            "reporting_date": self.reporting_date,
            "reporting_dates": self.reporting_dates,
            "type_reporting_dates": self.type_reporting_dates,
            "mixed_reporting_dates": self.has_mixed_reporting_dates,
            "total_balance": self.total_balance,
            "loan_count": self.loan_count,
            "portfolios": [p.to_dict() for p in self.portfolios],
            "type_slices": [s.to_dict() for s in self.type_slices],
            "fell_back_to_total": self.fell_back_to_total,
            "requested_scope": self.requested_scope,
        }


# --------------------------------------------------------------------------- #
# Construction.
# --------------------------------------------------------------------------- #

def _summaries(registry: Any, scope: Any) -> Tuple[PortfolioSummary, ...]:
    """The governed registry records inside a scope, as deck summaries."""
    try:
        from trakt_core.portfolio import scope_records
        records = scope_records(registry, scope)
    except Exception:  # noqa: BLE001 — a registry problem must not break the deck
        records = []
    out: List[PortfolioSummary] = []
    for rec in records:
        dates = tuple(getattr(rec, "reporting_dates", ()) or ())
        out.append(PortfolioSummary(
            portfolio_id=getattr(rec, "portfolio_id", ""),
            label=getattr(rec, "display_label", "") or getattr(rec, "portfolio_id", ""),
            portfolio_type=getattr(rec, "portfolio_type", None),
            reporting_date=dates[-1] if dates else None,
            reporting_dates=dates,
            row_count=getattr(rec, "row_count", None),
            present_in_data=bool(getattr(rec, "present_in_data", True)),
            originates=bool(getattr(rec, "originates", False)),
        ))
    return tuple(out)


def build_context(*, tenant_id: str, client_id: str, registry: Any, scope: Any,
                  snapshot: Optional[Mapping[str, Any]] = None,
                  type_snapshots: Optional[Mapping[str, Mapping[str, Any]]] = None,
                  reporting_date: Optional[str] = None) -> DeckPortfolioContext:
    """Compose the deck context from an already-resolved governed scope.

    The caller owns data resolution (it holds the frame and the environment);
    this function only assembles. ``type_snapshots`` maps a governed portfolio
    type to the funded snapshot computed over that type's rows — supplied only
    when more than one type is in scope, so a single-book deck costs nothing
    extra.
    """
    portfolios = _summaries(registry, scope)
    by_type: Dict[str, List[PortfolioSummary]] = {}
    for p in portfolios:
        if p.portfolio_type:
            by_type.setdefault(str(p.portfolio_type).lower(), []).append(p)

    slices: List[TypeSlice] = []
    for ptype in order_types(by_type.keys()):
        snap = (type_snapshots or {}).get(ptype)
        if snap is None:
            continue
        members = by_type[ptype]
        dates = tuple(sorted({p.reporting_date for p in members if p.reporting_date}))
        slices.append(TypeSlice(
            portfolio_type=ptype,
            portfolio_ids=tuple(p.portfolio_id for p in members),
            snapshot=dict(snap),
            reporting_date=dates[-1] if dates else None,
            reporting_dates=dates,
        ))

    return DeckPortfolioContext(
        tenant_id=str(tenant_id or client_id or ""),
        client_id=str(client_id or ""),
        report_scope=str(getattr(scope, "context_id", CONTEXT_TOTAL) or CONTEXT_TOTAL),
        scope_label=str(getattr(scope, "label", "Total") or "Total"),
        scope_kind=str(getattr(scope, "context_kind", "total") or "total"),
        portfolios=portfolios,
        type_slices=tuple(slices),
        reporting_date=reporting_date,
        fell_back_to_total=bool(getattr(scope, "fell_back_to_total", False)),
        requested_scope=getattr(scope, "requested_context_id", None),
        snapshot=dict(snapshot or {}),
    )
