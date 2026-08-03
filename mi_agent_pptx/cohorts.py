"""mi_agent_pptx.cohorts — vintage formation and static-pool seasoning.

The deck used to show one point-in-time cross-section of the current book grouped
by origination year, and said so honestly: *"Point-in-time composition of the
current book by origination year"*. That is not cohort analysis. It cannot answer
the question a funder actually asks — how has a cohort behaved since it formed —
and the platform already answers it for the dashboard.

**No cohort methodology lives here.** Both surfaces come from the same governed
services the React Cohorts tab consumes:

  * :func:`mi_agent_api.cohorts.cohort_analysis` — the point-in-time cohort table
    (``/mi/cohorts``), which defines the vintage assignment and the cohort basis;
  * :func:`mi_agent_api.evolution.funded_cohort_progression` — the static pool
    (``/mi/cohorts/progression``), which defines the cohort as a source-portfolio
    lens ± an origination vintage and tracks THAT cohort across reporting
    periods.

This module chooses which cohorts are worth a slide, derives presentation ratios
over the governed series (``balance[t] / balance[0]`` of the service's own
numbers), and formats. If it ever needs a figure the services do not supply, the
fix is in the service.

Three things it deliberately refuses to do:

* **It does not report exits or redemptions.** The progression service reports a
  loan COUNT per period. A count that falls is a net change: it cannot
  distinguish a redemption from a late-arriving correction, and calling it a
  redemption would be a fabrication dressed as an analytic.
* **It does not blend a seasoned acquired book with a young direct book into one
  cohort series.** Where the scope spans more than one portfolio type, the
  cohorts are reported per type, because a ten-year book and a two-year book
  averaged together describe neither.
* **It does not call anything "retention".** The governed service re-selects the
  cohort by origination vintage in every period rather than freezing the loans
  present at formation, so a cohort can gain balance when a later period boards
  a loan of that vintage. Retention asserts a closed pool that can only shrink;
  the ratio is reported as a change against the cohort's first observed period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: How many vintages the progression slide can carry and stay readable. The rest
#: are disclosed rather than dropped.
MAX_COHORT_SERIES = 4

#: A vintage below this share of the book is not worth its own seasoning curve.
MIN_COHORT_SHARE_PCT = 2.0

#: Governed progression metrics, in the order the slide prefers them.
PROGRESSION_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("funded_balance", "Funded balance", "gbp"),
    ("loan_count", "Loans", "count"),
    ("wa_ltv", "WA current LTV", "pct"),
    ("wa_interest_rate", "WA rate", "pct"),
    ("nneg_headroom_pct", "NNEG headroom", "pct"),
)


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Formation — what entered each vintage.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VintageRow:
    """One origination vintage, exactly as the governed cohort table reports it."""

    vintage: str
    loan_count: Optional[int]
    balance: Optional[float]
    share_pct: Optional[float]
    wa_ltv: Optional[float]
    wa_rate: Optional[float]
    wa_months_on_book: Optional[float]

    @property
    def average_balance(self) -> Optional[float]:
        if not self.balance or not self.loan_count:
            return None
        return self.balance / self.loan_count


@dataclass(frozen=True)
class Formation:
    """The vintage table for one portfolio type (or the whole scope)."""

    label: str
    portfolio_type: Optional[str]
    rows: Tuple[VintageRow, ...]
    reporting_date: Optional[str]
    metrics_available: Tuple[str, ...]
    cohort_basis: Optional[str]
    available: bool
    reason: Optional[str] = None

    @property
    def total_balance(self) -> float:
        return sum(r.balance or 0.0 for r in self.rows)

    @property
    def span(self) -> Optional[str]:
        named = [r.vintage for r in self.rows if r.vintage != "Unknown"]
        if not named:
            return None
        return named[0] if len(named) == 1 else f"{named[0]}–{named[-1]}"


def _pct(value: Any) -> Optional[float]:
    """A governed percentage in POINTS. The services emit fractions for rates and
    LTVs and points elsewhere; the deck displays points everywhere."""
    v = _f(value)
    if v is None:
        return None
    return v * 100.0 if abs(v) <= 1.5 else v


def adapt_formation(payload: Optional[Dict[str, Any]], *, label: str = "Total",
                    portfolio_type: Optional[str] = None) -> Formation:
    """The governed cohort table as presentation rows. Nothing is recomputed."""
    payload = payload or {}
    if not payload.get("available") or not payload.get("cohorts"):
        return Formation(label=label, portfolio_type=portfolio_type, rows=(),
                         reporting_date=payload.get("reportingDate"),
                         metrics_available=tuple(payload.get("metricsAvailable") or ()),
                         cohort_basis=payload.get("cohortBasis"), available=False,
                         reason=payload.get("reason")
                         or "no governed cohort composition for this book")
    rows = tuple(
        VintageRow(
            vintage=str(c.get("cohort") or c.get("vintage") or "Unknown"),
            loan_count=int(c["loanCount"]) if c.get("loanCount") is not None else None,
            balance=_f(c.get("balance")),
            share_pct=_f(c.get("sharePct")),
            wa_ltv=_pct(c.get("waLtv")),
            wa_rate=_pct(c.get("waRate")),
            wa_months_on_book=_f(c.get("waMonthsOnBook")),
        )
        for c in payload["cohorts"])
    # The governed table's own order is not guaranteed chronological; a vintage
    # axis that is not in time order is not a vintage axis.
    rows = tuple(sorted(rows, key=lambda r: (r.vintage == "Unknown", r.vintage)))
    return Formation(label=label, portfolio_type=portfolio_type, rows=rows,
                     reporting_date=payload.get("reportingDate"),
                     metrics_available=tuple(payload.get("metricsAvailable") or ()),
                     cohort_basis=payload.get("cohortBasis"), available=True)


# --------------------------------------------------------------------------- #
# Progression — how a cohort seasoned.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CohortPoint:
    """One reporting period of one cohort's static-pool series."""

    period: str
    reporting_date: Optional[str]
    loan_count: int
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CohortSeries:
    """One cohort tracked across reporting periods, from the governed service."""

    vintage: str
    lens: str
    points: Tuple[CohortPoint, ...]
    metrics_available: Tuple[str, ...]
    available: bool
    reason: Optional[str] = None

    @property
    def live(self) -> Tuple[CohortPoint, ...]:
        """Periods in which the cohort actually had loans.

        A vintage does not exist before it originates, and the governed service
        reports those earlier periods as zero rather than omitting them. Drawing
        the zeros would show every cohort rising from nothing at the left edge —
        an origination event mis-drawn as seasoning.
        """
        first = next((i for i, p in enumerate(self.points) if p.loan_count), None)
        return () if first is None else self.points[first:]

    def value(self, metric: str, index: int) -> Optional[float]:
        pts = self.live
        if not pts or not -len(pts) <= index < len(pts):
            return None
        return pts[index].metrics.get(metric)

    def change_since_first(self, metric: str) -> Optional[float]:
        """Latest as a percentage of the cohort's own FIRST observed period.

        Deliberately not called retention. Retention asserts a closed pool that
        can only shrink, and the governed service does not guarantee one: it
        re-selects the cohort by origination vintage in every period, so a loan
        boarded later with an older origination date joins the cohort and the
        ratio can exceed 100%. Reporting that as "balance retained 237%" would
        be an obviously false statement of an analytic the reader trusts.

        What the ratio IS: two figures the service produced, divided — the same
        class of presentation arithmetic as a share of book.
        """
        first, last = self.value(metric, 0), self.value(metric, -1)
        if not first or last is None:
            return None
        return last / first * 100.0

    @property
    def months_on_book(self) -> Tuple[int, ...]:
        """Seasoning axis: periods SINCE the cohort formed, not calendar dates.

        Comparing cohorts on a calendar axis compares a 2019 vintage at ninety
        months with a 2025 vintage at six; on a seasoning axis they line up.
        """
        return tuple(range(len(self.live)))


def adapt_progression(payload: Optional[Dict[str, Any]], vintage: str) -> CohortSeries:
    payload = payload or {}
    points = tuple(
        CohortPoint(period=str(p.get("period") or ""),
                    reporting_date=p.get("reporting_date"),
                    loan_count=int(p.get("loanCount") or 0),
                    metrics={k: _pct(v) if k in ("wa_ltv", "wa_interest_rate",
                                                 "nneg_headroom_pct") else _f(v)
                             for k, v in (p.get("metrics") or {}).items()})
        for p in payload.get("periods") or ())
    return CohortSeries(
        vintage=vintage, lens=str(payload.get("lens") or "Total"), points=points,
        metrics_available=tuple(payload.get("metricsAvailable") or ()),
        available=bool(payload.get("available")), reason=payload.get("reason"))


# --------------------------------------------------------------------------- #
# Selection + sufficiency.
# --------------------------------------------------------------------------- #

def select_cohorts(formation: Formation,
                   limit: int = MAX_COHORT_SERIES) -> Tuple[List[str], List[str]]:
    """(vintages worth a seasoning curve, vintages disclosed but not drawn).

    Largest first by balance, because a 0.3%-of-book vintage curve is noise on a
    slide an investor reads in thirty seconds. ``Unknown`` is never a cohort — it
    is the bucket for rows whose origination date the tape does not carry.
    """
    material = [r for r in formation.rows
                if r.vintage != "Unknown" and r.balance
                and (r.share_pct is None or r.share_pct >= MIN_COHORT_SHARE_PCT)]
    material.sort(key=lambda r: -(r.balance or 0.0))
    chosen = [r.vintage for r in material[:limit]]
    # Restore chronological order for display: rank decided WHICH, not the axis.
    chosen.sort()
    return chosen, [r.vintage for r in material[limit:]]


def progression_is_meaningful(series: Sequence[CohortSeries]) -> bool:
    """True only when at least one cohort has two or more live periods.

    One period is a formation snapshot, not seasoning, and joining a single point
    into a line is the misleading trend the sufficiency rules exist to prevent.
    """
    return any(s.available and len(s.live) >= 2 for s in series)


def formation_is_meaningful(formation: Formation) -> bool:
    return formation.available and bool(
        [r for r in formation.rows if r.vintage != "Unknown"])


def single_vintage(formation: Formation) -> bool:
    """A book with one origination vintage gets a summary, never a trend line."""
    return len([r for r in formation.rows if r.vintage != "Unknown"]) == 1
