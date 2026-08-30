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
* **It does not plot a cohort whose loan count RISES.** The static-pool contract
  is that a loan belongs to one vintage for life, the pool is fixed at formation,
  and the surviving count can hold or fall but never rise. The governed service
  implements membership by re-filtering each reporting period on origination
  vintage rather than by freezing a loan-id set, so on a book where a later
  period boards a loan of an earlier vintage the count CAN rise — proven with
  loan identifiers in ``test_cohort_static_pool.py``. Such a cohort is not a
  static pool, so it is omitted with a recorded reason and a publication gate
  blocks any deck that plots one. Nothing here re-derives membership: the deck
  either presents the governed series as a static pool or declines to.
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
    """One reporting period of one cohort's static-pool series.

    Every field is the governed payload's own. Nothing here is derived: exits and
    retention used to be computed in the deck from counts, which meant two
    channels could disagree about the same cohort by rounding differently.
    """

    period: str
    reporting_date: Optional[str]
    loan_count: int
    surviving_ids: Tuple[str, ...] = ()
    exits_count: Optional[int] = None
    loan_retention: Optional[float] = None
    balance_retention: Optional[float] = None
    seasoning: Optional[int] = None
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
    #: The governed static-pool contract, verbatim.
    methodology_version: Optional[str] = None
    membership_rule: Optional[str] = None
    formation_date: Optional[str] = None
    formation_ids: Tuple[str, ...] = ()
    formation_loan_count: Optional[int] = None
    formation_balance: Optional[float] = None
    data_quality: Dict[str, Any] = field(default_factory=dict)

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

    @property
    def violates_static_pool(self) -> bool:
        """True when the surviving count RISES between live periods.

        The static-pool contract permits a count to hold or fall. A rise means
        the governed series is not describing a fixed pool for this cohort, and
        every downstream measure — retention, exits, seasoning — would then be
        computed over a moving population. Detected, never corrected: correcting
        it here would be a second interpretation of the methodology.
        """
        counts = [p.loan_count for p in self.live]
        return any(b > a for a, b in zip(counts, counts[1:]))

    @property
    def formation_count(self) -> Optional[int]:
        """The service's formation count, not the first plotted point's."""
        if self.formation_loan_count is not None:
            return self.formation_loan_count
        pts = self.live
        return pts[0].loan_count if pts else None

    @property
    def surviving_count(self) -> Optional[int]:
        pts = self.live
        return pts[-1].loan_count if pts else None

    @property
    def exits(self) -> Optional[int]:
        """Loans in the pool at formation that are no longer in it.

        Meaningful only for a pool that never gained a member, which is why a
        cohort failing :attr:`violates_static_pool` is never plotted.
        """
        pts = self.live
        if pts and pts[-1].exits_count is not None:
            return pts[-1].exits_count           # the service's own figure
        if self.formation_count is None or self.surviving_count is None:
            return None
        return max(0, self.formation_count - self.surviving_count)

    def retention(self, metric: str) -> Optional[float]:
        """Latest as a percentage of the cohort's value at formation.

        Legitimate here because the only cohorts reaching a slide are those the
        governed series shows as never gaining a member, so the denominator is
        the pool the numerator is a subset of.

        NOTE ON WORDING. "Retention" is a survival idea, and it is only that for
        a COUNT: a loan is in the pool or it is not, and the ratio cannot exceed
        100%. A BALANCE ratio is a different measure wearing the same name — on
        a book where interest rolls up it exceeds 100% while loans are leaving —
        so the two are exposed under separate names below and the presentation
        layer must not call the balance one retention.
        """
        pts = self.live
        if pts:
            governed = {"funded_balance": pts[-1].balance_retention,
                        "loan_count": pts[-1].loan_retention}.get(metric)
            if governed is not None:
                return governed              # the service's own figure
        first, last = self.value(metric, 0), self.value(metric, -1)
        if not first or last is None:
            return None
        return last / first * 100.0

    @property
    def loan_survival(self) -> Optional[float]:
        """Surviving loans as a percentage of the loans at formation.

        This IS retention: a count-based survival ratio, bounded by 100%.
        """
        return self.retention("loan_count")

    @property
    def balance_vs_formation(self) -> Optional[float]:
        """Latest balance as a percentage of the balance at formation.

        NOT retention, and never labelled as such. It is the net of everything
        that happened to the pool's balance — exits, repayment, and on a
        roll-up book the interest that accrued — and it can legitimately exceed
        100% while loans are leaving. Trakt does not decompose it here; the
        attribution that would is not a capability this pack carries.
        """
        return self.retention("funded_balance")

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
                    surviving_ids=tuple(p.get("survivingLoanIds") or ()),
                    exits_count=p.get("exitsCount"),
                    loan_retention=_f(p.get("loanRetention")),
                    balance_retention=_f(p.get("balanceRetention")),
                    seasoning=p.get("seasoningPeriods"),
                    metrics={k: _pct(v) if k in ("wa_ltv", "wa_interest_rate",
                                                 "nneg_headroom_pct") else _f(v)
                             for k, v in (p.get("metrics") or {}).items()})
        for p in payload.get("periods") or ())
    return CohortSeries(
        vintage=vintage, lens=str(payload.get("lens") or "Total"), points=points,
        metrics_available=tuple(payload.get("metricsAvailable") or ()),
        available=bool(payload.get("available")), reason=payload.get("reason"),
        methodology_version=payload.get("methodologyVersion"),
        membership_rule=payload.get("membershipRule"),
        formation_date=payload.get("formationDate"),
        formation_ids=tuple(payload.get("formationLoanIds") or ()),
        formation_loan_count=payload.get("formationLoanCount"),
        formation_balance=_f(payload.get("formationBalance")),
        data_quality=dict(payload.get("dataQuality") or {}))


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


def plottable(series: Sequence[CohortSeries]) -> List[CohortSeries]:
    """The cohorts a seasoning slide may draw.

    Two conditions, both necessary: at least two live periods (one period is a
    formation snapshot, and joining a single point into a line is the misleading
    trend the sufficiency rules exist to prevent), and a surviving count that
    never rises (otherwise it is not a static pool).
    """
    return [s for s in series
            if s.available and len(s.live) >= 2 and not s.violates_static_pool]


def rejected(series: Sequence[CohortSeries]) -> List[Tuple[str, str]]:
    """``[(vintage, reason)]`` for cohorts the slide declined, for disclosure."""
    out: List[Tuple[str, str]] = []
    for s in series:
        if not s.available:
            out.append((s.vintage, s.reason or "no governed series"))
        elif s.violates_static_pool:
            out.append((s.vintage,
                        "surviving loan count rises between reporting periods, "
                        "so this is not a fixed pool"))
        elif len(s.live) < 2:
            out.append((s.vintage, "only one reporting period holds this cohort"))
    return out


def progression_is_meaningful(series: Sequence[CohortSeries]) -> bool:
    return bool(plottable(series))


def formation_is_meaningful(formation: Formation) -> bool:
    return formation.available and bool(
        [r for r in formation.rows if r.vintage != "Unknown"])


def single_vintage(formation: Formation) -> bool:
    """A book with one origination vintage gets a summary, never a trend line."""
    return len([r for r in formation.rows if r.vintage != "Unknown"]) == 1
