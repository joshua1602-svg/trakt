"""mi_workflows.analytical.narrative — findings in, prose out.

The one rule: **this module may only read findings.** It never touches a frame,
a registry or an engine, so a sentence it writes cannot contain a figure the
deterministic layer did not produce. Where a finding is unavailable it says so
using that finding's own note, rather than omitting the subject and leaving the
reader to assume it was fine.

Formatting decisions (currency symbol, percent scale) come from the same
helpers the rest of the answer surface uses, so an analytical answer and a
routed one render the same number the same way.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_workflows import engine

from .contract import (
    Finding,
    KIND_COHORT,
    KIND_COMPARISON,
    KIND_COMPOSITION,
    KIND_COMPOSITION_SHIFT,
    KIND_FORECAST,
    KIND_LIMIT,
    KIND_MEASURE,
    KIND_MOVEMENT,
    KIND_THRESHOLD,
    KIND_TIMING,
)
from .orchestrator import AnalyticalResult


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def money(value: Optional[float]) -> str:
    if value is None:
        return "—"
    from mi_agent_api import currency as currency_mod
    return currency_mod.format_money(value, suffixes=("bn", "m", "k"))


def _pct(value: Optional[float], decimals: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{decimals}f}%"


def _share(value: Optional[float]) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _share_2dp(value: Optional[float]) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _count(value: Optional[float]) -> str:
    return "—" if value is None else f"{int(round(float(value))):,}"


def value_text(finding: Finding, value: Optional[float] = None) -> str:
    """One finding's value, in its own unit."""
    raw = finding.value if value is None else value
    if raw is None:
        return "—"
    unit = finding.unit
    if unit == engine.UNIT_CURRENCY:
        return money(raw)
    if unit == engine.UNIT_PERCENT:
        return _pct(raw)
    if unit == engine.UNIT_SHARE:
        return _share(raw)
    if finding.aggregation in (engine.AGG_AVERAGE, engine.AGG_WEIGHTED_AVERAGE):
        # An AVERAGE of an integer field is not an integer. Rounding a mean
        # borrower age of 68.4 to "68" throws away the only digit that
        # distinguishes two books, which is precisely what these answers compare.
        return f"{float(raw):,.1f}"
    if unit == engine.UNIT_INTEGER or finding.aggregation == engine.AGG_COUNT:
        return _count(raw)
    return f"{float(raw):,.2f}"


def _delta_text(finding: Finding) -> str:
    if finding.change is None:
        return ""
    unit = finding.unit
    if unit == engine.UNIT_CURRENCY:
        sign = "+" if finding.change >= 0 else "−"
        return f"{sign}{money(abs(finding.change))}"
    if unit == engine.UNIT_PERCENT:
        return f"{finding.change:+.2f}pp"
    if unit == engine.UNIT_SHARE:
        return f"{finding.change * 100:+.1f}pp"
    if finding.aggregation in (engine.AGG_AVERAGE, engine.AGG_WEIGHTED_AVERAGE):
        return f"{finding.change:+,.1f}"
    if unit == engine.UNIT_INTEGER or finding.aggregation == engine.AGG_COUNT:
        return f"{int(round(finding.change)):+,}"
    return f"{finding.change:+,.2f}"


def _period_text(finding: Finding) -> str:
    period = finding.period
    if period is None:
        return ""
    if period.start and period.end and period.start != period.end:
        return f"{period.start} → {period.end}"
    return str(period.end or "")


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def competing_scopes(result: AnalyticalResult) -> Dict[Tuple[str, str], List[Finding]]:
    """Measures reported over more than one scope, with more than one value.

    A2. Two findings that name the SAME measure but were computed over DIFFERENT
    populations are not two facts — they are one question answered twice. Printed
    side by side they read as interchangeable figures, and a reader has no way to
    tell which one the rest of the answer used.

    The test is deliberately narrow, and deliberately about VALUES rather than
    labels: a measure is competing only when its findings disagree on the
    population predicate AND disagree on the number. Two scopes that produce the
    same figure are the same quantity described two ways, and there is nothing
    unsafe about that.

    This contract does not decide which scope is right. It cannot: that is a
    governance question about the population, not a presentation question. It
    declines to present either.
    """
    groups: Dict[Tuple[str, str], List[Finding]] = {}
    for finding in result.findings:
        if not finding.ok or finding.metric is None:
            continue
        value = finding.forecast_value if finding.forecast_value is not None else finding.value
        if value is None:
            continue
        groups.setdefault((finding.metric, finding.kind), []).append(finding)
    # Populations the plan deliberately set against each other. A comparison
    # finding IS the reconciliation: it names both sides and the delta between
    # them, so two figures over those two populations are an intended pair, not
    # one question answered twice. "How has direct moved relative to acquired?"
    # must keep both numbers.
    reconciled: List[frozenset] = []
    for finding in result.of_kind(KIND_COMPARISON):
        left = finding.population.key if finding.population else None
        right = finding.comparand.key if finding.comparand else None
        if left and right:
            reconciled.append(frozenset({left, right}))

    competing: Dict[Tuple[str, str], List[Finding]] = {}
    for key, findings in groups.items():
        if len(findings) < 2:
            continue
        predicates = {(f.population.predicate if f.population else None)
                      for f in findings}
        values = {round(float(f.forecast_value if f.forecast_value is not None
                              else f.value), 2) for f in findings}
        if len(predicates) <= 1 or len(values) <= 1:
            continue
        keys = {f.population.key for f in findings if f.population}
        if any(keys <= pair for pair in reconciled):
            continue
        competing[key] = findings
    return competing


def _competing_scope_sentences(competing) -> List[str]:
    """State that a measure was not presented, and why. No figures."""
    out: List[str] = []
    for (metric, _kind), findings in competing.items():
        label = findings[0].label or metric
        scopes = []
        for finding in findings:
            pop = finding.population
            rows = (f", {_count(pop.rows)} case(s)"
                    if pop is not None and pop.rows is not None else "")
            scopes.append(f"{(pop.label if pop else 'an unnamed scope')}"
                          f"{rows}")
        # No count of scopes here, deliberately: the scopes are named in full
        # below, so a bare numeral would add nothing and would be the one figure
        # in this sentence that no finding holds.
        out.append(
            f"{label} could not be reported: it was produced over more than one "
            f"governed population ({'; '.join(scopes)}) and they do not agree. "
            "Presenting either figure would imply a scope this answer cannot "
            "establish, so neither is shown.")
    return out


def compose(result: AnalyticalResult) -> str:
    """The answer text for one analytical result."""
    competing = competing_scopes(result)
    suppress = set(competing)
    parts: List[str] = []
    parts.extend(_movement_sentences(result))
    parts.extend(_comparison_sentences(result))
    parts.extend(_forecast_sentences(result, suppress=suppress))
    parts.extend(_composition_sentences(result))
    parts.extend(_limit_sentences(result))
    parts.extend(_competing_scope_sentences(competing))
    parts.extend(_unavailable_sentences(result))
    if not parts:
        return ("The governed analytical capabilities this question needs "
                "produced no computable result for this book.")
    return " ".join(p for p in parts if p)


def _movement_sentences(result: AnalyticalResult) -> List[str]:
    moves = [f for f in result.of_kind(KIND_MOVEMENT)
             if f.ok and f.change is not None]
    if not moves:
        return []
    # Group by population so a two-population plan reads as two clauses.
    by_pop: Dict[str, List[Finding]] = {}
    for finding in moves:
        by_pop.setdefault(finding.population.key if finding.population else "",
                          []).append(finding)
    out: List[str] = []
    for group in by_pop.values():
        head = group[0]
        pop = head.population
        label = pop.label if pop else "the book"
        if pop is not None and pop.membership_changed:
            # A3 — a ROLLING COHORT. Membership of this population is defined by
            # a time-relative predicate, so the two dates describe different row
            # sets. Narrating "X → Y" would imply runoff inside one standing
            # population; for a one-month origination window the two cohorts can
            # share no loans at all. Both populations are stated, and the result
            # is named for what it is.
            clauses = [f"{f.label} {value_text(f, f.prior_value)} against "
                       f"{value_text(f)}" for f in group[:5]]
            out.append(
                f"{label}: {_count(pop.rows_prior)} loans at "
                f"{head.period.start if head.period else 'the earlier date'} "
                f"against {_count(pop.rows)} loans at "
                f"{head.period.end if head.period else 'the later date'}. "
                + "; ".join(clauses)
                + ". These are rolling cohorts — membership is set by the "
                "origination window, so a loan joins or leaves it with the "
                "passage of time. This is a comparison of two cohorts, not "
                "movement within one population.")
            continue
        clauses = [f"{f.label} {value_text(f, f.prior_value)} → "
                   f"{value_text(f)} ({_delta_text(f)})" for f in group[:5]]
        rows = (f", {_count(pop.rows)} loans"
                if pop and pop.rows is not None else "")
        out.append(f"Across {_period_text(head)}, {label}{rows}: "
                   + "; ".join(clauses) + ".")
    return out


def _comparison_sentences(result: AnalyticalResult) -> List[str]:
    comparisons = [f for f in result.of_kind(KIND_COMPARISON) if f.ok]
    if not comparisons:
        return []
    left = comparisons[0].population
    right = comparisons[0].comparand
    clauses = []
    for finding in comparisons[:6]:
        clauses.append(
            f"{finding.label} {value_text(finding)} vs "
            f"{value_text(finding, finding.comparand_value)}"
            + (f" ({_delta_text(finding)})" if finding.change is not None else ""))
    lead = (f"{left.label if left else 'A'} against "
            f"{right.label if right else 'B'}")
    counts = ""
    if left is not None and right is not None and left.rows and right.rows:
        counts = f" ({_count(left.rows)} vs {_count(right.rows)} loans)"
    return [f"{lead}{counts}: " + "; ".join(clauses) + "."]


def _forecast_sentences(result: AnalyticalResult,
                        suppress: Optional[set] = None) -> List[str]:
    suppress = suppress or set()
    out: List[str] = []
    emitted_forward: set = set()
    forecasts = [f for f in result.of_kind(KIND_FORECAST)
                 if f.ok and (f.metric, f.kind) not in suppress]
    measures = [f for f in result.of_kind(KIND_MEASURE)
                if f.ok and f.value is not None
                and (f.metric, f.kind) not in suppress]
    for finding in measures:
        if finding.capability in ("funded_balance_forecast", "pipeline_stock"):
            rows = (f" across {_count(finding.population.rows)} case(s)"
                    if finding.population and finding.population.rows is not None
                    and finding.capability == "pipeline_stock" else "")
            out.append(f"{finding.label} is {value_text(finding)}{rows}"
                       + (f" as at {_period_text(finding)}." if _period_text(finding)
                          else "."))
    for finding in forecasts:
        if finding.capability == "completion_run_rate":
            annual = (finding.evidence or {}).get("annualisedRunRate")
            months = (finding.evidence or {}).get("observedMonths")
            out.append(f"{finding.label} is {value_text(finding)}"
                       + (f" ({money(annual)} annualised)" if annual else "")
                       + (f", from {months} observed month(s)." if months else "."))
            continue
        # Two capabilities can legitimately produce the SAME forward figure over
        # the same population — the bridge and the stage breakdown both report
        # the open-pipeline expectation once the governed exclusion applies.
        # A2 does not refuse them, correctly, because they agree; but printing
        # one number twice under two labels is noise, so the second is dropped.
        seen_key = (finding.metric,
                    round(float(finding.forecast_value), 2)
                    if finding.forecast_value is not None else None)
        if seen_key[1] is not None and seen_key in emitted_forward:
            continue
        emitted_forward.add(seen_key)
        excluded_cases = (finding.evidence or {}).get("excludedCaseCount") or 0
        excluded_amount = (finding.evidence or {}).get("excludedFromWeightingAmount")
        # B5 — the governed exclusion belongs in the answer, not only on the
        # envelope. A reader who cannot see which cases were left out cannot
        # tell whether the figure is a forward expectation or a restatement of
        # what has already happened.
        exclusion = ""
        if excluded_cases and excluded_amount:
            exclusion = (f" This excludes {_count(excluded_cases)} case(s) worth "
                         f"{money(excluded_amount)} the extract already shows as "
                         "completed or withdrawn.")
        out.append(f"{finding.label}: "
                   f"{value_text(finding, finding.forecast_value)}"
                   + (f", expected {finding.forecast_date}."
                      if finding.forecast_date else ".")
                   + exclusion)
    timing = [f for f in result.of_kind(KIND_TIMING)
              if f.ok and f.forecast_value is not None]
    if timing:
        clauses = [f"{f.forecast_date} {value_text(f, f.forecast_value)}"
                   for f in timing[:6]]
        out.append("Expected to land: " + "; ".join(clauses) + ".")
    for finding in result.of_kind(KIND_THRESHOLD):
        if not finding.ok:
            continue
        if (finding.evidence or {}).get("reached"):
            out.append(f"{finding.label}: already reached "
                       f"(current {value_text(finding)}).")
        else:
            out.append(f"{finding.label}: around {finding.forecast_date} "
                       f"(current {value_text(finding)}).")
    return out


def _composition_sentences(result: AnalyticalResult) -> List[str]:
    out: List[str] = []
    shifts = [f for f in result.of_kind(KIND_COMPOSITION_SHIFT)
              if f.ok and f.change is not None]
    if shifts:
        top = sorted(shifts, key=lambda f: -abs(f.change or 0.0))[:4]
        clauses = [f"{f.label} {_share(f.prior_share)} → {_share(f.share)} "
                   f"({_delta_text(f)})" for f in top]
        out.append("Composition shift: " + "; ".join(clauses) + ".")
    profile = [f for f in result.of_kind(KIND_COMPOSITION)
               if f.ok and f.share is not None]
    if profile:
        by_dim: Dict[str, List[Finding]] = {}
        for finding in profile:
            by_dim.setdefault(str(finding.metric), []).append(finding)
        clauses = []
        for group in by_dim.values():
            lead = group[0]
            head = lead.label.split(":", 1)
            dimension = head[0].strip()
            names = ", ".join(f"{f.label.split(':', 1)[-1].strip()} {_share(f.share)}"
                              for f in group[:3])
            clauses.append(f"{dimension} — {names}")
        out.append("Current profile: " + "; ".join(clauses) + ".")
    cohorts = [f for f in result.of_kind(KIND_COHORT) if f.ok and f.value is not None]
    if cohorts:
        oldest = cohorts[0]
        newest = cohorts[-1]
        def _ltv(f: Finding) -> str:
            # ``cohort_analysis`` normalises every percent metric to a FRACTION
            # (0.3471 == 34.71%) so one formatter serves a tape that stores LTV
            # as a fraction and the rate in points. Render it as one.
            return _share_2dp((f.evidence or {}).get("waLtv"))
        out.append(
            f"Across {len(cohorts)} governed origination vintage(s), "
            f"{oldest.label} holds {money(oldest.value)} at {_ltv(oldest)} "
            f"weighted-average LTV and {newest.label} holds {money(newest.value)} "
            f"at {_ltv(newest)}.")
    return out


def _limit_sentences(result: AnalyticalResult) -> List[str]:
    limits = [f for f in result.of_kind(KIND_LIMIT) if f.ok]
    if not limits:
        return []
    clauses = [f"{f.label} {value_text(f)} against {value_text(f, f.limit)} "
               f"(headroom {f.headroom:,.2f}, {f.limit_status})"
               for f in limits[:5] if f.headroom is not None]
    return ["Closest to limit: " + "; ".join(clauses) + "."] if clauses else []


def _unavailable_sentences(result: AnalyticalResult) -> List[str]:
    """Say what could not be computed, in the finding's own words.

    A composite answer that silently drops its unavailable half is the failure
    mode this layer exists to avoid: the reader would take the remaining figures
    as the whole answer.
    """
    seen: List[str] = []
    for finding in result.findings:
        if finding.ok or not finding.note:
            continue
        if finding.note not in seen:
            seen.append(finding.note)
    return seen[:4]
