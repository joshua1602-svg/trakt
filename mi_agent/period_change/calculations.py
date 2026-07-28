"""mi_agent.period_change.calculations — the deterministic movement engine.

Every number a Period Change Analysis publishes is produced here, from a governed
Business Semantics Registry entry and two dataframes. No LLM, no heuristics, no
client-specific rule, and no threshold.

Two layers:

* :func:`aggregate` — ONE governed aggregate at ONE snapshot, plus the population
  evidence that justifies it (valid rows, excluded rows, numerator, denominator,
  total weight). Aggregation follows ``default_aggregation`` exactly.
* :func:`metric_change` — the movement between two such aggregates, calculated
  according to the entry's ``temporality`` and expressed in the field's unit.

The temporality rules (§7) are the point of the module:

===================  =====================================================
``point_in_time``    aggregate at each date; movement is the difference
``period_flow``      each date's aggregate is that period's own flow; the
                     two are compared as independent period totals
``cumulative``       the movement IS the difference of the cumulative
                     positions; a fall is reported with a warning and is
                     never floored at zero
``static_baseline``  no "changed between periods" calculation at all
===================  =====================================================

Numeric parsing is delegated to ``analytics_lib.numeric.coerce_numeric``, the
platform's single deterministic coercion, so a period-change figure parses a
tape exactly as every other MI figure does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from ..business_semantics import (
    AGG_AVERAGE,
    AGG_DISTRIBUTION,
    AGG_SHARE,
    AGG_SUM,
    AGG_WEIGHTED_AVERAGE,
    DIRECTION_CONTEXT_DEPENDENT,
    DIRECTION_HIGHER_IS_BETTER,
    DIRECTION_HIGHER_IS_WORSE,
    DIRECTION_LOWER_IS_BETTER,
    DIRECTION_LOWER_IS_WORSE,
    DIRECTION_NEUTRAL,
    TEMPORALITY_CUMULATIVE,
    TEMPORALITY_PERIOD_FLOW,
    TEMPORALITY_POINT_IN_TIME,
    TEMPORALITY_STATIC_BASELINE,
    BusinessSemanticsEntry,
)
from ..mi_dataset_profile import PERCENT_POINTS, percent_storage_scale
from .models import (
    BASIS_CUMULATIVE_DIFFERENCE,
    BASIS_FLOW_LEVEL_COMPARISON,
    BASIS_POINT_IN_TIME_DIFFERENCE,
    BASIS_STATIC_BASELINE_REFERENCE,
    CUMULATIVE_DECLINE_NOTE,
    FLOW_COMPARISON_NOTE,
    INTERPRETATION_DETERIORATION,
    INTERPRETATION_IMPROVEMENT,
    INTERPRETATION_NO_MOVEMENT,
    INTERPRETATION_NOT_ASSESSED,
    STATUS_AVAILABLE,
    STATUS_INVALID_WEIGHT_POPULATION,
    STATUS_NOT_AVAILABLE,
    STATUS_NOT_COMPARABLE,
    STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY,
    STATUS_PARTIALLY_AVAILABLE,
    STATUS_SOURCE_CONVENTION_UNCERTAIN,
    STATUS_ZERO_DENOMINATOR,
    UNIT_CURRENCY,
    UNIT_PERCENTAGE_POINT,
    AggregateOutcome,
    MetricChange,
    SnapshotFrame,
    evidence_ref,
)
from .units import FieldUnit, resolve_unit

#: The governed balance field the platform already treats as portfolio exposure.
#: Used as a share denominator and as the bridge's balance. It is a canonical
#: field name, not a guess: every existing MI service uses this same column.
BALANCE_FIELD = "current_outstanding_balance"

#: Tape columns that may carry the exposure currency, most canonical first.
#: Mirrors ``mi_agent_api.currency._CURRENCY_FIELDS`` — duplicated as data (not
#: imported) so the domain layer keeps no dependency on the API package.
CURRENCY_FIELDS: Tuple[str, ...] = (
    "exposure_currency_denomination", "currency_denomination", "collateral_currency")

#: Controlled flag vocabulary for a ``share`` aggregation. A value outside BOTH
#: sets is NOT silently treated as false: it is excluded from the denominator and
#: reported, because counting an unrecognised code as "no" would understate every
#: flag share on a tape with a local convention.
TRUE_TOKENS: frozenset = frozenset({
    "y", "yes", "true", "t", "1", "1.0"})
FALSE_TOKENS: frozenset = frozenset({
    "n", "no", "false", "f", "0", "0.0"})

#: Share bases this workflow can calculate. ``count`` is the v2 default for flag
#: shares; ``balance`` is supported because the governed balance field exists.
SUPPORTED_SHARE_BASES: Tuple[str, ...] = ("count", "balance")

_MIXED_CURRENCY_NOTE = (
    "The snapshot reports more than one exposure currency and no governed "
    "currency-normalisation capability exists, so monetary amounts are not "
    "aggregated across currencies.")


# --------------------------------------------------------------------------- #
# Pair context — decisions taken ONCE for both snapshots
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairContext:
    """Scaling and unit decisions shared by both snapshots of one comparison.

    The percent storage scale is decided across BOTH frames together. Deciding it
    per snapshot would let a start value be read as a fraction and an end value
    as points on the same field, which silently manufactures a 5,000-point
    movement — the exact class of error this workflow exists to prevent.
    """

    entry: BusinessSemanticsEntry
    unit: FieldUnit
    multiplier: float = 1.0
    percent_scale: Optional[str] = None

    @property
    def field(self) -> str:
        return self.entry.field


def _combined_series(frames: Sequence[Any], column: str) -> pd.Series:
    parts = [f[column] for f in frames
             if f is not None and column in getattr(f, "columns", [])]
    if not parts:
        return pd.Series(dtype="float64")
    return pd.concat(parts, ignore_index=True)


def build_pair_context(entry: BusinessSemanticsEntry,
                       frames: Sequence[Any]) -> PairContext:
    """The scaling context for one field across the two snapshots."""
    unit = resolve_unit(entry)

    if entry.default_aggregation == AGG_SHARE:
        # A share is computed as a fraction here and reported in points.
        return PairContext(entry, unit, multiplier=100.0)

    if (unit.unit == UNIT_PERCENTAGE_POINT
            and entry.default_aggregation in (AGG_AVERAGE, AGG_WEIGHTED_AVERAGE)):
        series = _combined_series(frames, entry.field)
        scale = percent_storage_scale(series) if not series.empty else None
        multiplier = 1.0 if scale == PERCENT_POINTS else 100.0
        return PairContext(entry, unit, multiplier=multiplier, percent_scale=scale)

    return PairContext(entry, unit)


# --------------------------------------------------------------------------- #
# Population helpers
# --------------------------------------------------------------------------- #
def _row_count(df: Any) -> int:
    return int(len(df)) if df is not None else 0


def _mixed_currency(df: Any) -> bool:
    """True when the frame reports more than one exposure currency."""
    if df is None:
        return False
    for column in CURRENCY_FIELDS:
        if column not in getattr(df, "columns", []):
            continue
        values = df[column].astype("string").str.strip().str.upper()
        distinct = {v for v in values.dropna().unique() if v}
        if len(distinct) > 1:
            return True
    return False


def _flag_counts(series: pd.Series) -> Tuple[int, int, int]:
    """``(true_count, recognised_count, unrecognised_count)`` for a flag column."""
    text = series.astype("string").str.strip().str.lower()
    present = text.notna() & text.ne("") & ~text.isin(["nan", "none", "<na>"])
    values = text[present]
    is_true = values.isin(TRUE_TOKENS)
    is_false = values.isin(FALSE_TOKENS)
    recognised = int((is_true | is_false).sum())
    return int(is_true.sum()), recognised, int(len(values)) - recognised


# --------------------------------------------------------------------------- #
# Aggregation — §8
# --------------------------------------------------------------------------- #
def aggregate(df: Any, ctx: PairContext) -> AggregateOutcome:
    """The governed aggregate for one field at one snapshot.

    ``default_aggregation`` is followed exactly. In particular a
    ``weighted_average`` never falls back to an unweighted mean: an absent or
    empty weight population is reported as such, because an unweighted answer to
    a weighted question is a different number wearing the same label.
    """
    entry = ctx.entry
    aggregation = entry.default_aggregation
    total_rows = _row_count(df)

    if df is None or entry.field not in getattr(df, "columns", []):
        return AggregateOutcome(
            None, STATUS_NOT_AVAILABLE, aggregation,
            excluded_population=total_rows,
            notes=(f"Field {entry.field!r} is not present in this snapshot.",))

    if aggregation == AGG_DISTRIBUTION:
        return AggregateOutcome(
            None, STATUS_NOT_AVAILABLE, aggregation,
            notes=("A distribution is calculated by the distribution analysis, "
                   "not as a single aggregate.",))

    if aggregation == AGG_SHARE:
        return _aggregate_share(df, ctx, total_rows)
    if aggregation == AGG_WEIGHTED_AVERAGE:
        return _aggregate_weighted(df, ctx, total_rows)
    if aggregation in (AGG_SUM, AGG_AVERAGE):
        return _aggregate_numeric(df, ctx, total_rows)

    return AggregateOutcome(
        None, STATUS_NOT_AVAILABLE, aggregation,
        notes=(f"Aggregation {aggregation!r} is not a governed period-change "
               f"calculation.",))


def _currency_guard(df: Any, ctx: PairContext) -> Tuple[str, ...]:
    if ctx.unit.unit == UNIT_CURRENCY and _mixed_currency(df):
        return (_MIXED_CURRENCY_NOTE,)
    return ()


def _aggregate_numeric(df: Any, ctx: PairContext, total_rows: int
                       ) -> AggregateOutcome:
    entry = ctx.entry
    values = coerce_numeric(df[entry.field])
    valid_mask = values.notna()
    valid = int(valid_mask.sum())
    excluded = total_rows - valid
    notes = _currency_guard(df, ctx)

    if notes:
        return AggregateOutcome(
            None, STATUS_NOT_COMPARABLE, entry.default_aggregation,
            valid_population=valid, excluded_population=excluded, notes=notes)
    if valid == 0:
        return AggregateOutcome(
            None, STATUS_NOT_AVAILABLE, entry.default_aggregation,
            excluded_population=total_rows,
            notes=("No numerically valid value in this snapshot.",))

    usable = values[valid_mask]
    raw = float(usable.sum()) if entry.default_aggregation == AGG_SUM \
        else float(usable.mean())
    status = STATUS_AVAILABLE if excluded == 0 else STATUS_PARTIALLY_AVAILABLE
    if excluded:
        notes = notes + (f"{excluded} of {total_rows} rows carried no valid "
                         f"numeric value and were excluded.",)
    return AggregateOutcome(
        raw * ctx.multiplier, status, entry.default_aggregation,
        valid_population=valid, excluded_population=excluded,
        numerator=raw * ctx.multiplier if entry.default_aggregation == AGG_SUM else None,
        denominator=float(valid) if entry.default_aggregation == AGG_AVERAGE else None,
        notes=notes)


def _aggregate_weighted(df: Any, ctx: PairContext, total_rows: int
                        ) -> AggregateOutcome:
    entry = ctx.entry
    weight_field = entry.weight_field
    if not weight_field:
        return AggregateOutcome(
            None, STATUS_INVALID_WEIGHT_POPULATION, AGG_WEIGHTED_AVERAGE,
            excluded_population=total_rows,
            notes=("The registry requires a weighted average but declares no "
                   "weight field; an unweighted average is not substituted.",))
    if weight_field not in getattr(df, "columns", []):
        return AggregateOutcome(
            None, STATUS_INVALID_WEIGHT_POPULATION, AGG_WEIGHTED_AVERAGE,
            weight_field=weight_field, excluded_population=total_rows,
            notes=(f"Weight field {weight_field!r} is not present in this "
                   f"snapshot; an unweighted average is not substituted.",))

    values = coerce_numeric(df[entry.field])
    weights = coerce_numeric(df[weight_field])
    mask = values.notna() & weights.notna()
    valid = int(mask.sum())
    excluded = total_rows - valid

    if valid == 0:
        return AggregateOutcome(
            None, STATUS_INVALID_WEIGHT_POPULATION, AGG_WEIGHTED_AVERAGE,
            weight_field=weight_field, excluded_population=total_rows,
            notes=("No row carries both a valid metric value and a valid "
                   "weight; an unweighted average is not substituted.",))

    vv, ww = values[mask], weights[mask]
    total_weight = float(ww.sum())
    if total_weight == 0:
        return AggregateOutcome(
            None, STATUS_ZERO_DENOMINATOR, AGG_WEIGHTED_AVERAGE,
            weight_field=weight_field, valid_population=valid,
            excluded_population=excluded, total_weight=0.0, denominator=0.0,
            notes=("The total eligible weight is zero, so a weighted average is "
                   "undefined; an unweighted average is not substituted.",))

    numerator = float((vv * ww).sum())
    raw = numerator / total_weight
    status = STATUS_AVAILABLE if excluded == 0 else STATUS_PARTIALLY_AVAILABLE
    notes: Tuple[str, ...] = ()
    if excluded:
        notes += (f"{excluded} of {total_rows} rows lacked a valid metric value "
                  f"or a valid weight and were excluded from the weighted "
                  f"population.",)
    return AggregateOutcome(
        raw * ctx.multiplier, status, AGG_WEIGHTED_AVERAGE,
        valid_population=valid, excluded_population=excluded,
        numerator=numerator * ctx.multiplier, denominator=total_weight,
        total_weight=total_weight, weight_field=weight_field, notes=notes)


def _aggregate_share(df: Any, ctx: PairContext, total_rows: int
                     ) -> AggregateOutcome:
    entry = ctx.entry
    basis = (entry.share_basis or "count").strip().lower()
    if basis not in SUPPORTED_SHARE_BASES:
        return AggregateOutcome(
            None, STATUS_NOT_AVAILABLE, AGG_SHARE, share_basis=basis,
            excluded_population=total_rows,
            notes=(f"Share basis {basis!r} has no governed calculation; a count "
                   f"share is not substituted.",))

    series = df[entry.field]
    true_count, recognised, unrecognised = _flag_counts(series)
    missing = total_rows - (recognised + unrecognised)

    if basis == "count":
        numerator, denominator = float(true_count), float(recognised)
    else:
        if BALANCE_FIELD not in getattr(df, "columns", []):
            return AggregateOutcome(
                None, STATUS_NOT_AVAILABLE, AGG_SHARE, share_basis=basis,
                excluded_population=total_rows,
                notes=(f"A balance share needs {BALANCE_FIELD!r}, which is not "
                       f"present in this snapshot; a count share is not "
                       f"substituted.",))
        text = series.astype("string").str.strip().str.lower()
        balance = coerce_numeric(df[BALANCE_FIELD])
        is_true = text.isin(TRUE_TOKENS) & balance.notna()
        is_known = text.isin(TRUE_TOKENS | FALSE_TOKENS) & balance.notna()
        numerator = float(balance[is_true].sum())
        denominator = float(balance[is_known].sum())

    notes: Tuple[str, ...] = ()
    status = STATUS_AVAILABLE
    if unrecognised:
        notes += (f"{unrecognised} rows carried a flag value outside the "
                  f"governed yes/no vocabulary and were excluded from both the "
                  f"numerator and the denominator.",)
        status = STATUS_SOURCE_CONVENTION_UNCERTAIN
    if missing:
        notes += (f"{missing} of {total_rows} rows carried no flag value.",)
        if status == STATUS_AVAILABLE:
            status = STATUS_PARTIALLY_AVAILABLE

    if denominator == 0:
        return AggregateOutcome(
            None, STATUS_ZERO_DENOMINATOR, AGG_SHARE, share_basis=basis,
            valid_population=recognised, excluded_population=total_rows - recognised,
            numerator=numerator, denominator=0.0,
            notes=notes + ("The share denominator is zero.",))

    return AggregateOutcome(
        (numerator / denominator) * ctx.multiplier, status, AGG_SHARE,
        valid_population=recognised, excluded_population=total_rows - recognised,
        numerator=numerator, denominator=denominator, share_basis=basis,
        notes=notes)


# --------------------------------------------------------------------------- #
# Directionality — §12
# --------------------------------------------------------------------------- #
#: ``directionality`` → the interpretation of a POSITIVE movement. A negative
#: movement is the opposite; ``None`` means no judgement is asserted at all.
_POSITIVE_MEANS: Dict[str, Optional[str]] = {
    DIRECTION_HIGHER_IS_BETTER: INTERPRETATION_IMPROVEMENT,
    DIRECTION_LOWER_IS_WORSE: INTERPRETATION_IMPROVEMENT,
    DIRECTION_HIGHER_IS_WORSE: INTERPRETATION_DETERIORATION,
    DIRECTION_LOWER_IS_BETTER: INTERPRETATION_DETERIORATION,
    DIRECTION_NEUTRAL: None,
    DIRECTION_CONTEXT_DEPENDENT: None,
}

_OPPOSITE = {INTERPRETATION_IMPROVEMENT: INTERPRETATION_DETERIORATION,
             INTERPRETATION_DETERIORATION: INTERPRETATION_IMPROVEMENT}


def interpret(directionality: str, movement: Optional[float]) -> str:
    """The controlled interpretation of one movement.

    ``neutral`` and ``context_dependent`` never yield improvement or
    deterioration: the movement is reported and left to the reader. That is not
    a limitation — a rise in the interest rate, a prepayment or a redemption is
    genuinely not universally good or bad, and the registry says so.
    """
    if movement is None:
        return INTERPRETATION_NOT_ASSESSED
    if movement == 0:
        return INTERPRETATION_NO_MOVEMENT
    positive = _POSITIVE_MEANS.get(directionality)
    if positive is None:
        return INTERPRETATION_NOT_ASSESSED
    return positive if movement > 0 else _OPPOSITE[positive]


# --------------------------------------------------------------------------- #
# Movement — §7 and §11
# --------------------------------------------------------------------------- #
_TEMPORALITY_BASIS = {
    TEMPORALITY_POINT_IN_TIME: BASIS_POINT_IN_TIME_DIFFERENCE,
    TEMPORALITY_PERIOD_FLOW: BASIS_FLOW_LEVEL_COMPARISON,
    TEMPORALITY_CUMULATIVE: BASIS_CUMULATIVE_DIFFERENCE,
    TEMPORALITY_STATIC_BASELINE: BASIS_STATIC_BASELINE_REFERENCE,
}


def metric_change(entry: BusinessSemanticsEntry,
                  start_snapshot: SnapshotFrame,
                  end_snapshot: SnapshotFrame, *,
                  ctx: Optional[PairContext] = None) -> MetricChange:
    """The governed movement of one field between two snapshots."""
    ctx = ctx or build_pair_context(
        entry, [start_snapshot.frame, end_snapshot.frame])
    start = aggregate(start_snapshot.frame, ctx)
    end = aggregate(end_snapshot.frame, ctx)

    basis = _TEMPORALITY_BASIS.get(entry.temporality, BASIS_POINT_IN_TIME_DIFFERENCE)
    notes: List[str] = []
    movement: Optional[float] = None
    status = STATUS_AVAILABLE

    if entry.temporality == TEMPORALITY_STATIC_BASELINE:
        # §7: a static baseline is a reference value, not a series. It receives
        # no "changed between periods" calculation and no ranking.
        status = STATUS_NOT_COMPARABLE
        notes.append(
            "This field is a static origination-time baseline. It is reported "
            "as a reference value at each date and is excluded from "
            "period-change ranking; a baseline does not change between periods.")
    elif not start.ok and not end.ok:
        status = STATUS_NOT_AVAILABLE
    elif not start.ok or not end.ok:
        status = STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY
        notes.append(
            "The field is populated at only one of the two reporting dates, so "
            "no movement can be calculated. It is retained for visibility and "
            "is not substituted with a related field.")
    else:
        movement = float(end.value) - float(start.value)
        if entry.temporality == TEMPORALITY_PERIOD_FLOW:
            notes.append(FLOW_COMPARISON_NOTE)
        elif entry.temporality == TEMPORALITY_CUMULATIVE and movement < 0:
            notes.append(CUMULATIVE_DECLINE_NOTE)
            status = STATUS_SOURCE_CONVENTION_UNCERTAIN
        if STATUS_PARTIALLY_AVAILABLE in (start.status, end.status) \
                and status == STATUS_AVAILABLE:
            status = STATUS_PARTIALLY_AVAILABLE
        if STATUS_SOURCE_CONVENTION_UNCERTAIN in (start.status, end.status):
            status = STATUS_SOURCE_CONVENTION_UNCERTAIN

    relative, basis_points = _expressions(ctx.unit, start.value, movement, notes)

    return MetricChange(
        field=entry.field, display_name=entry.display_name,
        analytical_concept=entry.analytical_concept,
        analytical_role=entry.analytical_role, temporality=entry.temporality,
        aggregation=entry.default_aggregation, weight_field=entry.weight_field,
        share_basis=entry.share_basis,
        start_value=start.value, end_value=end.value,
        movement_value=movement, movement_unit=ctx.unit.unit,
        movement_basis=basis, relative_change=relative,
        basis_point_change=basis_points,
        directionality=entry.directionality,
        interpretation=interpret(entry.directionality, movement),
        status=status, start=start, end=end,
        confidence=entry.confidence, rationale=entry.rationale,
        evidence=(
            evidence_ref("aggregate", start_snapshot, field_name=entry.field,
                         detail={"role": "start", **start.to_dict()}),
            evidence_ref("aggregate", end_snapshot, field_name=entry.field,
                         detail={"role": "end", **end.to_dict()}),
        ),
        notes=tuple(notes))


def _expressions(unit: FieldUnit, start_value: Optional[float],
                 movement: Optional[float], notes: List[str]
                 ) -> Tuple[Optional[float], Optional[float]]:
    """``(relative_change, basis_point_change)`` for one movement.

    A relative change is only produced where dividing by the start value is a
    sound statement: not for a rate expressed in points, not for a category or a
    date, and never against a zero or absent start value.
    """
    basis_points = None
    if unit.unit == UNIT_PERCENTAGE_POINT and movement is not None:
        basis_points = movement * 100.0

    if movement is None or not unit.supports_relative_change:
        return None, basis_points
    if start_value is None:
        return None, basis_points
    if float(start_value) == 0.0:
        notes.append(
            "The starting value is zero, so a percentage change is undefined "
            "and is not reported. The absolute movement is unaffected.")
        return None, basis_points
    return movement / abs(float(start_value)), basis_points
