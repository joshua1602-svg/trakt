"""mi_workflows/engine — the shared deterministic calculation engine.

One implementation of every governed analytical primitive, shared by all
workflows so that two workflows can never disagree about what a weighted
average, a share or a distribution is. Every function is pure over the frame
it is given: no I/O, no LLM, no client-specific behaviour, no thresholds.

Primitives
----------
* :func:`aggregate` — sum / average / weighted average / share, driven by the
  Business Semantics Registry's ``default_aggregation`` / ``weight_field`` /
  ``share_basis`` (never by ad-hoc per-metric code);
* :func:`distribution` — governed categorical distribution with explicit
  count share, exposure share and unknown share;
* :func:`ranked_distribution` — the same governed distribution ordered into a
  deterministic concentration ranking: rank, cumulative share, top-N share and
  an explicit unknown block (never a silently dropped unknown);
* :func:`compare_values` — absolute + relative difference, one definition;
* :func:`directionality_verdict` — the observed relation between two values
  interpreted through governed directionality (never aggregated into a score);
* :func:`currency_profile` / :func:`mixed_currency_guard` — the
  mixed-currency guard: monetary metrics are only comparable where the
  governed currency profiles match, and no FX conversion is ever performed;
* :func:`unit_for_field` — unit resolution from the runtime MI semantics
  layer (``currency`` / ``percent`` / ``decimal`` / ``integer`` / ``boolean``).

Every aggregate discloses its population: rows in scope, rows that validly
contributed, rows excluded, and the denominator used — evidence generation is
part of the calculation, not an afterthought.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

#: Version of the deterministic calculation semantics in this module. Recorded
#: in every workflow audit block so a historical result can be tied to the
#: calculation rules that produced it. 1.1.0 added the ranked-distribution
#: (concentration) primitive; every existing primitive is unchanged.
CALCULATION_VERSION = "1.1.0"

#: Aggregations the engine implements (the BSR's ``default_aggregations``
#: taxonomy minus ``distribution``, which has its own primitive).
AGG_SUM = "sum"
AGG_AVERAGE = "average"
AGG_WEIGHTED_AVERAGE = "weighted_average"
AGG_SHARE = "share"
AGG_DISTRIBUTION = "distribution"

#: Units, resolved from the runtime MI semantics registry's ``format``.
UNIT_CURRENCY = "currency"
UNIT_PERCENT = "percent"
UNIT_DECIMAL = "decimal"
UNIT_INTEGER = "integer"
UNIT_SHARE = "share"        # engine-produced: a 0..1 proportion
UNIT_CATEGORY = "category"

#: The currency columns the platform recognises, most specific first — the
#: same precedence ``mi_agent_api.currency.resolve_currency_code`` uses.
CURRENCY_FIELDS = ("exposure_currency_denomination", "currency_denomination",
                   "collateral_currency")

#: Label under which null/blank categorical values are reported. Explicit —
#: an unknown value is disclosed, never dropped or guessed.
UNKNOWN_CATEGORY = "(unknown)"

#: Truthy / falsy vocabularies for governed Y/N share metrics.
_TRUE_TOKENS = frozenset({"y", "yes", "true", "1", "1.0"})
_FALSE_TOKENS = frozenset({"n", "no", "false", "0", "0.0"})


@dataclass(frozen=True)
class MetricAggregate:
    """One governed aggregate with its full population disclosure."""

    value: Optional[float]
    aggregation: str
    unit: str
    #: Weight field for a weighted average; share basis for a share; else None.
    weight_basis: Optional[str]
    #: Rows in the scope the aggregate was computed over.
    population: int
    #: Rows that validly contributed to the aggregate.
    valid_population: int
    #: Rows in scope that could not contribute (null metric, null/non-positive
    #: weight, unparseable flag).
    excluded_population: int
    #: The denominator actually used: Σweight for a weighted average, valid row
    #: count for average/share(count), Σbasis for share(weighted). None for sum.
    denominator: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "aggregation": self.aggregation,
            "unit": self.unit,
            "weight_basis": self.weight_basis,
            "population": self.population,
            "valid_population": self.valid_population,
            "excluded_population": self.excluded_population,
            "denominator": self.denominator,
        }


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _flag(series: pd.Series) -> pd.Series:
    """Parse a governed Y/N-style flag column to 1.0 / 0.0 / NaN."""
    tokens = series.astype("string").str.strip().str.lower()
    out = pd.Series(float("nan"), index=series.index, dtype="float64")
    out[tokens.isin(_TRUE_TOKENS)] = 1.0
    out[tokens.isin(_FALSE_TOKENS)] = 0.0
    return out


def aggregate(df: pd.DataFrame, metric_field: str, aggregation: str, *,
              weight_field: Optional[str] = None,
              share_basis: Optional[str] = None,
              unit: str = UNIT_DECIMAL) -> MetricAggregate:
    """One governed aggregate over one frame.

    The aggregation, weight field and share basis come from the Business
    Semantics Registry — this function implements them, it never chooses them.
    An unknown aggregation or a missing column yields a ``None`` value with the
    population recorded, never a guess.
    """
    population = int(len(df))

    def _empty(basis: Optional[str]) -> MetricAggregate:
        return MetricAggregate(None, aggregation, unit, basis, population,
                               0, population, None)

    if metric_field not in df.columns:
        return _empty(weight_field if aggregation == AGG_WEIGHTED_AVERAGE
                      else (share_basis if aggregation == AGG_SHARE else None))

    if aggregation == AGG_SUM:
        values = _numeric(df[metric_field])
        valid = values.notna()
        n = int(valid.sum())
        if n == 0:
            return _empty(None)
        return MetricAggregate(float(values[valid].sum()), aggregation, unit, None,
                               population, n, population - n, None)

    if aggregation == AGG_AVERAGE:
        values = _numeric(df[metric_field])
        valid = values.notna()
        n = int(valid.sum())
        if n == 0:
            return _empty(None)
        return MetricAggregate(float(values[valid].mean()), aggregation, unit, None,
                               population, n, population - n, float(n))

    if aggregation == AGG_WEIGHTED_AVERAGE:
        if not weight_field or weight_field not in df.columns:
            return _empty(weight_field)
        values = _numeric(df[metric_field])
        weights = _numeric(df[weight_field])
        valid = values.notna() & weights.notna() & (weights > 0)
        n = int(valid.sum())
        if n == 0:
            return _empty(weight_field)
        wsum = float(weights[valid].sum())
        value = float((values[valid] * weights[valid]).sum() / wsum)
        return MetricAggregate(value, aggregation, unit, weight_field,
                               population, n, population - n, wsum)

    if aggregation == AGG_SHARE:
        basis = share_basis or "count"
        flags = _flag(df[metric_field])
        valid = flags.notna()
        if basis == "count":
            n = int(valid.sum())
            if n == 0:
                return _empty(basis)
            return MetricAggregate(float(flags[valid].mean()), aggregation,
                                   UNIT_SHARE, basis, population, n,
                                   population - n, float(n))
        # A weighted share: basis is a canonical weight field.
        if basis not in df.columns:
            return _empty(basis)
        weights = _numeric(df[basis])
        valid = valid & weights.notna() & (weights > 0)
        n = int(valid.sum())
        if n == 0:
            return _empty(basis)
        wsum = float(weights[valid].sum())
        value = float((flags[valid] * weights[valid]).sum() / wsum)
        return MetricAggregate(value, aggregation, UNIT_SHARE, basis,
                               population, n, population - n, wsum)

    return _empty(None)


@dataclass(frozen=True)
class DistributionBucket:
    """One category of a governed distribution."""

    category: str
    count: int
    count_share: Optional[float]
    exposure: Optional[float]
    exposure_share: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "count": self.count,
                "count_share": self.count_share, "exposure": self.exposure,
                "exposure_share": self.exposure_share}


@dataclass(frozen=True)
class Distribution:
    """A governed categorical distribution over one frame."""

    field: str
    exposure_field: Optional[str]
    population: int
    unknown_count: int
    buckets: Tuple[DistributionBucket, ...]

    @property
    def unknown_share(self) -> Optional[float]:
        return (self.unknown_count / self.population) if self.population else None

    def bucket(self, category: str) -> Optional[DistributionBucket]:
        for b in self.buckets:
            if b.category == category:
                return b
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "exposure_field": self.exposure_field,
                "population": self.population, "unknown_count": self.unknown_count,
                "unknown_share": self.unknown_share,
                "buckets": [b.to_dict() for b in self.buckets]}


def distribution(df: pd.DataFrame, field: str, *,
                 exposure_field: Optional[str] = None) -> Optional[Distribution]:
    """The governed distribution of one categorical field over one frame.

    Count shares are shares of ALL rows in scope (unknowns included in the
    denominator and reported as their own ``(unknown)`` bucket), so shares
    always sum to 1 and a poorly-populated dimension cannot silently present
    itself as fully known. Exposure shares follow the same rule over the
    exposure column. Returns ``None`` when the field is not on the frame.
    """
    if field not in df.columns:
        return None
    population = int(len(df))
    labels = df[field].astype("string").str.strip()
    labels = labels.mask(labels.isna() | (labels == ""), UNKNOWN_CATEGORY)

    exposure = None
    total_exposure = None
    if exposure_field and exposure_field in df.columns:
        exposure = _numeric(df[exposure_field])
        total = float(exposure.fillna(0).sum())
        total_exposure = total if total > 0 else None
    else:
        exposure_field = None

    buckets: List[DistributionBucket] = []
    unknown_count = 0
    for category, group in df.groupby(labels, sort=True, dropna=False):
        cat = str(category)
        count = int(len(group))
        if cat == UNKNOWN_CATEGORY:
            unknown_count = count
        exp_value: Optional[float] = None
        exp_share: Optional[float] = None
        if exposure is not None:
            exp_value = float(exposure.loc[group.index].fillna(0).sum())
            exp_share = (exp_value / total_exposure) if total_exposure else None
        buckets.append(DistributionBucket(
            category=cat, count=count,
            count_share=(count / population) if population else None,
            exposure=exp_value, exposure_share=exp_share))
    return Distribution(field=field, exposure_field=exposure_field,
                        population=population, unknown_count=unknown_count,
                        buckets=tuple(buckets))


#: The two governed concentration bases. Exposure is the current outstanding
#: balance; count is loans. No other basis exists in the platform.
BASIS_EXPOSURE = "exposure"
BASIS_COUNT = "count"


@dataclass(frozen=True)
class RankedBucket:
    """One known category of a governed concentration ranking."""

    rank: int
    category: str
    count: int
    count_share: Optional[float]
    exposure: Optional[float]
    exposure_share: Optional[float]
    #: Cumulative share of the ranking basis from rank 1 to this rank,
    #: over ALL rows in scope (unknowns included in the denominator).
    cumulative_share: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "category": self.category,
                "count": self.count, "count_share": self.count_share,
                "exposure": self.exposure, "exposure_share": self.exposure_share,
                "cumulative_share": self.cumulative_share}


@dataclass(frozen=True)
class RankedDistribution:
    """A governed distribution ordered into a deterministic concentration
    ranking.

    Built on :func:`distribution`, so count shares, exposure shares and the
    unknown bucket mean exactly the same thing here as everywhere else. Known
    categories are ranked by the chosen basis (value descending, then category
    name ascending — total and stable, so the same frame always ranks the same
    way). The unknown bucket is never ranked and never dropped: it is disclosed
    as its own explicit block, and because every share keeps the full in-scope
    denominator, cumulative shares can only reach 1 − unknown share.
    """

    field: str
    basis: str
    exposure_field: Optional[str]
    population: int
    total_exposure: Optional[float]
    unknown_count: int
    unknown_count_share: Optional[float]
    unknown_exposure: Optional[float]
    unknown_exposure_share: Optional[float]
    buckets: Tuple[RankedBucket, ...]

    def top_share(self, n: int) -> Optional[float]:
        """Cumulative basis share of the top ``n`` known categories."""
        if n <= 0 or not self.buckets:
            return None
        capped = self.buckets[:n]
        return capped[-1].cumulative_share

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field, "basis": self.basis,
            "exposure_field": self.exposure_field,
            "population": self.population,
            "total_exposure": self.total_exposure,
            "unknown_count": self.unknown_count,
            "unknown_count_share": self.unknown_count_share,
            "unknown_exposure": self.unknown_exposure,
            "unknown_exposure_share": self.unknown_exposure_share,
            "buckets": [b.to_dict() for b in self.buckets],
        }


def ranked_distribution(df: pd.DataFrame, field: str, *,
                        exposure_field: Optional[str] = None,
                        basis: str = BASIS_EXPOSURE
                        ) -> Optional[RankedDistribution]:
    """The governed concentration ranking of one categorical field.

    A post-ordering of :func:`distribution` — no share or total is recomputed,
    so this primitive can never disagree with the distribution it ranks.
    Returns ``None`` when the field is not on the frame, or when the exposure
    basis is requested but the frame carries no positive exposure to rank by
    (the caller then decides whether to fall back to the count basis — the
    engine implements bases, it never chooses them).
    """
    if basis not in (BASIS_EXPOSURE, BASIS_COUNT):
        return None
    dist = distribution(df, field, exposure_field=exposure_field)
    if dist is None:
        return None
    if basis == BASIS_EXPOSURE and dist.exposure_field is None:
        return None

    unknown = dist.bucket(UNKNOWN_CATEGORY)
    known = [b for b in dist.buckets if b.category != UNKNOWN_CATEGORY]
    if basis == BASIS_EXPOSURE:
        if all((b.exposure_share is None) for b in known) and known:
            return None
        known.sort(key=lambda b: (-(b.exposure_share or 0.0), b.category))
    else:
        known.sort(key=lambda b: (-(b.count_share or 0.0), b.category))

    total_exposure: Optional[float] = None
    if dist.exposure_field is not None:
        total_exposure = float(sum(b.exposure or 0.0 for b in dist.buckets))

    ranked: List[RankedBucket] = []
    cumulative = 0.0
    cumulative_known = True
    for rank, bucket in enumerate(known, start=1):
        share = (bucket.exposure_share if basis == BASIS_EXPOSURE
                 else bucket.count_share)
        if share is None:
            cumulative_known = False
        else:
            cumulative += share
        ranked.append(RankedBucket(
            rank=rank, category=bucket.category, count=bucket.count,
            count_share=bucket.count_share, exposure=bucket.exposure,
            exposure_share=bucket.exposure_share,
            cumulative_share=cumulative if cumulative_known else None))

    return RankedDistribution(
        field=field, basis=basis, exposure_field=dist.exposure_field,
        population=dist.population, total_exposure=total_exposure,
        unknown_count=dist.unknown_count,
        unknown_count_share=dist.unknown_share,
        unknown_exposure=(unknown.exposure if unknown is not None
                          else (0.0 if dist.exposure_field else None)),
        unknown_exposure_share=(unknown.exposure_share if unknown is not None
                                else (0.0 if dist.exposure_field else None)),
        buckets=tuple(ranked))


def compare_values(a: Optional[float], b: Optional[float]) -> Dict[str, Optional[float]]:
    """Absolute and relative difference of two aggregates, one definition.

    ``absolute`` is A − B. ``relative`` is (A − B) / |B|, i.e. A relative to
    B; ``None`` when either side is missing or B is zero. No other comparison
    formula exists in the platform.
    """
    if a is None or b is None:
        return {"absolute": None, "relative": None}
    absolute = float(a) - float(b)
    relative = (absolute / abs(float(b))) if float(b) != 0 else None
    return {"absolute": absolute, "relative": relative}


#: Directionality interpretations (never aggregated into a score).
FAVOURS_A = "portfolio_a_more_favourable"
FAVOURS_B = "portfolio_b_more_favourable"
NO_DIFFERENCE = "no_observed_difference"
NEUTRAL = "neutral"
CONTEXT_DEPENDENT = "context_dependent"
NOT_ASSESSED = "not_assessed"


def directionality_verdict(directionality: str, a: Optional[float],
                           b: Optional[float]) -> Dict[str, Optional[str]]:
    """The observed relation of two values through governed directionality.

    Returns which portfolio holds the higher value and, where the registry
    declares an unambiguous direction, which side the observation favours.
    ``neutral`` and ``context_dependent`` directionality are reported as such —
    the engine never invents a preference, and verdicts are never combined
    into an overall portfolio score.
    """
    if a is None or b is None:
        return {"higher": None, "interpretation": NOT_ASSESSED}
    if float(a) == float(b):
        return {"higher": None, "interpretation": NO_DIFFERENCE}
    higher = "portfolio_a" if float(a) > float(b) else "portfolio_b"
    if directionality in ("higher_is_better", "lower_is_worse"):
        interpretation = FAVOURS_A if higher == "portfolio_a" else FAVOURS_B
    elif directionality in ("higher_is_worse", "lower_is_better"):
        interpretation = FAVOURS_B if higher == "portfolio_a" else FAVOURS_A
    elif directionality == "context_dependent":
        interpretation = CONTEXT_DEPENDENT
    else:
        interpretation = NEUTRAL
    return {"higher": higher, "interpretation": interpretation}


def currency_profile(df: pd.DataFrame,
                     fields: Sequence[str] = CURRENCY_FIELDS) -> Tuple[str, ...]:
    """The distinct declared currencies of one frame, sorted.

    Read from the first governed currency column present — the same precedence
    the display-currency resolver uses. An empty tuple means the tape declares
    no currency (the platform's documented single-currency book-level
    assumption then applies, and the guard discloses that it did).
    """
    for field in fields:
        if field in df.columns:
            values = df[field].astype("string").str.strip().str.upper()
            distinct = sorted(v for v in values.dropna().unique() if v)
            if distinct:
                return tuple(distinct)
    return ()


def mixed_currency_guard(profile_a: Tuple[str, ...],
                         profile_b: Tuple[str, ...]) -> Tuple[bool, Optional[str]]:
    """May monetary values from these two scopes be compared directly?

    Allowed only when neither scope is internally mixed and both declare the
    same single currency (or neither declares one — the platform's documented
    single-currency deployment assumption, which the caller must disclose).
    Never converts: a disallowed comparison is suppressed and explained, not
    FX-adjusted.
    """
    if len(profile_a) > 1 or len(profile_b) > 1:
        mixed = profile_a if len(profile_a) > 1 else profile_b
        return False, ("mixed currencies within a portfolio scope "
                       f"({', '.join(mixed)}); monetary comparison suppressed — "
                       "no FX conversion is performed")
    if profile_a and profile_b and profile_a != profile_b:
        return False, (f"currency mismatch between portfolios ({profile_a[0]} vs "
                       f"{profile_b[0]}); monetary comparison suppressed — "
                       "no FX conversion is performed")
    return True, None


def unit_for_field(field: str, mi_semantics: Optional[Mapping[str, Any]]) -> str:
    """The display/comparison unit of a canonical field.

    Resolved from the runtime MI semantics registry's per-field ``format``
    (``currency`` / ``percent`` / ``decimal`` / ``integer`` / ``boolean``).
    Falls back to ``decimal`` — the neutral unit — when the layer does not
    describe the field. Monetary behaviour keys off :data:`UNIT_CURRENCY`.
    """
    try:
        fields = (mi_semantics or {}).get("fields") or {}
        fmt = str((fields.get(field) or {}).get("format") or "").strip().lower()
    except Exception:  # noqa: BLE001 - unit resolution must never fail a metric
        fmt = ""
    if fmt in (UNIT_CURRENCY, UNIT_PERCENT, UNIT_INTEGER, UNIT_DECIMAL):
        return fmt
    if fmt == "boolean":
        return UNIT_SHARE
    return UNIT_DECIMAL


def is_monetary(unit: str) -> bool:
    return unit == UNIT_CURRENCY
