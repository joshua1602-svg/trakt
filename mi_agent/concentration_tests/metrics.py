"""mi_agent.concentration_tests.metrics — deterministic metric evaluators.

Every activated concentration test resolves to one of the registered
evaluators below, with explicit approved parameters. Each evaluator returns a
:class:`MetricComputation` carrying the value, the numerator and denominator it
was built from and the boolean row mask of the contributing loans — so
drill-through reconciles to the calculation by construction, not by a second
filter written elsewhere.

Conventions shared with the existing risk monitor:

* numbers parse through :func:`analytics_lib.numeric.coerce_numeric`;
* LTV series are normalised to percent when the median suggests a fraction
  (same rule as ``mi_agent_api.risk_limits._normalise_ltv``);
* a zero or missing denominator yields NO value (never a silent zero);
* a missing required column yields ``data_missing`` with the role and the
  candidate columns named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .library import ConcentrationLibrary, MetricDefinition
from .models import (
    DATA_EXTERNAL_UNCONFIGURED,
    DATA_MISSING,
    DATA_OK,
)


# --------------------------------------------------------------------------- #
# Result contract
# --------------------------------------------------------------------------- #
@dataclass
class MetricComputation:
    """The full deterministic account of one metric evaluation."""

    value: Optional[float] = None
    unit: str = "percent"
    numerator_value: Optional[float] = None
    denominator_value: Optional[float] = None
    denominator_basis: str = ""
    loans_in_numerator: int = 0
    total_loans: int = 0
    data_status: str = DATA_OK
    missing_fields: List[str] = field(default_factory=list)
    notes: str = ""
    #: Boolean mask over the frame's index selecting the contributing loans.
    mask: Optional[pd.Series] = None
    #: Column the numeric comparison / weighting actually read, for disclosure.
    resolved_columns: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def missing(cls, role: str, candidates: List[str], *, unit: str,
                total_loans: int = 0) -> "MetricComputation":
        return cls(value=None, unit=unit, data_status=DATA_MISSING,
                   missing_fields=candidates or [role], total_loans=total_loans,
                   notes=(f"No populated column for '{role}' "
                          f"(looked for: {', '.join(candidates) or role})."))


class ExternalIndexProvider(Protocol):
    """Governed access to an approved external index series.

    The platform does not simulate index feeds: when no provider is
    configured, index metrics report ``external_reference_unconfigured``.
    """

    def index_value(self, source: str, series: str, on_date: str
                    ) -> Optional[float]: ...


# --------------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------------- #
def resolve_role_column(df: pd.DataFrame, lib: ConcentrationLibrary,
                        role: str) -> Optional[str]:
    """First candidate column that exists and holds any non-null value."""
    spec = lib.role(role)
    if spec is None:
        return None
    for col in spec.candidates:
        if col in getattr(df, "columns", []) and df[col].notna().any():
            return col
    return None


def role_candidates(lib: ConcentrationLibrary, role: str) -> List[str]:
    spec = lib.role(role)
    return list(spec.candidates) if spec else [role]


def _normalise_ltv(series: pd.Series) -> pd.Series:
    v = coerce_numeric(series)
    if v.notna().any() and float(v.dropna().median()) <= 1.5:
        return v * 100.0
    return v


_BALANCE_ROLE_FOR_DENOMINATOR = {
    "current_balance": "balance_current",
    "original_balance": "balance_original",
}

#: Which governed role a basis parameter value reads, per parameter name.
_BASIS_ROLES: Dict[str, Dict[str, str]] = {
    "value_basis": {"original": "valuation_original",
                    "current": "valuation_current",
                    "indexed": "valuation_indexed"},
    "balance_basis": {"original": "balance_original",
                      "current": "balance_current"},
    "ltv_basis": {"original": "ltv_original", "current": "ltv_current",
                  "indexed": "ltv_indexed"},
    "age_basis": {"youngest": "borrower_age_youngest",
                  "oldest": "borrower_age_oldest"},
}

_DIMENSION_ROLES = {
    "originator": "originator", "seller": "seller", "source_type": "source_type",
    "source_portfolio": "source_portfolio", "servicer": "servicer", "spv": "spv",
    "loan_purpose": "loan_purpose", "property_type": "property_type",
    "tenure": "tenure", "product": "product", "payment_option": "payment_option",
    "rate_type": "rate_type", "region": "region", "broker": "broker",
}

#: Parameter names that carry the categorical values for share_of_balance.
_VALUE_LIST_PARAMS = ("regions", "countries", "counties", "areas", "values",
                      "years")

#: Parameter names that carry the numeric threshold for numeric-share metrics.
_THRESHOLD_PARAMS = ("amount", "ltv_percent", "age")


def _param_default(metric: MetricDefinition, name: str) -> Any:
    spec = (metric.parameters or {}).get(name) or {}
    return spec.get("default")


def _param(metric: MetricDefinition, params: Dict[str, Any], name: str) -> Any:
    if name in params and params[name] not in (None, ""):
        return params[name]
    return _param_default(metric, name)


def _denominator(df: pd.DataFrame, lib: ConcentrationLibrary,
                 metric: MetricDefinition, params: Dict[str, Any]
                 ) -> "tuple[Optional[pd.Series], Optional[float], str, List[str]]":
    """Resolve the denominator basis balance series and its total.

    Returns (balance_series, total, basis_label, missing_candidates).
    """
    basis = str(_param(metric, params, "denominator") or "current_balance")
    role = _BALANCE_ROLE_FOR_DENOMINATOR.get(basis, "balance_current")
    col = resolve_role_column(df, lib, role)
    if col is None:
        return None, None, basis, role_candidates(lib, role)
    bal = coerce_numeric(df[col])
    total = float(bal.sum(skipna=True))
    if not total:
        return bal, None, basis, []
    return bal, total, basis, []


def _values_from_params(params: Dict[str, Any]) -> List[str]:
    for name in _VALUE_LIST_PARAMS:
        v = params.get(name)
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    return []


def _norm_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _share_result(df: pd.DataFrame, mask: pd.Series, bal: pd.Series,
                  total: float, metric: MetricDefinition, basis: str,
                  resolved: Dict[str, str]) -> MetricComputation:
    numer = float(bal[mask].sum(skipna=True))
    value = round(numer / total * 100.0, metric.output_precision)
    return MetricComputation(
        value=value, unit=metric.unit, numerator_value=round(numer, 2),
        denominator_value=round(total, 2), denominator_basis=basis,
        loans_in_numerator=int(mask.sum()), total_loans=int(len(df)),
        data_status=DATA_OK, mask=mask, resolved_columns=resolved)


def _no_denominator(metric: MetricDefinition, basis: str, total_loans: int
                    ) -> MetricComputation:
    return MetricComputation(
        value=None, unit=metric.unit, denominator_basis=basis,
        total_loans=total_loans, data_status=DATA_MISSING,
        notes="The denominator balance is zero or missing; the share is not "
              "reported as zero.")


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #
def _categorical_role(metric: MetricDefinition) -> Optional[str]:
    for role in metric.required_roles:
        if not role.startswith("balance_"):
            return role
    return None


def _eval_share_of_balance(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    role = _categorical_role(metric)
    col = resolve_role_column(df, lib, role) if role else None
    if col is None:
        return MetricComputation.missing(role or "dimension",
                                         role_candidates(lib, role or ""),
                                         unit=metric.unit, total_loans=len(df))
    values = [v.strip().lower() for v in _values_from_params(params)]
    if not values:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df),
            notes="No filter values configured for this test.")
    mask = _norm_text(df[col]).isin(values)
    return _share_result(df, mask, bal, total, metric, basis, {role: col})


def _eval_dimension_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    dimension = str(params.get("dimension") or "")
    role = _DIMENSION_ROLES.get(dimension)
    col = resolve_role_column(df, lib, role) if role else None
    if col is None:
        return MetricComputation.missing(role or dimension,
                                         role_candidates(lib, role or dimension),
                                         unit=metric.unit, total_loans=len(df))
    values = [str(v).strip().lower() for v in (params.get("values") or [])]
    mask = _norm_text(df[col]).isin(values)
    return _share_result(df, mask, bal, total, metric, basis, {role: col})


def _numeric_basis_role(metric: MetricDefinition, params: Dict[str, Any]
                        ) -> Optional[str]:
    for pname, mapping in _BASIS_ROLES.items():
        if pname in (metric.parameters or {}):
            basis = str(_param(metric, params, pname) or "")
            return mapping.get(basis)
    return None


def _eval_share_of_balance_numeric(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    role = _numeric_basis_role(metric, params)
    if role is None:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df),
            notes="The value basis for this test is not configured.")
    col = resolve_role_column(df, lib, role)
    if col is None:
        return MetricComputation.missing(role, role_candidates(lib, role),
                                         unit=metric.unit, total_loans=len(df))
    threshold = None
    for pname in _THRESHOLD_PARAMS:
        if params.get(pname) is not None:
            threshold = float(params[pname])
            break
    if threshold is None:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="No numeric threshold configured.")
    series = _normalise_ltv(df[col]) if role.startswith("ltv_") else coerce_numeric(df[col])
    comparison = str(_param(metric, params, "comparison") or "above")
    mask = (series < threshold) if comparison == "below" else (series > threshold)
    mask = mask.fillna(False)
    return _share_result(df, mask, bal, total, metric, basis, {role: col})


def _eval_postcode_area_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    col = resolve_role_column(df, lib, "postcode")
    if col is None:
        return MetricComputation.missing("postcode", role_candidates(lib, "postcode"),
                                         unit=metric.unit, total_loans=len(df))
    areas = [a.strip().upper() for a in (params.get("areas") or [])]
    outward = df[col].astype(str).str.strip().str.upper().str.split().str[0].fillna("")
    letters = outward.str.extract(r"^([A-Z]+)", expand=False).fillna("")
    mask = outward.isin(areas) | letters.isin(areas)
    return _share_result(df, mask, bal, total, metric, basis, {"postcode": col})


def _group_role(metric: MetricDefinition, params: Dict[str, Any]) -> Optional[str]:
    dimension = params.get("dimension")
    if dimension:
        return _DIMENSION_ROLES.get(str(dimension))
    return _categorical_role(metric)


def _eval_largest_group_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    role = _group_role(metric, params)
    col = resolve_role_column(df, lib, role) if role else None
    if col is None:
        return MetricComputation.missing(role or "dimension",
                                         role_candidates(lib, role or ""),
                                         unit=metric.unit, total_loans=len(df))
    excluded = {str(v).strip().lower() for v in (params.get("exclude_values") or [])}
    groups = _norm_text(df[col])
    sums = bal.groupby(groups).sum()
    sums = sums[~sums.index.isin(excluded)]
    sums = sums[sums.index.notna() & (sums.index != "nan") & (sums.index != "")]
    if sums.empty:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), denominator_basis=basis,
            notes="No populated groups to measure.")
    top_group = sums.idxmax()
    mask = groups == top_group
    out = _share_result(df, mask, bal, total, metric, basis, {role: col})
    out.notes = f"Largest group: {str(top_group).title()}."
    return out


def _eval_distinct_group_count(df, lib, metric, params, external=None):
    role = _categorical_role(metric)
    col = resolve_role_column(df, lib, role) if role else None
    if col is None:
        return MetricComputation.missing(role or "dimension",
                                         role_candidates(lib, role or ""),
                                         unit=metric.unit, total_loans=len(df))
    groups = _norm_text(df[df[col].notna()][col])
    groups = groups[(groups != "") & (groups != "nan")]
    min_share = params.get("min_share")
    if min_share is not None:
        bal_col = resolve_role_column(df, lib, "balance_current")
        if bal_col is None:
            return MetricComputation.missing(
                "balance_current", role_candidates(lib, "balance_current"),
                unit=metric.unit, total_loans=len(df))
        bal = coerce_numeric(df[bal_col])
        total = float(bal.sum(skipna=True))
        if not total:
            return _no_denominator(metric, "current_balance", len(df))
        shares = bal.groupby(_norm_text(df[col])).sum() / total * 100.0
        groups = pd.Series(shares[shares >= float(min_share)].index)
    count = int(groups.nunique())
    mask = df[col].notna() if col in df.columns else None
    return MetricComputation(
        value=float(count), unit=metric.unit, numerator_value=float(count),
        denominator_value=None, loans_in_numerator=int(mask.sum()) if mask is not None else 0,
        total_loans=len(df), data_status=DATA_OK, mask=mask,
        resolved_columns={role: col})


def _eval_weighted_average(df, lib, metric, params, external=None):
    # Which numeric series is being averaged.
    role = _numeric_basis_role(metric, params)
    if role is None and "interest_rate" in metric.required_roles:
        role = "interest_rate"
    if role is None:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="The value basis is not configured.")
    col = resolve_role_column(df, lib, role)
    if col is None:
        return MetricComputation.missing(role, role_candidates(lib, role),
                                         unit=metric.unit, total_loans=len(df))
    values = _normalise_ltv(df[col]) if role.startswith("ltv_") else coerce_numeric(df[col])
    deduction = params.get("deduction_percent")
    if deduction is not None:
        values = values - float(deduction)
    weighting = str(_param(metric, params, "weighting") or "current_balance")
    if weighting == "unweighted":
        mask = values.notna()
        if not int(mask.sum()):
            return MetricComputation(
                value=None, unit=metric.unit, data_status=DATA_MISSING,
                total_loans=len(df), notes="No populated values to average.")
        value = round(float(values[mask].mean()), metric.output_precision)
        return MetricComputation(
            value=value, unit=metric.unit, numerator_value=round(float(values[mask].sum()), 4),
            denominator_value=float(int(mask.sum())), denominator_basis="loan_count",
            loans_in_numerator=int(mask.sum()), total_loans=len(df),
            data_status=DATA_OK, mask=mask, resolved_columns={role: col})
    w_role = _BALANCE_ROLE_FOR_DENOMINATOR.get(weighting, "balance_current")
    w_col = resolve_role_column(df, lib, w_role)
    if w_col is None:
        return MetricComputation.missing(w_role, role_candidates(lib, w_role),
                                         unit=metric.unit, total_loans=len(df))
    weights = coerce_numeric(df[w_col])
    mask = values.notna() & weights.notna()
    denom = float(weights[mask].sum())
    if not denom:
        return _no_denominator(metric, weighting, len(df))
    value = round(float((values[mask] * weights[mask]).sum() / denom),
                  metric.output_precision)
    return MetricComputation(
        value=value, unit=metric.unit,
        numerator_value=round(float((values[mask] * weights[mask]).sum()), 2),
        denominator_value=round(denom, 2), denominator_basis=weighting,
        loans_in_numerator=int(mask.sum()), total_loans=len(df),
        data_status=DATA_OK, mask=mask,
        resolved_columns={role: col, w_role: w_col})


def _eval_field_average(df, lib, metric, params, external=None):
    role = _numeric_basis_role(metric, params) or "balance_current"
    col = resolve_role_column(df, lib, role)
    if col is None:
        return MetricComputation.missing(role, role_candidates(lib, role),
                                         unit=metric.unit, total_loans=len(df))
    values = coerce_numeric(df[col])
    mask = values.notna()
    count = int(mask.sum())
    if not count:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="No populated values to average.")
    total = float(values[mask].sum())
    return MetricComputation(
        value=round(total / count, metric.output_precision), unit=metric.unit,
        numerator_value=round(total, 2), denominator_value=float(count),
        denominator_basis="loan_count", loans_in_numerator=count,
        total_loans=len(df), data_status=DATA_OK, mask=mask,
        resolved_columns={role: col})


def _eval_field_extremum(df, lib, metric, params, *, take_max: bool,
                         external=None):
    role = _numeric_basis_role(metric, params)
    if role is None:
        role = "balance_current" if "balance_basis" in (metric.parameters or {}) else None
    if role is None:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="The value basis is not configured.")
    col = resolve_role_column(df, lib, role)
    if col is None:
        return MetricComputation.missing(role, role_candidates(lib, role),
                                         unit=metric.unit, total_loans=len(df))
    values = _normalise_ltv(df[col]) if role.startswith("ltv_") else coerce_numeric(df[col])
    if not values.notna().any():
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="No populated values.")
    idx = values.idxmax() if take_max else values.idxmin()
    mask = pd.Series(False, index=df.index)
    mask.loc[idx] = True
    return MetricComputation(
        value=round(float(values.loc[idx]), metric.output_precision),
        unit=metric.unit, numerator_value=round(float(values.loc[idx]), 2),
        denominator_value=None, loans_in_numerator=1, total_loans=len(df),
        data_status=DATA_OK, mask=mask, resolved_columns={role: col})


def _eval_largest_by_field_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    role = _numeric_basis_role(metric, params)
    col = resolve_role_column(df, lib, role) if role else None
    if col is None:
        return MetricComputation.missing(role or "value",
                                         role_candidates(lib, role or ""),
                                         unit=metric.unit, total_loans=len(df))
    values = coerce_numeric(df[col])
    if not values.notna().any():
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="No populated values.")
    idx = values.idxmax()
    mask = pd.Series(False, index=df.index)
    mask.loc[idx] = True
    return _share_result(df, mask, bal, total, metric, basis, {role: col})


def _eval_top_n_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    n = int(params.get("n") or 0)
    if n < 1:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="Top-N size is not configured.")
    top_idx = bal.fillna(0).sort_values(ascending=False, kind="mergesort").head(n).index
    mask = pd.Series(df.index.isin(top_idx), index=df.index)
    return _share_result(df, mask, bal, total, metric, basis, {})


def _eval_max_count_per_group(df, lib, metric, params, external=None):
    role = _categorical_role(metric) or "borrower_id"
    col = resolve_role_column(df, lib, role)
    if col is None:
        return MetricComputation.missing(role, role_candidates(lib, role),
                                         unit=metric.unit, total_loans=len(df))
    groups = _norm_text(df[df[col].notna()][col])
    groups = groups[(groups != "") & (groups != "nan")]
    if groups.empty:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="No populated identifiers.")
    counts = groups.groupby(groups).size()
    top_group = counts.idxmax()
    mask = _norm_text(df[col]) == top_group
    return MetricComputation(
        value=float(int(counts.max())), unit=metric.unit,
        numerator_value=float(int(counts.max())), denominator_value=None,
        loans_in_numerator=int(mask.sum()), total_loans=len(df),
        data_status=DATA_OK, mask=mask, resolved_columns={role: col},
        notes=f"Borrower with most loans: {top_group}.")


def _eval_multi_loan_borrower_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    col = resolve_role_column(df, lib, "borrower_id")
    if col is None:
        return MetricComputation.missing("borrower_id",
                                         role_candidates(lib, "borrower_id"),
                                         unit=metric.unit, total_loans=len(df))
    min_loans = int(params.get("min_loans") or 0)
    if min_loans < 2:
        return MetricComputation(
            value=None, unit=metric.unit, data_status=DATA_MISSING,
            total_loans=len(df), notes="Minimum loan count is not configured.")
    groups = _norm_text(df[col])
    counts = groups.map(groups.groupby(groups).size())
    mask = df[col].notna() & (counts >= min_loans)
    return _share_result(df, mask, bal, total, metric, basis,
                         {"borrower_id": col})


def _eval_joint_borrower_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    joint_basis = str(_param(metric, params, "joint_basis") or "structure_flag")
    if joint_basis == "borrower_count":
        col = resolve_role_column(df, lib, "borrower_count")
        if col is None:
            return MetricComputation.missing(
                "borrower_count", role_candidates(lib, "borrower_count"),
                unit=metric.unit, total_loans=len(df))
        counts = coerce_numeric(df[col])
        joint = (counts >= 2).fillna(False)
        resolved = {"borrower_count": col}
    else:
        col = resolve_role_column(df, lib, "borrower_structure")
        if col is None:
            return MetricComputation.missing(
                "borrower_structure", role_candidates(lib, "borrower_structure"),
                unit=metric.unit, total_loans=len(df))
        text = _norm_text(df[col])
        joint = text.str.contains("joint", na=False) | text.isin(("2", "two"))
        resolved = {"borrower_structure": col}
    if bool(params.get("invert")):
        joint = df[col].notna() & ~joint
    return _share_result(df, joint, bal, total, metric, basis, resolved)


def _eval_arrears_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    min_days = params.get("min_days")
    if min_days is not None:
        col = resolve_role_column(df, lib, "days_past_due")
        if col is None:
            return MetricComputation.missing(
                "days_past_due", role_candidates(lib, "days_past_due"),
                unit=metric.unit, total_loans=len(df))
        mask = (coerce_numeric(df[col]) > float(min_days)).fillna(False)
        resolved = {"days_past_due": col}
    else:
        col = resolve_role_column(df, lib, "arrears_balance")
        if col is None:
            return MetricComputation.missing(
                "arrears_balance", role_candidates(lib, "arrears_balance"),
                unit=metric.unit, total_loans=len(df))
        mask = (coerce_numeric(df[col]) > 0).fillna(False)
        resolved = {"arrears_balance": col}
    return _share_result(df, mask, bal, total, metric, basis, resolved)


def _eval_vintage_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    col = resolve_role_column(df, lib, "origination_date")
    if col is None:
        return MetricComputation.missing(
            "origination_date", role_candidates(lib, "origination_date"),
            unit=metric.unit, total_loans=len(df))
    years = {str(y).strip() for y in (params.get("years") or [])}
    parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
    mask = parsed.dt.year.astype("Int64").astype(str).isin(years)
    return _share_result(df, mask.fillna(False), bal, total, metric, basis,
                         {"origination_date": col})


def _eval_external_index_ratio(df, lib, metric, params, external=None):
    source = str(params.get("index_source") or "")
    series = str(params.get("index_series") or "")
    reference_date = str(params.get("reference_date") or "")
    reporting_date = str(params.get("_reporting_date") or "")
    if external is None:
        return MetricComputation(
            value=None, unit=metric.unit,
            data_status=DATA_EXTERNAL_UNCONFIGURED, total_loans=len(df),
            notes=("No approved external index source is configured for "
                   f"'{source or 'this test'}'. The ratio is not simulated."))
    current = external.index_value(source, series, reporting_date)
    base = external.index_value(source, series, reference_date)
    if current is None or base is None or not base:
        return MetricComputation(
            value=None, unit=metric.unit,
            data_status=DATA_EXTERNAL_UNCONFIGURED, total_loans=len(df),
            notes=(f"Index series '{series}' has no value for the reporting "
                   "or reference date."))
    value = round(current / base * 100.0, metric.output_precision)
    return MetricComputation(
        value=value, unit=metric.unit, numerator_value=current,
        denominator_value=base, denominator_basis="reference_index_value",
        total_loans=len(df), data_status=DATA_OK,
        notes=f"{series}: {current} vs {base} at {reference_date}.")


def _eval_filtered_share(df, lib, metric, params, external=None):
    bal, total, basis, missing = _denominator(df, lib, metric, params)
    if missing:
        return MetricComputation.missing("balance", missing, unit=metric.unit,
                                         total_loans=len(df))
    if total is None:
        return _no_denominator(metric, basis, len(df))
    mask = pd.Series(True, index=df.index)
    resolved: Dict[str, str] = {}
    for f in params.get("filters") or []:
        role = str(f.get("role") or "")
        col = resolve_role_column(df, lib, role)
        if col is None:
            return MetricComputation.missing(role, role_candidates(lib, role),
                                             unit=metric.unit,
                                             total_loans=len(df))
        resolved[role] = col
        values = f.get("values")
        if values:
            mask &= _norm_text(df[col]).isin(
                [str(v).strip().lower() for v in values])
        else:
            series = (_normalise_ltv(df[col]) if role.startswith("ltv_")
                      else coerce_numeric(df[col]))
            if f.get("min") is not None:
                mask &= (series >= float(f["min"])).fillna(False)
            if f.get("max") is not None:
                mask &= (series <= float(f["max"])).fillna(False)
    return _share_result(df, mask, bal, total, metric, basis, resolved)


EVALUATORS: Dict[str, Callable[..., MetricComputation]] = {
    "share_of_balance": _eval_share_of_balance,
    "share_of_balance_numeric": _eval_share_of_balance_numeric,
    "postcode_area_share": _eval_postcode_area_share,
    "largest_group_share": _eval_largest_group_share,
    "distinct_group_count": _eval_distinct_group_count,
    "weighted_average": _eval_weighted_average,
    "field_average": _eval_field_average,
    "field_maximum": lambda df, lib, m, p, external=None:
        _eval_field_extremum(df, lib, m, p, take_max=True, external=external),
    "field_minimum": lambda df, lib, m, p, external=None:
        _eval_field_extremum(df, lib, m, p, take_max=False, external=external),
    "largest_by_field_share": _eval_largest_by_field_share,
    "top_n_share": _eval_top_n_share,
    "max_count_per_group": _eval_max_count_per_group,
    "multi_loan_borrower_share": _eval_multi_loan_borrower_share,
    "joint_borrower_share": _eval_joint_borrower_share,
    "arrears_share": _eval_arrears_share,
    "dimension_share": _eval_dimension_share,
    "vintage_share": _eval_vintage_share,
    "external_index_ratio": _eval_external_index_ratio,
    "filtered_share": _eval_filtered_share,
}


def evaluate_metric(df: pd.DataFrame, lib: ConcentrationLibrary,
                    metric: MetricDefinition, params: Dict[str, Any],
                    *, external: Optional[ExternalIndexProvider] = None
                    ) -> MetricComputation:
    """Evaluate one registered metric with explicit approved parameters.

    Raises for an unregistered evaluator (a registry defect, caught by the
    library tests); analytical shortfalls come back as controlled
    ``data_status`` values, never as fabricated numbers.
    """
    if metric.evaluator is None or metric.evaluator not in EVALUATORS:
        raise ValueError(
            f"Metric '{metric.metric_id}' has no registered deterministic "
            "evaluator.")
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    return EVALUATORS[metric.evaluator](frame, lib, metric, dict(params or {}),
                                        external=external)
