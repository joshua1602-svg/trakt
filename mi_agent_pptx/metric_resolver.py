"""mi_agent_pptx.metric_resolver — resolve KPI metrics from the registry layer.

Resolves the scalar KPIs the deck renders (total pipeline, funded balance, loan
count, WA ticket size, WA current LTV, WA borrower age, largest exposures, data
quality …). Resolution order honours the source-of-truth principles:

1. If a post-pipeline analytics / metric-registry artifact already carries the
   value, use it verbatim (the MI Agent computed it — the deck must not
   recompute).
2. Otherwise compute a **registry-authorised** aggregation: the aggregation
   method (``sum`` / ``avg`` / ``count`` / ``weighted_avg``) and the weighting
   field come from the semantic registry field entry, not from ad-hoc code in
   the PPTX layer. Weighted averages use the registry ``weight_field``.
3. If neither is possible (field absent), return an *unavailable* result so the
   slide renders a branded placeholder and an appendix note — never a crash.

No economic derivation lives here beyond the aggregation methods the registry
already declares for each field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_resolver import ResolvedData
from .registry_loader import RegistryLoader


@dataclass
class MetricResult:
    """A resolved KPI value plus provenance for the appendix."""

    key: str
    label: str
    value: Any = None
    fmt: str = "string"
    available: bool = False
    basis: str = "unavailable"   # analytics_artifact | registry_computed | unavailable
    note: str = ""
    display: str = "—"

    @property
    def ok(self) -> bool:
        return self.available and self.value is not None


# --------------------------------------------------------------------------- #
# Formatting (mirrors mi_chart_factory compact formatters).
# --------------------------------------------------------------------------- #

def compact_currency(value: float, symbol: str = "£") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{sign}{symbol}{a / 1e9:.2f}bn"
    if a >= 1e6:
        return f"{sign}{symbol}{a / 1e6:.1f}m"
    if a >= 1e3:
        return f"{sign}{symbol}{a / 1e3:.0f}k"
    return f"{sign}{symbol}{a:,.0f}"


def compact_number(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{int(round(float(value))):,}"


def format_percent(value: float, *, already_fraction: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    v = float(value)
    pct = v * 100.0 if already_fraction else v
    return f"{pct:.1f}%"


def format_value(value: Any, fmt: str) -> str:
    """Render *value* per a registry format string."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if fmt == "currency":
        return compact_currency(value)
    if fmt == "percent":
        return format_percent(value)
    if fmt == "rate":
        # Interest rate is stored as whole-number points (e.g. 9.55), not a
        # fraction — render as-is with a percent sign.
        return f"{float(value):.2f}%"
    if fmt in ("integer", "count"):
        return compact_number(value)
    if fmt in ("decimal", "ratio"):
        return f"{float(value):,.2f}"
    if fmt == "years":
        return f"{float(value):.1f} yrs"
    return str(value)


# --------------------------------------------------------------------------- #
# Aggregation (registry-authorised only).
# --------------------------------------------------------------------------- #

def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna()
    denom = w[mask].sum()
    if denom == 0 or not mask.any():
        return float("nan")
    return float((v[mask] * w[mask]).sum() / denom)


class MetricResolver:
    """Resolve deck KPIs from analytics artifacts or registry aggregations."""

    def __init__(
        self,
        data: ResolvedData,
        registries: RegistryLoader,
        analytics: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.reg = registries
        self.analytics = analytics or {}

    # ------------------------------------------------------------------ core
    def resolve(self, spec: Dict[str, Any]) -> MetricResult:
        """Resolve a single metric spec.

        Spec keys: ``key``, ``label`` (optional), ``field`` (semantic key),
        ``aggregation`` (overrides the registry default), ``format`` (overrides
        registry format), ``dimension`` (for ``largest_*`` metrics),
        ``kind`` (``field`` | ``largest`` | ``data_quality``).
        """
        key = spec.get("key", spec.get("id", "metric"))
        kind = spec.get("kind", "field")
        field_key = spec.get("field")
        label = spec.get("label") or (
            self.reg.label_for(field_key) if field_key else key.replace("_", " ").title()
        )
        fmt = spec.get("format") or (
            self.reg.format_for(field_key) if field_key else "string"
        )

        # 1) analytics artifact override -------------------------------------
        art_val = self._from_analytics(key)
        if art_val is not None:
            return self._finish(key, label, art_val, fmt, "analytics_artifact",
                                 "Sourced from MI Agent analytics artifact.")

        # 2) registry-authorised computation --------------------------------
        if kind == "data_quality":
            return self._data_quality(key, label)
        if kind == "largest":
            return self._largest_exposure(key, label, spec)
        return self._field_metric(key, label, field_key, spec, fmt)

    def resolve_all(self, specs: List[Dict[str, Any]]) -> List[MetricResult]:
        return [self.resolve(s) for s in specs]

    # ------------------------------------------------------------- computations
    def _field_metric(self, key, label, field_key, spec, fmt) -> MetricResult:
        df = self.data.df
        agg = spec.get("aggregation")

        # count needs no field.
        if agg == "count" or spec.get("kind") == "count":
            return self._finish(key, label, int(len(df)), fmt or "integer",
                                 "registry_computed", "Loan count from tape.")

        if not field_key:
            return self._unavailable(key, label, fmt, "No field bound to metric.")

        fspec = self.reg.field_spec(field_key)
        canonical = fspec.canonical_field if fspec else field_key
        if canonical not in df.columns or df[canonical].notna().sum() == 0:
            return self._unavailable(
                key, label, fmt,
                f"Field '{canonical}' not present in tape.")

        agg = agg or (fspec.default_aggregation if fspec else None) or "sum"
        series = pd.to_numeric(df[canonical], errors="coerce")

        if agg == "sum":
            val = float(series.sum())
        elif agg in ("avg", "mean"):
            val = float(series.mean())
        elif agg == "median":
            val = float(series.median())
        elif agg == "weighted_avg":
            weight_field = spec.get("weight_field") or (
                fspec.weight_field if fspec else None
            ) or self.reg.default_weight_field
            if weight_field not in df.columns:
                # Fall back to a simple mean rather than fail.
                val = float(series.mean())
                note = (f"Weight field '{weight_field}' absent; used simple mean.")
                return self._finish(key, label, val, fmt, "registry_computed", note)
            val = _weighted_avg(df[canonical], df[weight_field])
            if val is None or (isinstance(val, float) and math.isnan(val)):
                # Weight column present but unusable (e.g. all-NaN balances) —
                # fall back to a registry-allowed simple mean.
                mean_val = float(series.mean())
                if not math.isnan(mean_val):
                    return self._finish(
                        key, label, mean_val, fmt, "registry_computed",
                        f"weighted_avg weight '{weight_field}' unusable; used "
                        f"simple mean of {canonical}.")
        else:
            val = float(series.sum())

        if val is None or (isinstance(val, float) and math.isnan(val)):
            return self._unavailable(key, label, fmt, "Aggregation produced no value.")

        return self._finish(
            key, label, val, fmt, "registry_computed",
            f"{agg} of {canonical} (registry-authorised).")

    def _largest_exposure(self, key, label, spec) -> MetricResult:
        """Largest single-category exposure share for a dimension."""
        dim = spec.get("dimension")
        fmt = spec.get("format", "percent")
        df = self.data.df
        bal = self.data.balance_col
        if not dim or dim not in df.columns or bal is None:
            return self._unavailable(key, label, fmt,
                                     f"Dimension '{dim}' or balance unavailable.")
        try:
            from analytics_lib.concentration import group_shares
            table = group_shares(df, dim, bal)
        except Exception as exc:  # pragma: no cover
            return self._unavailable(key, label, fmt, f"Concentration failed: {exc}")
        if table is None or table.empty:
            return self._unavailable(key, label, fmt, "No groups to rank.")
        top = table.iloc[0]
        share = float(top.get("balance_share", float("nan")))
        name = str(top.get(dim, "—"))
        display = f"{name} · {format_percent(share)}"
        return MetricResult(
            key=key, label=label, value=share, fmt=fmt, available=True,
            basis="registry_computed",
            note=f"Largest {dim} exposure by balance share.", display=display,
        )

    def _data_quality(self, key, label) -> MetricResult:
        """Deterministic data-quality status from validation artifact or coverage."""
        val_art = self.analytics.get("validation") if self.analytics else None
        status = None
        if isinstance(val_art, dict):
            status = (val_art.get("status") or val_art.get("overallStatus")
                      or val_art.get("summary", {}).get("status"))
        if not status:
            # Derive a coarse status from field coverage / issues.
            errs = [i for i in self.data.issues if i.get("severity") == "error"]
            status = "Amber" if errs else "Green"
        status = str(status).title()
        return MetricResult(
            key=key, label=label, value=status, fmt="string", available=True,
            basis="registry_computed", note="Pipeline validation status.",
            display=status,
        )

    # -------------------------------------------------------------- analytics
    def _from_analytics(self, key: str) -> Any:
        """Look up a metric value in the loaded analytics artifact(s)."""
        for container in (self.analytics.get("metrics"), self.analytics.get("analytics")):
            if isinstance(container, dict):
                if key in container:
                    entry = container[key]
                    if isinstance(entry, dict):
                        return entry.get("value")
                    return entry
        return None

    # ----------------------------------------------------------------- helpers
    def _finish(self, key, label, value, fmt, basis, note) -> MetricResult:
        return MetricResult(
            key=key, label=label, value=value, fmt=fmt, available=True,
            basis=basis, note=note, display=format_value(value, fmt),
        )

    def _unavailable(self, key, label, fmt, note) -> MetricResult:
        return MetricResult(
            key=key, label=label, value=None, fmt=fmt, available=False,
            basis="unavailable", note=note, display="—",
        )
