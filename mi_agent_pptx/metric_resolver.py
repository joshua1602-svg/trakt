"""mi_agent_pptx.metric_resolver — lens-aware, registry-authorised KPI metrics.

Resolves the deck's scalar KPIs from post-pipeline artifacts / registry
aggregations, with three properties the investor pack requires:

* **Lens separation** — every metric declares a lens (``funded`` / ``pipeline``
  / ``forecast``). A metric resolves against *that lens's* frame only, so the
  pipeline total is never the funded total (and vice-versa). If a lens's frame
  is absent for the run, the metric is *unavailable* (branded placeholder), not
  silently backfilled from another lens.
* **Prior-period deltas** — when a prior-period run is supplied, each metric
  also reports its change (Δ absolute, Δ%, direction) so KPI tiles can show a
  "+£0.7MM vs prior" chip, mirroring the React dashboard. No prior data ⇒ no
  fabricated delta.
* **No new economics** — aggregation methods and weights come from the semantic
  registry; the only composite is the registry-declared forecast bridge
  (funded + Σ weighted pipeline).

Value formatting mirrors the dashboard (`lib/utils.formatGBP`): £X.XXBN / £X.XMM
/ £XK, percentages to one decimal, tabular figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_resolver import ResolvedData
from .registry_loader import RegistryLoader

# Weighted-pipeline column used by the registry forecast bridge.
_WEIGHTED_PIPELINE_COL = "weighted_expected_funded_amount"


# --------------------------------------------------------------------------- #
# Formatting (mirrors the React dashboard's formatGBP / formatPercent).
# --------------------------------------------------------------------------- #

def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def compact_currency(value: float, symbol: str = "£") -> str:
    if value is None or _is_nan(value):
        return "—"
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{sign}{symbol}{a / 1e9:.2f}BN"
    if a >= 1e6:
        return f"{sign}{symbol}{a / 1e6:.1f}MM"
    if a >= 1e3:
        return f"{sign}{symbol}{a / 1e3:.0f}K"
    return f"{sign}{symbol}{a:,.0f}"


def signed_currency(value: float) -> str:
    if value is None or _is_nan(value):
        return "—"
    body = compact_currency(abs(value))
    sign = "−" if value < 0 else "+"  # U+2212 minus, per dashboard
    return f"{sign}{body}"


def compact_number(value: float) -> str:
    if value is None or _is_nan(value):
        return "—"
    return f"{int(round(float(value))):,}"


def format_percent(value: float) -> str:
    if value is None or _is_nan(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def format_value(value: Any, fmt: str) -> str:
    """Render *value* per a registry format string (dashboard parity)."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if _is_nan(value):
        return "—"
    if fmt == "currency":
        return compact_currency(value)
    if fmt == "percent":
        return format_percent(value)
    if fmt == "rate":
        return f"{float(value):.2f}%"
    if fmt in ("integer", "count"):
        return compact_number(value)
    if fmt in ("decimal", "ratio"):
        return f"{float(value):,.2f}"
    if fmt == "years":
        return f"{float(value):.1f} yrs"
    return str(value)


# --------------------------------------------------------------------------- #
# Result type.
# --------------------------------------------------------------------------- #

@dataclass
class MetricResult:
    key: str
    label: str
    value: Any = None
    fmt: str = "string"
    available: bool = False
    basis: str = "unavailable"
    note: str = ""
    display: str = "—"
    lens: str = "funded"
    hint: str = ""                       # secondary context line for the tile
    # prior-period delta
    delta: Optional[float] = None
    delta_display: str = ""
    delta_dir: str = "flat"              # up | down | flat
    delta_pct: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.available and self.value is not None

    @property
    def has_delta(self) -> bool:
        return self.delta is not None and self.delta_display != ""


# --------------------------------------------------------------------------- #
# Aggregation helpers (registry-authorised only).
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
    """Resolve deck KPIs across lenses, with optional prior-period deltas."""

    def __init__(
        self,
        lenses: Dict[str, Optional[ResolvedData]],
        registries: RegistryLoader,
        analytics: Optional[Dict[str, Any]] = None,
        prior_lenses: Optional[Dict[str, Optional[ResolvedData]]] = None,
        default_lens: str = "funded",
    ):
        self.lenses = lenses or {}
        self.reg = registries
        self.analytics = analytics or {}
        self.prior_lenses = prior_lenses or {}
        self.default_lens = default_lens

    # ------------------------------------------------------------------ public
    def resolve(self, spec: Dict[str, Any]) -> MetricResult:
        key = spec.get("key", spec.get("id", "metric"))
        lens = spec.get("lens", self.default_lens)
        field_key = spec.get("field")
        label = spec.get("label") or (
            self.reg.label_for(field_key) if field_key else key.replace("_", " ").title())
        fmt = spec.get("format") or (
            self.reg.format_for(field_key) if field_key else "string")

        # 1) analytics-artifact override -----------------------------------
        art_val = self._from_analytics(key)
        if art_val is not None:
            return self._finish(key, label, art_val, fmt, lens,
                                "analytics_artifact",
                                "Sourced from MI Agent analytics artifact.", spec)

        data = self.lenses.get(lens)
        if data is None or data.df is None or data.df.empty:
            return self._unavailable(
                key, label, fmt, lens,
                f"{lens.title()} lens data not available for this run.")

        value, basis, note = self._compute(spec, data)
        if value is None or _is_nan(value):
            return self._unavailable(key, label, fmt, lens, note or "No value.")

        return self._finish(key, label, value, fmt, lens, basis, note, spec)

    def resolve_all(self, specs: List[Dict[str, Any]]) -> List[MetricResult]:
        return [self.resolve(s) for s in specs]

    # ------------------------------------------------------------- computation
    def _compute(self, spec, data: ResolvedData):
        """Return ``(value, basis, note)`` for a metric spec against *data*."""
        kind = spec.get("kind", "field")
        if kind == "data_quality":
            return self._data_quality(data)
        if kind == "largest":
            return self._largest(spec, data)
        if kind == "forecast_funded":
            return self._forecast_funded(spec)
        if kind == "count":
            return float(len(data.df)), "registry_computed", "Loan/case count."

        field_key = spec.get("field")
        if not field_key:
            return None, "unavailable", "No field bound."
        fspec = self.reg.field_spec(field_key)
        canonical = fspec.canonical_field if fspec else field_key
        if canonical not in data.df.columns or data.df[canonical].notna().sum() == 0:
            return None, "unavailable", f"Field '{canonical}' not present."

        agg = spec.get("aggregation") or (
            fspec.default_aggregation if fspec else None) or "sum"
        series = pd.to_numeric(data.df[canonical], errors="coerce")
        if agg == "sum":
            return float(series.sum()), "registry_computed", f"sum of {canonical}."
        if agg in ("avg", "mean"):
            return float(series.mean()), "registry_computed", f"mean of {canonical}."
        if agg == "median":
            return float(series.median()), "registry_computed", f"median of {canonical}."
        if agg == "weighted_avg":
            wf = spec.get("weight_field") or (fspec.weight_field if fspec else None) \
                or self.reg.default_weight_field
            if wf in data.df.columns:
                val = _weighted_avg(data.df[canonical], data.df[wf])
                if not _is_nan(val):
                    return val, "registry_computed", f"{canonical} weighted by {wf}."
            mean_val = float(series.mean())
            return mean_val, "registry_computed", f"mean of {canonical} (weight absent)."
        return float(series.sum()), "registry_computed", f"sum of {canonical}."

    def _forecast_funded(self, spec):
        """Registry forecast bridge: funded balance + Σ weighted pipeline."""
        funded = self.lenses.get("funded")
        pipe = self.lenses.get("pipeline")
        if funded is None or funded.df.empty or funded.balance_col is None:
            return None, "unavailable", "Funded lens unavailable."
        funded_sum = float(pd.to_numeric(
            funded.df[funded.balance_col], errors="coerce").sum())
        if pipe is None or pipe.df.empty:
            return None, "unavailable", "Pipeline lens unavailable for forecast bridge."
        if _WEIGHTED_PIPELINE_COL in pipe.df.columns:
            weighted = float(pd.to_numeric(
                pipe.df[_WEIGHTED_PIPELINE_COL], errors="coerce").sum())
        elif "completion_probability" in pipe.df.columns and pipe.balance_col:
            weighted = float((pd.to_numeric(pipe.df[pipe.balance_col], errors="coerce")
                              * pd.to_numeric(pipe.df["completion_probability"],
                                              errors="coerce")).sum())
        else:
            return None, "unavailable", "Pipeline weighting fields absent."
        return (funded_sum + weighted), "registry_computed", \
            "Registry forecast bridge: funded + Σ(weighted pipeline)."

    def _largest(self, spec, data: ResolvedData):
        dim = spec.get("dimension")
        if not dim or dim not in data.df.columns or data.balance_col is None:
            return None, "unavailable", f"Dimension '{dim}' or balance unavailable."
        try:
            from analytics_lib.concentration import group_shares
            table = group_shares(data.df, dim, data.balance_col)
        except Exception as exc:  # pragma: no cover
            return None, "unavailable", f"Concentration failed: {exc}"
        if table is None or table.empty:
            return None, "unavailable", "No groups to rank."
        top = table.iloc[0]
        return float(top.get("balance_share", float("nan"))), "registry_computed", \
            f"Largest {dim} exposure share · {top.get(dim)}"

    def _data_quality(self, data: ResolvedData):
        val_art = self.analytics.get("validation") if self.analytics else None
        status = None
        if isinstance(val_art, dict):
            status = (val_art.get("status") or val_art.get("overallStatus")
                      or (val_art.get("summary", {}) or {}).get("status"))
        if not status:
            errs = [i for i in data.issues if i.get("severity") == "error"]
            status = "Amber" if errs else "Green"
        return str(status).title(), "registry_computed", "Pipeline validation status."

    # -------------------------------------------------------------- analytics
    def _from_analytics(self, key: str) -> Any:
        for container in (self.analytics.get("metrics"), self.analytics.get("analytics")):
            if isinstance(container, dict) and key in container:
                entry = container[key]
                return entry.get("value") if isinstance(entry, dict) else entry
        return None

    # ----------------------------------------------------------------- finish
    def _finish(self, key, label, value, fmt, lens, basis, note, spec) -> MetricResult:
        res = MetricResult(
            key=key, label=label, value=value, fmt=fmt, available=True,
            basis=basis, note=note, display=format_value(value, fmt), lens=lens)
        # Prior-period delta (numeric metrics only; not largest/quality strings).
        if isinstance(value, (int, float)) and not _is_nan(value) \
                and spec.get("kind") not in ("largest", "data_quality"):
            self._attach_delta(res, spec, lens)
        # Secondary hint line (e.g. concentration category, MoM context).
        if spec.get("kind") == "largest":
            res.hint = str(note.split("·")[-1]).strip() if "·" in note else ""
        return res

    def _attach_delta(self, res: MetricResult, spec, lens) -> None:
        prior = self.prior_lenses.get(lens)
        if prior is None or prior.df is None or prior.df.empty:
            return
        pv, _b, _n = self._compute(spec, prior)
        if pv is None or _is_nan(pv) or pv == 0:
            return
        delta = float(res.value) - float(pv)
        res.delta = delta
        res.delta_pct = delta / abs(pv) if pv else None
        # Treat a delta that rounds to zero at display precision as "no change".
        eps = {"currency": 500.0, "percent": 5e-4, "rate": 5e-3,
               "integer": 0.5, "count": 0.5, "years": 0.05}.get(res.fmt, 1e-6)
        if abs(delta) < eps:
            res.delta_dir = "flat"
            res.delta_display = "no change vs prior"
            return
        res.delta_dir = "up" if delta > 0 else "down"
        sign = "−" if delta < 0 else "+"
        if res.fmt == "currency":
            mag = signed_currency(delta)
        elif res.fmt == "percent":
            mag = f"{sign}{abs(delta) * 100:.1f}pp"
        elif res.fmt in ("integer", "count"):
            mag = f"{sign}{compact_number(abs(delta))}"
        elif res.fmt == "rate":
            mag = f"{sign}{abs(delta):.2f}pp"
        else:
            mag = f"{sign}{abs(delta):.1f}"
        res.delta_display = f"{mag} vs prior"

    def _unavailable(self, key, label, fmt, lens, note) -> MetricResult:
        return MetricResult(key=key, label=label, value=None, fmt=fmt,
                            available=False, basis="unavailable", note=note,
                            display="—", lens=lens)
