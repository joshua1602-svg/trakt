"""mi_agent.portfolio_scope — dataframe-facing portfolio scope resolution.

The bridge between the pure governed contract (:mod:`trakt_core.portfolio`) and
the canonical dataframes the MI engine actually reads. Everything that has to
*touch data* to answer a governed question about portfolios lives here, and only
here:

  * discover the portfolios present in a frame (:func:`portfolio_records`);
  * build the governed registry for a frame (:func:`registry_for_frame`);
  * filter a frame down to a resolved scope (:func:`apply_scope`);
  * work out, per portfolio, whether a requested field can actually answer
    (:func:`field_availability`) and assemble the disclosure facts
    (:func:`coverage_for_frame`).

No channel — React, Copilot, PPTX, drill-through — repeats any of this. They
receive the resolved structures and format them.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from trakt_core.portfolio import (
    FIELD_PORTFOLIO_ID,
    FIELD_PORTFOLIO_LABEL,
    FIELD_PORTFOLIO_TYPE,
    PortfolioRegistry,
    PortfolioScope,
    ScopeCoverage,
    build_coverage,
    build_registry,
    resolve_scope,
)

#: Canonical column carrying a row's reporting date, where present.
_REPORTING_DATE_COLUMNS = ("reporting_date", "data_cut_off_date", "pool_cut_off_date")


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "nat", "<na>", "null"):
        return None
    return text


def has_provenance(df: Optional[pd.DataFrame]) -> bool:
    """True when the frame carries the source-portfolio provenance contract."""
    if df is None:
        return False
    cols = set(getattr(df, "columns", []))
    return FIELD_PORTFOLIO_ID in cols or FIELD_PORTFOLIO_TYPE in cols


def _reporting_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in _REPORTING_DATE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def portfolio_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Distinct provenance records (+ row counts and reporting dates) in a frame.

    These are the raw inputs to :func:`trakt_core.portfolio.build_registry`. A
    frame with no provenance yields ``[]``, which keeps a single-portfolio
    deployment working exactly as before.
    """
    if df is None or FIELD_PORTFOLIO_ID not in getattr(df, "columns", []):
        return []
    date_col = _reporting_date_column(df)
    out: List[Dict[str, Any]] = []
    for pid, group in df.groupby(df[FIELD_PORTFOLIO_ID].astype("string"), dropna=True):
        clean_id = _clean(pid)
        if not clean_id:
            continue
        rec: Dict[str, Any] = {
            FIELD_PORTFOLIO_ID: clean_id,
            "row_count": int(len(group)),
        }
        if FIELD_PORTFOLIO_TYPE in group.columns:
            types = [t for t in (_clean(v) for v in group[FIELD_PORTFOLIO_TYPE].unique()) if t]
            if types:
                rec[FIELD_PORTFOLIO_TYPE] = types[0]
        if FIELD_PORTFOLIO_LABEL in group.columns:
            labels = [t for t in (_clean(v) for v in group[FIELD_PORTFOLIO_LABEL].unique()) if t]
            if labels:
                rec[FIELD_PORTFOLIO_LABEL] = labels[0]
        if date_col:
            dates = sorted({d for d in (_clean(v) for v in group[date_col].unique()) if d})
            if dates:
                rec["reporting_date"] = dates[-1]
                rec["reporting_dates"] = dates
        out.append(rec)
    return out


def _prefix_type_resolver(portfolio_id: str) -> Optional[str]:
    """Governed last-resort type derivation.

    Delegates to ``engine.provenance.derive_portfolio_type`` — the SAME prefix
    contract onboarding validates against and fails closed on — so this is a
    governed fallback for legacy frames that predate the type column, not an
    ad-hoc string match. It is used only when neither the data nor the metadata
    carries a type.
    """
    try:
        from engine.provenance import derive_portfolio_type
    except Exception:  # noqa: BLE001 - engine unavailable (e.g. slim deploy)
        return None
    return derive_portfolio_type(portfolio_id)


def registry_for_frame(df: Optional[pd.DataFrame], *,
                       client_id: Optional[str] = None,
                       metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
                       load_metadata: bool = True) -> PortfolioRegistry:
    """The governed portfolio registry for one canonical frame."""
    overlay: Mapping[str, Mapping[str, Any]] = metadata or {}
    if load_metadata and metadata is None:
        from .portfolio_metadata import load_portfolio_metadata
        overlay = load_portfolio_metadata(client_id)
    return build_registry(portfolio_records(df), metadata=overlay,
                          client_id=client_id, type_resolver=_prefix_type_resolver)


def resolve(df: Optional[pd.DataFrame], context_id: Optional[Any] = None, *,
            client_id: Optional[str] = None,
            metadata: Optional[Mapping[str, Mapping[str, Any]]] = None
            ) -> tuple[PortfolioRegistry, PortfolioScope]:
    """Registry + resolved scope for a frame and a requested context."""
    registry = registry_for_frame(df, client_id=client_id, metadata=metadata)
    return registry, resolve_scope(registry, context_id)


def apply_scope(df: Optional[pd.DataFrame],
                scope: Optional[PortfolioScope]) -> Optional[pd.DataFrame]:
    """Filter a frame to the portfolios in ``scope``.

    Total (or a frame without provenance) is returned unchanged. Any other scope
    filters on the RESOLVED portfolio id list, so a group is exactly the sum of
    its current members — there is no type-string filter that could quietly pick
    up a portfolio the registry did not resolve.
    """
    if df is None or scope is None or scope.is_total:
        return df
    if FIELD_PORTFOLIO_ID not in getattr(df, "columns", []):
        return df
    wanted = {str(p).strip().lower() for p in scope.portfolio_ids}
    if not wanted:
        return df.iloc[0:0]
    ids = df[FIELD_PORTFOLIO_ID].astype("string").str.strip().str.lower()
    return df[ids.isin(wanted)]


def portfolios_with_rows(df: Optional[pd.DataFrame],
                         scope: PortfolioScope) -> List[str]:
    """Portfolios in scope that actually carry rows in ``df``."""
    if df is None:
        return []
    if FIELD_PORTFOLIO_ID not in getattr(df, "columns", []):
        # No provenance: the frame is one undifferentiated book, so every
        # portfolio in scope is treated as contributing (single-portfolio shape).
        return list(scope.portfolio_ids) if len(df) else []
    present = {_clean(v) for v in df[FIELD_PORTFOLIO_ID].unique()}
    present.discard(None)
    lowered = {p.lower() for p in present if p}
    return [pid for pid in scope.portfolio_ids if pid.lower() in lowered]


def field_availability(df: Optional[pd.DataFrame], scope: PortfolioScope,
                       fields: Sequence[str]) -> Dict[str, List[str]]:
    """Per-portfolio availability of each requested field.

    A field is "available in" a portfolio when the column exists AND that
    portfolio has at least one non-null value for it. A column that is present
    but entirely null for a portfolio is *unavailable* there — which is what
    stops a missing value being aggregated as a zero.
    """
    out: Dict[str, List[str]] = {}
    if df is None or not fields:
        return out
    cols = set(getattr(df, "columns", []))
    has_pid = FIELD_PORTFOLIO_ID in cols
    ids = (df[FIELD_PORTFOLIO_ID].astype("string").str.strip().str.lower()
           if has_pid else None)
    for fname in fields:
        if not fname or fname in out:
            continue
        if fname not in cols:
            out[fname] = []
            continue
        populated = df[fname].notna()
        if not has_pid:
            out[fname] = list(scope.portfolio_ids) if bool(populated.any()) else []
            continue
        available: List[str] = []
        for pid in scope.portfolio_ids:
            mask = ids == pid.lower()
            if bool((populated & mask).any()):
                available.append(pid)
        out[fname] = available
    return out


def coverage_for_frame(df: Optional[pd.DataFrame], scope: PortfolioScope, *,
                       fields: Optional[Sequence[str]] = None,
                       field_labels: Optional[Mapping[str, str]] = None
                       ) -> ScopeCoverage:
    """The governed disclosure facts for an answer computed over ``df``.

    ``df`` is the frame BEFORE the field-level filtering an answer applies, so a
    portfolio that has rows but cannot answer for a requested field is reported
    as excluded-with-a-reason rather than silently missing.
    """
    return build_coverage(
        scope,
        portfolios_with_rows=portfolios_with_rows(df, scope),
        field_availability=field_availability(df, scope, fields or ()),
        field_labels=field_labels)


def requested_fields(spec: Any) -> List[str]:
    """The canonical fields an MI query spec actually reads.

    Used to drive field coverage from the SAME resolved fields the executor
    calculates on, so a coverage statement can never disagree with the answer.
    """
    if spec is None:
        return []
    fields: List[str] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value and value not in fields:
                fields.append(value)
            return
        if isinstance(value, Mapping):
            for v in value.values():
                _add(v)
            return
        if isinstance(value, Iterable):
            for v in value:
                _add(v)

    for attr in ("metric", "dimension", "dimensions", "weight_field",
                 "secondary_metric", "sort_field"):
        _add(getattr(spec, attr, None) if not isinstance(spec, Mapping)
             else spec.get(attr))
    resolved = (getattr(spec, "resolved_fields", None)
                if not isinstance(spec, Mapping) else spec.get("resolved_fields"))
    _add(resolved)
    return fields
