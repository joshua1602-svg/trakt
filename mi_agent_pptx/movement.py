"""mi_agent_pptx.movement — where the period's movement came from.

A total that moved £554k tells an investor almost nothing. *Which* regions,
channels, LTV bands and ticket bands moved, and by how much, is the analysis.

All of it comes from one governed service:
:func:`mi_agent_api.evolution.funded_bridge`. Its per-category deltas sum
EXACTLY to (closing − opening), and it aggregates the tail into an explicit
"Other" so a many-category view stays legible and still reconciles. This module
calls it once per dimension per scope and formats the result. **No movement is
computed here.**

Two deliberate constraints:

* **Dimensions are candidate lists, never a fixed column.** ``funded_bridge``
  resolves the first candidate actually present in the data, so a client whose
  tape carries ``broker_name`` rather than ``broker_channel`` still attributes.
  Nothing here assumes a particular client's schema.
* **Terminology stays cautious.** A balance that fell is a *reduction*, not a
  redemption; a balance that rose is an *increase*, not a funding. Naming a
  cause the decomposition does not prove would be a fabrication, however
  plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Governed dimensions the deck attributes movement across. Each entry is an
#: ordered candidate list; the first column present in the funded data wins.
#: Ordering mirrors the canonical field families the MI layer already reads.
DIMENSIONS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("region", "Region", ("geographic_region_collateral",
                          "geographic_region_obligor", "region")),
    ("broker", "Broker / channel", ("broker_channel", "broker_name", "broker",
                                    "origination_channel")),
    ("ltv", "LTV band", ("ltv_bucket", "ltv_band")),
    ("ticket", "Ticket-size band", ("ticket_bucket", "ticket_band")),
    ("borrower_type", "Borrower type", ("borrower_type",)),
    ("age", "Borrower age band", ("age_bucket", "age_band")),
    ("product", "Product", ("product", "product_type", "erm_product_type")),
    ("property_value", "Property-value band", ("property_value_bucket",)),
)

#: Below this share of the opening balance a category movement is noise.
MATERIAL_SHARE_PCT = 0.25


@dataclass(frozen=True)
class Contributor:
    """One category's contribution to the period movement."""

    category: str
    start: float
    end: float
    delta: float
    is_other: bool = False


@dataclass(frozen=True)
class MovementBridge:
    """The governed attribution for one dimension, ready to render."""

    key: str
    label: str
    available: bool
    dimension_col: Optional[str] = None
    opening: Optional[float] = None
    closing: Optional[float] = None
    start_period: Optional[str] = None
    end_period: Optional[str] = None
    #: The governed reporting dates behind the two periods, so a movement can
    #: always state the window it was measured over.
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    contributors: Tuple[Contributor, ...] = ()
    reason: Optional[str] = None

    @property
    def total_delta(self) -> Optional[float]:
        if self.opening is None or self.closing is None:
            return None
        return self.closing - self.opening

    def reconciles(self, tolerance: float = 0.51) -> bool:
        """Contributor deltas must sum to the headline movement.

        ``funded_bridge`` guarantees this by construction (the "Other" bucket
        carries the residual); the check exists so a future change that broke it
        would fail a gate rather than quietly misattribute.
        """
        if self.total_delta is None or not self.contributors:
            return False
        return abs(sum(c.delta for c in self.contributors) - self.total_delta) <= tolerance

    def share_change_pp(self, contributor: Contributor) -> Optional[float]:
        """Change in a category's share of the book, in percentage points."""
        if not self.opening or not self.closing:
            return None
        return (contributor.end / self.closing - contributor.start / self.opening) * 100.0

    def movers(self, limit: int = 3) -> Tuple[List[Contributor], List[Contributor]]:
        """(largest increases, largest reductions), material ones only."""
        floor = abs(self.opening or 0.0) * MATERIAL_SHARE_PCT / 100.0
        real = [c for c in self.contributors
                if not c.is_other and abs(c.delta) >= floor]
        ups = sorted([c for c in real if c.delta > 0],
                     key=lambda c: -c.delta)[:limit]
        downs = sorted([c for c in real if c.delta < 0],
                       key=lambda c: c.delta)[:limit]
        return ups, downs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "available": self.available,
            "dimension_col": self.dimension_col, "opening": self.opening,
            "closing": self.closing, "start_period": self.start_period,
            "end_period": self.end_period, "start_date": self.start_date,
            "end_date": self.end_date, "reason": self.reason,
            "contributors": [{"category": c.category, "start": c.start,
                              "end": c.end, "delta": c.delta,
                              "is_other": c.is_other}
                             for c in self.contributors],
        }


def _adapt(key: str, label: str, payload: Mapping[str, Any]) -> MovementBridge:
    if not payload or not payload.get("available"):
        return MovementBridge(key=key, label=label, available=False,
                              reason=(payload or {}).get("reason")
                              or "attribution is not available for this dimension")
    contributors = tuple(
        Contributor(category=str(c.get("category", "")),
                    start=float(c.get("start") or 0.0),
                    end=float(c.get("end") or 0.0),
                    delta=float(c.get("delta") or 0.0),
                    is_other=bool(c.get("isOther")))
        for c in payload.get("contributions") or payload.get("contributors") or ())
    start = payload.get("start") or {}
    end = payload.get("end") or {}
    # The governed bridge falls back to the first candidate column when none is
    # present, which yields an empty grouping and a 0 → 0 "movement". That is an
    # absent dimension, not a book that did not move, and rendering it as the
    # latter would be a quiet misstatement.
    opening, closing = _f(start.get("total")), _f(end.get("total"))
    if not contributors or (not opening and not closing):
        return MovementBridge(
            key=key, label=label, available=False,
            dimension_col=payload.get("dimensionCol"),
            reason=f"the funded data carries no {label.lower()} for this book")
    return MovementBridge(
        key=key, label=label, available=True,
        dimension_col=payload.get("dimensionCol"),
        opening=opening,
        closing=closing,
        start_period=start.get("period"),
        end_period=end.get("period"),
        start_date=start.get("reporting_date"),
        end_date=end.get("reporting_date"),
        contributors=contributors)


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def build_bridges(output_root, client_id: str, run_id: Optional[str], *,
                  start_period: Optional[str] = None,
                  lens_filters: Optional[Dict[str, str]] = None,
                  lens_label: str = "Total",
                  keys: Optional[Sequence[str]] = None,
                  top_n: int = 6,
                  note=None) -> Dict[str, MovementBridge]:
    """Governed movement attribution per dimension, computed once per scope.

    Only the requested ``keys`` are computed; each is one call to the governed
    bridge. The caller holds the result for the whole deck, so no slide triggers
    a recomputation.

    ``start_period`` is the opening period the attribution is measured FROM, and
    the caller must supply the same period its headline movement uses. Left
    unset, the governed bridge opens at the EARLIEST period it can find — which
    on a two-period book is the prior period and looks correct, and on a
    fourteen-period book attributes fourteen months of movement beside a
    one-month headline.
    """
    from mi_agent_api import evolution

    wanted = set(keys) if keys else {k for k, _l, _c in DIMENSIONS}
    out: Dict[str, MovementBridge] = {}
    for key, label, candidates in DIMENSIONS:
        if key not in wanted:
            continue
        try:
            payload = evolution.funded_bridge(
                output_root, client_id, list(candidates), to_run_id=run_id,
                start_period=start_period,
                lens_filters=lens_filters, lens_label=lens_label, top_n=top_n)
        except Exception as exc:  # noqa: BLE001 — one dimension must not break the deck
            if note:
                note(f"movement[{key}]: {type(exc).__name__}: {exc}")
            continue
        out[key] = _adapt(key, label, payload)
    return out


# --------------------------------------------------------------------------- #
# Deterministic narration.
# --------------------------------------------------------------------------- #

def _money(v: Optional[float]) -> str:
    """Prose notation, matching the governed insight generators.

    The deck deliberately runs two notations, one per medium:

      * TILES and CHART LABELS use ``compact_currency`` (£833.5MM) — the
        dashboard's own ``formatGBP``, so an exported tile and the screen it
        mirrors are character-for-character identical;
      * SENTENCES use £833.5m, matching ``mi_agent_api.insight_generators``,
        which also writes React's weekly brief.

    These sentences are prose and feed the executive summary, so they follow the
    prose rule. Formatting them as chart labels made one summary card disagree
    with the five beside it.
    """
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"£{v / 1e9:.2f}bn"
    if a >= 1e6:
        return f"£{v / 1e6:.1f}m"
    if a >= 1e3:
        return f"£{v / 1e3:.0f}k"
    return f"£{v:,.0f}"


def _signed(v: Optional[float]) -> str:
    return "—" if v is None else ("+" if v >= 0 else "−") + _money(abs(v))


def takeaways(bridge: MovementBridge, limit: int = 2) -> List[str]:
    """Deterministic sentences for one dimension's movement.

    Cautious by construction: an increase is an increase. Nothing here claims a
    redemption, a funding or a cause, because the balance decomposition does not
    evidence one.
    """
    if not bridge.available or not bridge.contributors:
        return []
    ups, downs = bridge.movers(limit=limit)
    lines: List[str] = []
    for c in ups:
        pp = bridge.share_change_pp(c)
        share = f", taking its share to {c.end / bridge.closing * 100:.1f}%" \
            if bridge.closing else ""
        lines.append(f"{c.category} increased {_money(abs(c.delta))}{share}"
                     + (f" ({pp:+.1f}pp)" if pp is not None else "") + ".")
    for c in downs:
        pp = bridge.share_change_pp(c)
        lines.append(f"{c.category} reduced {_money(abs(c.delta))}"
                     + (f" ({pp:+.1f}pp share)" if pp is not None else "") + ".")
    if not lines:
        lines.append(f"No individual {bridge.label.lower()} moved materially "
                     f"over the period.")
    return lines[: limit * 2]


def headline(bridge: MovementBridge) -> Optional[str]:
    """One sentence naming the largest contributor in either direction."""
    if not bridge.available:
        return None
    ups, downs = bridge.movers(limit=1)
    if ups and downs:
        return (f"{ups[0].category} contributed the largest increase "
                f"({_signed(ups[0].delta)}); {downs[0].category} the largest "
                f"reduction ({_signed(downs[0].delta)}).")
    if ups:
        return (f"{ups[0].category} contributed the largest increase "
                f"({_signed(ups[0].delta)}).")
    if downs:
        return (f"{downs[0].category} contributed the largest reduction "
                f"({_signed(downs[0].delta)}).")
    return None
