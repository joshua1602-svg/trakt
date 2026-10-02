"""mi_agent_pptx.forecast_accuracy — WORDING for the forecaster's track record.

THE MEASURE MOVED. ``evolution.forecast_evolution`` now computes per-period
forecast error and the ``forecastAccuracy`` track record over it — because the
one number a funder uses to judge a forecaster is an analytical result, not a
slide's arithmetic, and while it lived here React could not have shown it at all.

What remains here is presentation: reading the engine's structured finding and
choosing the English for it. The engine says whether there is a lean and how
large; this module decides whether to call it "overstated" or "understated", and
says nothing at all where the engine reports no track record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: Below this the lean is not a lean. A forecaster whose mean signed error is a
#: fifth of a percent of the book is not biased in any sense a reader should act
#: on, and naming a direction there manufactures a finding. This is a
#: PRESENTATION threshold — it decides whether to say a word, not what is true.
BIAS_FLOOR_PCT = 0.5


@dataclass(frozen=True)
class Accuracy:
    """The engine's track record, in the shape the slides read."""

    observations: int = 0
    bias_pct: Optional[float] = None
    error_pct: Optional[float] = None
    worst_pct: Optional[float] = None
    worst_period: Optional[str] = None
    reason: str = ""
    available: bool = False

    @property
    def lean(self) -> str:
        """"over", "under", or "" where the lean is inside the floor."""
        if self.bias_pct is None or abs(self.bias_pct) < BIAS_FLOOR_PCT:
            return ""
        # A NEGATIVE error is actual below forecast: the forecast was high.
        return "over" if self.bias_pct < 0 else "under"

    def to_dict(self) -> dict:
        return {"observations": self.observations, "biasPct": self.bias_pct,
                "errorPct": self.error_pct, "worstPct": self.worst_pct,
                "worstPeriod": self.worst_period, "reason": self.reason}


def measure(evolution: Optional[Mapping[str, Any]]) -> Accuracy:
    """Read the governed track record off the forecast evolution payload."""
    payload = (evolution or {}).get("forecastAccuracy") or {}
    return Accuracy(
        observations=int(payload.get("observations") or 0),
        bias_pct=payload.get("biasPct"),
        error_pct=payload.get("errorPct"),
        worst_pct=payload.get("worstPct"),
        worst_period=payload.get("worstPeriod"),
        reason=str(payload.get("reason") or
                   "no forecast history is available for this book"),
        available=bool(payload.get("available")),
    )


def describe(accuracy: Accuracy) -> str:
    """One sentence a funder can act on, or the reason there is not one."""
    if not accuracy.available:
        return accuracy.reason
    lean = accuracy.lean
    sentence = (f"Across {accuracy.observations} periods the published forecast "
                f"was typically {accuracy.error_pct:.1f}% from the outturn")
    if lean:
        sentence += (f", and {lean}stated it on average by "
                     f"{abs(accuracy.bias_pct):.1f}%")
    else:
        sentence += ", with no consistent lean in either direction"
    if accuracy.worst_period:
        sentence += (f"; the widest miss was {abs(accuracy.worst_pct):.1f}% "
                     f"at {accuracy.worst_period}")
    return sentence + "."
