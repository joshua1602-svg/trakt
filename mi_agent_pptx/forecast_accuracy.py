"""mi_agent_pptx.forecast_accuracy — did the published forecast hold?

A funder reads a forecast page and asks one question before any other: has this
forecaster been right before? The pack drew two lines and left the reader to
answer it by eye.

Everything here is arithmetic over figures the governed evolution service has
ALREADY reconciled — ``prior_forecast`` at period N is ``forecast_funded_balance``
at period N-1, and the variance between them is already computed. No new model,
no new economics, no projection: this module summarises a track record that
exists, and refuses to summarise one that does not.

The two measures are deliberately the plain ones:

    BIAS   the mean signed error. Says which way the forecaster leans —
           persistently under or persistently over — which a mean absolute
           error hides completely.
    ERROR  the mean absolute error. Says how far off it typically was,
           regardless of direction.

Both are reported as a percentage of the actual balance, because a funder
comparing a 40m book with a 400m one cares about the proportion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

#: A track record needs more than one observation. One period in which the
#: forecast happened to be close is luck, and reporting it as "0.4% mean error"
#: dresses a single coincidence as a measured property of the process.
MIN_OBSERVATIONS = 2

#: Below this the lean is not a lean. A forecaster whose mean signed error is a
#: fifth of a percent of the book is not biased in any sense a reader should act
#: on, and naming a direction there manufactures a finding.
BIAS_FLOOR_PCT = 0.5


@dataclass(frozen=True)
class Accuracy:
    """The forecaster's track record on this book, or the reason there is none."""

    observations: int = 0
    bias_pct: Optional[float] = None
    error_pct: Optional[float] = None
    worst_pct: Optional[float] = None
    worst_period: Optional[str] = None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.observations >= MIN_OBSERVATIONS and self.error_pct is not None

    @property
    def lean(self) -> str:
        """"over", "under", or "" where the lean is inside the floor."""
        if self.bias_pct is None or abs(self.bias_pct) < BIAS_FLOOR_PCT:
            return ""
        # A NEGATIVE variance is actual below forecast: the forecast was high.
        return "over" if self.bias_pct < 0 else "under"

    def to_dict(self) -> dict:
        return {"observations": self.observations, "biasPct": self.bias_pct,
                "errorPct": self.error_pct, "worstPct": self.worst_pct,
                "worstPeriod": self.worst_period, "reason": self.reason}


def measure(evolution: Optional[Mapping[str, Any]]) -> Accuracy:
    """Summarise the published forecast against what actually happened."""
    periods: Sequence[Mapping[str, Any]] = (evolution or {}).get("periods") or ()
    errors = []
    for period in periods:
        metrics = period.get("metrics") or {}
        prior, actual = metrics.get("prior_forecast"), metrics.get("funded_balance")
        if prior in (None, 0) or actual is None:
            continue
        try:
            pct = (float(actual) - float(prior)) / abs(float(prior)) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        errors.append((str(period.get("period")
                           or period.get("reporting_date") or ""), pct))
    if len(errors) < MIN_OBSERVATIONS:
        return Accuracy(
            observations=len(errors),
            reason=(f"a forecast track record needs at least {MIN_OBSERVATIONS} "
                    f"periods carrying a prior forecast; {len(errors)} available"))
    values = [pct for _, pct in errors]
    worst_period, worst = max(errors, key=lambda e: abs(e[1]))
    return Accuracy(
        observations=len(errors),
        bias_pct=round(sum(values) / len(values), 2),
        error_pct=round(sum(abs(v) for v in values) / len(values), 2),
        worst_pct=round(worst, 2),
        worst_period=worst_period)


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
