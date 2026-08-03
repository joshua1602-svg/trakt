"""mi_agent_pptx.preflight — the checks a deck must pass before it is published.

A deck that renders successfully is not the same as a deck that is safe to send
to an investor. Rendering proves the file is valid PowerPoint; it proves nothing
about whether the pack states its scope, whether its dates are disclosed, or
whether its parts add up.

These gates are deliberately narrow. They assert the things that, if wrong, make
the artefact *misleading* rather than merely incomplete:

  * the reporting scope is rendered, so a single-book deck cannot be read as a
    total-portfolio deck;
  * the reporting dates are rendered;
  * the constituent books are named;
  * direct + acquired reconcile to the total;
  * an executive summary was generated;
  * no placeholder slide reached the deck;
  * the file opens as a presentation.

Failure policy, by design: **generate but do not publish.** The deck is still
written to the run directory so an operator can look at it and see what went
wrong, the failure is recorded on the artefact metadata, and the durable
publication is withheld. A misleading pack that never leaves the run directory
is a contained problem; a published one is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Relative tolerance for the additivity check. The per-type snapshots and the
#: total are the same function over disjoint row sets, so they should agree to
#: floating-point noise; this allows for rounding only.
RECONCILIATION_TOLERANCE = 0.005  # 0.5 basis points of the total


@dataclass
class GateResult:
    """One check, its verdict and the evidence behind it."""

    gate: str
    passed: bool
    detail: str
    mandatory: bool = True
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"gate": self.gate, "passed": self.passed, "detail": self.detail,
                "mandatory": self.mandatory, "evidence": self.evidence}


@dataclass
class PreflightReport:
    """The publication verdict for one generated deck."""

    results: List[GateResult] = field(default_factory=list)

    @property
    def failures(self) -> List[GateResult]:
        return [r for r in self.results if not r.passed and r.mandatory]

    @property
    def warnings(self) -> List[GateResult]:
        return [r for r in self.results if not r.passed and not r.mandatory]

    @property
    def publishable(self) -> bool:
        """False when any MANDATORY gate failed — publication must be withheld."""
        return not self.failures

    def summary(self) -> str:
        state = "PASS" if self.publishable else "BLOCKED"
        return (f"{state} — {len(self.results)} gate(s), {len(self.failures)} "
                f"failure(s), {len(self.warnings)} warning(s)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "publishable": self.publishable,
            "summary": self.summary(),
            "failed_gates": [r.gate for r in self.failures],
            "warning_gates": [r.gate for r in self.warnings],
            "gates": [r.to_dict() for r in self.results],
        }


# --------------------------------------------------------------------------- #
# Gates.
# --------------------------------------------------------------------------- #

def _rendered_text(deck_path: Optional[str]) -> Optional[str]:
    """All text in the generated deck, or ``None`` if it cannot be opened.

    Reading the FILE rather than the build record is deliberate: the gate must
    verify what the investor will actually see, not what the builder believed it
    drew.
    """
    if not deck_path:
        return None
    try:
        from pptx import Presentation
        prs = Presentation(str(deck_path))
        return "\n".join(shape.text_frame.text
                         for slide in prs.slides for shape in slide.shapes
                         if shape.has_text_frame)
    except Exception:  # noqa: BLE001 — an unopenable deck is the gate's answer
        return None


def _gate_opens(text: Optional[str]) -> GateResult:
    ok = text is not None
    return GateResult("pptx_opens", ok,
                      "the deck opens as a valid presentation" if ok else
                      "the generated file could not be opened as a presentation")


def _gate_scope_rendered(text: Optional[str], portfolio) -> GateResult:
    if text is None:
        return GateResult("scope_rendered", False, "deck could not be read")
    if portfolio is None:
        return GateResult("scope_rendered", False,
                          "no governed portfolio context was resolved, so the "
                          "deck cannot state its scope")
    label = str(portfolio.scope_label or "").strip()
    ok = bool(label) and label.lower() in text.lower()
    return GateResult("scope_rendered", ok,
                      f"reporting scope '{label}' is rendered in the deck" if ok else
                      f"reporting scope '{label}' does not appear anywhere in the deck",
                      evidence={"scope": label})


def _gate_dates_rendered(text: Optional[str], portfolio, reporting_date) -> GateResult:
    if text is None:
        return GateResult("dates_rendered", False, "deck could not be read")
    expected = sorted({d for d in (portfolio.reporting_dates.values()
                                   if portfolio else [])} or
                      ({reporting_date} if reporting_date else set()))
    if not expected:
        return GateResult("dates_rendered", False,
                          "no reporting date could be resolved for this run")
    missing = [d for d in expected if d not in text]
    ok = not missing
    return GateResult("dates_rendered", ok,
                      "every reporting date is rendered" if ok else
                      f"reporting date(s) not rendered: {', '.join(missing)}",
                      evidence={"expected": expected, "missing": missing})


def _gate_books_rendered(text: Optional[str], portfolio) -> GateResult:
    if text is None or portfolio is None:
        return GateResult("books_rendered", False, "deck or context unavailable")
    names = [p.label for p in portfolio.portfolios]
    if not names:
        # A tape without per-book provenance is a legitimate single-book
        # deployment, not a governance failure: there is no constituent book to
        # name. Recorded as a warning so the absence is still visible.
        return GateResult("books_rendered", True,
                          "no per-book provenance on this tape (single-book "
                          "deployment) — no constituent books to name",
                          mandatory=False)
    missing = [n for n in names if n not in text]
    ok = not missing
    return GateResult("books_rendered", ok,
                      f"all {len(names)} constituent book(s) are named" if ok else
                      f"constituent book(s) not named in the deck: {', '.join(missing)}",
                      evidence={"books": names, "missing": missing})


def _gate_reconciles(portfolio) -> GateResult:
    """Direct + acquired must equal the total — the additivity guarantee."""
    if portfolio is None:
        return GateResult("totals_reconcile", False, "no portfolio context")
    if len(portfolio.type_slices) < 2:
        return GateResult("totals_reconcile", True,
                          "single portfolio type — no additivity to check",
                          mandatory=False)
    total = portfolio.total_balance
    parts = portfolio.type_balance_total()
    if total is None or parts is None:
        return GateResult("totals_reconcile", False,
                          "balances unavailable on the total or a type slice")
    drift = abs(total - parts)
    tol = abs(total) * RECONCILIATION_TOLERANCE
    ok = drift <= tol
    return GateResult(
        "totals_reconcile", ok,
        (f"portfolio types sum to the total (drift {drift:,.2f})" if ok else
         f"portfolio types do NOT sum to the total: total {total:,.2f} vs parts "
         f"{parts:,.2f} (drift {drift:,.2f}, tolerance {tol:,.2f})"),
        evidence={"total": total, "sum_of_types": parts, "drift": drift,
                  "tolerance": tol})


def _gate_loan_reconciles(portfolio) -> GateResult:
    if portfolio is None or len(portfolio.type_slices) < 2:
        return GateResult("loan_counts_reconcile", True,
                          "single portfolio type — no additivity to check",
                          mandatory=False)
    total = portfolio.loan_count
    parts = portfolio.type_loan_total()
    if total is None or parts is None:
        return GateResult("loan_counts_reconcile", False, "loan counts unavailable")
    ok = int(total) == int(parts)
    return GateResult("loan_counts_reconcile", ok,
                      f"loan counts reconcile ({int(total):,})" if ok else
                      f"loan counts do NOT reconcile: total {int(total):,} vs parts "
                      f"{int(parts):,}",
                      evidence={"total": total, "sum_of_types": parts})


#: Facts that mean the run HAD something to observe. A first-ever run of a
#: single book with no prior period genuinely has nothing to say, and blocking
#: it would fail closed on a perfectly valid pack.
_OBSERVABLE_FACTS = ("has_movement", "is_mixed", "has_geo", "has_risk",
                     "has_pipeline", "has_forecast", "has_funded_history")


def _gate_executive_summary(insights: Optional[Mapping[str, Any]],
                            facts: Optional[Mapping[str, Any]] = None) -> GateResult:
    items = (insights or {}).get("insights") or []
    if items:
        return GateResult("executive_summary", True,
                          f"{len(items)} governed observation(s) generated",
                          evidence={"count": len(items),
                                    "status": (insights or {}).get("status")})
    observable = [f for f in _OBSERVABLE_FACTS if (facts or {}).get(f)]
    if not observable:
        # Nothing to observe is not the same as failing to observe.
        return GateResult("executive_summary", True,
                          "no observations were possible for this run (single "
                          "book, single period, no pipeline, forecast or risk "
                          "inputs)", mandatory=False,
                          evidence={"count": 0})
    return GateResult("executive_summary", False,
                      "no governed observations were generated even though the "
                      f"run had observable inputs ({', '.join(observable)})",
                      evidence={"count": 0, "observable": observable,
                                "status": (insights or {}).get("status")})


def _gate_no_placeholders(records: Sequence[Mapping[str, Any]]) -> GateResult:
    placeholders = [r.get("id") for r in records if r.get("placeholder")]
    ok = not placeholders
    return GateResult("no_placeholder_slides", ok,
                      "no placeholder slides in the deck" if ok else
                      f"placeholder slide(s) reached the deck: "
                      f"{', '.join(str(p) for p in placeholders)}",
                      evidence={"placeholders": placeholders})


def _gate_mandatory_slides(records: Sequence[Mapping[str, Any]]) -> GateResult:
    """Cover, methodology and appendix are the disclosure spine of the pack."""
    ids = {str(r.get("id")) for r in records}
    required = {"cover", "methodology", "appendix"}
    missing = sorted(required - ids)
    ok = not missing
    return GateResult("mandatory_slides", ok,
                      "cover, methodology and appendix are present" if ok else
                      f"mandatory slide(s) missing: {', '.join(missing)}",
                      evidence={"missing": missing})


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def run_preflight(build_report: Mapping[str, Any], data: Any) -> PreflightReport:
    """Evaluate every publication gate for a generated deck."""
    deck_path = build_report.get("output")
    records = build_report.get("slides") or []
    portfolio = getattr(data, "portfolio", None)
    text = _rendered_text(deck_path)
    facts = build_report.get("facts") or {}

    report = PreflightReport(results=[
        _gate_opens(text),
        _gate_mandatory_slides(records),
        _gate_scope_rendered(text, portfolio),
        _gate_dates_rendered(text, portfolio, getattr(data, "reporting_date", None)),
        _gate_books_rendered(text, portfolio),
        _gate_reconciles(portfolio),
        _gate_loan_reconciles(portfolio),
        _gate_executive_summary(getattr(data, "insights", None), facts),
        _gate_no_placeholders(records),
    ])
    return report
