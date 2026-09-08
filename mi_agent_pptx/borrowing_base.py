"""mi_agent_pptx.borrowing_base — presenting the governed borrowing base.

Adapts the ``borrowingBase`` block of the ``/mi/concentration-tests`` envelope
(:func:`mi_agent.borrowing_base.service.evaluate`, injected at
``concentration_tests_api``) into the rows an investor slide can carry.

**No borrowing-base arithmetic lives here.** The advance rate, the gross and
available base, the facility cap, headroom, deficiency and both utilisations are
produced by ``mi_agent.borrowing_base.calculator`` and travel on the envelope.
This module only *selects* what a slide can hold and *formats* it. The React
panel is held to the same rule — it "does no arithmetic: it does not divide,
does not cap, does not floor a negative headroom, and does not convert a missing
input into a zero" — so the two surfaces cannot disagree about a covenant
figure. If a number is wanted that the evaluator does not supply, the fix is in
the evaluator.

``NOT_CALCULABLE`` IS A GOVERNED ANSWER, NOT A GAP. A measure the facility
configuration cannot support comes back as the string ``"NOT_CALCULABLE"``, and
it is rendered as a named missing input — "facility drawings not supplied" —
never as a dash and never as a zero. A dash would read as "nil"; a zero on a
headroom tile would read as "fully drawn".
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

#: The evaluator's sentinel for a measure the configuration cannot support.
NOT_CALCULABLE = "NOT_CALCULABLE"

#: Which missing input caused it, in the reader's words. Mirrors
#: ``MISSING_INPUT_LABEL`` in ``BorrowingBasePanel.tsx`` — one vocabulary, so a
#: funder reading the pack and the dashboard is told the same thing.
MISSING_INPUT_LABEL: Dict[str, str] = {
    "current_drawn_amount": "facility drawings not supplied",
    "facility_commitment": "commitment not configured",
    "advance_rate": "advance rate not configured",
    "concentration_denominator_floor": "denominator floor not configured",
}

#: Utilisation at or above which the tile stops being reassuring. The same two
#: thresholds the panel uses (:100 rose, :90 amber).
UTILISATION_BREACH = 100.0
UTILISATION_WARNING = 90.0


def available(envelope: Optional[Mapping[str, Any]]) -> bool:
    """Whether this book has a governed borrowing base at all."""
    return bool(snapshot(envelope).get("available"))


def snapshot(envelope: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The ``borrowingBase`` block, or an empty mapping."""
    block = (envelope or {}).get("borrowingBase")
    return dict(block) if isinstance(block, Mapping) else {}


def reason(envelope: Optional[Mapping[str, Any]]) -> str:
    """Why there is no borrowing base, in the engine's own words."""
    return str(snapshot(envelope).get("reason")
               or "No funding facility is configured for this portfolio.")


def _num(value: Any) -> Optional[float]:
    """A real number, or ``None`` — ``NOT_CALCULABLE`` included."""
    if value is None or value == NOT_CALCULABLE or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def money(value: Any) -> Optional[str]:
    """A governed money figure in the currency in force, or ``None``.

    ``None`` rather than a dash, so the caller decides what an absent value
    says — a tile says why it is missing, a table cell says nothing.
    """
    v = _num(value)
    if v is None:
        return None
    from mi_agent_api.insight_generators import money as _governed
    return _governed(v)


def pct(value: Any, places: int = 1) -> Optional[str]:
    v = _num(value)
    return None if v is None else f"{v:.{places}f}%"


def count(value: Any) -> str:
    v = _num(value)
    return "—" if v is None else f"{v:,.0f}"


def missing_phrase(snap: Mapping[str, Any], key: str) -> Optional[str]:
    """The named reason a measure could not be calculated, if it is this one."""
    if key in set(snap.get("missingInputs") or ()):
        return MISSING_INPUT_LABEL.get(key, f"{key} not supplied")
    return None


def over_drawn(snap: Mapping[str, Any]) -> bool:
    """Drawings exceed the available base. Presentation only — the governed
    headroom stays negative on the envelope."""
    headroom = _num(snap.get("borrowingBaseHeadroom"))
    return headroom is not None and headroom < 0


def utilisation_status(snap: Mapping[str, Any]) -> str:
    """``breach`` / ``warning`` / ``pass`` / ``unavailable`` for the facility.

    The same two thresholds the dashboard tile uses, so a pack and a screen do
    not disagree about whether a facility is comfortable.
    """
    value = _num(snap.get("facilityUtilisationPct"))
    if value is None:
        return "unavailable"
    if value >= UTILISATION_BREACH:
        return "breach"
    if value >= UTILISATION_WARNING:
        return "warning"
    return "pass"


def headline(snap: Mapping[str, Any]) -> str:
    """The facility line for the slide's strapline — advance rate and size."""
    facility = snap.get("facility") or {}
    bits: List[str] = []
    label = str(facility.get("facilityLabel") or "").strip()
    ftype = str(facility.get("facilityType") or "").replace("_", " ").strip()
    if label:
        bits.append(label + (f" · {ftype}" if ftype else ""))
    rate = pct(snap.get("advanceRatePct"), 0)
    if rate:
        bits.append(f"advance rate {rate}")
    commitment = money(snap.get("facilityCommitment"))
    if commitment:
        bits.append(f"commitment {commitment}")
    return " · ".join(bits) or "Governed facility eligibility and headroom"


def tiles(snap: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The five measures the dashboard leads with, in its order.

    Same measures, same order, same sublabels — a reader who knows the panel
    reads the slide without relearning it.
    """
    loans = count(snap.get("eligibleLoanCount"))
    gross = money(snap.get("grossBorrowingBase"))
    base_sub = ("capped at facility commitment" if snap.get("facilityCapBinding")
                else (f"gross {gross}" if gross else None))

    deficiency = money(snap.get("borrowingBaseDeficiency"))
    drawn_missing = missing_phrase(snap, "current_drawn_amount")

    bb_util = pct(snap.get("borrowingBaseUtilisationPct"))
    status = utilisation_status(snap)

    return [
        {"label": "ELIGIBLE COLLATERAL",
         "value": money(snap.get("eligibleCurrentBalance")),
         "sub": f"{loans} loans", "status": "neutral"},
        {"label": "BORROWING BASE",
         "value": money(snap.get("availableBorrowingBase")),
         "sub": base_sub, "missing": missing_phrase(snap, "advance_rate"),
         "status": "neutral"},
        {"label": "FACILITY DRAWN",
         "value": money(snap.get("currentDrawnAmount")),
         "sub": (snap.get("facility") or {}).get("currentDrawnAmountAsOf"),
         "missing": drawn_missing, "status": "neutral"},
        # An over-drawn facility shows nil headroom beside the deficiency,
        # exactly as the panel does. The negative governed figure is not shown
        # as a negative headroom, which reads as a rebate rather than a breach.
        {"label": "HEADROOM",
         "value": (money(0) if over_drawn(snap)
                   else money(snap.get("borrowingBaseHeadroom"))),
         "sub": (f"over-drawn — deficiency {deficiency}"
                 if over_drawn(snap) and deficiency else None),
         "missing": drawn_missing,
         "status": "breach" if over_drawn(snap) else "pass"},
        {"label": "FACILITY UTILISATION",
         "value": pct(snap.get("facilityUtilisationPct")),
         "sub": (f"{bb_util} of borrowing base" if bb_util else None),
         "missing": drawn_missing, "status": status},
    ]


def split(snap: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The eligibility split, in the panel's order and vocabulary."""
    rows = [("Eligible", "eligible", "pass"),
            ("Ineligible", "ineligible", "neutral"),
            ("Undetermined", "undetermined", "warning")]
    out = []
    for label, prefix, status in rows:
        out.append({
            "label": label, "status": status,
            "count": count(snap.get(f"{prefix}LoanCount")),
            "balance": money(snap.get(f"{prefix}CurrentBalance")) or "—",
            "share": pct(snap.get(
                f"{prefix}ShareOfFinancingPortfolioPct")) or "—",
        })
    return out


def population_line(snap: Mapping[str, Any]) -> str:
    """What the split is a split OF, and the denominator limits are tested on."""
    balance = money(snap.get("financingPortfolioBalance")) or "—"
    loans = count(snap.get("financingPortfolioLoanCount"))
    line = f"Financing Portfolio {balance} over {loans} loans"
    denominator = money(snap.get("concentrationLimitDenominator"))
    if denominator:
        line += f" · Concentration Limit Denominator {denominator}"
        if snap.get("concentrationDenominatorFloorBinding"):
            line += " (contractual floor binding)"
    return line


def alerts(snap: Mapping[str, Any]) -> List[Dict[str, str]]:
    """What must be said before the figures are read, most serious first.

    An unreconciled population is first because it disqualifies everything
    below it: the figures are then diagnostics, not a governed borrowing base,
    and a pack that showed them without saying so would be making a covenant
    claim it cannot support.
    """
    out: List[Dict[str, str]] = []
    if snap.get("reconciles") is False:
        out.append({"tone": "breach", "text":
                    "The eligibility population does not reconcile, so these "
                    "figures are shown as diagnostics and are NOT a governed "
                    "borrowing base."})
    if over_drawn(snap):
        deficiency = money(snap.get("borrowingBaseDeficiency"))
        out.append({"tone": "breach", "text":
                    f"Borrowing-base deficiency {deficiency} — drawings exceed "
                    f"the available borrowing base."})
    prototype = [str(p) for p in (snap.get("prototypeAssumptionsUsed") or ()) if p]
    if prototype:
        out.append({"tone": "warning",
                    "text": "Prototype assumption in use. " + " ".join(prototype)})
    return out


def concentration_note(snap: Mapping[str, Any]) -> str:
    """The binding concentration limit and how a breach is treated.

    The treatment matters as much as the limit: this facility MONITORS breaches
    rather than deducting for them, so a reader must not assume the base above
    is already net of one.
    """
    bits: List[str] = []
    nearest = snap.get("nearestConcentrationLimit")
    if nearest and nearest != NOT_CALCULABLE:
        line = f"Closest concentration limit: {nearest}"
        headroom_pct = pct(snap.get("nearestConcentrationHeadroomPct"), 2)
        if headroom_pct:
            line += f" · {headroom_pct} headroom"
        amount = money(snap.get("nearestConcentrationHeadroomAmount"))
        if amount:
            line += f" ({amount})"
        breached = _num(snap.get("breachedConcentrationCount")) or 0
        if breached:
            line += f" · {breached:,.0f} breached"
        bits.append(line + ".")
    note = ((snap.get("concentrationAdjustment") or {}).get("note") or "").strip()
    if note:
        bits.append(note)
    return " ".join(bits)


#: The derivation's own closed vocabulary, in the reader's words. A configured
#: rule supplies its own code, which is humanised rather than guessed at.
REASON_LABEL: Dict[str, str] = {
    "no_approved_eligibility_rules": "No approved eligibility rules",
    "prototype_financing_portfolio_assumption": "Prototype financing-portfolio assumption",
    "eligibility_rule_input_missing": "Rule input missing from the tape",
    "outside_financing_portfolio": "Outside the financing portfolio",
}

#: The reason a loan is ELIGIBLE. It is not a reason for exclusion and never
#: appears in the breakdown of why loans are out.
REASON_SATISFIED = "all_approved_eligibility_rules_satisfied"

#: How many reasons a slide can carry and stay readable.
MAX_REASONS = 5


#: Words that are acronyms in this domain, so humanising a configured rule id
#: does not print "Ltv above facility cap" on a funder's page.
_ACRONYMS = {"ltv", "dscr", "wa", "ere", "erm", "kfi", "nneg", "ppy", "cpr"}


def reason_label(code: str) -> str:
    """A reason code as a reader would say it.

    Known codes have written labels. A CONFIGURED rule supplies its own code,
    which cannot be known in advance, so it is humanised rather than guessed
    at — the code is still recognisably the code, which is what an operator
    checking the pack against the configuration needs.
    """
    if code is None:
        return "Unattributed"
    known = REASON_LABEL.get(str(code))
    if known:
        return known
    words = str(code).replace("_", " ").split()
    if not words:
        return "Unattributed"
    out = [w.upper() if w.lower() in _ACRONYMS else w.lower() for w in words]
    if out[0] not in ("",) and out[0].lower() not in _ACRONYMS:
        out[0] = out[0].capitalize()
    return " ".join(out)


def exclusion_reasons(snap: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Why loans are not eligible, largest group first.

    THE DERIVATION'S OWN COUNTS, carried on the receipt — the deck does not
    classify a loan and does not add anything up beyond ordering the groups it
    was handed. The dashboard answers this with a drill-down button; a pack
    cannot, so it carries the shape of the answer instead of a link to it.

    Returns ``[]`` where the derivation did not run, which is not the same as
    "no loans are ineligible" and is left to the caller to say.
    """
    derivation = ((snap.get("receipt") or {}).get("eligibility_derivation") or {})
    if not derivation.get("applied"):
        return []
    counts = derivation.get("reason_counts") or {}
    if not isinstance(counts, Mapping):
        return []
    rows = [{"code": str(code), "label": reason_label(code), "count": int(n)}
            for code, n in counts.items()
            if str(code) != REASON_SATISFIED and int(n or 0) > 0]
    # Largest first, then by code so equal groups do not reorder between runs.
    rows.sort(key=lambda r: (-r["count"], r["code"]))
    return rows[:MAX_REASONS]
