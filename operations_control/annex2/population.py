"""operations_control.annex2.population — can every required field be filled?

The question this answers is not "does a rule mention this code" but "will a
VALUE reach the regulator". Those are different, and confusing them is how a
delivery could be declared complete here and then refused twenty-one times by
the XML builder: the projector emits a column for every registry-mapped code, so
a code always had a column, and a blank column is not a populated field.

So the assessment is made against the frame that will actually be delivered,
using the same authority the builder uses:

  * the effective Annex 2 contract (engine.regime_contract.annex2_contract) —
    which codes the workbook makes mandatory, and which no-data codes the
    regulator permits;
  * the projected frame itself — what the layered configuration and the lender's
    data actually produced.

A mandatory code blocks when the frame has no value for it and nothing in the
contract will supply one. Whether an operator could answer it, and how, is the
decision surface's business (see ``nd_treatments``); this module reports the
truth about the delivery in front of it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]

MECH_SOURCE = "value in the delivery frame"
MECH_DERIVED = "deterministic derivation"
MECH_ND = "permitted no-data value"
MECH_ATTRIBUTE = "carried as an XML attribute (no element of its own)"
MECH_OPTIONAL = "optional — may be absent"
MECH_UNSUPPORTED = "no value and no permitted representation"

_BLANK = {"", "nan", "none", "null", "<na>"}


@dataclass
class CodeAssessment:
    annex2_code: str
    field_name: str
    canonical_field: str
    mandatory: bool
    nd_permitted: str
    population_mechanism: str
    source: str
    expected_final_treatment: str
    blocking: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _column_state(df, code: str) -> str:
    """``populated`` | ``blank`` | ``absent`` for one code in the frame."""
    if df is None or code not in getattr(df, "columns", []):
        return "absent"
    col = df[code].astype(str).str.strip().str.lower()
    if col.isin(_BLANK).all():
        return "blank"
    if col.isin(_BLANK).any():
        return "partial"
    return "populated"


def _derivation_inputs_present(df, derive: Dict[str, Any]) -> bool:
    """Will this derivation actually produce a value on this frame?

    A declared derivation is not an answer unless its inputs are there. A
    lifetime mortgage's original term is months between origination and
    maturity, and a lifetime mortgage has no maturity — so the rule exists, and
    on this book it computes nothing. Reporting that as "covered by a
    derivation" is how a gap reaches the XML builder unannounced.
    """
    inputs = [c for c in (list(derive.get("fields") or [])
                          + [derive.get("start_field"), derive.get("end_field")])
              if c]
    if df is None or not inputs:
        return bool(inputs)
    for column in inputs:
        if _column_state(df, column) not in ("populated", "partial"):
            return False
        # A no-data code is a statement that there is no value; a calculation
        # cannot be performed on one.
        values = df[column].astype(str).str.strip().str.upper()
        if values.str.fullmatch(r"ND[1-5](-\d{4}-\d{2}-\d{2})?").all():
            return False
    return True


def assess_population(projected_csv: Optional[str | Path] = None,
                      *, performance_mode: Optional[str] = None,
                      ) -> List[CodeAssessment]:
    """Assess every Annex 2 code against the frame that will be delivered.

    With no frame the assessment is structural — what the contract requires and
    what the regulator permits — and nothing is reported as blocking, because
    without data there is nothing yet to be missing.
    """
    from engine.regime_contract.annex2_contract import contract

    df = None
    if projected_csv and Path(projected_csv).exists():
        import pandas as pd
        df = pd.read_csv(projected_csv, dtype=str).fillna("")

    c = contract(performance_mode)
    out: List[CodeAssessment] = []
    for code in c.codes():
        fc = c.fields[code]
        state = _column_state(df, code)
        nd = ", ".join(fc.nd_allowed) or "none"

        if not fc.emitting:
            mech, blocking = MECH_ATTRIBUTE, False
            source = "auth.099 XSD — the schema defines no element at this path"
            treatment = "disclosed as an attribute of the value it qualifies"
            expl = ("The schema carries this concept as an attribute of another "
                    "field, so it has no element that could be missing.")
        elif df is not None and state in ("populated", "partial"):
            mech = MECH_SOURCE
            blocking = bool(fc.mandatory and state == "partial")
            source = "the projected delivery frame"
            treatment = ("a value on every record" if state == "populated"
                         else "a value on some records only")
            expl = ("The delivery carries a value." if state == "populated" else
                    "The delivery carries a value on some records and not "
                    "others; the regulator requires one on every record.")
        elif fc.derive and _derivation_inputs_present(df, fc.derive):
            mech, blocking = MECH_DERIVED, False
            source = f"derivation ({fc.derive.get('type', '')})"
            treatment = "calculated during delivery preparation"
            expl = "Calculated from other delivered values; nothing to supply."
        elif not fc.mandatory:
            mech, blocking = MECH_OPTIONAL, False
            source = "workbook multiplicity"
            treatment = "may be absent from the submission"
            expl = "The schema permits this field to be absent."
        elif fc.nd_allowed:
            mech, blocking = MECH_ND, df is not None
            source = "field universe — the regulator permits a no-data code"
            treatment = f"needs an approved no-data treatment ({nd})"
            expl = ("Required and not supplied. The regulator permits a no-data "
                    "code here, so an operator can say why it is empty — until "
                    "one is approved the delivery has no value for it.")
        else:
            mech, blocking = MECH_UNSUPPORTED, df is not None
            source = "none found"
            treatment = "cannot be validly represented"
            expl = ("Required, not supplied, and the regulator permits no "
                    "no-data code: only a real value will do.")

        out.append(CodeAssessment(
            annex2_code=code, field_name=fc.field_name,
            canonical_field=fc.canonical_field, mandatory=bool(fc.mandatory),
            nd_permitted=nd, population_mechanism=mech, source=source,
            expected_final_treatment=treatment, blocking=blocking,
            explanation=expl))
    return out


def reconciliation_document(assessments: Optional[List[CodeAssessment]] = None,
                            *, projected_csv: Optional[str | Path] = None,
                            performance_mode: Optional[str] = None,
                            ) -> Dict[str, Any]:
    """The persistable reconciliation artefact + operator summary."""
    rows = assessments if assessments is not None else assess_population(
        projected_csv, performance_mode=performance_mode)
    blockers = [r for r in rows if r.blocking]
    from collections import Counter
    mech_counts = Counter(r.population_mechanism for r in rows)
    return {
        "universe_count": len(rows),
        "assessed_against": str(projected_csv) if projected_csv else "",
        "mandatory_count": sum(1 for r in rows if r.mandatory),
        "blocking_count": len(blockers),
        "blocking_codes": [r.annex2_code for r in blockers],
        "mechanism_counts": dict(mech_counts),
        "summary_sentence": (
            "Every required regulatory field has a value or a permitted way of "
            "being filled."
            if not blockers else
            f"{len(blockers)} required field"
            f"{'s' if len(blockers) != 1 else ''} cannot be filled and "
            "need attention."),
        "rows": [r.to_dict() for r in rows],
    }
