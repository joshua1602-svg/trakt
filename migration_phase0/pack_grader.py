"""migration_phase0/pack_grader.py — the grader that produces MI_REVIEW_PACK_166.

IN THE REPOSITORY BECAUSE IT BURIED VERDICTS FOR MONTHS AND NOBODY COULD SEE IT.
It lived in an ephemeral scratch directory while the pack it grades is reviewed
by hand and committed here. A review artefact whose oracle is not reviewable is
an assertion, not evidence — and this one was silently overruling a human's
recorded judgement (see `_FROZEN_GRADE`).

Stage 4 grader — correct / wrong / false refusal / true refusal.

Deterministic, from the PRE-REGISTERED oracles: the 75-bank's
`independent_truth` figures computed from the tape, and the CFO bank's frozen
`expect` / `must` / `must_not` / `rows`.

It must be able to produce every class it reports, and it is calibrated against
the frozen deterministic run before it is used on anything else.
"""
import json, re
from typing import Any, Dict, List, Optional

CORRECT, WRONG, FALSE_REFUSAL, TRUE_REFUSAL = (
    "CORRECT", "WRONG", "FALSE_REFUSAL", "TRUE_REFUSAL")
NO_ORACLE = "NO_COMPUTABLE_TRUTH"

#: THE FROZEN HUMAN VERDICT, mapped into this grader's vocabulary.
#:
#: F3 SAYS an instrument that cannot be measured must report NOT MEASURED, never
#: clean. It does NOT say not-measured may overrule a measurement somebody else
#: made. This grader was doing exactly that: `independent_truth` is null on 24 of
#: the 75 rows, and it returned NO_COMPUTABLE_TRUTH the moment it was null —
#: over the top of a `grade` field sitting in the SAME ROW of the same file,
#: recorded by a human who had read the answer.
#:
#: Five verdicts were buried that way, and the label made them invisible in every
#: pack this programme has published:
#:
#:     Q10A  WRONG / SILENT              answered from the FUNDED book; the
#:                                       question named the pipeline dataset
#:     Q07B  WRONG / SILENT              both scopes dropped; a whole-book figure
#:                                       answered a comparison question
#:     Q25A/B/C  CURRENT-STATE SUBSTITUTION
#:                                       a FORWARD question answered with today's
#:                                       risk-limit status
#:
#: Q07B has nothing to do with the dataset class it was found alongside. It was
#: buried for four months of this programme by a label that reads as innocuous.
_FROZEN_GRADE = {
    "EXACT": CORRECT,
    "SUBSTANTIVELY CORRECT": CORRECT,
    "WRONG / SILENT": WRONG,
    "WRONG / DISCLOSED": WRONG,
    "CURRENT-STATE SUBSTITUTION": WRONG,
    "FALSE REFUSAL": FALSE_REFUSAL,
    "CAPABILITY GAP": TRUE_REFUSAL,
}


def _normalise(text: Any) -> str:
    """An answer reduced to what a human graded, so two runs can be compared."""
    return " ".join(str(text or "").split())


def frozen_verdict(row: Dict[str, Any], frozen: Optional[Dict[str, Any]]
                   ) -> Optional[Dict[str, str]]:
    """The recorded human grade — ONLY where it is still about THIS answer.

    DEFERENCE IS GATED ON FIDELITY, and it has to be. A frozen grade is a
    judgement of a PARTICULAR answer, not of a question. Code has shipped since
    that run — the interest-rate declaration, the fragment property, the
    value-collision rule — and several answers have legitimately moved. Deferring
    to a grade recorded against a different answer would assert a stale verdict
    with a human's authority behind it, which is a worse failure than the one
    this fixes.

    So the recorded answer must match today's, byte for byte after whitespace
    normalisation. Where it does not, the frozen grade is STALE and this returns
    None: NO_COMPUTABLE_TRUTH is then the honest report, and it says so.

    This is the same discipline the merge arm's replay already uses — assert
    fidelity, never assume it.
    """
    if not frozen:
        return None
    grade = _FROZEN_GRADE.get(str(frozen.get("grade") or "").strip().upper())
    if grade is None:
        return None
    if _normalise(row.get("answer")) != _normalise(frozen.get("answer")):
        return None
    why = str(frozen.get("grader_rationale") or "").strip()
    return {"grade": grade,
            "why": "frozen human grade %r on a byte-identical answer%s"
                   % (frozen.get("grade"), (" — " + why) if why else "")}

_MONEY = re.compile(r"£\s*([\d,]+(?:\.\d+)?)\s*(bn|b|mm|m|k)?", re.I)
_PCT = re.compile(r"([\d]+(?:\.\d+)?)\s*%")
_INT = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(?![\d.,%])")
_SCALE = {"bn": 1e9, "b": 1e9, "mm": 1e6, "m": 1e6, "k": 1e3, None: 1.0, "": 1.0}


def _monies(text: str) -> List[float]:
    out = []
    for raw, unit in _MONEY.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")) * _SCALE[(unit or "").lower()])
        except (ValueError, KeyError):
            continue
    return out


def _ints(text: str) -> List[int]:
    out = []
    for raw in _INT.findall(text or ""):
        try:
            out.append(int(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def _pcts(text: str) -> List[float]:
    return [float(x) for x in _PCT.findall(text or "")]


def _near(got: float, want: float, rel: float = 0.01) -> bool:
    if want == 0:
        return abs(got) < 1e-9
    return abs(got - want) / abs(want) <= rel


def _artefact_rows(artefacts) -> int:
    return max([int(a.get("rows") or 0) for a in (artefacts or [])] or [0])


def grade_75(row: Dict[str, Any], truth: Optional[Dict[str, Any]],
             frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One 75-bank answer against its pre-registered truth.

    PRECEDENCE, and the order is the point:

      1. a refusal is a refusal;
      2. the COMPUTED truth, where one exists — it is the stronger oracle and it
         is what caught Q19C, so a frozen human grade never overrules it;
      3. the FROZEN HUMAN GRADE, where no truth is computable AND the answer is
         still the one that was graded (see `frozen_verdict`);
      4. NO_COMPUTABLE_TRUTH, which now means what it says — nobody has measured
         this answer — rather than "I could not, so nothing else counts".
    """
    answer = row.get("answer") or ""
    ok = bool(row.get("ok"))
    if not ok:
        return {"grade": FALSE_REFUSAL, "why": "expected an answer, got a refusal"}
    if not truth:
        deferred = frozen_verdict(row, frozen)
        if deferred is not None:
            return deferred
        stale = bool(frozen and _FROZEN_GRADE.get(
            str(frozen.get("grade") or "").strip().upper()))
        return {"grade": NO_ORACLE,
                "why": ("no independent truth, and the frozen grade %r was "
                        "recorded against a different answer" % frozen.get("grade"))
                       if stale else
                       "no independent truth was computed for this case"}

    checks: List[str] = []
    if "count" in truth:
        want = int(truth["count"])
        hit = want in _ints(answer)
        checks.append("count %d %s" % (want, "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "balance" in truth and "wa_ltv" not in truth:
        want = float(truth["balance"])
        hit = any(_near(g, want) for g in _monies(answer))
        checks.append("balance %.2f %s" % (want, "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "wa_ltv" in truth:
        want = float(truth["wa_ltv"])
        hit = any(abs(g - want) <= 0.06 for g in _pcts(answer))
        checks.append("wa_ltv %.2f%% %s" % (want, "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "cells" in truth:
        want = int(truth["cells"])
        rows = _artefact_rows(row.get("artefacts") or row.get("artifacts"))
        hit = rows == want or want in _ints(answer)
        checks.append("cells %d (artefact rows %d) %s" % (want, rows,
                                                          "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "delta" in truth:
        want = float(truth["delta"])
        hit = any(_near(g, want, 0.02) for g in _monies(answer))
        checks.append("delta %.2f %s" % (want, "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "top_region" in truth:
        want = str(truth["top_region"])
        hit = want.lower() in answer.lower()
        checks.append("top region %s %s" % (want, "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if not checks:
        return {"grade": NO_ORACLE, "why": "truth carries no checkable figure"}
    return {"grade": CORRECT, "why": "; ".join(checks)}


def grade_cfo(row: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """One CFO-91 answer against the frozen expectation."""
    answer = row.get("answer") or ""
    ok = bool(row.get("ok"))
    expect = str(spec.get("expect") or "DELIVER").upper()
    if expect == "REFUSE":
        if ok:
            return {"grade": WRONG,
                    "why": "the bank expects a refusal and this answered"}
        return {"grade": TRUE_REFUSAL, "why": "refused, as the bank expects"}
    if not ok:
        return {"grade": FALSE_REFUSAL, "why": "the bank expects an answer"}
    low = answer.lower()
    for needle in (spec.get("must") or []):
        if str(needle).lower() not in low:
            return {"grade": WRONG, "why": "answer omits required %r" % needle}
    for needle in (spec.get("must_not") or []):
        if str(needle).lower() in low:
            return {"grade": WRONG, "why": "answer contains forbidden %r" % needle}
    want_rows = spec.get("rows")
    if want_rows:
        rows = _artefact_rows(row.get("artefacts") or row.get("artifacts"))
        if rows < int(want_rows):
            return {"grade": WRONG,
                    "why": "expected at least %s rows, got %d" % (want_rows, rows)}
    return {"grade": CORRECT, "why": "answered and met every frozen assertion"}
