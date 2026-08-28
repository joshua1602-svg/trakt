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


#: A narrated measure: "Weighted-average Current LTV: 55.59%".
_NARRATED_PCT = re.compile(r"([A-Za-z][A-Za-z /()-]{2,40}?):\s*(-?[\d,]+\.?\d*)\s*%")
_MEASURE_PREFIXES = ("weighted-average ", "weighted average ", "average ", "wa ")


def _kpi_percentages(row: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    arte = row.get("artefacts") or row.get("artifacts") or []
    for a in arte:
        for k in (a.get("kpis") or []):
            value = str(k.get("value") or "")
            if not value.endswith("%"):
                continue
            try:
                out[str(k.get("label") or "").strip().lower()] = float(
                    value[:-1].replace(",", ""))
            except ValueError:
                continue
    return out


def coherent_rendering(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """WRONG where the answer's own prose contradicts its own rendered figure.

    EVERY FACTUAL ASSERTION IN AN ANSWER IS GRADED, not only the one the reader
    asked for. A count can be right while the sentence beside it states a
    governed measure the reader did not ask about and states it wrongly — and
    the oracle used to pass that, because the requested figure was present.

    This needs no truth file and no question id: the prose and the KPI tile are
    two renderings of ONE executed row, so a disagreement between them is a
    defect no matter what the question was. It is how the weighted-average LTV
    published as "0.56%" beside a tile reading "55.6%" becomes visible.

    Silent where the answer renders no percentage tile, which is most answers.
    """
    kpis = _kpi_percentages(row)
    if not kpis:
        return None
    for match in _NARRATED_PCT.finditer(row.get("answer") or ""):
        label, stated = match.group(1).strip(), float(match.group(2).replace(",", ""))
        key = label.lower()
        for prefix in _MEASURE_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        rendered = next((v for k, v in kpis.items() if key in k or k in key), None)
        if rendered is None:
            continue
        if abs(rendered - stated) > 0.15:
            return {"grade": WRONG,
                    "why": ("the answer states %s as %s%% and renders it as %s%% "
                            "in the same result" % (label, stated, rendered))}
    return None


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
    incoherent = coherent_rendering(row)
    if incoherent is not None:
        return incoherent
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
        arte = row.get("artefacts") or row.get("artifacts")
        rows = _artefact_rows(arte)
        # THE ARTEFACT IS THE EVIDENCE WHERE THERE IS ONE. The prose fallback
        # below asks only whether the number appears SOMEWHERE in the sentence,
        # and a sentence contains many numbers: Q10B returned five groups
        # against a truth of eight and passed, because it also said "8 loans".
        # An instrument a coincidence can satisfy is not measuring, so the
        # fallback now applies only where nothing was rendered to count.
        hit = (rows == want) if arte else (want in _ints(answer))
        checks.append("cells %d (artefact rows %d%s) %s"
                      % (want, rows, "" if arte else ", none rendered",
                         "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "delta" in truth:
        want = float(truth["delta"])
        hit = any(_near(g, want, 0.02) for g in _monies(answer))
        checks.append("delta %.2f %s" % (want, "found" if hit else "ABSENT"))
        if not hit:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "must_state" in truth:
        # THE ANSWER MUST CARRY THE OPERATION'S OWN EVIDENCE, not merely some
        # figure. A question asking which governed limit TESTS are most at risk
        # is not answered by a ranking of largest exposures, however accurate
        # that ranking is — the two are different operations over the same book,
        # and only one of them consults the governing document. Each entry names
        # something only the requested operation can produce.
        missing = [str(x) for x in truth["must_state"]
                   if str(x).lower() not in answer.lower()]
        checks.append("states %s%s" % (", ".join(map(str, truth["must_state"])),
                                       "" if not missing else
                                       " — ABSENT: " + ", ".join(missing)))
        if missing:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "must_not_state" in truth:
        # THE OPERATION'S NEGATIVE EVIDENCE. A milestone question about a target
        # the book has NOT reached must not report it as reached, and the
        # positive assertions alone cannot catch that — both answers name the
        # target. Same mechanism as `must_state`, read the other way.
        present = [str(x) for x in truth["must_not_state"]
                   if str(x).lower() in answer.lower()]
        checks.append("does not state %s%s" % (", ".join(map(str, truth["must_not_state"])),
                                               "" if not present else
                                               " — PRESENT: " + ", ".join(present)))
        if present:
            return {"grade": WRONG, "why": "; ".join(checks)}
    if "cohorts" in truth:
        # A COMPARISON IS ADJUDICATED AS A COMPARISON. The truth carries each
        # cohort's open, close and delta, and which one is larger; an answer that
        # names one of them and not the other has not compared anything. Every
        # delta must appear, and the winner must be named.
        cohorts = truth["cohorts"] or {}
        for name, figures in sorted(cohorts.items()):
            want = float(figures["delta"])
            hit = any(_near(g, want, 0.02) for g in _monies(answer))
            checks.append("%s delta %.2f %s" % (name, want, "found" if hit else "ABSENT"))
            if not hit:
                return {"grade": WRONG, "why": "; ".join(checks)}
        winner = str(truth.get("larger") or "")
        if winner:
            hit = winner.lower() in answer.lower()
            checks.append("names %s as larger %s" % (winner, "yes" if hit else "NO"))
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
    incoherent = coherent_rendering(row)
    if incoherent is not None:
        return incoherent
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
    ranked = spec.get("ranked_artefact")
    if ranked:
        verdict = _ranked_artefact_verdict(row, ranked)
        if verdict is not None:
            return verdict
    return {"grade": CORRECT, "why": "answered and met every frozen assertion"}


def _ranked_artefact_verdict(row: Dict[str, Any], spec: Dict[str, Any]
                             ) -> Optional[Dict[str, str]]:
    """THE ARTEFACT IS THE ANSWER when a question asks for a ranked list.

    "Show the largest 10 loan exposures" is answered by ten ranked rows, not by
    a sentence about the largest one — and the sentence is what the oracle used
    to read. This checks the thing the reader was actually given: the row count
    the question asked for, the ranking order, and that the cumulative share
    climbs to the pre-registered total.

    Needs the FULL artefact rows, so a capture that records only row counts
    reports NOT MEASURED rather than passing.
    """
    arte = [a for a in (row.get("artefacts") or row.get("artifacts") or [])
            if isinstance(a.get("row_data"), list) and a["row_data"]]
    if not arte:
        return {"grade": NO_ORACLE,
                "why": "the ranked artefact was not captured, so it was not measured"}
    table = max(arte, key=lambda a: len(a["row_data"]))
    rows = table["row_data"]
    want_n = int(spec["exact_rows"])
    if len(rows) != want_n:
        return {"grade": WRONG,
                "why": "expected exactly %d ranked rows, got %d" % (want_n, len(rows))}
    key, order = spec["key"], list(spec["order"])
    got = [str(r.get(key)) for r in rows]
    if got != order:
        return {"grade": WRONG,
                "why": "ranking is %s, expected %s" % (got, order)}
    cum_key = spec.get("cumulative_key")
    if cum_key:
        try:
            cums = [float(str(r.get(cum_key)).rstrip("%")) for r in rows]
        except (TypeError, ValueError):
            return {"grade": WRONG, "why": "cumulative share is not readable"}
        if any(b < a - 1e-9 for a, b in zip(cums, cums[1:])):
            return {"grade": WRONG, "why": "cumulative share does not increase: %s" % cums}
        want_cum = spec.get("cumulative_total")
        if want_cum is not None and abs(cums[-1] - float(want_cum)) > 0.05:
            return {"grade": WRONG,
                    "why": "cumulative share ends at %.2f%%, expected %.2f%%"
                           % (cums[-1], float(want_cum))}
    return None
