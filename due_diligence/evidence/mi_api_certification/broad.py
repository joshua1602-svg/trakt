#!/usr/bin/env python3
"""Scoring for the BROAD live certification.

The core suite in `certify_mi_api` is a release smoke gate: a small set of
questions whose verdicts were argued one at a time. This module is the other
half — a wide, DATA-DRIVEN sweep whose cases live in `broad_cases.json` as
QUESTIONS AND EXPECTATIONS ONLY, and whose scoring reads the response's own
evidence rather than a figure remembered from a previous run.

Three rules govern everything here.

FIRST, a bank replay is not a certification. Counting how many questions came
back with ``ok: true`` cannot see a wrong population, a lost filter, a wrong
aggregation, or a breakdown that quietly went missing — all of which arrive as
``ok: true``. Every case below therefore carries an expectation, and every
answered response is additionally put through `coherence`, which needs no
expectation at all.

SECOND, an expectation must survive the book changing. So the assertions are
identities between two live responses (a paraphrase agrees, a narrowing never
widens, the parts of a partition sum to the whole) and never a monetary total
copied into git. Nothing in `broad_cases.json` records a figure.

THIRD, where the response does not expose enough to decide a check, the case is
reported NOT ESTABLISHED. It is not reported as a pass. A harness that cannot
tell "checked and correct" from "could not look" is worse than no harness,
because it reports confidence it did not earn.

A refusal is not a failure. A book that does not carry `erm_product_type`
cannot answer a lump-sum question however well it understands one, and the
suite is deliberately not tuned for answer rate: MAY_REFUSE_FOR_DATA cases are
reported, not failed. What a refusal must never be is a broader figure.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CASES_PATH = Path(__file__).with_name("broad_cases.json")

#: Per-group ratio columns. Additive reasoning does not apply to them, so they
#: are kept out of every measure fingerprint rather than silently summed.
_RATIO_COLUMNS = {"concentration_pct", "share_pct", "pct", "percentage"}

#: Aggregations whose group values sum to the ungrouped value. Anything else
#: (an average, a weighted average, a median) does NOT reconcile by addition,
#: and a reconciliation check over one is reported NOT ESTABLISHED rather than
#: computed wrongly.
_ADDITIVE = {"sum", "count"}

#: The estate's own wording for "this book does not report that field".
_DATA_MARKERS = (
    "not available in this dataset",
    "unavailable in this dataset",
    "is not in this dataset",
    "does not report it",
    "field is unavailable",
    "no reporting periods are available",
    "not supported",
    "cannot be applied",
    "could not be applied",
    "no loans in this book match",
)


# --------------------------------------------------------------------------- #
# Evidence: what one response actually publishes about itself
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    question: str
    envelope: Dict[str, Any]
    seconds: float = 0.0

    # transport
    transport_error: bool = False
    http_status: Optional[int] = None

    ok: bool = False
    population: Optional[int] = None
    population_total: Optional[int] = None
    group_count: Optional[int] = None
    aggregation: Optional[str] = None
    fingerprint: Dict[str, float] = field(default_factory=dict)
    cells: Optional[Dict[frozenset, float]] = None
    dimension_columns: List[str] = field(default_factory=list)
    spec_filters: Dict[str, Any] = field(default_factory=dict)
    spec_dimensions: List[str] = field(default_factory=list)
    spec_measures: List[Dict[str, Any]] = field(default_factory=list)
    dropped_filters: List[str] = field(default_factory=list)
    dropped_dimensions: List[str] = field(default_factory=list)
    validation_ok: Optional[bool] = None
    #: The response shows only part of the breakdown, by the question's own
    #: request (a top-N or a limit) or by the chart collapsing a tail.
    truncated: bool = False
    answer: str = ""
    data_refusal: bool = False


def _table_artifact(envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    artifacts = envelope.get("artifacts") or []
    for artifact in artifacts:
        if artifact.get("type") == "table" and isinstance(artifact.get("rows"), list):
            return artifact
    for artifact in artifacts:
        if isinstance(artifact.get("rows"), list):
            return artifact
    return None


def _dimension_columns(envelope: Dict[str, Any],
                       rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The columns that identify a GROUP, not a measure of one.

    Taken from the spec's own dimension list where the response publishes one,
    because that is the product's statement of what it grouped by; falling back
    to the table's declared text columns, which is how a response that carries a
    table but no dimension list still yields a group key.
    """
    spec = envelope.get("spec") or {}
    declared = [str(d) for d in (spec.get("dimensions") or []) if d]
    if not declared and spec.get("dimension"):
        declared = [str(spec["dimension"])]
    present = [d for d in declared if any(d in row for row in rows)]
    if present:
        return present
    table = _table_artifact(envelope) or {}
    return [str(column.get("key")) for column in (table.get("columns") or [])
            if column.get("format") == "text" and column.get("key")]


def _fingerprint(envelope: Dict[str, Any], dimension_columns: Sequence[str]
                 ) -> Dict[str, float]:
    """Every numeric quantity the response publishes, keyed by its field.

    A KPI answer publishes them as artefact KPIs; a grouped answer publishes
    them per row, and the fingerprint holds their column totals. Ratio columns
    are excluded — adding percentages across groups is arithmetic that means
    nothing. Two responses are compared on the keys they SHARE, so a grouped
    shape and a KPI shape can still be checked against each other on the
    quantities both of them state.
    """
    out: Dict[str, float] = {}
    for artifact in (envelope.get("artifacts") or []):
        for kpi in (artifact.get("kpis") or []) + (artifact.get("items") or []):
            name = kpi.get("field") or kpi.get("label")
            value = kpi.get("rawValue", kpi.get("value"))
            if not name:
                continue
            try:
                out[str(name)] = float(str(value).replace(",", "").replace("£", ""))
            except (TypeError, ValueError):
                continue
    table = _table_artifact(envelope)
    if table:
        skip = set(dimension_columns) | _RATIO_COLUMNS
        totals: Dict[str, float] = {}
        for row in table.get("rows") or []:
            for key, value in row.items():
                if key in skip or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)
        for key, value in totals.items():
            out.setdefault(key, value)
    return out


def _cells(envelope: Dict[str, Any], dimension_columns: Sequence[str]
           ) -> Optional[Dict[frozenset, Tuple[Tuple[str, float], ...]]]:
    """Group cells keyed so that DIMENSION ORDER cannot change the key.

    ``by region and LTV band`` and ``by LTV band and region`` are the same
    question about the same book; keying a cell by the unordered set of its
    (column, value) pairs is what lets the two be compared without asserting
    which one the product should put on which axis.

    A cell's VALUE is every non-ratio measure the row publishes, not one
    designated column. Reading a single `valueKey` looked right and was not:
    the table artefact does not publish one, so a two-measure breakdown fell
    through to "no comparable cells" and five paraphrase groups were reported
    NOT ESTABLISHED when the evidence to decide them was in the rows all along.
    Comparing the whole row also makes the check stricter — a breakdown that
    agrees on balance and disagrees on loan count is now caught.
    """
    table = _table_artifact(envelope)
    if not table or not dimension_columns:
        return None
    rows = table.get("rows") or []
    if not rows:
        return None
    skip = set(dimension_columns) | _RATIO_COLUMNS
    measures = sorted(key for key, value in rows[0].items()
                      if key not in skip and isinstance(value, (int, float))
                      and not isinstance(value, bool))
    if not measures:
        return None
    cells: Dict[frozenset, Tuple[Tuple[str, float], ...]] = {}
    for row in rows:
        if not all(isinstance(row.get(m), (int, float)) for m in measures):
            return None
        key = frozenset((column, str(row.get(column))) for column in dimension_columns)
        cells[key] = tuple((m, round(float(row[m]), 2)) for m in measures)
    return cells


def _population_from_answer(text: str) -> Optional[int]:
    match = re.search(r"([\d,]+)\s+loans?\b", text or "")
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def read_evidence(question: str, envelope: Dict[str, Any],
                  seconds: float = 0.0) -> Evidence:
    ev = Evidence(question=question, envelope=envelope, seconds=seconds)
    ev.transport_error = bool(envelope.get("__transport_error__"))
    ev.http_status = envelope.get("__http_status__")
    ev.ok = bool(envelope.get("ok"))
    ev.answer = str(envelope.get("answer") or envelope.get("error") or "")
    ev.data_refusal = any(m in ev.answer.lower() for m in _DATA_MARKERS)
    if ev.transport_error:
        return ev

    spec = envelope.get("spec") or {}
    ev.spec_filters = dict(spec.get("filters") or {})
    ev.spec_dimensions = [str(d) for d in (spec.get("dimensions") or []) if d]
    ev.spec_measures = list(spec.get("measures") or [])

    summary = envelope.get("executionSummary") or {}
    recon = envelope.get("reconciliation") or {}
    ev.aggregation = summary.get("aggregation") or spec.get("aggregation")
    ev.group_count = summary.get("groupCount")
    for candidate in (summary.get("population"), recon.get("records_included"),
                      recon.get("records_after_filters")):
        if isinstance(candidate, int):
            ev.population = candidate
            break
    for candidate in (summary.get("populationTotal"), recon.get("total_records")):
        if isinstance(candidate, int):
            ev.population_total = candidate
            break

    table = _table_artifact(envelope)
    rows = (table or {}).get("rows") or []
    ev.dimension_columns = _dimension_columns(envelope, rows)
    ev.fingerprint = _fingerprint(envelope, ev.dimension_columns)
    ev.cells = _cells(envelope, ev.dimension_columns)
    if ev.group_count is None and rows and ev.dimension_columns:
        ev.group_count = len(rows)
    if ev.population is None:
        loans = ev.fingerprint.get("loan_count")
        ev.population = int(loans) if isinstance(loans, float) else None
    if ev.population is None:
        ev.population = _population_from_answer(ev.answer)

    ev.truncated = bool(spec.get("top_n") or spec.get("limit")
                        or spec.get("ranking_mode")
                        or any(a.get("otherCategories")
                               for a in (envelope.get("artifacts") or [])))

    filt = envelope.get("filterInvariant") or {}
    ev.dropped_filters = ([str(x) for x in (filt.get("rejected_filters") or [])]
                          + [str(x) for x in (filt.get("dropped") or [])])
    dim = envelope.get("dimensionInvariant") or {}
    ev.dropped_dimensions = ([str(x) for x in (dim.get("rejected") or [])]
                             + [str(x) for x in (dim.get("dropped") or [])])
    validation = envelope.get("validation")
    if isinstance(validation, dict):
        ev.validation_ok = bool(validation.get("ok"))
    return ev


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
OK = "ok"
FAIL = "FAIL"
DATA = "data"
NOT_ESTABLISHED = "n/e"
SKIP = "skip"


@dataclass
class Result:
    case_id: str
    cls: str
    status: str
    detail: str
    #: A failure reached while the service was answering CONFIDENTLY. This is
    #: the class the whole programme exists to prevent, and it is tracked
    #: separately from an honest refusal that happens to disappoint a case.
    silent_wrong: bool = False


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, 1e-6 * max(abs(a), abs(b)))


# --------------------------------------------------------------------------- #
# Coherence: what every answered response must satisfy, with no expectation
# --------------------------------------------------------------------------- #
def coherence(ev: Evidence) -> List[str]:
    """Contradictions inside ONE confident answer.

    This is what makes the holdout section meaningful. There is no oracle for a
    question nobody has scored before, but a response can still convict itself:
    a population larger than the book, a filter the parse claimed and the
    execution dropped, a breakdown whose rows do not add up to the population it
    says it covered. Each of those is a wrong answer delivered with ``ok: true``.
    """
    problems: List[str] = []
    if not ev.ok:
        return problems
    if (ev.population is not None and ev.population_total is not None
            and ev.population > ev.population_total):
        problems.append(f"population {ev.population} exceeds book "
                        f"{ev.population_total}")
    if ev.dropped_filters:
        problems.append("answered while dropping filters "
                        f"{ev.dropped_filters}")
    if ev.dropped_dimensions:
        problems.append("answered while dropping dimensions "
                        f"{ev.dropped_dimensions}")
    if ev.validation_ok is False:
        problems.append("answered while validation reports not ok")
    # A one-dimensional additive breakdown publishes a per-group loan count.
    # Those counts are a partition of the population the same response claims —
    # UNLESS the response deliberately shows only part of it. "Show me the top 3
    # regions" answers with three groups over a 36-loan population and is
    # perfectly honest about it; scoring that as an incoherent answer would have
    # reported a defect in a question the product gets right.
    if (ev.ok and len(ev.dimension_columns) == 1 and not ev.truncated
            and ev.aggregation in _ADDITIVE and ev.population is not None):
        counted = ev.fingerprint.get("loan_count")
        if counted is not None and not _close(counted, float(ev.population)):
            problems.append(f"group loan counts total {counted:.0f} but the "
                            f"response claims a population of {ev.population}")
    return problems


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class Session:
    """One live request per DISTINCT question, timed, reused across cases.

    The suite asks the same question from several cases on purpose — a
    paraphrase group and a subset pair share endpoints — and asking it once
    keeps the run honest in two ways: the relation is compared against the same
    observation both cases saw, and the latency figures count real distinct
    requests instead of inflating themselves with repeats.
    """

    def __init__(self, ask: Callable[[str], Dict[str, Any]],
                 progress: bool = False) -> None:
        self._ask = ask
        self._seen: Dict[str, Evidence] = {}
        self.timings: List[Tuple[str, float]] = []
        self.transport_failures: List[Tuple[str, Optional[int]]] = []
        self.statuses: List[int] = []
        # A live sweep runs under a CI time limit. If it is killed, the report
        # is never written — so the per-request line goes to stderr as it
        # happens, and a run that dies still leaves behind the timings that say
        # why. That is the difference between "the job timed out" and "the job
        # timed out at request 96, which took 41 seconds".
        self._progress = progress

    def get(self, question: str) -> Evidence:
        if question in self._seen:
            return self._seen[question]
        started = time.perf_counter()
        envelope = self._ask(question)
        elapsed = time.perf_counter() - started
        ev = read_evidence(question, envelope, elapsed)
        self.timings.append((question, elapsed))
        if self._progress:
            import sys as _sys
            print(f"[{len(self.timings):3d}] {elapsed:6.2f}s  "
                  f"{'ok ' if ev.ok else ('ERR' if ev.transport_error else 'ref')}  "
                  f"{question[:60]}", file=_sys.stderr, flush=True)
        if ev.transport_error:
            self.transport_failures.append((question, ev.http_status))
            if isinstance(ev.http_status, int):
                self.statuses.append(ev.http_status)
        self._seen[question] = ev
        return ev

    @property
    def requests(self) -> int:
        return len(self.timings)


def load_cases(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads(Path(path or CASES_PATH).read_text(encoding="utf-8"))


def _transport(case_id: str, cls: str, evs: Sequence[Evidence]) -> Optional[Result]:
    for ev in evs:
        if ev.transport_error:
            return Result(case_id, cls, FAIL,
                          f"{ev.question!r} [transport] "
                          f"{ev.answer[:70]} status={ev.http_status}")
    return None


def _coherence_result(case_id: str, cls: str, evs: Sequence[Evidence]
                      ) -> Optional[Result]:
    for ev in evs:
        problems = coherence(ev)
        if problems:
            return Result(case_id, cls, FAIL,
                          f"{ev.question!r} incoherent: {'; '.join(problems)}",
                          silent_wrong=True)
    return None


# ---- singles -------------------------------------------------------------- #
def score_single(session: Session, case: Dict[str, Any]) -> Result:
    cid, cls, question = case["id"], case["cls"], case["q"]
    ev = session.get(question)
    bad = _transport(cid, cls, [ev])
    if bad:
        return bad
    expect = case["expect"]

    if expect == "MUST_REFUSE":
        if ev.ok:
            return Result(cid, cls, FAIL,
                          f"{question!r} ANSWERED — an ungoverned term was bound "
                          f"to a population: {ev.answer[:80]}", silent_wrong=True)
        return Result(cid, cls, OK, f"{question!r} refused")

    bad = _coherence_result(cid, cls, [ev])
    if bad:
        return bad
    if ev.ok:
        return Result(cid, cls, OK, f"{question!r} answered coherently")
    if expect == "MUST_ANSWER":
        reason = refusal_class(ev.envelope)
        if reason == DATA_UNAVAILABLE:
            return Result(cid, cls, DATA,
                          f"{question!r} {reason}: {ev.answer[:70]}")
        if reason == CAPABILITY_UNAVAILABLE:
            return Result(cid, cls, DATA,
                          f"{question!r} {reason}: {ev.answer[:70]}")
        return Result(cid, cls, FAIL,
                      f"{question!r} refused as {reason}: {ev.answer[:80]}")
    return Result(cid, cls, DATA,
                  f"{question!r} {refusal_class(ev.envelope)} "
                  f"(needs {case.get('needs', 'a field')})")


# ---- relations ------------------------------------------------------------ #
def _compare_fingerprints(a: Evidence, b: Evidence) -> Optional[str]:
    shared = set(a.fingerprint) & set(b.fingerprint)
    if not shared:
        return None                      # nothing shared -> NOT ESTABLISHED
    for key in sorted(shared):
        if not _close(a.fingerprint[key], b.fingerprint[key]):
            return (f"{key}: {a.question!r}={a.fingerprint[key]:.2f} vs "
                    f"{b.question!r}={b.fingerprint[key]:.2f}")
    return ""


def score_equivalence(session: Session, case: Dict[str, Any]) -> Result:
    cid, cls = case["id"], case["cls"]
    evs = [session.get(q) for q in case["questions"]]
    bad = _transport(cid, cls, evs)
    if bad:
        return bad
    bad = _coherence_result(cid, cls, evs)
    if bad:
        return bad

    answered = [e for e in evs if e.ok]
    if not answered:
        return Result(cid, cls, DATA,
                      f"all {len(evs)} wordings refuse (needs "
                      f"{case.get('needs', 'a field')})")
    if len(answered) != len(evs):
        refused = [e.question for e in evs if not e.ok]
        return Result(cid, cls, FAIL,
                      "paraphrases disagree about whether the question can be "
                      f"answered; refused: {refused}", silent_wrong=True)

    first = answered[0]
    for other in answered[1:]:
        if first.population != other.population:
            return Result(cid, cls, FAIL,
                          f"{first.question!r} covers {first.population} loans, "
                          f"{other.question!r} covers {other.population}",
                          silent_wrong=True)
        if case.get("population_only"):
            continue
        mismatch = _compare_fingerprints(first, other)
        if mismatch is None:
            return Result(cid, cls, NOT_ESTABLISHED,
                          f"populations agree ({first.population}) but the two "
                          "responses publish no shared measure to compare")
        if mismatch:
            return Result(cid, cls, FAIL, mismatch, silent_wrong=True)
        if case["relation"] == "SAME_RESULT_AS":
            if first.cells is None or other.cells is None:
                return Result(cid, cls, NOT_ESTABLISHED,
                              f"{first.question!r} agrees on population and "
                              "measures, but the responses expose no comparable "
                              "group cells")
            if first.cells != other.cells:
                return Result(cid, cls, FAIL,
                              f"{first.question!r} and {other.question!r} agree "
                              "on totals but not per group", silent_wrong=True)
    label = "same population" if case.get("population_only") else "agree"
    return Result(cid, cls, OK,
                  f"{len(answered)} wordings {label} ({first.population} loans)")


def score_subset(session: Session, case: Dict[str, Any]) -> Result:
    cid, cls = case["id"], case["cls"]
    whole, part = session.get(case["whole"]), session.get(case["part"])
    bad = _transport(cid, cls, [whole, part])
    if bad:
        return bad
    bad = _coherence_result(cid, cls, [whole, part])
    if bad:
        return bad
    if not part.ok:
        return Result(cid, cls, DATA,
                      f"{case['part']!r} refused (needs "
                      f"{case.get('needs', 'a field')})")
    if not whole.ok:
        return Result(cid, cls, FAIL,
                      f"{case['part']!r} was answered but the population it "
                      f"narrows, {case['whole']!r}, was not", silent_wrong=True)
    if whole.population is None or part.population is None:
        return Result(cid, cls, NOT_ESTABLISHED,
                      "neither response publishes a population size")
    if part.population > whole.population:
        return Result(cid, cls, FAIL,
                      f"{case['part']!r}={part.population} exceeds "
                      f"{case['whole']!r}={whole.population}", silent_wrong=True)
    if whole.aggregation in _ADDITIVE and part.aggregation in _ADDITIVE:
        for key in sorted(set(whole.fingerprint) & set(part.fingerprint)):
            if key in _RATIO_COLUMNS:
                continue
            if part.fingerprint[key] > whole.fingerprint[key] + 0.01:
                return Result(cid, cls, FAIL,
                              f"{key}: narrowed {part.fingerprint[key]:.2f} "
                              f"exceeds whole {whole.fingerprint[key]:.2f}",
                              silent_wrong=True)
    return Result(cid, cls, OK,
                  f"{part.population} <= {whole.population}")


def score_algebra(session: Session, case: Dict[str, Any]) -> Result:
    cid, cls, relation = case["id"], case["cls"], case["relation"]
    if relation == "SUBSET_OF":
        return score_subset(session, case)
    if relation == "SUBSET_OF_BOTH":
        results = []
        for atom in case["atoms"]:
            results.append(score_subset(session, {
                "id": cid, "cls": cls, "whole": atom, "part": case["part"],
                "needs": case.get("needs")}))
        for r in results:
            if r.status == FAIL:
                return r
        if all(r.status == DATA for r in results):
            return Result(cid, cls, DATA, f"{case['part']!r} refused")
        if any(r.status == NOT_ESTABLISHED for r in results):
            return Result(cid, cls, NOT_ESTABLISHED,
                          "population sizes not published for both atoms")
        return Result(cid, cls, OK,
                      f"{case['part']!r} sits inside both atoms")
    if relation in ("ATOMIC_EQUIVALENCE", "SAME_RESULT_AS", "EQUAL_TO"):
        return score_equivalence(session, dict(case, relation="SAME_RESULT_AS"
                                 if relation != "ATOMIC_EQUIVALENCE" else "EQUAL_TO"))
    if relation == "SAME_CELLS_AS":
        evs = [session.get(q) for q in case["questions"]]
        bad = _transport(cid, cls, evs) or _coherence_result(cid, cls, evs)
        if bad:
            return bad
        if not all(e.ok for e in evs):
            if any(e.ok for e in evs):
                return Result(cid, cls, FAIL,
                              "one dimension ordering answers and the other "
                              "refuses", silent_wrong=True)
            return Result(cid, cls, DATA,
                          f"both orderings refuse (needs "
                          f"{case.get('needs', 'a field')})")
        first, other = evs[0], evs[1]
        if first.cells is None or other.cells is None:
            return Result(cid, cls, NOT_ESTABLISHED,
                          "responses expose no comparable group cells")
        if first.cells != other.cells:
            return Result(cid, cls, FAIL,
                          "reordering the grouping dimensions changed the cells",
                          silent_wrong=True)
        return Result(cid, cls, OK,
                      f"{len(first.cells)} cells identical under either ordering")
    if relation == "SAME_POPULATION_AS":
        evs = [session.get(q) for q in case["questions"]]
        bad = _transport(cid, cls, evs) or _coherence_result(cid, cls, evs)
        if bad:
            return bad
        if not all(e.ok for e in evs):
            return Result(cid, cls, DATA, "not all wordings answered")
        if any(e.population is None for e in evs):
            return Result(cid, cls, NOT_ESTABLISHED, "population not published")
        if len({e.population for e in evs}) != 1:
            return Result(cid, cls, FAIL,
                          "grouping changed the population: "
                          + ", ".join(f"{e.question!r}={e.population}" for e in evs),
                          silent_wrong=True)
        return Result(cid, cls, OK,
                      f"population unchanged at {evs[0].population}")
    return Result(cid, cls, NOT_ESTABLISHED, f"unknown relation {relation}")


# ---- outputs -------------------------------------------------------------- #
def _output_tokens(ev: Evidence) -> List[str]:
    """What the response actually PUBLISHES as an output, lower-cased."""
    tokens: List[str] = []
    for measure in ev.spec_measures:
        if isinstance(measure, dict):
            tokens.append(str(measure.get("field") or "").lower())
        else:
            tokens.append(str(measure).lower())
    tokens.extend(k.lower() for k in ev.fingerprint)
    summary = ev.envelope.get("executionSummary") or {}
    if summary.get("measure"):
        tokens.append(str(summary["measure"]).lower())
    tokens.append(str((ev.envelope.get("spec") or {}).get("metric") or "").lower())
    return [t for t in tokens if t]


#: What a requested output is called in a question, and the substrings that
#: identify it in the response. Deliberately a small governed map rather than a
#: fuzzy match: a check that can convince itself an output is present is not a
#: check.
_OUTPUT_SYNONYMS = {
    "balance": ("balance",),
    "loan count": ("loan_count", "count", "loans"),
    "average loan size": ("avg", "average", "mean"),
    "LTV": ("ltv",),
}


def score_outputs(session: Session, case: Dict[str, Any]) -> Result:
    cid, cls, question = case["id"], case["cls"], case["q"]
    ev = session.get(question)
    bad = _transport(cid, cls, [ev]) or _coherence_result(cid, cls, [ev])
    if bad:
        return bad
    if not ev.ok:
        return Result(cid, cls, DATA,
                      f"{question!r} refused (needs {case.get('needs', 'a field')})")
    tokens = _output_tokens(ev)
    missing = []
    for wanted in case["expect_outputs"]:
        needles = _OUTPUT_SYNONYMS.get(wanted, (wanted.lower(),))
        if not any(any(n in token for token in tokens) for n in needles):
            missing.append(wanted)
    if missing:
        return Result(cid, cls, FAIL,
                      f"{question!r} answered but published no "
                      f"{', '.join(missing)}", silent_wrong=True)
    return Result(cid, cls, OK,
                  f"{question!r} published all "
                  f"{len(case['expect_outputs'])} requested outputs")


def score_output_local(session: Session, case: Dict[str, Any]) -> Result:
    """A narrowing attached to one output must not cost the breakdown.

    Two failure modes, and they are opposite: the narrowing is silently ignored
    (the narrowed answer covers the same population as the broad one), or the
    narrowing eats the grouping (the answer comes back as a single figure with
    the breakdown gone). Both arrive as ``ok: true``.
    """
    cid, cls = case["id"], case["cls"]
    broad, narrow = session.get(case["broad"]), session.get(case["narrowed"])
    bad = _transport(cid, cls, [broad, narrow]) or \
        _coherence_result(cid, cls, [broad, narrow])
    if bad:
        return bad
    if not narrow.ok:
        return Result(cid, cls, DATA,
                      f"{case['narrowed']!r} refused (needs "
                      f"{case.get('needs', 'a field')})")
    if not broad.ok:
        return Result(cid, cls, FAIL,
                      f"{case['narrowed']!r} answered but the un-narrowed "
                      f"{case['broad']!r} did not", silent_wrong=True)
    if not narrow.spec_dimensions:
        return Result(cid, cls, FAIL,
                      f"{case['narrowed']!r} lost the breakdown: the narrowing "
                      "consumed the grouping", silent_wrong=True)
    if broad.population is None or narrow.population is None:
        return Result(cid, cls, NOT_ESTABLISHED, "population not published")
    if narrow.population > broad.population:
        return Result(cid, cls, FAIL,
                      f"narrowed population {narrow.population} exceeds "
                      f"{broad.population}", silent_wrong=True)
    if narrow.population == broad.population and not narrow.spec_filters:
        return Result(cid, cls, FAIL,
                      f"{case['narrowed']!r} kept every loan and recorded no "
                      "filter — the narrowing was dropped", silent_wrong=True)
    return Result(cid, cls, OK,
                  f"breakdown kept ({narrow.group_count} groups) over "
                  f"{narrow.population} of {broad.population} loans")


# ---- numeric identities --------------------------------------------------- #
def score_reconciliation(session: Session, case: Dict[str, Any]) -> Result:
    """The parts of a partition sum to the whole — both sides from THIS run.

    No figure is remembered and no independent engine is consulted: the check is
    that the product's own grouped answer and its own ungrouped answer describe
    the same book. A grouping that drops rows, double counts, or quietly filters
    breaks this identity and nothing else in the suite would see it.
    """
    cid, cls = case["id"], case["cls"]
    total, grouped = session.get(case["total"]), session.get(case["grouped"])
    bad = _transport(cid, cls, [total, grouped]) or \
        _coherence_result(cid, cls, [total, grouped])
    if bad:
        return bad
    if not grouped.ok:
        return Result(cid, cls, DATA,
                      f"{case['grouped']!r} refused (needs "
                      f"{case.get('needs', 'a field')})")
    if not total.ok:
        return Result(cid, cls, FAIL,
                      f"{case['grouped']!r} answered but {case['total']!r} did not",
                      silent_wrong=True)
    if grouped.aggregation not in _ADDITIVE:
        return Result(cid, cls, NOT_ESTABLISHED,
                      f"{grouped.aggregation!r} does not reconcile by addition")
    mismatch = _compare_fingerprints(total, grouped)
    if mismatch is None:
        return Result(cid, cls, NOT_ESTABLISHED,
                      "the grouped and ungrouped responses publish no shared "
                      "quantity to reconcile")
    if mismatch:
        return Result(cid, cls, FAIL,
                      f"grouped parts do not sum to the whole — {mismatch}",
                      silent_wrong=True)
    return Result(cid, cls, OK,
                  f"{grouped.group_count} groups sum to {case['total']!r}")


def score_arithmetic(session: Session, case: Dict[str, Any]) -> Result:
    cid, cls, identity = case["id"], case["cls"], case["identity"]
    if identity.startswith("mean"):
        mean = session.get(case["mean"])
        count = session.get(case["count"])
        total = session.get(case["total"])
        bad = _transport(cid, cls, [mean, count, total]) or \
            _coherence_result(cid, cls, [mean, count, total])
        if bad:
            return bad
        if not all(e.ok for e in (mean, count, total)):
            return Result(cid, cls, DATA, "not all three questions answered")
        avg = next((v for k, v in mean.fingerprint.items()
                    if k.endswith("_avg") or k.endswith("_mean")), None)
        if avg is None:
            avg = next((v for k, v in mean.fingerprint.items()
                        if k != "loan_count"), None)
        sums = next((v for k, v in total.fingerprint.items()
                     if k.endswith("_sum")), None)
        if avg is None or sums is None or count.population is None:
            return Result(cid, cls, NOT_ESTABLISHED,
                          "the three responses do not publish a mean, a total "
                          "and a population to multiply")
        product = avg * count.population
        if abs(product - sums) > max(1.0, 0.005 * abs(sums)):
            return Result(cid, cls, FAIL,
                          f"mean x population = {product:.2f} but the total is "
                          f"{sums:.2f}", silent_wrong=True)
        return Result(cid, cls, OK,
                      f"mean x {count.population} reconciles to the total")
    if identity.startswith("group shares"):
        grouped = session.get(case["grouped"])
        bad = _transport(cid, cls, [grouped]) or _coherence_result(cid, cls, [grouped])
        if bad:
            return bad
        if not grouped.ok:
            return Result(cid, cls, DATA, f"{case['grouped']!r} refused")
        table = _table_artifact(grouped.envelope) or {}
        shares = [row.get("concentration_pct") for row in (table.get("rows") or [])]
        shares = [s for s in shares if isinstance(s, (int, float))]
        if not shares:
            return Result(cid, cls, NOT_ESTABLISHED,
                          "the response publishes no per-group share")
        # `otherCategories` means the chart collapsed a tail; the shares then
        # legitimately fall short of 100 and the check cannot be decided.
        if grouped.truncated:
            return Result(cid, cls, NOT_ESTABLISHED,
                          "the response collapsed a tail into 'other'")
        if abs(sum(shares) - 100.0) > 0.5:
            return Result(cid, cls, FAIL,
                          f"per-group shares sum to {sum(shares):.2f}, not 100",
                          silent_wrong=True)
        return Result(cid, cls, OK,
                      f"{len(shares)} group shares sum to 100")
    return Result(cid, cls, NOT_ESTABLISHED, f"unknown identity {identity}")


# ---- the sweep ------------------------------------------------------------ #
_SECTIONS = (
    ("singles", score_single),
    ("outputs", score_outputs),
    ("output_local", score_output_local),
    ("equivalence_groups", score_equivalence),
    ("subset_pairs", score_subset),
    ("algebra", score_algebra),
    ("reconciliations", score_reconciliation),
    ("arithmetic", score_arithmetic),
)


def run_broad(ask: Callable[[str], Dict[str, Any]],
              cases: Optional[Dict[str, Any]] = None,
              include_holdout: bool = True,
              progress: bool = False
              ) -> Tuple[Session, List[Result], List[Result]]:
    cases = cases or load_cases()
    session = Session(ask, progress=progress)
    results: List[Result] = []
    for key, scorer in _SECTIONS:
        for case in cases.get(key, []):
            results.append(scorer(session, case))
    holdout: List[Result] = []
    if include_holdout:
        for case in cases.get("holdout", []):
            holdout.append(score_single(session, case))
    return session, results, holdout


def latency(session: Session) -> Dict[str, float]:
    times = sorted(t for _, t in session.timings)
    if not times:
        return {}

    def pick(fraction: float) -> float:
        index = min(len(times) - 1, int(round(fraction * (len(times) - 1))))
        return times[index]

    return {"requests": float(len(times)), "total_s": float(sum(times)),
            "p50_s": pick(0.50), "p95_s": pick(0.95), "max_s": times[-1]}


# --------------------------------------------------------------------------- #
# Why a refusal happened — four classes, kept apart
# --------------------------------------------------------------------------- #
#: The certification report must not describe an UNSUPPORTED CAPABILITY as
#: missing data, and it must not describe a SEMANTIC refusal as either. They are
#: three different conversations with an operator: one is "load the column", one
#: is "this product cannot do that yet", and one is "the question named something
#: no governed vocabulary carries" — which is the only one that is a safety
#: success rather than a limitation.
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
SEMANTIC_UNRESOLVED = "SEMANTIC_UNRESOLVED"
GOVERNED_REFUSAL = "GOVERNED_REFUSAL"

#: The estate's own sentences for "the reader named something no governed
#: vocabulary claims". The first is written by
#: `llm_query_parser.unknown_category_refusal`; the second is the measure and
#: dimension owners saying the same thing about a different role. Matched on the
#: wording those owners author, so this reader and those writers stay one fact.
_SEMANTIC_MARKERS = (
    "no loans in this book match that filter",
    "is not a governed measure",
    "is not a governed dimension",
    "is not a governed statistic",
)

#: The estate's own wording for "this book does not report that field". Checked
#: BEFORE the capability markers, because a field-unavailability refusal also
#: says the concept "could not be applied to the calculation" — and the clause
#: that names the missing field is the one that tells an operator what to do.
_DATA_CLASS_MARKERS = (
    "not available in this dataset",
    "unavailable in this dataset",
    "is not in this dataset",
    "does not report it",
    "field is unavailable",
    "no reporting periods are available",
    "no governed pipeline data is available",
    "no governed pipeline source is available",
)

#: The estate's own wording for "understood, governed, and this product cannot
#: do it". THE DISTINCTION THIS EXISTS FOR: "Loan count and balance by region
#: and LTV band" refuses while both `Loan count by LTV band` and `Total balance
#: by region and LTV band` answer, and the reason is that multi-measure and
#: two-dimensional grouping are not supported TOGETHER. Reporting that as
#: missing data would send an operator to load a column that is already there.
_CAPABILITY_MARKERS = (
    "could not be applied to the calculation",
    "was not applied",
    "not supported",
    "cannot compute",
    "can't compute",
    "cannot be answered from",
    "no capability",
)


def refusal_class(envelope: Dict[str, Any]) -> Optional[str]:
    """Which of the four a refusal is, or None when the response answered.

    Read from the response's own words and its own `metadata` flags, in the
    order that keeps the classes disjoint and keeps the ADVICE right:

      1. SEMANTIC first — the estate has one authored sentence for an unresolved
         term, and a book that also lacks a column would otherwise hide it;
      2. DATA next — a refusal naming a missing FIELD is a data limit even
         though it also says the concept could not be applied;
      3. CAPABILITY last — understood, governed, and not supported.

    The order is the whole content of the function. Getting it wrong does not
    produce a wrong count, it produces wrong advice: "load the column" for a
    feature that does not exist, or "build the feature" for a column nobody
    mapped.
    """
    if envelope.get("ok"):
        return None
    text = str(envelope.get("answer") or envelope.get("error") or "").lower()
    for marker in _SEMANTIC_MARKERS:
        if marker in text:
            return SEMANTIC_UNRESOLVED
    for marker in _DATA_CLASS_MARKERS:
        if marker in text:
            return DATA_UNAVAILABLE
    if (envelope.get("metadata") or {}).get("controlledUnsupported"):
        return CAPABILITY_UNAVAILABLE
    for marker in _CAPABILITY_MARKERS:
        if marker in text:
            return CAPABILITY_UNAVAILABLE
    return GOVERNED_REFUSAL
