#!/usr/bin/env python3
"""question_interpretation/completeness.py — is every stated concept carried?

THE DEFECT THIS EXISTS TO CATCH
-------------------------------
A question states some governed concepts. The estate decides a contract. The
answer is delivered. Nothing in that chain asks the one question a reader would
ask: *did everything I said survive into what you computed?*

Measured over the 75-question acceptance bank, 21 of the 25 failures were one
cause — a concept the sentence plainly states that never reaches the contract.
Six of those 21 answered CONFIDENTLY on a broader population and disclosed
nothing: "How many drawdown loans have LTV above 50%?" applied the LTV
threshold, dropped `drawdown`, and returned a figure over the whole book.

WHAT THIS IS NOT
----------------
It is NOT a new reader of the question. Every reading below is a call to an
owner that already ships:

    requested_dimension_terms       the axes the sentence named
    categorical_spans.value_field   the book's own value catalogue
    portfolio_lens                  the scope, after the value owner has
                                    claimed its spans (`Gamma Direct` is a
                                    broker, and reading `direct` out of it
                                    raised a scope the reader never named)
    workspace.resolve_dataset       funded or pipeline
    llm_query_parser._detect_metric the measure
    llm_query_parser._forecast_target_value
                                    the milestone, consulted ONLY where
                                    `answer_type.asked` says the question asks
                                    for a DATE — `£300,000` in "how many loans
                                    are above £300,000" is a threshold, and the
                                    forecast owner never claimed it
    detect_requested_facets         thresholds, lost narrowings, geography,
                                    comparison periods, projections, rankings,
                                    groupings

and it is NOT an oracle. It does not know what the right answer is. It compares
two records that already exist.

THE ONE RULE
------------
    A concept the sentence states is CARRIED only when the executed contract
    POSITIVELY RECORDS it. Silence is a finding, not a pass.

That rule is uniform, and it is what makes the check honest rather than tuned.
It is also what found the defect it was calibrated against. Five questions
across the standing banks answered CORRECTLY and recorded nothing —
`period_movement` and `concentration_analysis` narrowed to a book, `evolution`
read the pipeline, and none of them published a population, a narrowing or a
facet to say so. Their envelopes were indistinguishable from Q19C's, where the
same scope was DROPPED and the answer was wrong by £10.2m.

Reading them as false positives and tuning them away would have suppressed
Q19C with them. They were fixed instead: the two scope routes now publish
`metadata.scopeApplied`, the dataset axis is read from `reconciliation.dataset`
(what the answer was reconciled against) rather than `metadata.datasetContext`
(what the request decided), and the lens that named a book and narrowed nothing
now raises rather than returning the frame. Across the 75-question bank, the
frozen CFO 91 and the two composition banks, the check now fires on no
delivering question at all.

WHAT IT COMPARES IS PRESENCE, NOT CORRECTNESS. Q21C bound the function word
`among` as a categorical value; every concept the sentence states is still in
the contract, so this is silent. A concept carried into the WRONG governed
field is invisible here by construction, and reporting disagreement between a
proposed concept and the field the registry binds it to is what covers that
case.

WHAT IT CANNOT SEE
------------------
ITS RECALL IS THE OWNERS' RECALL. "Break Direct-book balance down by both
broker channel and loan type" loses `Direct-book` and this check is silent,
because `portfolio_lens` does not read the hyphenated form outside a selector
position — `For Direct-book` and `of Direct-book balance` carry a selector mark
and `Break Direct-book balance` does not — and no other owner claims it either.
A concept NO OWNER RESOLVES cannot be seen lost by any deterministic detector.

That is a stated limit, not a tuning shortfall, and it is the argument for a
proposal step upstream: a reader that PROPOSES concepts does not need the
grammar to reach them, and this check then has something to compare.

ROLE DISAGREEMENT IS NOT LOSS
-----------------------------
"How many owner occupied loans do we have?" makes the axis owner ask for
`occupancy_type` as a GROUPING and the contract binds it as a FILTER. The
concept reached the contract; only its role differs. That is a different
finding with a different fix, and conflating the two cost five false positives
on the composition banks before the distinction was drawn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as _field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["StatedConcept", "ExecutedContract", "stated_concepts",
           "unresolved_concepts", "from_envelope", "SCOPE_FIELDS"]

#: The governed fields that carry a SOURCE PORTFOLIO. A value of one of these
#: is carried by the lens rather than by a row filter, and testing it as a row
#: filter is what made every scoped question look incomplete.
SCOPE_FIELDS: Tuple[str, ...] = ("source_portfolio_id", "source_portfolio_type")

#: Facet kinds that name a concept THE SENTENCE STATED. `granularity` and
#: `requested_statistic` are shape rather than a stated narrowing: the receipt
#: raises them for its own reasons and they are not the reader's words.
_STATED_FACET_KINDS: Tuple[str, ...] = (
    "threshold", "lost_narrowing", "geographic_scope", "comparison_period",
    "projection", "ranking", "row_population", "share", "grouping_dimension")

_GROUPING = "grouping_dimension"
_NARROWING = ("lost_narrowing", "geographic_scope")

_WORD_RE = re.compile(r"[A-Za-z0-9_%£\.\-]+")


@dataclass(frozen=True)
class StatedConcept:
    """One governed concept an existing owner resolved out of the question."""

    kind: str
    field: str
    value: str
    term: str
    owner: str
    #: Other governed fields that CARRY this concept when the contract holds
    #: them — the registry's own `derived_from` relation, resolved where the
    #: semantics are in hand so `_carried` never needs them. See ROLE
    #: DISAGREEMENT IS NOT LOSS in the module docstring.
    carried_by: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "field": self.field, "value": self.value,
                "term": self.term, "owner": self.owner}

    def describe(self) -> str:
        what = self.term or self.value or self.field
        return "%s (%s) — resolved by %s" % (what, self.kind, self.owner)


@dataclass(frozen=True)
class ExecutedContract:
    """What the estate RECORDS having executed.

    Deliberately a record of the estate's own statements, not of the data. A
    field absent here is absent from the record — which is the finding, whether
    or not the answer happened to be right.
    """

    filters: Tuple[str, ...] = ()
    dimensions: Tuple[str, ...] = ()
    metric: Optional[str] = None
    forecast_target: Optional[float] = None
    scope_context: Optional[str] = None
    dataset_context: Optional[str] = None
    dataset_reconciled: Optional[str] = None
    route: Optional[str] = None
    narrowed: bool = False
    population_total: Optional[int] = None
    population_applied: bool = False
    scope_applied: bool = False
    applied_fields: Tuple[str, ...] = ()
    facets: Tuple[Tuple[str, str, str], ...] = ()   # (kind, label, status)

    @property
    def scoped(self) -> bool:
        # `scope_applied` counts, and it is the STRONGEST of the three: it is a
        # route's own record of a narrowing it PERFORMED, where `scope_context`
        # is only the scope the request resolved. Requiring the weaker signal as
        # a precondition for reading the stronger one made a route that applied
        # a lens and said so — but published no `portfolioScope` — read as
        # unscoped, which is the opposite of what the evidence says.
        return (self.scope_applied
                or self.scope_context not in (None, "total")
                or any(f in self.filters for f in SCOPE_FIELDS))

    @property
    def narrowing_recorded(self) -> bool:
        """Did the estate record that it narrowed ANYTHING?

        `applied_fields` is deliberately NOT consulted: it carries the field of
        every applied facet, GROUPING AXES INCLUDED, and an axis is not a
        narrowing. Reading it here let "show product concentration for the
        direct book" count its `erm_product_type` axis as evidence that the
        Direct scope had been applied, when the route recorded no narrowing at
        all.
        """
        return bool(self.narrowed or self.population_applied or self.filters)

    def facet_applied(self, kind: str, label: str) -> bool:
        """Did ANYTHING apply this facet?

        ANY, not the first match. The estate publishes its own record — a
        refused answer publishes `("threshold", "LTV over 55", "lost")` — and a
        merge that later satisfies that threshold appends its own. Returning on
        the FIRST match made the second record unreachable, so a threshold the
        merge had filled with exactly the right bound still read as lost, and
        reach on threshold losses stayed pinned at zero after the rule that was
        supposed to unpin it. Two records of one facet is not a contradiction to
        resolve by position; the question is whether any of them applied it.
        """
        return any(k == kind and lbl == label and status == "applied"
                   for k, lbl, status in self.facets)


def from_envelope(envelope: Dict[str, Any]) -> ExecutedContract:
    """THE ONE ADAPTER from an answer envelope. Nothing else reads its shape."""
    spec = envelope.get("spec") or {}
    meta = envelope.get("metadata") or {}
    ex = envelope.get("executionSummary") or {}
    scope = envelope.get("portfolioScope") or {}
    pop = meta.get("populationApplied") or {}
    scope_ledger = meta.get("scopeApplied") or {}

    dims = list(spec.get("dimensions") or [])
    if spec.get("dimension"):
        dims.append(spec["dimension"])

    facets = tuple((f.get("kind") or "", f.get("label") or "", f.get("status") or "")
                   for f in (ex.get("facets") or []))
    applied = tuple(sorted({f.get("field") for f in (ex.get("facets") or [])
                            if f.get("status") == "applied" and f.get("field")}))
    return ExecutedContract(
        filters=tuple(spec.get("filters") or ()),
        dimensions=tuple(dims),
        metric=spec.get("metric"),
        forecast_target=spec.get("forecast_target_value"),
        scope_context=scope.get("context_id"),
        dataset_context=meta.get("datasetContext"),
        dataset_reconciled=(envelope.get("reconciliation") or {}).get("dataset"),
        route=meta.get("route"),
        narrowed=bool(ex.get("narrowed")),
        population_total=ex.get("populationTotal"),
        population_applied=bool(pop.get("applied") or meta.get("applied_filter_fields")),
        scope_applied=bool(scope_ledger),
        applied_fields=applied,
        facets=facets,
    )


def stated_concepts(question: str, semantics: Dict[str, Any], *,
                    available_values: Any = None,
                    available_columns: Optional[Iterable[str]] = None,
                    frame: Any = None,
                    default_dataset: str = "funded") -> List[StatedConcept]:
    """Every governed concept an EXISTING owner resolves out of ``question``.

    Owner precedence is the estate's, not this module's: the axis owner claims
    its terms before the value owner reads them as a filter, and the value owner
    masks its spans before the scope owner reads them as a book.
    """
    from mi_agent import answer_type as AT
    from mi_agent import categorical_spans as CS
    from mi_agent import execution_receipt as R
    from mi_agent import llm_query_parser as LQ
    from mi_agent import portfolio_lens as PL

    q = question or ""
    out: List[StatedConcept] = []

    # ---- the AXIS owner, first, so its terms are not re-read as values ---- #
    dim_terms = R.requested_dimension_terms(q, semantics, available_columns) or []
    axis_words = set()
    for term in dim_terms:
        key, matched = term[0], str(term[1])
        alts = tuple(term[2]) if len(term) > 2 else ()
        axis_words.update(re.findall(r"[a-z0-9]+", matched.lower()))
        out.append(StatedConcept("dimension", key, "|".join((key,) + alts),
                                 matched, "requested_dimension_terms",
                                 carried_by=_derived_sources(key, semantics)))

    # ---- the SCOPE owner, on a question the value owner has claimed ------- #
    scope_q = PL.mask_claimed_value_spans(q, available_values)
    comparison = PL.resolve_comparison_lenses(scope_q) or [] \
        if PL.is_comparison(scope_q) else []
    lens = PL.resolve_lens(scope_q)
    lens_name = getattr(lens, "name", None) if lens is not None else None
    if lens_name not in (None, "total"):
        out.append(StatedConcept("scope", "portfolio_lens", lens_name, "",
                                 "portfolio_lens.resolve_lens"))
    if len(comparison) > 1:
        out.append(StatedConcept(
            "scope_comparison", "portfolio_lens",
            ",".join(sorted(getattr(l, "name", "?") for l in comparison)), "",
            "portfolio_lens.resolve_comparison_lenses"))

    # ---- the VALUE owner: longest phrase first, owners consulted first ---- #
    words = _WORD_RE.findall(q)
    taken = [False] * len(words)
    for size in range(4, 0, -1):
        for i in range(0, len(words) - size + 1):
            if any(taken[i:i + size]):
                continue
            phrase = " ".join(words[i:i + size])
            hit = CS.value_field(phrase, available_values)
            if not hit:
                continue
            if all(w.lower() in axis_words for w in words[i:i + size]):
                continue        # the axis owner already claimed these words
            if len(comparison) > 1 and hit[0] in SCOPE_FIELDS:
                continue        # the comparison owner already claimed this book
            for j in range(i, i + size):
                taken[j] = True
            out.append(StatedConcept("value", hit[0], hit[1], phrase,
                                     "categorical_spans.value_field"))

    # ---- the DATASET owner ------------------------------------------------ #
    try:
        from mi_agent_api import workspace as W
        view = W.resolve_dataset(q)
        default_dataset = getattr(W, "DEFAULT_VIEW", default_dataset)
    except Exception:  # noqa: BLE001 - no owner reachable, no claim
        view = None
    if view and view != default_dataset:
        out.append(StatedConcept("dataset", "view", view, "",
                                 "workspace.resolve_dataset"))

    # ---- the MEASURE owner ------------------------------------------------ #
    metric_key, _agg, matched_terms = LQ._detect_metric(q, semantics)
    if metric_key:
        out.append(StatedConcept("measure", metric_key, metric_key,
                                 ",".join(matched_terms),
                                 "llm_query_parser._detect_metric",
                                 carried_by=_bands_of(metric_key, semantics)))

    # ---- the TARGET owner, gated by the ANSWER-TYPE owner ----------------- #
    if AT.asked(q) == AT.DATE:
        target = LQ._forecast_target_value(q.lower())
        if target is not None:
            out.append(StatedConcept("target", "forecast_target_value",
                                     str(target), "",
                                     "llm_query_parser._forecast_target_value"))

    # ---- the FACET owner -------------------------------------------------- #
    for facet in R.detect_requested_facets(q, semantics, frame=frame,
                                           requested_dimensions=dim_terms):
        if facet.kind in _STATED_FACET_KINDS:
            out.append(StatedConcept("facet:" + facet.kind, facet.field_key or "",
                                     facet.label or "", facet.label or "",
                                     "detect_requested_facets",
                                     carried_by=_derived_sources(
                                         facet.field_key or "", semantics)))
    return out


def _derived_sources(field: str, semantics: Dict[str, Any]) -> Tuple[str, ...]:
    """The field this band is DERIVED FROM, if the registry declares one.

    The inverse of `_bands_of`, from the same declaration. A question naming a
    band and a contract narrowing its source field are the same concept.
    """
    entry = ((semantics or {}).get("fields") or {}).get(field) or {}
    src = entry.get("derived_from")
    return (str(src),) if src else ()


def _bands_of(field: str, semantics: Dict[str, Any]) -> Tuple[str, ...]:
    """Every governed dimension the registry declares DERIVED FROM ``field``.

    `age_bucket` is `derived_from: youngest_borrower_age`; `ltv_bucket` is
    `derived_from: current_loan_to_value`. Nineteen such fields are declared, so
    this is read rather than listed, and a band added tomorrow needs no edit
    here. Used by `_carried` to decide that a question stating a field and an
    answer banding it are talking about the same concept.
    """
    fields = (semantics or {}).get("fields") or {}
    return tuple(sorted(
        k for k, e in fields.items() if (e or {}).get("derived_from") == field))


def _carried(concept: StatedConcept, contract: ExecutedContract) -> bool:
    kind, field, value = concept.kind, concept.field, concept.value
    filters, dims = set(contract.filters), set(contract.dimensions)
    applied = set(contract.applied_fields)

    if kind == "facet:" + _GROUPING:
        # ROLE DISAGREEMENT IS NOT LOSS. See the module docstring. The band
        # relation applies here for the same reason it applies to the dimension
        # concept this facet twins: a contract narrowing the field a band is
        # derived from carries that band's concept.
        return (field in dims or field in applied or field in filters
                or contract.facet_applied(_GROUPING, value)
                or bool(set(concept.carried_by) & (dims | filters | applied)))
    if kind in ("facet:" + k for k in _NARROWING):
        return field in filters or field in dims or field in applied
    if kind.startswith("facet:"):
        return contract.facet_applied(kind.split(":", 1)[1], value)
    if kind == "value":
        if field in filters or field in dims or field in applied:
            return True
        return field in SCOPE_FIELDS and contract.scoped
    if kind == "scope":
        # SCOPED IS NOT APPLIED. `portfolioScope` reports the scope the request
        # RESOLVED; Q19C published `direct` and answered the whole book. Only an
        # executed narrowing counts — `metadata.scopeApplied` is the route's own
        # record of the one it performed.
        return contract.scoped and (contract.scope_applied
                                    or any(f in filters for f in SCOPE_FIELDS)
                                    or contract.narrowing_recorded)
    if kind == "scope_comparison":
        return bool(contract.route) or contract.scoped
    if kind == "dataset":
        # `datasetContext` is the DECISION; `reconciliation.dataset` is what the
        # answer was RECONCILED AGAINST, which is the estate's own record of
        # what it read. "Summarise the current pipeline" publishes
        # `datasetContext: pipeline` beside `reconciliation.dataset: funded` —
        # the contradiction is already in the envelope, and reading the decision
        # alone would have called that carried.
        return contract.dataset_reconciled == value
    if kind == "dimension":
        want = set(value.split("|"))
        if bool(want & dims) or bool(want & applied) or bool(want & filters):
            return True
        # THE BAND RELATION, READ THE OTHER WAY. `_carried` already accepts a
        # contract that BANDS a stated field ("balance by borrower age bucket"
        # states `youngest_borrower_age`, the contract groups `age_bucket`).
        # The inverse is the same relation and the same governed fact: "for
        # tickets larger than £150k" states the band `ticket_bucket`, and the
        # contract narrows `current_outstanding_balance`, which is precisely
        # what the registry declares `ticket_bucket` is `derived_from`. The
        # concept reached the contract, as a row predicate on its own source
        # field. Read from the registry, so a band added tomorrow is covered.
        return bool(set(concept.carried_by) & (dims | filters | applied))
    if kind == "measure":
        if contract.metric == field or field in filters or field in applied:
            return True
        # A BAND OF A FIELD IS THAT FIELD. "Show balance by borrower age bucket"
        # states `youngest_borrower_age`; the contract groups by `age_bucket`,
        # which the registry declares `derived_from: youngest_borrower_age`. The
        # concept reached the contract — banded, and as an axis rather than a
        # measure — so this is ROLE DISAGREEMENT, not loss, and the module
        # docstring already says role disagreement is not loss.
        #
        # Measured before this existed: fourteen delivering questions across the
        # corpora were reported incomplete for a concept they had plainly
        # answered by, including "Show balance by interest rate bucket" and
        # "balance by ltv band". Every one is a bucket naming its own source
        # field. The relation is read from `derived_from`, which the registry
        # already declares for all nineteen derived fields, so there is no list
        # here and a bucket added tomorrow is covered without an edit.
        if set(concept.carried_by) & (dims | filters | applied):
            return True
        # A ROUTE THAT SUPPLIES ITS OWN MEASURE is not a loss. A parser that
        # bound a DIFFERENT measure is — which is how Q21B's mis-binding of
        # `balance` to `current_loan_to_value` shows up as an incompleteness.
        return contract.metric is None and bool(contract.route)
    if kind == "target":
        return contract.forecast_target is not None or bool(contract.route)
    return True


# --------------------------------------------------------------------------- #
# The coverage ledger
# --------------------------------------------------------------------------- #
#: The concept reached the contract and execution records it.
RESOLVED = "resolved"
#: The ESTATE ITSELF declined the concept — a facet it stamped unsupported or
#: unavailable. Kept distinct from `UNACCOUNTED` on purpose: "we cannot do this
#: and said so" and "this vanished without trace" are different events, and
#: collapsing them would let a governed refusal read as a silent loss.
UNSUPPORTED_BY_ESTATE = "unsupported"
#: Stated by an owner, and nothing in the executed contract records it. THE
#: FINDING. An answer carrying one of these widened, dropped or substituted
#: something the reader asked for, and nothing downstream can see it because
#: no claim was ever made to lose.
UNACCOUNTED = "unaccounted"

#: Facet statuses that mean the estate GAVE AN ANSWER ABOUT the concept rather
#: than losing it. Read from `mi_agent.execution_receipt`'s own vocabulary.
_DECLINED_STATUSES = ("unsupported", "unavailable")

#: There is deliberately no NON_SEMANTIC state. Ordinary words never enter this
#: ledger at all: it is built from what governed OWNERS name, not from
#: tokenising the sentence, so a word no owner claims is absent rather than
#: dispositioned. That is why the measured false-positive rate on correct
#: answers is zero rather than tuned.


def coverage_report(question: str, envelope: Dict[str, Any],
                    semantics: Dict[str, Any], *,
                    available_values: Any = None,
                    available_columns: Optional[Iterable[str]] = None,
                    frame: Any = None) -> Dict[str, Any]:
    """Every governed concept the question states, and how it was accounted for.

    THE ONE OWNER of "all material user meaning has been accounted for". Routes
    publish what they executed; this decides whether that covers what was asked.
    The decision is not distributed.

    Independent BY CONSTRUCTION, which is the whole reason this exists:
    `stated_concepts` receives the question and the governed registry/value
    catalogue and never the spec, the contract or this envelope. So a concept
    the interpretation dropped is still named here — which is exactly the case
    no existing control can see, because every one of them compares two records
    that both exist and a dropped concept leaves none.

    CONCEPT-FOUNDED, NEVER SPAN-FOUNDED. `question_interpretation.schema.Slot`
    records that 170 of 690 measured claims carry no recoverable span and that a
    consumer "must never require it". Spans ride along for diagnostics where an
    owner supplied one; coverage never depends on them.

    Returns the ledger. Decides nothing about the answer: the caller does that.
    """
    concepts = stated_concepts(question, semantics,
                               available_values=available_values,
                               available_columns=available_columns, frame=frame)
    contract = from_envelope(envelope)
    declined = {label for kind, label, status in contract.facets
                if status in _DECLINED_STATUSES}
    entries: List[Dict[str, Any]] = []
    for c in concepts:
        if _carried(c, contract):
            disposition = RESOLVED
        elif c.term and c.term in declined:
            disposition = UNSUPPORTED_BY_ESTATE
        else:
            disposition = UNACCOUNTED
        entries.append({"kind": c.kind, "field": c.field, "value": str(c.value),
                        "term": c.term, "owner": c.owner,
                        "disposition": disposition})
    return {"version": 1, "concepts": entries,
            "unaccounted": [e for e in entries if e["disposition"] == UNACCOUNTED]}


def unaccounted_concepts(ledger: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ledger's unaccounted entries, or ``[]``. The gate reads only this."""
    if not isinstance(ledger, dict):
        return []
    return list(ledger.get("unaccounted") or ())


def unresolved_concepts(concepts: Sequence[StatedConcept],
                        contract: ExecutedContract) -> List[StatedConcept]:
    """The stated concepts the executed contract does not record carrying."""
    return [c for c in concepts if not _carried(c, contract)]
