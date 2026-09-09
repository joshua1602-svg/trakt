"""THE SEMANTIC CLAIMS LEDGER — what a question was understood to ask, once.

Built ONCE per request, after the deterministic parse and the model merge and
before any capability executes, by ASKING the owners of each semantic fact:

  * the temporal-aspect owner (`question_interpretation.lexical`) — level or
    movement;
  * the period owners (the contract's `time` claim, filled from
    `period_change.recognition` and `period_request`) — which period(s);
  * the lending-window owner (`mi_agent.seasoning`) — which origination window;
  * the status owner (`mi_agent.states.models`) — whether a pipeline status
    was named;
  * the stage-movement owner (`mi_agent_api.stage_movement_query`);
  * the analytical-intent boundary (`mi_workflows.analytical.intent`) — the
    non-temporal families and what they structurally require.

Every guard downstream compares what was CLAIMED here with what was EXECUTED,
and refuses on the difference. No guard reads the sentence again: a second
raw reader is how a level question about "the LTV the portfolio is running
at" came to be refused as a trend while a question about "new loans" was
answered from the whole book. The ledger is carried on the parse metadata
(`semanticClaims`) and on every route request, so the receipt can show it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

KIND_PERIOD = "period"
KIND_POPULATION = "population"
KIND_DATASET = "dataset"
KIND_STAGE_MOVEMENT = "stage_movement"
KIND_FAMILY = "analytical_family"

#: Requirements this ledger OWNS (beside the intent boundary's own constants).
REQ_PERIOD_COMPARISON = "period_comparison"
REQ_PERIOD_AS_AT = "period_as_at"
REQ_STAGE_MOVEMENT = "stage_movement"
REQ_POPULATION_PREFIX = "population:"
REQ_DATASET_PREFIX = "dataset:"

REASONS: Dict[str, str] = {
    REQ_PERIOD_COMPARISON: ("it asks for a change between two governed reporting "
                            "periods, and this figure was calculated from one"),
    REQ_PERIOD_AS_AT: ("it asks about an earlier reporting period, and this figure "
                       "was calculated as at the current one"),
    REQ_STAGE_MOVEMENT: ("it asks about movement between pipeline stages, which "
                         "a single snapshot cannot show"),
}


@dataclass(frozen=True)
class Claim:
    kind: str
    owner: str
    value: str
    text: str = ""
    requirement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "owner": self.owner, "value": self.value,
                "text": self.text, "requirement": self.requirement}


@dataclass
class SemanticClaims:
    question: str
    temporal_aspect: str
    temporal_evidence: Tuple[str, ...] = ()
    claims: Tuple[Claim, ...] = ()
    requirements: Tuple[str, ...] = ()
    #: The intent boundary's own reading, as it publishes it.
    intent: Dict[str, Any] = field(default_factory=dict)
    materially_analytical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question,
                "temporal_aspect": self.temporal_aspect,
                "temporal_evidence": list(self.temporal_evidence),
                "claims": [c.to_dict() for c in self.claims],
                "requirements": list(self.requirements),
                "intent": dict(self.intent),
                "materially_analytical": self.materially_analytical}

    def claims_for(self, requirement: str) -> List[Claim]:
        return [c for c in self.claims if c.requirement == requirement]


def from_dict(data: Mapping[str, Any]) -> SemanticClaims:
    return SemanticClaims(
        question=str(data.get("question") or ""),
        temporal_aspect=str(data.get("temporal_aspect") or "level"),
        temporal_evidence=tuple(data.get("temporal_evidence") or ()),
        claims=tuple(Claim(str(c.get("kind")), str(c.get("owner")),
                           str(c.get("value")), str(c.get("text") or ""),
                           c.get("requirement"))
                     for c in (data.get("claims") or ())),
        requirements=tuple(data.get("requirements") or ()),
        intent=dict(data.get("intent") or {}),
        materially_analytical=bool(data.get("materially_analytical")))


def as_claims(value: Any) -> Optional[SemanticClaims]:
    if isinstance(value, SemanticClaims):
        return value
    if isinstance(value, Mapping):
        return from_dict(value)
    return None


# --------------------------------------------------------------------------- #
# Building — every fact from its owner
# --------------------------------------------------------------------------- #
def _period_claims(question: str, interpretation: Any, spec: Any,
                   aspect: str) -> Tuple[List[Claim], List[str]]:
    from mi_agent.period_change import recognition as R

    claims: List[Claim] = []
    reqs: List[str] = []
    time = getattr(interpretation, "time", None)
    if time is not None:
        mode = getattr(time, "relative_mode", None)
        pair = tuple(getattr(time, "comparison_periods", None) or ())
        window = getattr(time, "window_periods", None)
    else:
        mode = R.relative_mode(question)
        pair = tuple(str(p) for p in (getattr(spec, "compare_periods", None) or []))
        window = None
    reference = R.period_reference(question)
    if pair:
        claims.append(Claim(KIND_PERIOD, "parser.compare_periods", ", ".join(pair),
                            text="the periods " + " and ".join(pair)))
    if mode:
        wording = R.relative_wording(question)
        claims.append(Claim(KIND_PERIOD, "period_change.recognition", str(mode),
                            text=("the current reporting period" if reference == "current"
                                  else (f"the {wording}" if wording
                                        else f"a {str(mode).replace('_', '-')} period")),
                            requirement=(None if reference == "current" or aspect == "movement"
                                         else REQ_PERIOD_AS_AT)))
    if window:
        claims.append(Claim(KIND_PERIOD, "period_request.requested_span",
                            f"window:{int(window)}",
                            text=f"a window of {int(window)} reporting period(s)"))

    if aspect == "movement":
        reqs.append(REQ_PERIOD_COMPARISON)
    elif len(pair) >= 2:
        reqs.append(REQ_PERIOD_COMPARISON)
    elif mode and reference == "other":
        # A LEVEL at an earlier period: "the balance in the prior period".
        reqs.append(REQ_PERIOD_AS_AT)
    return claims, reqs


def _population_claims(question: str) -> Tuple[List[Claim], List[str]]:
    from mi_agent import seasoning

    claims: List[Claim] = []
    reqs: List[str] = []
    for key in seasoning.lending_windows_named(question):
        req = REQ_POPULATION_PREFIX + str(key)
        claims.append(Claim(KIND_POPULATION, "seasoning.lending_windows_named",
                            str(key), text=f"{str(key).replace('_', ' ')} lending",
                            requirement=req))
        reqs.append(req)
    return claims, reqs


def _dataset_claims(question: str) -> Tuple[List[Claim], List[str]]:
    from question_interpretation.normalise import normalise_question

    from mi_agent.states import models as states

    claims: List[Claim] = []
    reqs: List[str] = []
    seen = set()
    for word in normalise_question(question).replace("?", " ").split():
        word = word.strip(".,;:!'\"()")
        if word in seen or not states.status_word(word):
            continue
        seen.add(word)
        funded = states.classify_funded_value(word)
        if funded is False:
            req = REQ_DATASET_PREFIX + "pipeline"
            claims.append(Claim(KIND_DATASET, "states.models", "pipeline",
                                text=f"{word} (a pipeline status)", requirement=req))
            if req not in reqs:
                reqs.append(req)
        elif funded is True:
            claims.append(Claim(KIND_DATASET, "states.models", "funded",
                                text=f"{word} (the funded book)"))
    return claims, reqs


def _stage_movement_claims(question: str) -> Tuple[List[Claim], List[str]]:
    try:
        from mi_agent_api import stage_movement_query as SMQ

        reading = SMQ.read(question)
    except Exception:  # noqa: BLE001 - the owner absent claims nothing
        reading = None
    if reading is None:
        return [], []
    subtype = str(getattr(reading, "subtype", "") or "stage movement")
    return ([Claim(KIND_STAGE_MOVEMENT, "stage_movement_query.read", subtype,
                   text=f"{subtype.replace('_', ' ')} between pipeline stages",
                   requirement=REQ_STAGE_MOVEMENT)],
            [REQ_STAGE_MOVEMENT])


def _intent_claims(question: str, available_values: Any
                   ) -> Tuple[List[Claim], List[str], Dict[str, Any], bool]:
    from mi_workflows.analytical import intent as I

    masked = question
    if available_values:
        try:
            from mi_agent.categorical_spans import mask_value_spans

            masked = mask_value_spans(question, available_values)
        except Exception:  # noqa: BLE001 - no catalogue, the sentence as written
            masked = question
    reading = I.classify(masked)
    block = reading.to_dict()
    claims = [Claim(KIND_FAMILY, "analytical.intent", str(f),
                    text=str(f).replace("_", " ").lower())
              for f in reading.families]
    # THE PERIOD REQUIREMENT IS THE TEMPORAL OWNERS'. The boundary asserts
    # `period_comparison` on a run-rate signal, and "the LTV the portfolio is
    # running at" is a level. Its other requirements are its own.
    reqs = [r for r in reading.requirements if r != I.REQ_PERIOD_COMPARISON]
    return claims, reqs, block, bool(reading.materially_analytical)


def build(question: str, *, interpretation: Any = None, spec: Any = None,
          parse_meta: Any = None, available_values: Any = None) -> SemanticClaims:
    """The ledger for ``question``, from its owners. Never raises."""
    from question_interpretation.lexical import temporal_aspect

    aspect = temporal_aspect(question)
    claims: List[Claim] = []
    requirements: List[str] = []

    def _take(pair):
        cs, rs = pair
        claims.extend(cs)
        for r in rs:
            if r not in requirements:
                requirements.append(r)

    _take(_period_claims(question, interpretation, spec, aspect.verdict))
    _take(_population_claims(question))
    _take(_dataset_claims(question))
    _take(_stage_movement_claims(question))
    intent_block: Dict[str, Any] = {}
    material = False
    try:
        cs, rs, intent_block, material = _intent_claims(question, available_values)
        _take((cs, rs))
    except Exception:  # noqa: BLE001 - the boundary absent claims nothing
        pass
    return SemanticClaims(question=question, temporal_aspect=aspect.verdict,
                          temporal_evidence=tuple(aspect.evidence),
                          claims=tuple(claims), requirements=tuple(requirements),
                          intent=intent_block, materially_analytical=material)


# --------------------------------------------------------------------------- #
# Comparing CLAIMED with EXECUTED
# --------------------------------------------------------------------------- #
def unmet(claims: SemanticClaims, evidence: Mapping[str, Any]) -> List[str]:
    """The claimed requirements ``evidence`` does not demonstrate.

    ``evidence`` describes what was ACTUALLY executed: ``dataset``, ``periods``
    (governed snapshots compared), ``as_at`` ("current" or a period label),
    ``executed_fields`` (row-predicate fields applied), ``stage_movement``,
    ``forecast``, ``limits``, ``grouping``, ``populations``.
    """
    from mi_workflows.analytical import intent as I

    out: List[str] = []
    periods = int(evidence.get("periods") or 0)
    dataset = str(evidence.get("dataset") or "")
    executed = {str(f) for f in (evidence.get("executed_fields") or ())}
    for req in claims.requirements:
        if req == REQ_PERIOD_COMPARISON:
            if periods < 2:
                out.append(req)
        elif req == REQ_PERIOD_AS_AT:
            if str(evidence.get("as_at") or "current") == "current":
                out.append(req)
        elif req == REQ_STAGE_MOVEMENT:
            if not evidence.get("stage_movement"):
                out.append(req)
        elif req.startswith(REQ_POPULATION_PREFIX):
            if not _window_executed(req[len(REQ_POPULATION_PREFIX):], executed):
                out.append(req)
        elif req.startswith(REQ_DATASET_PREFIX):
            if dataset != req[len(REQ_DATASET_PREFIX):]:
                out.append(req)
        else:
            # The intent boundary's own requirements, checked by its own
            # checker — one rule for "is this structurally met".
            if claims.materially_analytical and I.unmet_from_requirements(
                    (req,), evidence=evidence):
                out.append(req)
    return out


def _window_executed(key: str, executed_fields: Iterable[str]) -> bool:
    from mi_agent import seasoning

    try:
        window = seasoning.load_seasoning_config().lending_window(key)
    except Exception:  # noqa: BLE001 - no config, nothing demonstrated
        return False
    if window is None:
        return False
    wanted = set((window.predicate() or {}).keys())
    return bool(wanted) and wanted <= set(executed_fields)


def refusal_message(claims: SemanticClaims, unmet_requirements: Sequence[str]) -> str:
    from mi_workflows.analytical import intent as I

    understood: List[str] = []
    reasons: List[str] = []
    for req in unmet_requirements:
        for c in claims.claims_for(req):
            if c.text and c.text not in understood:
                understood.append(c.text)
        if req in REASONS:
            reasons.append(REASONS[req])
        elif req.startswith(REQ_POPULATION_PREFIX):
            reasons.append(f"it narrows to {req[len(REQ_POPULATION_PREFIX):].replace('_', ' ')} "
                           "lending, and no such narrowing was applied")
        elif req.startswith(REQ_DATASET_PREFIX):
            reasons.append(f"it names the {req[len(REQ_DATASET_PREFIX):]} dataset, "
                           "and this figure was calculated from a different one")
        elif req in I.REQUIREMENT_REASONS:
            reasons.append(I.REQUIREMENT_REASONS[req])
    if not understood:
        understood = [c.text for c in claims.claims
                      if c.kind in (KIND_PERIOD, KIND_FAMILY) and c.text][:3]
    what = ", ".join(understood) if understood else "this question"
    body = "; and ".join(reasons) if reasons else (
        "no governed analytic could be established for it")
    return (f"I understood that you asked for {what}, but I have not answered it: "
            f"{body}. I have NOT substituted a current-position figure, because "
            "that would answer a different question from the one you asked.")


__all__ = ["SemanticClaims", "Claim", "build", "from_dict", "as_claims", "unmet",
           "refusal_message", "REQ_PERIOD_COMPARISON", "REQ_PERIOD_AS_AT",
           "REQ_STAGE_MOVEMENT", "REQ_POPULATION_PREFIX", "REQ_DATASET_PREFIX"]
