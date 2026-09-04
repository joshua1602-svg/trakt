"""mi_workflows.analytical.intent — the governed ANALYTICAL INTENT BOUNDARY.

What this is
------------
A small recognition layer that sits in front of the answer, not inside it. Its
only job is to say three things about a question:

    1. which governed analytical FAMILY (or families) it belongs to;
    2. which governed OPERATION(s) within that family it asks for;
    3. whether it is MATERIALLY ANALYTICAL — i.e. whether a single
       point-in-time aggregate could honestly be the answer.

It builds nothing, calculates nothing and owns no data. Everything it names maps
onto a capability or a route that already exists (see :data:`FAMILIES`), so this
module can never introduce an analytic the platform does not already govern.

Why a family layer at all
-------------------------
Measured across 752 runs of ordinary CFO phrasings, every unsafe outcome came
from the same structural gap: a question that no specialist recogniser claimed
fell through to the generic point-in-time executor, which has no concept of a
run rate, a limit, a pipeline or a forecast, and answered with whatever measure
and dimension the parse happened to produce. *"How many loans are we completing
at the moment?"* returned the total number of loans on the book, with ok=True
and a green guard.

The gap is not that the analytics are missing — they exist and are correct. The
gap is that nothing was asking *"is this question even the kind of thing the
thing about to answer it can answer?"* That is this module.

Concept signals, not question templates
---------------------------------------
The vocabularies below are CONCEPT sets: the words a book, a control or a
movement is described with. They are deliberately not sentence patterns, and no
entry was added because a particular test phrasing needed to pass. The test of
a candidate entry is: *would a credit or finance professional use this word to
name this concept, in a question nobody has written yet?* If not, it does not
belong here.

Two consequences follow, and both are intended. Recognition is broad — many
phrasings reach the same family. Ownership is narrow — a family only ever hands
the question to the governed route or capability that already owns it.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_agent import portfolio_lens as _lens
from mi_agent import seasoning as _seasoning

# --------------------------------------------------------------------------- #
# §1 The six governed analytical families
# --------------------------------------------------------------------------- #
FAMILY_MIX_PROFILE = "MIX_PROFILE"
FAMILY_PIPELINE = "PIPELINE"
FAMILY_LIMITS_CONCENTRATION = "LIMITS_CONCENTRATION"
FAMILY_FORECAST_PROJECTION = "FORECAST_PROJECTION"
FAMILY_MOVEMENT_TREND = "MOVEMENT_TREND"
FAMILY_VINTAGE_COHORT = "VINTAGE_COHORT"

ALL_FAMILIES: Tuple[str, ...] = (
    FAMILY_MIX_PROFILE, FAMILY_PIPELINE, FAMILY_LIMITS_CONCENTRATION,
    FAMILY_FORECAST_PROJECTION, FAMILY_MOVEMENT_TREND, FAMILY_VINTAGE_COHORT,
)

# --------------------------------------------------------------------------- #
# §2 The governed operations, per family
# --------------------------------------------------------------------------- #
# MIX_PROFILE
OP_SNAPSHOT = "SNAPSHOT"
OP_COMPOSITION = "COMPOSITION"
OP_COMPARISON = "COMPARISON"
OP_CHANGE = "CHANGE"
OP_DIVERGENCE = "DIVERGENCE"
OP_ATTRIBUTION = "ATTRIBUTION"
# PIPELINE
OP_STOCK = "STOCK"
OP_MOVEMENT = "MOVEMENT"
OP_CONVERSION = "CONVERSION"
OP_RUN_RATE = "RUN_RATE"
OP_EXPECTED_COMPLETION = "EXPECTED_COMPLETION"
OP_TIMING = "TIMING"
OP_MIX = "MIX"
# LIMITS_CONCENTRATION
OP_CONCENTRATION = "CONCENTRATION"
OP_STATUS = "STATUS"
OP_HEADROOM = "HEADROOM"
OP_RANKING = "RANKING"
OP_FORECAST_BREACH = "FORECAST_BREACH"
# FORECAST_PROJECTION
OP_PROJECT_VALUE = "PROJECT_VALUE"
OP_MILESTONE = "MILESTONE"
OP_HORIZON = "HORIZON"
OP_SCENARIO = "SCENARIO"
# MOVEMENT_TREND
OP_DELTA = "DELTA"
OP_TREND = "TREND"
OP_ACCELERATION = "ACCELERATION"
# VINTAGE_COHORT
OP_EVOLUTION = "EVOLUTION"


# --------------------------------------------------------------------------- #
# §6 Concept signals. Five groups, exactly the five the brief names.
# --------------------------------------------------------------------------- #
SIGNAL_COMPARISON = "comparison"
SIGNAL_CHANGE = "change_trend"
SIGNAL_RUN_RATE = "run_rate"
SIGNAL_LIMITS = "limits"
SIGNAL_FORECAST = "forecast"

#: A question is COMPARATIVE when it sets one thing against another. Includes
#: "relative to", which the shared portfolio-lens vocabulary deliberately does
#: not carry, because widening that vocabulary would change how every other
#: route reads a question.
_COMPARISON_TERMS: Tuple[str, ...] = (
    " vs ", " vs. ", " versus ", " compared with ", " compared to ",
    " comparing ", " relative to ", " against ", " than ", " compare ",
    " comparison ", " differently ", " different ", " differences ",
    " difference ", " differ ", " differs ", " how does ", " how do ",
    " each other ", " one another ", " side by side ",
)

#: A question is about CHANGE when it asks how something moved rather than what
#: it is. "over time" is included because it is the plainest way to say it.
_CHANGE_TERMS: Tuple[str, ...] = (
    " changed ", " change ", " changes ", " changing ", " moved ", " move ",
    " moves ", " moving ", " movement ", " shifted ", " shift ", " shifts ",
    " shifting ", " trend ", " trends ", " trending ", " trajectory ",
    " evolved ", " evolve ", " evolving ", " evolution ", " developed ",
    " develop ", " developing ", " development ", " grown ", " growth ",
    " grew ", " growing ", " declined ", " decline ", " declining ",
    " fallen ", " fell ", " falling ", " risen ", " rose ", " rising ",
    " increased ", " increase ", " increasing ", " decreased ", " decrease ",
    " decreasing ", " progressed ", " progressing ", " progression ",
    " over time ", " month on month ", " year on year ",
    " accelerating ", " accelerated ", " acceleration ", " slowing ",
)

#: A question is about a RATE OR FLOW when it asks how fast something is
#: happening rather than how much of it there is.
#: Kept deliberately tight. " monthly ", " a month " and " level of " were
#: candidates and were REJECTED: "what is the monthly payment?" and "what level
#: of arrears do we have?" are point-in-time questions, and a rate signal there
#: would refuse an answer the product gives correctly today.
_RUN_RATE_TERMS: Tuple[str, ...] = (
    " run rate ", " run-rate ", " runrate ", " per month ", " per week ",
    " per quarter ", " each month ", " every month ", " each week ",
    " pace ", " velocity ", " throughput ", " how fast ", " how quickly ",
    " running at ", " completion rate ", " origination rate ",
    " conversion rate ", " rate we are ", " rate are we ",
)

#: A comparand that names a PAST period. A comparative question whose comparand
#: is a past period is a CHANGE question however it is worded — "compared with a
#: few months ago" and "changed in the last few months" ask the same thing, and
#: only one of them contains a change verb.
_PRIOR_PERIOD_TERMS: Tuple[str, ...] = (
    " months ago ", " a few months ago ", " weeks ago ", " a year ago ",
    " earlier in the year ", " earlier this year ", " earlier in the period ",
    " last month ", " last quarter ", " last year ", " previously ",
    " this time last ", " back then ", " historically ", " used to ",
    " before ", " prior period ", " the prior ", " the previous ",
)

#: A question is about a LIMIT when it names a governed CONTROL. "concentration"
#: alone is deliberately NOT enough: a concentration is a measure ("balance
#: concentration by region") and only becomes a control when a limit,
#: headroom, breach, covenant, tolerance or threshold is named with it.
_LIMIT_TERMS: Tuple[str, ...] = (
    " limit ", " limits ", " headroom ", " breach ", " breaches ",
    " breached ", " breaching ", " covenant ", " covenants ", " tolerance ",
    " tolerances ", " threshold ", " thresholds ", " within limits ",
    " limit test ", " limit tests ", " concentration limit ",
    " concentration limits ",
)

#: A question is a FORECAST when it asks about a state that has not happened.
#: "at risk of BREACHING" is a forward breach, not a current ranking.
#:
#: Scoped to limits questions and to this exact construction, deliberately.
#: `_LIMIT_HEADROOM_TERMS` already contains " at risk ", which is right for
#: "which limits are most at risk?" — a ranking of today's headroom. But
#: "at risk of breaching" names a breach that has NOT happened, which is the same
#: thing "do we expect to breach" and "are any limits projected to breach" name,
#: and those two already carry FORECAST_PROJECTION + FORECAST_BREACH. Measured
#: before this: "which concentration tests are we at risk of breaching?"
#: classified LIMITS_CONCENTRATION only, matching just ('breaching',), and was
#: answered with today's risk-limit status — the CURRENT-STATE SUBSTITUTION the
#: frozen bank recorded against Q25A/B/C.
#:
#: NOT added to `_FORECAST_TERMS`. That vocabulary is read by every family, and
#: a phrase that means "forward" only in front of a breach would widen
#: forecasting across the whole classifier.
_AT_RISK_OF_BREACH_TERMS: Tuple[str, ...] = (
    " at risk of breach ", " at risk of breaching ",
)


_FORECAST_TERMS: Tuple[str, ...] = (
    " forecast ", " forecasts ", " forecast to ", " forecasting ",
    " project ", " projected ", " projection ", " projections ",
    " expect ", " expected ", " expecting ", " likely ", " outlook ",
    " will be ", " will we ", " will our ", " will the ", " going to ",
    " anticipate ", " anticipated ", " when will ", " when do we ",
    " when are we ", " when should ", " how long until ", " by when ",
    " milestone ", " scenario ", " what if ", " should convert ",
    " should we ", " grow to ", " get to ", " reach ", " land ", " lands ",
)


# --------------------------------------------------------------------------- #
# Family vocabularies. Nouns that NAME the family's subject.
# --------------------------------------------------------------------------- #
_PROFILE_TERMS: Tuple[str, ...] = (
    " profile ", " profiles ", " mix ", " composition ", " composed of ",
    " characteristics ", " make-up ", " makeup ", " made up of ",
    " breakdown ", " broken down ", " distribution ", " spread across ",
    " what kind of ", " what kinds of ", " what sort of ", " what sorts of ",
    " what type of ", " what types of ", " types of loans ",
    " kinds of loans ", " shape of the book ", " weighted towards ",
)

#: Risk framing is a PROFILE question about the credit measures the book
#: carries, so it names the same family by a different word.
_RISK_PROFILE_TERMS: Tuple[str, ...] = (
    " risk profile ", " credit profile ", " risk characteristics ",
    " how risky ", " riskiness ", " riskier ", " risky ", " credit quality ",
    " quality of ", " risk perspective ", " from a risk ", " credit risk ",
)

#: PIPELINE names the pre-funding dataset — applications, offers and the
#: transition into funded. Deliberately excludes bare "funded"/"fund", which
#: name the book itself rather than the flow into it.
#:
#: AND IT EXCLUDES A BARE `case`/`cases`, which it used to contain. That word is
#: DATASET-NEUTRAL in this estate: it names a pipeline case and a funded loan
#: about equally often, and the P1C golden bank uses it for the second —
#:
#:     "Which region gained the most cases since last month?"
#:
#: is filed there under `# -- loan count --`, beside "Which region added the
#: most loans month-on-month?", and expects a ranked FUNDED movement. This
#: layer read it as PIPELINE and so asserted `REQ_PIPELINE_DATASET` for it,
#: which `unmet_requirements` then tests against a funded dataset and finds
#: unmet. `workspace.resolve_dataset` — the authoritative dataset owner — does
#: not read the bare word, so the two disagreed about the same sentence.
#:
#: Removing it costs this layer nothing measurable: all 7 corpus questions
#: containing `case`/`cases` ALSO say "pipeline" outright, so every one of them
#: still reaches this family through `" pipeline "` above. `caseload` stays —
#: it is a different word and names the pipeline unambiguously.
#:
#: Explicit pipeline context still makes a case a pipeline case. That is the
#: house rule, and it is the same one the dataset owner applies.
_PIPELINE_TERMS: Tuple[str, ...] = (
    " pipeline ", " pipelines ", " application ", " applications ", " kfi ",
    " kfis ", " offer ", " offers ", " offered ", " at offer ",
    " caseload ", " in flight ", " in-flight ", " underwriting ",
    " decision in principle ", " dip ",
)

#: COMPLETION names the pipeline→funded transition. It is a FLOW concept: a
#: completion is an event, never a stock of the funded book, so the funded
#: point-in-time executor structurally cannot answer a question about one.
_COMPLETION_TERMS: Tuple[str, ...] = (
    " complete ", " completes ", " completing ", " completed ",
    " completion ", " completions ", " convert ", " converts ",
    " converting ", " conversion ", " conversions ", " drawdown ",
    " drawdowns ", " draw down ", " drawn down ", " fund out ", " funding out ",
)

#: Population nouns. A completion term standing immediately before one of these
#: is describing WHICH LOANS, not an event.
#:
#: "drawdown" is a pipeline→funded transition AND, in equity release, a governed
#: LOAN TYPE. Read only as the event, "how many drawdown loans do we have?" —
#: a plain count of 244 loans on the funded book — was declined as a pipeline
#: question the funded executor structurally cannot answer.
#:
#: The rule is generic and names no term: a governed category used as an
#: adjective in front of a population noun is a qualifier. "How many loans are
#: we drawing down at the moment?" has no population noun after the term and
#: stays the flow question it is.
_POPULATION_NOUNS: Tuple[str, ...] = (
    "loan", "loans", "case", "cases", "mortgage", "mortgages",
    "account", "accounts", "book", "balance", "balances",
)

_QUALIFIER_RE = re.compile(
    r"\b(" + "|".join(t.strip() for t in (
        " complete ", " completes ", " completing ", " completed ",
        " completion ", " completions ", " convert ", " converts ",
        " converting ", " conversion ", " conversions ", " drawdown ",
        " drawdowns ", " draw down ", " drawn down ")) + r")\s+(?:"
    + "|".join(_POPULATION_NOUNS) + r")\b", re.I)


#: The same rule in the COPULAR form. A completion term PREDICATED OF a
#: population — "what share of the book IS DRAWDOWN?" — is describing which
#: loans just as surely as the attributive form is, and it carries no population
#: noun after the term for `_QUALIFIER_RE` to anchor on.
#:
#: Still generic and still names no term. The verb must be followed IMMEDIATELY
#: by the term, so "how many loans are we drawing down at the moment?" — verb,
#: then a subject — stays the flow question it is.
_COMPLEMENT_RE = re.compile(
    r"\b(?:" + "|".join(_POPULATION_NOUNS) + r")\b[^?.!]{0,24}?"
    r"\b(?:is|are|was|were)\s+(" + "|".join(t.strip() for t in (
        " complete ", " completes ", " completing ", " completed ",
        " completion ", " completions ", " convert ", " converts ",
        " converting ", " conversion ", " conversions ", " drawdown ",
        " drawdowns ", " draw down ", " drawn down ")) + r")\b", re.I)


def _is_population_qualifier(text: str) -> bool:
    """True when every completion term in the text qualifies a population."""
    hits = [t.strip() for t in _COMPLETION_TERMS if t in text]
    if not hits:
        return False
    qualified = {m.group(1).lower() for m in _QUALIFIER_RE.finditer(text)}
    qualified |= {m.group(1).lower() for m in _COMPLEMENT_RE.finditer(text)}
    return all(h.lower() in qualified for h in hits)


_VINTAGE_TERMS: Tuple[str, ...] = (
    " vintage ", " vintages ", " cohort ", " cohorts ", " seasoning ",
    " seasoned ", " months on book ", " time on book ", " loan age ",
    " by year of origination ", " origination year ", " when they were written ",
)

#: PRESENT-TENSE framing. "How much X is happening now" is a question about the
#: RATE of X, not about a stock of it — which is why "how many loans are we
#: completing at the moment?" is a run-rate question and not a loan count.
_CURRENT_TERMS: Tuple[str, ...] = (
    " currently ", " at the moment ", " right now ", " at present ",
    " these days ", " lately ", " recent ", " recently ", " today ",
    " last few weeks ", " last few months ", " so far ",
)

#: A question that asks for a NUMBER OF THINGS rather than an amount.
_COUNT_TERMS: Tuple[str, ...] = (
    " how many ", " number of ", " count of ", " how many loans ",
    " how many cases ", " how many completions ",
)

#: ORIGINATION FLOW framing — the role signal that makes "lending" a measure of
#: volume rather than a population of loans (§4 of the ruling).
_ORIGINATION_FLOW_TERMS: Tuple[str, ...] = (
    " how much did we originate ", " how much have we originated ",
    " how much are we originating ", " how much lending ", " volume of ",
    " volumes ", " amount of new ", " value of new ", " originated in ",
    " written in ", " advanced in ", " gross lending ", " net lending ",
    " lending volume ", " origination volume ",
)


# --------------------------------------------------------------------------- #
# §4 The governed LENDING role
# --------------------------------------------------------------------------- #
#: "Lending" is a loan / origination domain concept whose ROLE depends on the
#: analytical context it appears in. The ruling is explicit that it must not be
#: globally mapped to one role.
ROLE_POPULATION = "population"
ROLE_FLOW = "origination_flow"


# --------------------------------------------------------------------------- #
# Structural requirements — what an answer must CARRY to be a safe answer
# --------------------------------------------------------------------------- #
REQ_PIPELINE_DATASET = "pipeline_dataset"
REQ_LIMIT_EVIDENCE = "limit_evidence"
REQ_FORECAST = "forecast"
REQ_PERIOD_COMPARISON = "period_comparison"
REQ_POPULATION_COMPARISON = "population_comparison"

#: Plain-English reason per requirement, used in the controlled refusal. Written
#: for a CFO, naming what could not be established rather than what code did.
REQUIREMENT_REASONS: Dict[str, str] = {
    REQ_PIPELINE_DATASET:
        "this asks about the pipeline (applications, offers or completions), "
        "which is a different governed dataset from the funded book",
    REQ_LIMIT_EVIDENCE:
        "this asks about concentration limits, which are governed by the "
        "portfolio's limit schedule rather than by the loan tape",
    REQ_FORECAST:
        "this asks for a forward-looking figure, which needs a governed "
        "forecast rather than a current position",
    REQ_PERIOD_COMPARISON:
        "this asks how something changed, which needs two governed reporting "
        "snapshots to compare",
    REQ_POPULATION_COMPARISON:
        "this compares two populations, which needs both of them measured "
        "separately",
}


# --------------------------------------------------------------------------- #
# §5 Family definitions — operations, and the GOVERNED OWNERS they map onto
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilyDefinition:
    """One governed family: its operations, and who already owns them.

    ``routes`` and ``capabilities`` are declarations, not dispatch. They exist so
    a reader (and a test) can confirm that every operation this layer recognises
    maps onto something the platform already governs — the guarantee that no new
    mathematics enters through this door.
    """

    name: str
    operations: Tuple[str, ...]
    routes: Tuple[str, ...]
    capabilities: Tuple[str, ...]
    note: str


FAMILIES: Dict[str, FamilyDefinition] = {
    FAMILY_MIX_PROFILE: FamilyDefinition(
        name=FAMILY_MIX_PROFILE,
        operations=(OP_SNAPSHOT, OP_COMPOSITION, OP_COMPARISON, OP_CHANGE,
                    OP_DIVERGENCE, OP_ATTRIBUTION),
        routes=("analytical_composition", "period_change_analysis",
                "portfolio_summary", "period_movement"),
        capabilities=("population_profile", "portfolio_snapshot",
                      "period_movement"),
        note="what the book is made of, and how that make-up moved"),
    FAMILY_PIPELINE: FamilyDefinition(
        name=FAMILY_PIPELINE,
        operations=(OP_STOCK, OP_MOVEMENT, OP_CONVERSION, OP_RUN_RATE,
                    OP_EXPECTED_COMPLETION, OP_TIMING, OP_MIX),
        routes=("analytical_composition", "forecast_extrapolation",
                "cohort_conversion", "scenario"),
        capabilities=("pipeline_stock", "pipeline_completion_forecast",
                      "completion_run_rate"),
        note="the pre-funding dataset and the flow out of it into the book"),
    FAMILY_LIMITS_CONCENTRATION: FamilyDefinition(
        name=FAMILY_LIMITS_CONCENTRATION,
        operations=(OP_CONCENTRATION, OP_STATUS, OP_HEADROOM, OP_RANKING,
                    OP_MOVEMENT, OP_FORECAST_BREACH),
        routes=("risk_limits", "analytical_composition"),
        capabilities=("concentration_limits", "threshold_projection"),
        note="governed controls and the distance to them"),
    FAMILY_FORECAST_PROJECTION: FamilyDefinition(
        name=FAMILY_FORECAST_PROJECTION,
        operations=(OP_PROJECT_VALUE, OP_MILESTONE, OP_HORIZON, OP_SCENARIO),
        routes=("forecast_extrapolation", "scenario", "analytical_composition"),
        capabilities=("funded_balance_forecast", "threshold_projection",
                      "completion_run_rate", "pipeline_completion_forecast"),
        note="a value, a date or a condition that has not happened yet"),
    FAMILY_MOVEMENT_TREND: FamilyDefinition(
        name=FAMILY_MOVEMENT_TREND,
        operations=(OP_DELTA, OP_TREND, OP_RANKING, OP_ACCELERATION,
                    OP_ATTRIBUTION),
        routes=("analytical_composition", "period_change_analysis",
                "period_movement", "evolution", "temporal_compare",
                "funded_bridge"),
        capabilities=("period_movement", "portfolio_snapshot"),
        note="how a measure moved across governed reporting snapshots"),
    FAMILY_VINTAGE_COHORT: FamilyDefinition(
        name=FAMILY_VINTAGE_COHORT,
        operations=(OP_SNAPSHOT, OP_COMPARISON, OP_EVOLUTION, OP_RANKING,
                    OP_DIVERGENCE),
        routes=("analytical_composition", "cohort_progression"),
        capabilities=("vintage_analysis", "portfolio_snapshot",
                      "population_profile"),
        note="origination cohorts, and how they differ from one another"),
}


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalise(question: Optional[str]) -> str:
    """Lower-cased, single-spaced, space-padded, punctuation neutralised.

    Identical treatment to the planner's own ``_norm``, for the same reasons:
    padding lets a term be matched as a whole phrase, and neutralising
    punctuation stops a trailing question mark from hiding the last word.
    """
    text = "".join(c if (c.isalnum() or c in "&'-£%") else " "
                   for c in str(question or "").lower())
    return " " + " ".join(text.split()) + " "


def _matches(text: str, term: str) -> bool:
    """Whether ``term`` occurs in ``text`` without the sentence ruling it out.

    B21 — THE THIRD READER of "does this question ask about the pipeline?".
    `resolve_active_view` picks the frame, `_dataset_for` picks the dataset,
    and this picks the STRUCTURAL REQUIREMENT that makes the question refusable
    — so "the balance by seasoning segment excluding pipeline cases" was
    refused as a pipeline question after the first two had correctly sent it to
    the funded book. Each of the three had its own vocabulary; none of them is
    wrong about its own job; all three were reading a mention the sentence had
    ruled out.

    The disclaiming window is `portfolio_lens`'s, not a fourth copy. Measured
    over all fourteen vocabularies here and all 661 corpus questions, exactly
    one question/term pair stops signalling, and it is a constructed case.
    """
    at = text.find(term)
    while at != -1:
        # The vocabularies are space-padded, so the TERM starts one past the hit.
        if not _lens.is_disclaimed_span(text, at + 1):
            return True
        at = text.find(term, at + 1)
    return False


def _any(text: str, terms: Sequence[str]) -> bool:
    return any(_matches(text, term) for term in terms)


def _hits(text: str, terms: Sequence[str]) -> List[str]:
    return [t.strip() for t in terms if _matches(text, t)]


def is_comparative(question: Optional[str]) -> bool:
    """Whether a question sets one thing against another.

    Public because the planner asks it rather than keeping a second comparison
    vocabulary — which is how the same question came to resolve two different
    ways depending on which of the two lists happened to contain its wording.
    """
    text = normalise(question)
    return _any(text, _COMPARISON_TERMS) or names_both_sides_of_a_pair(text)


def names_both_sides_of_a_pair(text: str) -> bool:
    """Whether the sentence names BOTH values of one governed binary dimension.

    NAMING BOTH SIDES *IS* THE COMPARISON. A comparison verb adds nothing to
    "direct and acquired" — there is no other thing two values of one dimension
    coordinated by "and" could be asking for once a measure is attached.

    Requiring a verb made this layer disagree with the guard that reads the same
    sentence. "How have direct and acquired balances moved over the periods?"
    carries no verb in `_COMPARISON_TERMS`, so no comparison signal was raised,
    `_plan_population_movement_comparison` declined, and the question fell
    through to a route returning a whole-book series —
    `execution_receipt.segments_named_in` then read the SAME sentence against
    the SAME governed values, found `direct` and `acquired`, and refused
    "Direct and Acquired tracked separately … could not be applied". True of
    the route, false of the product: the identical request with the word
    "compare" in it composes and answers.

    Deliberately narrow. It asks only about the two GOVERNED BINARY dimensions
    whose vocabularies are owned elsewhere — the portfolio lens (direct /
    acquired) and seasoning (front book / back book). It says nothing about two
    DIMENSIONS coordinated by "and": "balance by month by region and LTV band"
    names no value of either pair, raises no signal here, and keeps its P0
    refusal.
    """
    try:
        from mi_agent import portfolio_lens as _lens
    except Exception:  # noqa: BLE001 - recognition must never break on an import
        return False
    both_provenance = (_lens._contains_any(text, _lens._DIRECT_TERMS)
                       and _lens._contains_any(text, _lens._ACQUIRED_TERMS))
    if both_provenance:
        return True
    return " front book " in text and " back book " in text


# --------------------------------------------------------------------------- #
# The recognised intent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalyticalIntent:
    """What the boundary recognised. A description, never an instruction."""

    families: Tuple[str, ...] = ()
    operations: Tuple[str, ...] = ()
    signals: Tuple[str, ...] = ()
    #: Governed lending windows the question names, in the order it names them.
    lending_windows: Tuple[str, ...] = ()
    #: The role "lending" plays here, or ``None`` when context does not settle it.
    lending_role: Optional[str] = None
    #: Structure an answer MUST carry for this question to be safely answered.
    requirements: Tuple[str, ...] = ()
    #: The governed routes/capabilities that own the recognised families.
    owners: Tuple[str, ...] = ()
    #: Concept terms actually matched, for the receipt. Evidence, not decoration.
    matched: Tuple[str, ...] = ()
    #: Whether the question asks for a NUMBER OF THINGS rather than an amount.
    #: Kept because the governed run-rate capability produces a currency rate and
    #: nothing else: answering "how many loans are we completing?" with a pounds
    #: figure would be a measure substitution, so the boundary declines to hand
    #: that question over and the fail-closed rule applies instead.
    counts_requested: bool = False

    @property
    def materially_analytical(self) -> bool:
        """Whether a single point-in-time aggregate could honestly answer this.

        True when the question carries at least one of the five governed concept
        signals, or belongs to a family whose dataset the point-in-time executor
        does not even read. False for a plain snapshot question — "balance by
        region" is not analytical and must keep the answer it has always had.
        """
        return bool(self.requirements)

    @property
    def recognised(self) -> bool:
        return bool(self.families)

    def to_dict(self) -> Dict[str, Any]:
        return {"countsRequested": self.counts_requested,
                "families": list(self.families),
                "operations": list(self.operations),
                "signals": list(self.signals),
                "lendingWindows": list(self.lending_windows),
                "lendingRole": self.lending_role,
                "requirements": list(self.requirements),
                "owners": list(self.owners),
                "materiallyAnalytical": self.materially_analytical,
                "matched": list(self.matched)}


NOT_ANALYTICAL = AnalyticalIntent()


# --------------------------------------------------------------------------- #
# §4 Lending role resolution
# --------------------------------------------------------------------------- #
def lending_role(text: str, families: Sequence[str]) -> Optional[str]:
    """The role a named lending window plays in THIS question.

    The governed ruling: a PROFILE / MIX / RISK / CHARACTERISTICS context makes
    "new lending" a POPULATION OF LOANS; a RUN RATE / AMOUNT / VOLUME / FLOW /
    ORIGINATED-IN-PERIOD context makes it an ORIGINATION FLOW. Resolved from
    analytical context, never from the sentence.

    The FLOW test runs first and is the narrow one: a window named alongside a
    rate, a volume or an amount-originated-in-a-period is a measure of what was
    written, and narrowing the book to it would answer a different question.
    Everything else is a population, because a named origination window IS a
    cohort of loans — that is what the words mean when nothing reframes them.

    ``None`` is returned only when the caller passes no window at all; callers
    ask this question only when one was named. It is kept in the signature
    because an unresolved role must mean "create no population", and a caller
    reading a role should never have to distinguish "no window" from "a window
    whose role I guessed".
    """
    if _any(text, _ORIGINATION_FLOW_TERMS) or _any(text, _RUN_RATE_TERMS):
        return ROLE_FLOW
    if (FAMILY_MIX_PROFILE in families or FAMILY_VINTAGE_COHORT in families
            or _any(text, _RISK_PROFILE_TERMS) or _any(text, _PROFILE_TERMS)):
        return ROLE_POPULATION
    # A comparison or a movement BETWEEN two named windows is a comparison of
    # two populations of loans — there is nothing else two cohorts could be.
    if _any(text, _COMPARISON_TERMS) or _any(text, _CHANGE_TERMS):
        return ROLE_POPULATION
    return None


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify(question: Optional[str], *, spec: Any = None) -> AnalyticalIntent:
    """The governed family, operations and requirements for a question.

    ``spec`` is the single governed parse, consulted only for flags the parser
    has ALREADY resolved (a forecast mode, a risk-limit query, a temporal
    comparison). It is never re-parsed and never overridden: a parser that has
    settled a question is not second-guessed here.
    """
    text = normalise(question)
    if not text.strip():
        return NOT_ANALYTICAL

    matched: List[str] = []
    signals: List[str] = []

    # Through the SHARED definition, not a second copy of the vocabulary —
    # which is the drift `is_comparative`'s own docstring warns about, and which
    # is how this line and the planner came to answer the same question two
    # different ways.
    comparative = _any(text, _COMPARISON_TERMS) or names_both_sides_of_a_pair(text)
    changing = _any(text, _CHANGE_TERMS)
    # A comparison against a PAST period is a change question, whether or not
    # the sentence contains a change verb.
    if comparative and _any(text, _PRIOR_PERIOD_TERMS):
        changing = True
        matched_prior = _hits(text, _PRIOR_PERIOD_TERMS)
    else:
        matched_prior = []
    rating = _any(text, _RUN_RATE_TERMS)
    limiting = _any(text, _LIMIT_TERMS)
    forecasting = _any(text, _FORECAST_TERMS) or _spec_forecast(spec)
    # A LIMIT question asking about the risk of BREACHING is forward, and is
    # classified with the other forward-breach phrasings rather than as a
    # ranking of today's headroom. Gated on `limiting` so the phrase cannot
    # widen any other family.
    if limiting and _any(text, _AT_RISK_OF_BREACH_TERMS):
        forecasting = True
        matched += _hits(text, _AT_RISK_OF_BREACH_TERMS)

    if comparative:
        signals.append(SIGNAL_COMPARISON); matched += _hits(text, _COMPARISON_TERMS)
    if changing:
        signals.append(SIGNAL_CHANGE)
        matched += _hits(text, _CHANGE_TERMS) + matched_prior
    if rating:
        signals.append(SIGNAL_RUN_RATE); matched += _hits(text, _RUN_RATE_TERMS)
    if limiting:
        signals.append(SIGNAL_LIMITS); matched += _hits(text, _LIMIT_TERMS)
    if forecasting:
        signals.append(SIGNAL_FORECAST); matched += _hits(text, _FORECAST_TERMS)

    windows = tuple(_seasoning.lending_windows_named(question))

    profiling = _any(text, _PROFILE_TERMS) or _any(text, _RISK_PROFILE_TERMS)
    completing = (_any(text, _COMPLETION_TERMS)
                  and not _is_population_qualifier(text))
    pipelining = _any(text, _PIPELINE_TERMS) or completing
    vintaging = _any(text, _VINTAGE_TERMS) or bool(windows)

    families: List[str] = []
    operations: List[str] = []
    requirements: List[str] = []

    # ---- MIX_PROFILE ---------------------------------------------------- #
    if profiling:
        families.append(FAMILY_MIX_PROFILE)
        matched += _hits(text, _PROFILE_TERMS) + _hits(text, _RISK_PROFILE_TERMS)
        operations.append(OP_COMPOSITION if not (comparative or changing)
                          else OP_SNAPSHOT)
        if comparative:
            operations.append(OP_COMPARISON)
        if changing:
            operations += [OP_CHANGE, OP_ATTRIBUTION]
            requirements.append(REQ_PERIOD_COMPARISON)
        if comparative and changing:
            operations.append(OP_DIVERGENCE)

    # ---- PIPELINE ------------------------------------------------------- #
    # Always materially analytical: the pipeline is a DIFFERENT GOVERNED
    # DATASET from the funded book, so the funded point-in-time executor cannot
    # answer any question in this family — not narrowly, not partially.
    if pipelining:
        families.append(FAMILY_PIPELINE)
        matched += _hits(text, _PIPELINE_TERMS) + _hits(text, _COMPLETION_TERMS)
        requirements.append(REQ_PIPELINE_DATASET)
        if completing:
            operations.append(OP_CONVERSION if forecasting else OP_MOVEMENT)
        else:
            operations.append(OP_STOCK)
        if rating or (completing and _any(text, _CURRENT_TERMS)):
            operations.append(OP_RUN_RATE)
            # A rate is a change per unit of time: two governed snapshots, never
            # one, whatever else the question also needs.
            requirements.append(REQ_PERIOD_COMPARISON)
        if forecasting:
            operations.append(OP_EXPECTED_COMPLETION)
            requirements.append(REQ_FORECAST)
        if _asks_when(text):
            operations.append(OP_TIMING)
        if profiling:
            operations.append(OP_MIX)

    # ---- LIMITS_CONCENTRATION ------------------------------------------- #
    if limiting or _spec_risk_limit(spec):
        families.append(FAMILY_LIMITS_CONCENTRATION)
        matched += _hits(text, _LIMIT_TERMS)
        requirements.append(REQ_LIMIT_EVIDENCE)
        operations.append(OP_STATUS)
        if _any(text, (" headroom ", " closest ", " nearest ", " close to ",
                       " least ", " most at risk ", " at risk ", " tightest ")):
            operations += [OP_HEADROOM, OP_RANKING]
        if forecasting:
            operations.append(OP_FORECAST_BREACH)
            requirements.append(REQ_FORECAST)
        if changing:
            operations.append(OP_MOVEMENT)
        operations.append(OP_CONCENTRATION)

    # ---- FORECAST_PROJECTION -------------------------------------------- #
    if forecasting:
        families.append(FAMILY_FORECAST_PROJECTION)
        requirements.append(REQ_FORECAST)
        if _asks_when(text):
            operations += [OP_MILESTONE, OP_HORIZON]
        else:
            operations.append(OP_PROJECT_VALUE)
        if _any(text, (" what if ", " scenario ", " if we ", " assuming ")):
            operations.append(OP_SCENARIO)

    # ---- MOVEMENT_TREND -------------------------------------------------- #
    if changing:
        families.append(FAMILY_MOVEMENT_TREND)
        requirements.append(REQ_PERIOD_COMPARISON)
        operations.append(OP_DELTA)
        if _any(text, (" trend ", " trends ", " trending ", " over time ",
                       " trajectory ", " evolution ", " evolving ")):
            operations.append(OP_TREND)
        if _any(text, (" accelerating ", " accelerated ", " acceleration ",
                       " slowing ", " speeding up ")):
            operations.append(OP_ACCELERATION)
        if comparative:
            operations += [OP_RANKING, OP_ATTRIBUTION]

    # ---- VINTAGE_COHORT --------------------------------------------------- #
    if vintaging:
        families.append(FAMILY_VINTAGE_COHORT)
        matched += _hits(text, _VINTAGE_TERMS)
        operations.append(OP_SNAPSHOT)
        if comparative:
            operations += [OP_COMPARISON, OP_DIVERGENCE]
        if changing:
            operations.append(OP_EVOLUTION)
        if _any(text, (" which vintage ", " worst ", " best ", " ranked ",
                       " ranking ", " rank ")):
            operations.append(OP_RANKING)

    # A comparison BETWEEN two governed populations must measure both of them.
    if comparative and len(windows) >= 2:
        requirements.append(REQ_POPULATION_COMPARISON)

    # A RATE is a change per unit of time. It cannot be computed from a single
    # governed snapshot in any family, so it always requires two — and a bare
    # rate question with no other family named is a movement question.
    if rating:
        requirements.append(REQ_PERIOD_COMPARISON)
        if not families:
            # RUN_RATE is a governed operation of PIPELINE, not of
            # MOVEMENT_TREND. A rate question that names no pipeline concept is
            # a movement question, and is reported as one — an operation is
            # never reported outside the family that governs it.
            families.append(FAMILY_MOVEMENT_TREND)
            operations += [OP_DELTA, OP_TREND]
        matched += _hits(text, _RUN_RATE_TERMS)

    if not families:
        return NOT_ANALYTICAL

    role = lending_role(text, families) if windows else None
    owners: List[str] = []
    for family in families:
        for route in FAMILIES[family].routes:
            if route not in owners:
                owners.append(route)

    return AnalyticalIntent(
        families=tuple(dict.fromkeys(families)),
        operations=tuple(dict.fromkeys(operations)),
        signals=tuple(dict.fromkeys(signals)),
        lending_windows=windows,
        lending_role=role,
        requirements=tuple(dict.fromkeys(requirements)),
        owners=tuple(owners),
        counts_requested=_any(text, _COUNT_TERMS),
        matched=tuple(dict.fromkeys(matched)))


def _asks_when(text: str) -> bool:
    return _any(text, (" when ", " by when ", " how soon ", " how long until ",
                       " which month ", " what timeframe ", " over what ",
                       " timing ", " timeframe ", " time frame "))


def _spec_forecast(spec: Any) -> bool:
    return getattr(spec, "forecast_mode", None) is not None


def _spec_risk_limit(spec: Any) -> bool:
    return bool(getattr(spec, "risk_limit_query", False))


# --------------------------------------------------------------------------- #
# §5 / §8 Governed flag normalisation — route contention, resolved by ownership
# --------------------------------------------------------------------------- #
#: The governed spec flags the boundary is allowed to settle, and nothing else.
#: Each is a flag an EXISTING route already recognises on; setting one hands the
#: question to the route that owns the family, rather than answering it here.
#: Nothing analytical is decided by this — it is routing, and only ever toward a
#: capability that already exists.
FLAG_RISK_LIMIT = "risk_limit_query"
#: The category that flag is SCOPED to, settled from the parser's own reader.
#: `_route_risk` already narrows to it and already refuses honestly when a
#: category has no configured tests; leaving it open made the route answer every
#: category for a question that named one.
FLAG_RISK_LIMIT_CATEGORY = "risk_limit_category"
FLAG_FORECAST_MODE = "forecast_mode"


def governed_flags(reading: AnalyticalIntent, spec: Any,
                   question: Optional[str] = None) -> Dict[str, Any]:
    """Governed intent flags the boundary can settle that the parser did not.

    The measured failure this exists for: *"Which concentration limits have the
    least headroom?"* sets ``risk_limit_query`` and is answered correctly by the
    ``risk_limits`` route, while *"Where are we closest to our limits?"* does
    not and falls through to the funded executor, which answers with weighted
    average LTV by region. Same family, same operation, same governed owner —
    and two different answers, because one phrasing reached the flag and the
    other did not.

    Three rules keep this safe:

    * **Never overrides.** A flag the parser already set is left exactly as it
      is. The boundary fills gaps; it does not second-guess a settled parse.
    * **Never invents an analytic.** Every flag names a route that already
      exists and already owns the recognised family.
    * **Never widens a measure.** The governed run-rate capability produces a
      currency rate and nothing else, so a question asking *how many* is not
      handed to it — that question falls to the fail-closed rule instead, and
      is refused rather than answered in the wrong unit.
    """
    flags: Dict[str, Any] = {}

    if (FAMILY_LIMITS_CONCENTRATION in reading.families
            and not getattr(spec, FLAG_RISK_LIMIT, False)):
        flags[FLAG_RISK_LIMIT] = True

    # AND THE CATEGORY THE QUESTION NAMED, if it named one and the parse left it
    # open. Measured: "What is the largest geographic concentration versus
    # limit?" is claimed HERE — `_RISK_LIMIT_RE` never matches it — and reached
    # `_route_risk` with no category, which answered "5 passed … Nearest to
    # limit: Top 3 brokers" for a question about geography. Handing over the
    # flag without its scope hands over a wider question than the reader asked.
    #
    # The reader is the parser's own, so nothing here decides what a category
    # is; and it never overrides a category a parse already settled.
    if (FAMILY_LIMITS_CONCENTRATION in reading.families and question
            and getattr(spec, FLAG_RISK_LIMIT_CATEGORY, None) is None):
        try:
            from mi_agent.llm_query_parser import risk_limit_category
            category = risk_limit_category(question)
        except Exception:  # noqa: BLE001 - no reader, no scope; the flag stands
            category = None
        if category:
            flags[FLAG_RISK_LIMIT_CATEGORY] = category

    # Deliberately narrow: only a PIPELINE run rate. `forecast_extrapolation`
    # computes the COMPLETION run rate — the flow of cases out of the pipeline
    # into the funded book — and nothing else.
    #
    # This narrowness is load-bearing, not fastidious. `forecast_mode` is a flag
    # three other recognisers DECLINE on (`concentration_analysis`,
    # `portfolio_risk_comparison`, `period_change_analysis`), so setting it on a
    # question they own would divert it to a route that cannot answer it. "How
    # has concentration by region changed per month?" carries a rate concept and
    # belongs to period change; it must not become a completion forecast.
    wants_rate = (OP_RUN_RATE in reading.operations
                  and FAMILY_PIPELINE in reading.families)
    if (wants_rate and not reading.counts_requested
            and getattr(spec, FLAG_FORECAST_MODE, None) is None):
        flags[FLAG_FORECAST_MODE] = "extrapolation"

    return flags


# --------------------------------------------------------------------------- #
# §7 The fail-closed safety rule
# --------------------------------------------------------------------------- #
def unmet_requirements(reading: AnalyticalIntent, *,
                       evidence: Mapping[str, Any]) -> List[str]:
    """The structural requirements an answer did NOT satisfy.

    ``evidence`` describes what was ACTUALLY executed, not what was intended —
    the same discipline the P0 execution receipt applies. A requirement is met
    only when the executed answer demonstrably carries the structure; an answer
    that cannot prove it leaves the requirement unmet and the caller refuses.

    Keys, all optional and all defaulting to "not demonstrated":

    ``dataset``           the governed dataset the answer was computed from
    ``periods``           how many governed reporting snapshots it compared
    ``forecast``          whether it produced a forward-looking figure
    ``limits``            whether it evaluated governed limits
    ``grouping``          the dimension it grouped by, if any
    ``populations``       how many governed populations it measured separately
    """
    unmet: List[str] = []
    dataset = str(evidence.get("dataset") or "")
    periods = int(evidence.get("periods") or 0)
    grouping = str(evidence.get("grouping") or "")
    populations = int(evidence.get("populations") or 0)

    for requirement in reading.requirements:
        if requirement == REQ_PIPELINE_DATASET and dataset != "pipeline":
            unmet.append(requirement)
        elif requirement == REQ_LIMIT_EVIDENCE and not evidence.get("limits"):
            unmet.append(requirement)
        elif requirement == REQ_FORECAST and not evidence.get("forecast"):
            unmet.append(requirement)
        elif requirement == REQ_PERIOD_COMPARISON and periods < 2:
            unmet.append(requirement)
        elif requirement == REQ_POPULATION_COMPARISON:
            # Two populations may be measured side by side EITHER as two
            # separate populations OR as two values of one grouping dimension.
            # The second is how the point-in-time executor legitimately answers
            # "how does the front book compare with the back book?" — grouped by
            # seasoning segment, both sides present, nothing passed off as the
            # other. That is a real answer and must not be refused.
            if populations < 2 and not grouping:
                unmet.append(requirement)
    return unmet


def refusal_message(reading: AnalyticalIntent, unmet: Sequence[str]) -> str:
    """The controlled refusal for a materially analytical question nothing owns.

    Names what was understood and what could not be established, and asks for the
    one thing that would let the governed capability answer. It never presents a
    figure, and it never suggests the question was meaningless — the question was
    understood; the answer was not available.
    """
    families = ", ".join(f.replace("_", " ").lower() for f in reading.families)
    reasons = [REQUIREMENT_REASONS[r] for r in unmet if r in REQUIREMENT_REASONS]
    body = "; and ".join(reasons) if reasons else (
        "no governed analytic could be established for it")
    return (f"I understood this as a {families} question, but I have not answered "
            f"it: {body}. I have NOT substituted a current-position figure, "
            "because that would answer a different question from the one you "
            "asked. Ask for the governed analytic directly — for example the "
            "concentration limit tests, the completion run-rate, the pipeline at "
            "a named stage, or a named measure compared across two reporting "
            "periods — and I will compute it.")


def settle(question: Optional[str], spec: Any) -> Tuple[AnalyticalIntent, Dict[str, Any]]:
    """Read the question, and settle the governed flags the parser left open.

    Returns the boundary's reading and the flags it actually set, so the caller
    can publish both as evidence. Mutating the governed spec is deliberate and
    is the whole mechanism: the routes below already know how to recognise these
    flags, and a flag set here reaches every one of them without a single route
    being changed.
    """
    reading = classify(question, spec=spec)
    if not reading.recognised or spec is None:
        return reading, {}
    applied: Dict[str, Any] = {}
    for name, value in governed_flags(reading, spec, question).items():
        try:
            setattr(spec, name, value)
        except Exception:  # noqa: BLE001 - a spec that will not take a flag is
            continue       # simply left alone; the fail-closed rule still holds
        applied[name] = value
    return reading, applied
