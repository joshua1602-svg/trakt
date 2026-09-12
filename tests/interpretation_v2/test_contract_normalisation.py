"""One economic meaning, one canonical contract representation.

Three redundancies, each proved three ways: the rewrite happens, it converges the
cases that were diverging, and it loses nothing. The third is the one worth
writing tests for — a normalisation that quietly dropped a semantic element would
converge beautifully and be worse than the redundancy it replaced.

WHAT IS ASSERTED HERE
---------------------
    1. labels         two spellings of the user's words -> one plan identity
    2. relative period  two spellings of one relationship -> one canonical form
    3. owner          the implementation owner is DERIVED, never claimed

    and for all three:
        no semantic information is lost
        the model's own claim survives in provenance
        compilation stays deterministic and idempotent
        no outcome changes from PLAN to a refusal
        nothing outside the contract boundary moved
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from mi_agent.interpretation_v2.compiler import (
    CAPABILITY_OPERATIONS,
    CompilerContext,
    DeterministicCompiler,
)
from mi_agent.interpretation_v2.compiler import (
    _CAPABILITY_OWNED_PERIOD_OPERATIONS,
    _MULTI_PERIOD_FORMS,
    _MULTI_PERIOD_OPERATIONS,
)
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.metadata import metadata_tool_schemas
from mi_agent.interpretation_v2.opus_interpreter import SYSTEM_PROMPT
from mi_agent.interpretation_v2.normalise import (
    BOUNDED,
    CANONICAL_PAIR_FORM,
    CANONICAL_PAIR_PERIODS_BACK,
    NORMAL_FORM_VERSION,
    PAIR_IMPLYING_OPERATIONS,
    canonical_intent,
)
from mi_agent.interpretation_v2.plan import LABELS_ARE_WORDING_ONLY, identity_labels
from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary

_REPO_ROOT = Path(__file__).resolve().parents[2]
_START = "5ae1f73"


@pytest.fixture(scope="module")
def vocabulary():
    return load_governed_vocabulary()


@pytest.fixture(scope="module")
def compiler():
    return DeterministicCompiler(CompilerContext())


def movement_intent(**over):
    """The Q19/Q20 shape: a period-on-period movement of a generic measure.

    `change_form` is STATED because since the completeness gate a change-oriented
    temporal request that does not say which analytical form it is no longer
    compiles, and a period-on-period movement of one named generic measure is a
    metric delta. This completes an AUTHORED fixture; it does not move the
    subject, which is normalisation — labels, the relative period, and the derived
    implementation owner. Call sites naming a SPECIALIST measure override it with
    the form that measure's owner belongs to.

    The RECORDED Q19/Q20 payloads are not edited. They stay exactly as the model
    emitted them, and their authorised migration is pinned case by case in
    `test_slice3_portfolio_affordance`.
    """
    payload = {
        "schema_version": "candidate_intent/1.0",
        "capability": "period_movement",
        "operation": "movement",
        "change_form": "metric_delta",
        "measures": [{"concept": "current_outstanding_balance"}],
        "time": {"form": "relative_pair", "grain": "monthly",
                 "periods_back": 1, "labels": ["last month"]},
    }
    payload.update(over)
    return parse_candidate_intent(payload)


# --------------------------------------------------------------------------- #
# 1. period labels
# --------------------------------------------------------------------------- #

def test_the_users_wording_does_not_change_what_was_authorised(compiler):
    """Q19's whole residual divergence, in one assertion.

    Q19A/B carried labels=["last month"] and Q19C carried ["month-on-month"].
    Every other field was identical and the plans still hashed differently.
    """
    a = compiler.compile(movement_intent())
    b = compiler.compile(movement_intent(
        time={"form": "relative_pair", "grain": "monthly", "periods_back": 1,
              "labels": ["month-on-month"]}))
    assert a.plan is not None and b.plan is not None
    assert a.plan.plan_id == b.plan.plan_id


def test_the_plan_still_carries_the_words_it_was_given(compiler):
    """Identity stops depending on labels. The plan does not stop having them.

    A disclosure that cannot quote the period the user named is a worse answer,
    so the label travels — it is simply not part of what was authorised.
    """
    plan = compiler.compile(movement_intent()).plan
    assert plan.period.labels == ("last month",)
    assert plan.provenance.intent_claims["time"][1] == ("last month",)


def test_a_label_that_IS_the_period_stays_in_identity():
    """The limit of normalisation 1, and the reason it is not a blanket strip.

    "April" and "May" are explicit_period labels. They are not wording about a
    window the contract already knows — they ARE the window, and collapsing them
    would make two different months one plan.
    """
    assert identity_labels("explicit_period", ("April",)) == ("April",)
    assert identity_labels("range", ("2024",)) == ("2024",)
    assert identity_labels("series", ("last six months",)) == ("last six months",)
    assert identity_labels("forward_looking", ("by year end",)) == ("by year end",)
    # ...and the forms where the contract does settle it.
    for form in LABELS_ARE_WORDING_ONLY:
        assert identity_labels(form, ("whatever they said",)) == ()


def test_two_different_named_months_remain_two_different_plans(compiler):
    """The failure mode the previous test rules out, end to end."""
    april = compiler.compile(movement_intent(
        time={"form": "explicit_period", "grain": "monthly", "labels": ["April"]}))
    may = compiler.compile(movement_intent(
        time={"form": "explicit_period", "grain": "monthly", "labels": ["May"]}))
    assert april.plan is not None and may.plan is not None
    assert april.plan.plan_id != may.plan.plan_id


# --------------------------------------------------------------------------- #
# 2. relative period
# --------------------------------------------------------------------------- #

def test_one_relative_relationship_has_one_canonical_form(compiler, vocabulary):
    """Q20A against Q20B/C: two spellings, one economic relationship."""
    spelled_short = movement_intent(
        time={"form": "previous_reporting_period", "grain": "monthly",
              "labels": ["last month"]})
    result = canonical_intent(spelled_short, vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.time.form == CANONICAL_PAIR_FORM
    assert result.intent.time.periods_back == CANONICAL_PAIR_PERIODS_BACK
    assert any("relative_period" in a for a in result.applied)

    a = compiler.compile(spelled_short)
    b = compiler.compile(movement_intent())
    assert a.plan is not None and b.plan is not None
    assert a.plan.plan_id == b.plan.plan_id


def test_the_pair_rewrite_is_confined_to_operations_that_span_two_periods():
    """Derived from the compiler's own sets, so the two cannot drift apart.

    An operation that owns its own window is not ours to rewrite, and a series is
    not a pair. What survives is exactly `movement`.
    """
    derived = (_MULTI_PERIOD_OPERATIONS
               - _CAPABILITY_OWNED_PERIOD_OPERATIONS
               - {"series"})
    assert PAIR_IMPLYING_OPERATIONS == derived, (
        "the pair-implying set no longer follows from the compiler's contracts")


def test_a_single_period_question_keeps_its_single_period(compiler, vocabulary):
    """Normalisation removes freedom; it does not add a period.

    `previous_reporting_period` on a point-in-time question means ONE period —
    last month's balance, not the change in it. Rewriting that to a pair would be
    inventing a second period nobody asked for, which is the one thing a
    normalisation may never do.
    """
    single = parse_candidate_intent({
        "schema_version": "candidate_intent/1.0",
        "capability": "portfolio_summary", "operation": "summary",
        "measures": [{"concept": "current_outstanding_balance"}],
        "time": {"form": "previous_reporting_period", "grain": "monthly"}})
    result = canonical_intent(single, vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.time.form == "previous_reporting_period"
    assert result.applied == ()


def test_a_pair_with_no_distance_is_the_adjacent_pair(compiler, vocabulary):
    """The third spelling, and the reason the first attempt converged nothing.

    Normalising only `previous_reporting_period` -> `relative_pair+1` left
    `relative_pair` with `periods_back` absent as a distinct third form, so Q19C
    and Q20C still diverged from their siblings over nothing else. The compiler
    accepts a pair with no distance as COMPLETE — unlike a range or a series, it
    raises no AMBIGUOUS_PERIOD — which leaves adjacent as the only reading.
    """
    absent = movement_intent(time={"form": "relative_pair", "grain": "monthly",
                                   "labels": ["month-on-month"]})
    result = canonical_intent(absent, vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.time.periods_back == CANONICAL_PAIR_PERIODS_BACK
    assert compiler.compile(absent).plan.plan_id == \
        compiler.compile(movement_intent()).plan.plan_id


def test_a_pair_zero_periods_apart_is_left_alone(vocabulary):
    """Scope limit: absent is not the same as zero.

    A pair zero periods apart is not the adjacent pair, and folding it in would
    be changing a stated distance rather than supplying an implied one.
    """
    zero = movement_intent(time={"form": "relative_pair", "grain": "monthly",
                                 "periods_back": 0})
    result = canonical_intent(zero, vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.time.periods_back == 0
    assert result.applied == ()


def test_no_other_period_form_gains_an_implied_distance(vocabulary):
    """The backlog this sprint is told not to broaden into stays untouched.

    NL5 diverged over `periods_back` 0 against absent on a FORWARD-LOOKING
    horizon. That is a different question from the adjacent-pair default and it
    is not in scope here.
    """
    for form in ("forward_looking", "series", "range", "current"):
        intent = movement_intent(
            capability="forecast" if form == "forward_looking" else "period_movement",
            time={"form": form, "grain": "quarterly", "labels": ["next quarter"]})
        result = canonical_intent(intent, vocabulary,
                                  capability_operations=CAPABILITY_OPERATIONS)
        assert result.intent.time.periods_back is None, form


def test_a_stated_period_distance_is_not_overwritten(vocabulary):
    """Two periods back is not one period back.

    The canonical form supplies periods_back=1 only where nothing was stated.
    """
    intent = movement_intent(
        time={"form": "previous_reporting_period", "grain": "monthly",
              "periods_back": 2})
    result = canonical_intent(intent, vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.time.periods_back == 2


def test_both_spellings_remain_valid_multi_period_forms():
    """Proof the rewrite cannot change an outcome.

    Either spelling satisfies the composition check, so normalising between them
    can never turn a plan into an unsupported composition.
    """
    assert {"previous_reporting_period", CANONICAL_PAIR_FORM} <= _MULTI_PERIOD_FORMS


# --------------------------------------------------------------------------- #
# 3. implementation owner
# --------------------------------------------------------------------------- #

def test_the_implementation_owner_is_derived_from_the_measure(compiler, vocabulary):
    """The model should never have been choosing between internal owners.

    `funded_balance_movement` is owned by `funded_bridge`. Whatever capability
    the model names, the owner of the measure it asked for is the owner.
    """
    intent = movement_intent(capability="period_movement",
                             change_form="attribution",
                             measures=[{"concept": "funded_balance_movement"}])
    result = canonical_intent(intent, vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.capability == "funded_bridge"
    assert any("implementation_owner" in a for a in result.applied)
    assert compiler.compile(intent).plan.capability == "funded_bridge"


def test_a_generic_measure_derives_no_owner_and_is_left_alone(vocabulary):
    """`period_movement` owns no measures, so ownership settles nothing here."""
    result = canonical_intent(movement_intent(), vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.capability == "period_movement"
    assert not any("implementation_owner" in a for a in result.applied)


def test_an_owner_that_cannot_do_the_operation_is_not_bound(vocabulary):
    """Outcome neutrality, enforced rather than hoped for.

    Rewriting the capability to an owner that does not support the stated
    operation would convert a plan into UNSUPPORTED_OPERATION. A normalisation
    that turns answers into refusals is not a normalisation.
    """
    impossible = "breakdown"
    assert impossible not in CAPABILITY_OPERATIONS["funded_bridge"]
    intent = movement_intent(
        capability="period_movement", operation=impossible,
        dimensions=["erm_product_type"],
        change_form="attribution",
        measures=[{"concept": "funded_balance_movement"}])
    result = canonical_intent(intent, vocabulary,
                              capability_operations=CAPABILITY_OPERATIONS)
    assert result.intent.capability == "period_movement"
    assert result.applied == ()


def test_what_normalisation_3_deliberately_does_not_collapse():
    """The bound, recorded in the source so it cannot be mistaken for an oversight.

    `period_movement`/`current_outstanding_balance` and
    `funded_bridge`/`funded_balance_movement` name two different governed
    MEASURES. Deciding which of them answers "how did the book change?" is
    arithmetic, not representation, and this sprint may not change arithmetic.
    """
    assert "movement_measure_choice" in BOUNDED
    assert "analytical behaviour" in BOUNDED["movement_measure_choice"]


def test_two_different_measures_remain_two_different_plans(compiler):
    """Because the bound above is real, not decorative."""
    generic = compiler.compile(movement_intent())
    specialist = compiler.compile(movement_intent(
        change_form="attribution",
        measures=[{"concept": "funded_balance_movement"}]))
    assert generic.plan is not None and specialist.plan is not None
    assert generic.plan.plan_id != specialist.plan.plan_id


# --------------------------------------------------------------------------- #
# properties that must survive all three
# --------------------------------------------------------------------------- #

def test_normalisation_is_idempotent(vocabulary):
    """A canonical intent is its own canonical form.

    Without this, plan identity would depend on how many times the compiler ran,
    which is a worse defect than the one being fixed.
    """
    for intent in (movement_intent(),
                   movement_intent(time={"form": "previous_reporting_period",
                                         "grain": "monthly"}),
                   movement_intent(change_form="attribution",
                                   measures=[{"concept": "funded_balance_movement"}])):
        once = canonical_intent(intent, vocabulary,
                                capability_operations=CAPABILITY_OPERATIONS).intent
        twice = canonical_intent(once, vocabulary,
                                 capability_operations=CAPABILITY_OPERATIONS)
        assert twice.applied == (), f"second pass changed something: {twice.applied}"
        assert twice.intent.semantic_key() == once.semantic_key()


def test_compilation_remains_deterministic(compiler):
    """Rule I, re-asserted across the new step."""
    intent = movement_intent(time={"form": "previous_reporting_period",
                                   "grain": "monthly"})
    ids = {compiler.compile(intent).plan.plan_id for _ in range(5)}
    assert len(ids) == 1


def test_the_model_s_own_claim_survives_every_rewrite(compiler):
    """The audit question this architecture is judged on stays answerable.

    A provenance that recorded only the canonical form could not say what the
    model proposed, and "did the model pick this?" would become unanswerable.
    """
    intent = movement_intent(
        capability="period_movement",
        change_form="attribution",
        measures=[{"concept": "funded_balance_movement"}],
        time={"form": "previous_reporting_period", "grain": "monthly"})
    plan = compiler.compile(intent).plan
    claims = plan.provenance.intent_claims
    assert claims["capability"] == "period_movement"
    assert claims["time"][0] == "previous_reporting_period"
    # ...and what the compiler did to it is recorded on the compiler's side.
    applied = plan.provenance.compiler_bindings["normalisation"]["applied"]
    assert len(applied) == 2
    assert plan.provenance.compiler_bindings["normalisation"][
        "normal_form_version"] == NORMAL_FORM_VERSION


def test_the_compile_result_reports_the_intent_the_model_emitted(compiler):
    """A caller must not receive a rewritten intent as though it were the model's."""
    intent = movement_intent(time={"form": "previous_reporting_period",
                                   "grain": "monthly"})
    result = compiler.compile(intent)
    assert result.intent.time.form == "previous_reporting_period"


def test_normalisation_never_reads_the_question():
    """The rule the compiler obeys, extended to the module in front of it.

    `normalise.py` has no access to the question text and must never acquire one:
    a normal form derived from wording would be a recogniser with a new name.
    """
    source = (_REPO_ROOT / "mi_agent/interpretation_v2/normalise.py").read_text()
    tree = ast.parse(source)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                 for a in n.names}
    assert "re" not in imported, "normalisation must not pattern-match language"
    # Read the code, not the prose: the docstring is allowed to discuss
    # provenance, and an attribute access would be reading the question.
    reached = {node.attr for node in ast.walk(tree)
               if isinstance(node, ast.Attribute)}
    for forbidden in ("provenance", "question", "evidence"):
        assert forbidden not in reached, (
            f"normalisation reaches .{forbidden}, which carries the question's "
            f"own words")


# --------------------------------------------------------------------------- #
# the sprint boundary
# --------------------------------------------------------------------------- #

def test_only_contract_modules_moved(vocabulary):
    """Contract normalisation may touch the contract. Nothing else.

    Named explicitly because the brief's boundary is the deliverable as much as
    the normalisations are: an interpreter, metadata or registry edit smuggled in
    here would make the replay evidence meaningless.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-status", _START, "--", "mi_agent", "engine",
         "analytics_lib", "mi_agent_api", "frontend", "trakt_notifications",
         "mi_workflows", "trakt_tools"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")
    allowed_modified = {
        "mi_agent/interpretation_v2/compiler.py",
        "mi_agent/interpretation_v2/plan.py",
    }
    modified = set()
    for line in diff.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        if status.startswith("M") and "/evidence/" not in path:
            modified.add(path.strip())
    assert modified <= allowed_modified, (
        f"outside the contract boundary: {sorted(modified - allowed_modified)}")


def test_the_interpreter_policy_did_not_move(vocabulary):
    """Opus policy unchanged is a success condition, not an assumption.

    THE PROMPT OWNER IS STILL PINNED HARD. `SYSTEM_PROMPT` lives in
    `opus_interpreter.py`, the metadata service in `metadata.py`, the reason
    codes in `outcomes.py` and the frozen question banks under `banks/`. None of
    those may move, and this still fails if any of them does — which is what
    mechanically enforces "do not broadly retune interpretation_v2".

    `vocabulary.py` IS ALLOWED, AND ONLY FOR WHAT IT WAS ALLOWED FOR. The slice
    2 boundary correction was instructed to state the generic_analysis /
    period_movement ownership explicitly, after a live run measured four "how
    has X changed over N months" questions routed to `period_movement` on the
    strength of the word "changed". So the file moves, and a file-name check
    alone would now say nothing. The substance is asserted instead: the
    model-facing orientation block may differ from the pre-correction one by the
    AUTHORISED new keys and by nothing else. Another key, a changed enumeration
    or a reworded existing entry fails here.

    THE PROMPT AND THE TOOL SURFACE MOVED TOO, ONCE, AND THE SAME TREATMENT IS
    APPLIED. The slice 3 portfolio affordance was instructed to make
    `source_reference` a first-class model-facing concept and to give the model
    read-only access to the client's governed source names, after a live
    boundary run measured four named-portfolio questions that the affordance
    could not express — the schema field carried no description, the prompt
    never mentioned it, and `get_allowed_values` told the model in terms to
    record a blocking ambiguity instead. A file-name check on those two files
    would now say nothing either, so both are asserted on substance:

      * the SYSTEM_PROMPT may differ from the measured one by exactly the three
        amended paragraphs below and by nothing else. Every other paragraph — 23
        of 26 — must still be present word for word. A reworded rule, a dropped
        rule or an extra one fails here, and that is what mechanically enforces
        "no general retuning of interpretation_v2";
      * the metadata tool surface may gain `get_source_portfolios` and nothing
        else, and every pre-existing tool schema must be byte-identical.

    `outcomes.py` MOVED ONCE MORE, for one reason code, and is asserted on
    substance rather than by file name. The change-form sprint was instructed to
    represent a broad "what changed?" request as a first-class semantic AND to
    refuse it safely until the funded composition owner exists — "return a
    governed NOT_CONNECTED / REFUSE outcome ... do not optimise the benchmark by
    selecting a substitute analysis". A refusal with no code of its own would
    have had to borrow CAPABILITY_UNAVAILABLE, which says something different:
    that a capability is not registered on this book, rather than that an
    analytical form has no implementation anywhere. So the file may gain exactly
    that one code and nothing else; a second new code, or a changed or removed
    existing one, still fails here.

    `banks/` stays pinned by file name. The frozen question banks were not in
    scope and are not in scope now.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-only", _START, "--",
         "mi_agent/interpretation_v2/banks"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    assert changed == [], f"outside this sprint's boundary: {changed}"

    # -- reason codes: one added, every other one untouched ------------------ #
    was_outcomes = subprocess.run(
        ["git", "show", f"{_START}:mi_agent/interpretation_v2/outcomes.py"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if was_outcomes.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")
    import re as _re_codes
    was_codes = set(_re_codes.findall(r'^([A-Z][A-Z0-9_]+)\s*=\s*"',
                                      was_outcomes.stdout, _re_codes.M))
    from mi_agent.interpretation_v2 import outcomes as _outcomes
    now_codes = set(_re_codes.findall(
        r'^([A-Z][A-Z0-9_]+)\s*=\s*"',
        __import__("inspect").getsource(_outcomes), _re_codes.M))
    assert now_codes - was_codes == {"CHANGE_FORM_NOT_CONNECTED"}, (
        f"reason codes gained {sorted(now_codes - was_codes)}; exactly one was "
        f"authorised")
    assert was_codes - now_codes == set(), (
        f"reason codes lost {sorted(was_codes - now_codes)}")

    # -- the prompt: additive, and only where it was authorised to be -------- #
    import re as _re

    was_source = subprocess.run(
        ["git", "show", f"{_START}:mi_agent/interpretation_v2/opus_interpreter.py"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if was_source.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")
    was_prompt = _re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""',
                            was_source.stdout, _re.S).group(1).replace("\\\n", "")

    def _paragraphs(text):
        return [_re.sub(r"\s+", " ", block).strip()
                for block in text.split("\n\n") if block.strip()]

    was_paras, now_paras = _paragraphs(was_prompt), _paragraphs(SYSTEM_PROMPT)
    amended = [p for p in was_paras if p not in now_paras]
    #: The three paragraphs the slice 3 affordance was authorised to amend: the
    #: tool list (one tool added), the RULES block (rules 3a-i and 9 added), and
    #: preservation check A (the named source portfolio added to what must never
    #: be dropped). Identified by their opening words so this reads as intent
    #: rather than as a hash nobody can check.
    assert len(amended) == 3, (
        f"{len(amended)} paragraphs of the measured prompt changed; three were "
        f"authorised")
    assert amended[0].startswith("* `search_concepts`")
    assert amended[1].startswith("1. Name governed concept identifiers")
    assert amended[2].startswith("A. PRESERVATION.")
    assert len(now_paras) == len(was_paras), (
        "the prompt gained or lost a paragraph; the three authorised changes "
        "are all amendments to existing ones")

    # -- the tool surface: one tool added, the rest untouched ---------------- #
    was_metadata = subprocess.run(
        ["git", "show", f"{_START}:mi_agent/interpretation_v2/metadata.py"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if was_metadata.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")

    import importlib.util as _ilu
    import sys as _sys
    import tempfile as _tf

    with _tf.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metadata_before.py"
        path.write_text(was_metadata.stdout, encoding="utf-8")
        spec = _ilu.spec_from_file_location(
            "mi_agent.interpretation_v2.metadata_before", path)
        module = _ilu.module_from_spec(spec)
        _sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            was_tools = {t["name"]: t for t in module.metadata_tool_schemas()}
        finally:
            _sys.modules.pop(spec.name, None)

    now_tools = {t["name"]: t for t in metadata_tool_schemas()}
    assert set(now_tools) - set(was_tools) == {"get_source_portfolios"}
    assert set(was_tools) - set(now_tools) == set(), "a metadata tool was removed"
    for name, schema in was_tools.items():
        assert now_tools[name] == schema, f"the {name!r} tool schema was reworded"

    # The substantive half: what the model is SHOWN moved by exactly one key.
    before = subprocess.run(
        ["git", "show", f"{_START}:mi_agent/interpretation_v2/vocabulary.py"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if before.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")

    import importlib.util
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocabulary_before.py"
        path.write_text(before.stdout, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(
            "mi_agent.interpretation_v2.vocabulary_before", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            was = module.load_governed_vocabulary().orientation_payload()
        finally:
            sys.modules.pop(spec.name, None)

    now = vocabulary.orientation_payload()
    added = set(now) - set(was)
    assert added == {"capability_boundaries", "portfolio_scope_axes"}, (
        f"the orientation block gained {sorted(added)}; slice 2 was authorised "
        f"to add capability_boundaries and slice 3 portfolio_scope_axes, and "
        f"nothing else")
    assert set(was) - set(now) == set(), "the orientation block lost a key"

    #: ONE AUTHORISED WIDENING, WRITTEN OUT RATHER THAN WAIVED. The funded
    #: material-change sprint was instructed to connect `change_form =
    #: material_summary` to a governed owner. That owner is `period_movement` —
    #: `period_change.workflow` run in MODE_PORTFOLIO_OVERVIEW, the same owner
    #: `metric_delta` already used, distinguished from it by the MODE and not by
    #: a new capability. The operation that form carries is `summary`, which
    #: `period_movement` did not list while nothing composed that output into
    #: findings; `mi_agent_api.insight_funded` now does. Without this the form is
    #: unreachable: `movement` and `compare` both require a named measure, and a
    #: named measure is a metric delta, so the composition could never be
    #: entered. Nothing else about what the model is shown may move — a second
    #: added operation, a removed one, or any other reworded key still fails.
    AUTHORISED_OPERATION_ADDITIONS = {"period_movement": {"summary"}}

    for key in sorted(set(was) & set(now)):
        if key == "vocabulary_version":
            continue                       # moves with the block, by design
        if key == "capability_operations":
            for capability in sorted(set(was[key]) | set(now[key])):
                before_ops = set(was[key].get(capability, ()))
                after_ops = set(now[key].get(capability, ()))
                authorised = AUTHORISED_OPERATION_ADDITIONS.get(capability, set())
                assert after_ops - before_ops == (authorised & after_ops), (
                    f"capability {capability!r} was shown operations "
                    f"{sorted(after_ops - before_ops)}; only "
                    f"{sorted(authorised)} were authorised")
                assert before_ops - after_ops == set(), (
                    f"capability {capability!r} lost operations "
                    f"{sorted(before_ops - after_ops)}")
            continue
        assert was[key] == now[key], f"orientation key {key!r} was reworded"


def test_the_model_still_cannot_author_an_executable_binding():
    """The invariant the whole architecture rests on, re-checked after a contract edit.

    Normalisation gave the compiler one more decision to make. It must not have
    given the intent one more slot to fill.
    """
    from mi_agent.interpretation_v2 import candidate_intent_json_schema

    schema = candidate_intent_json_schema()
    assert schema["additionalProperties"] is False
    # `change_form` is the one slot the change-form sprint was authorised to
    # add: which analytical question a change request is asking, stated
    # separately from what is measured and from who executes it. It is an
    # ENUM of governed analytical forms, so it cannot carry a column, a
    # snapshot or an expression — which is the invariant this test is
    # actually about, asserted below rather than left to the set.
    assert set(schema["properties"]) == {
        "schema_version", "capability", "operation", "change_form",
        "population", "measures", "dimensions", "filters", "geography",
        "time", "comparison", "target", "outputs", "ambiguity", "evidence"}
    assert schema["properties"]["change_form"]["enum"]
