"""The interpretation POLICY, and proof that nothing else moved to get it.

Run 6 measured seven canonicals where the model understood the request and then
volunteered analytics nobody asked for — six added a measure, two added a
dimension, none ever dropped something requested. Two others (Q16, Q21) showed
the opposite risk: an explicit filter or grouping lost while a PLAN was still
produced.

Both are one policy: **minimum sufficient governed intent, with explicit
semantics preserved.** This phase changes the interpreter's instruction and
nothing else, so the tests here come in two halves — the instruction carries the
required invariants, and every other subsystem is byte-for-byte where it was.

WHAT IS NOT TESTED HERE, DELIBERATELY
-------------------------------------
Whether the policy WORKS. That is a question about a language model's behaviour
over 135 frozen questions, and it is answered by a live benchmark, not by a unit
test. Writing a deterministic English checker to enforce the policy would
rebuild the recogniser cascade this whole architecture replaced — so these tests
assert the instruction's content and the absence of collateral change, and the
evidence files carry the behavioural answer.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from mi_agent.interpretation_v2 import (
    INTENT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    build_system_blocks,
    build_tool_schema,
    candidate_intent_json_schema,
    load_governed_vocabulary,
)
from mi_agent.interpretation_v2.metadata import SOURCES, metadata_tool_schemas
from mi_agent.interpretation_v2.opus_interpreter import SYSTEM_PROMPT

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# the policy itself
# --------------------------------------------------------------------------- #

def test_the_instruction_states_the_necessity_test():
    """The defect was elaboration, so the instruction must draw the line."""
    assert "MINIMUM SUFFICIENT GOVERNED INTENT" in SYSTEM_PROMPT
    assert "necessary to answer the question" in SYSTEM_PROMPT
    assert "merely useful to add" in SYSTEM_PROMPT
    # The specific invitations measured in Run 6's policy cases.
    for invitation in ("useful", "interesting", "analytically related",
                       "available in Trakt"):
        assert invitation in SYSTEM_PROMPT, (
            f"the instruction does not rule out adding something because it is "
            f"{invitation!r}")


def test_minimum_sufficient_is_not_defined_as_literalism():
    """The recalibration's whole subject.

    Run 7 produced intents carrying an operation and no measure at all, because
    the question had not said the measure's name. The compiler refused them —
    correctly — but the omission was the interpreter's, and it came from reading
    "smallest" as "only the words present". The instruction must rule that out
    in terms, not merely hint at it.
    """
    assert "does NOT mean explicit words only" in SYSTEM_PROMPT
    assert "not a reason to omit it" in SYSTEM_PROMPT
    # Restraint has a scope, and the instruction must say what it is.
    assert "never governs what the request itself entails" in SYSTEM_PROMPT


def test_the_instruction_ranks_the_four_interpretation_bands():
    """Priority, not a single yes/no test.

    One test ("is it necessary?") cannot separate an element that is required
    and clear from one that is required and open — the first must be included
    and the second must clarify. Four bands can.
    """
    for band in ("EXPLICITLY REQUESTED", "REQUIRED BY THE REQUESTED OPERATION",
                 "OPTIONAL COMPANION"):
        assert band in SYSTEM_PROMPT, f"no band for {band!r}"
    # Required-and-open is the CLARIFY band; required-and-clear is the include
    # band. Both appear under the same heading, so check their dispositions.
    assert "more than one governed reading is materially plausible" in SYSTEM_PROMPT
    assert "exactly ONE materially" in SYSTEM_PROMPT


def test_the_instruction_names_what_an_operation_requires():
    """Generic entailments, not bank-specific ones.

    The prompt may not say "a movement needs a balance" — that is tuning. It
    may say that an operation reporting figures needs a measure, which is a
    property of the contract and holds for the next bank too.
    """
    assert "requires at least one measure" in SYSTEM_PROMPT
    assert "is not minimal, it is INCOMPLETE" in SYSTEM_PROMPT


def test_the_instruction_says_an_owned_measure_list_is_not_a_menu():
    """The mechanical cause, named.

    ``get_capability_metadata`` hands the model every measure a capability owns
    — eight for pipeline stage movement, three for limit assessment — and
    nothing told it that list was an inventory. Four of the seven policy
    divergences are measure sets drawn from exactly those lists.
    """
    assert "INVENTORY, not a menu" in SYSTEM_PROMPT
    assert "asks to RECEIVE" in SYSTEM_PROMPT


def test_the_instruction_requires_explicit_semantics_to_be_preserved():
    """Q16 dropped an explicit product filter and planned anyway; Q21 dropped an
    explicit grouping and planned anyway. Either is worse than clarifying."""
    assert "PRESERVATION" in SYSTEM_PROMPT
    for element in ("measure", "statistic", "weight", "filter",
                    "geography basis", "temporal", "comparison", "target"):
        assert element in SYSTEM_PROMPT, f"preservation omits {element!r}"
    assert "never simply absent" in SYSTEM_PROMPT


def test_the_pre_submission_check_covers_all_three_directions():
    """Preservation, completeness and restraint are three different failures.

    Run 6 failed preservation (an explicit filter dropped). Run 7 failed
    completeness (a required measure never named) while passing restraint. A
    check that only asks "did I add too much?" cannot catch the second, which is
    why NECESSITY alone was not enough.
    """
    for check in ("PRESERVATION", "COMPLETENESS", "RESTRAINT"):
        assert check in SYSTEM_PROMPT, f"the submission check omits {check}"
    assert "Silently leaving it out" in SYSTEM_PROMPT
    assert "If it is neither" in SYSTEM_PROMPT


def test_the_instruction_guards_against_becoming_clarification_heavy():
    """A policy that turns restraint into timidity is not an improvement.

    The failure mode to avoid is converting optional enrichment into CLARIFY,
    which would look like discipline and behave like uselessness.
    """
    assert "Terseness is not ambiguity" in SYSTEM_PROMPT
    assert "materially different governed readings" in SYSTEM_PROMPT
    assert "faithful reading over an unnecessary question" in SYSTEM_PROMPT


def test_the_policy_does_not_ask_the_model_to_expose_its_reasoning():
    assert "do not narrate them" in SYSTEM_PROMPT.lower()


def test_the_policy_names_no_bank_question_and_no_lexical_rule():
    """Evidence sentinels are not training examples.

    A prompt that named Q16 or "drawdown" would be tuned to this bank and would
    tell us nothing about the next one.
    """
    lowered = SYSTEM_PROMPT.lower()
    for canonical in ("q08", "q10", "q16", "q21", "q24", "q25", "nl6", "nl7",
                      "sm09", "bb01"):
        assert canonical not in lowered, f"the prompt names {canonical!r}"
    # Bank-specific vocabulary the policy must not hard-code.
    for word in ("drawdown", "lump_sum", "lump sum", "acquired book",
                 "front book", "back book", "kfi", "itl3"):
        assert word not in lowered, f"the prompt hard-codes {word!r}"


def test_the_policy_reaches_the_model():
    """An instruction that is not in the payload is not a policy."""
    blocks = build_system_blocks(load_governed_vocabulary())
    assert any("MINIMUM SUFFICIENT GOVERNED INTENT" in b["text"] for b in blocks)
    assert any("PRESERVATION" in b["text"] for b in blocks)


# --------------------------------------------------------------------------- #
# nothing else moved
# --------------------------------------------------------------------------- #

def test_the_candidate_intent_schema_did_not_change():
    """One slot added, by instruction; everything else still pinned.

    `change_form` says which analytical question a change request is asking —
    material summary, metric delta, attribution or level comparison — and it
    exists because that distinction previously had no home. It was being spread
    across `capability`, `operation` and the MEASURE, three slots that can
    disagree, and on the signed-off corpus they did: one business question
    reached the compiler as three different contracts because three paraphrases
    chose three different governed measures and a specialist measure determines
    its owner.

    The slot is OPTIONAL, so every intent recorded before it still parses, and
    it is an ENUM, so it can no more carry a column or a snapshot than any other
    slot here. The schema version is unchanged for the same reason: an optional
    additive enum is backward compatible with every existing reading.
    """
    assert INTENT_SCHEMA_VERSION == "candidate_intent/1.0"
    schema = candidate_intent_json_schema()
    assert set(schema["properties"]) == {
        "schema_version", "capability", "operation", "change_form",
        "population", "measures", "dimensions", "filters", "geography", "time",
        "comparison", "target", "outputs", "ambiguity", "evidence"}
    assert schema["additionalProperties"] is False
    assert "change_form" not in schema["required"]
    assert schema["properties"]["change_form"]["enum"]


def test_the_plan_schema_did_not_change():
    assert PLAN_SCHEMA_VERSION == "governed_query_plan/1.0"


def test_the_metadata_tool_surface_is_the_authorised_one():
    """Seven retrieval tools, plus the one slice 3 was authorised to add.

    `get_source_portfolios` hands over the governed NAMES of the client's source
    portfolios and nothing else — no identifier, no row count, no date — from a
    registry the caller supplies per request. It reads; it fetches nothing.
    `test_the_interpreter_policy_did_not_move` is the guard that every OTHER
    tool's schema is still byte-identical, and
    `test_a_named_portfolio_reaches_the_model_as_a_name_and_nothing_else` is the
    guard on what this one may return.
    """
    names = [t["name"] for t in metadata_tool_schemas()]
    assert names == ["search_concepts", "get_concept_metadata",
                     "get_allowed_values", "search_capabilities",
                     "get_capability_metadata", "get_asset_metadata",
                     "get_portfolio_semantic_context", "get_source_portfolios"]


def test_the_governed_registries_did_not_change():
    """The policy is a prompt change. It may not quietly edit the estate."""
    import subprocess

    tracked = [str(p.relative_to(_REPO_ROOT)) for p in SOURCES.values()
               if p.exists() and _REPO_ROOT in p.parents or p.exists()]
    diff = subprocess.run(
        ["git", "diff", "--name-only", "ea8c65b", "--"] + tracked,
        cwd=_REPO_ROOT, capture_output=True, text=True)
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    assert changed == [], f"governed sources changed: {changed}"


def test_the_measured_policy_has_not_moved_since_it_was_measured():
    """The policy phase's guard, pointed at its own conclusion instead of its start.

    Until `5ae1f73` this test asserted the compiler had not moved since
    `28422ec`, which was the right guard while the interpreter was the thing
    being changed. Contract normalisation inverts that: the compiler and the
    contract ARE the subject now, and the interpreter is what must hold still.

    So the assertion is repointed rather than dropped. `5ae1f73` is the commit
    whose 57-question probe produced the behaviour the evidence files describe;
    if `opus_interpreter.py` moves after it, those numbers stop describing this
    code, and the next benchmark would be measuring two changes at once — which
    is the same failure the original guard existed to prevent.
    """
    import subprocess

    # Repointed once, at the slice 3 portfolio affordance. The interpreter moved
    # to describe the portfolio axes, so the 57-question probe at 5ae1f73 stopped
    # describing this code and the guard would otherwise let the NEXT change ride
    # in unmeasured beside it. `test_the_interpreter_policy_did_not_move` is what
    # constrains WHAT moved: 23 of the prompt's 26 paragraphs word for word, and
    # every pre-existing metadata tool schema byte-identical.
    measured_at = "88fdf7f9"
    diff = subprocess.run(
        ["git", "diff", "--name-only", measured_at, "--",
         "mi_agent/interpretation_v2/opus_interpreter.py"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("measured commit not reachable in this checkout")
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    assert changed == [], (
        f"the interpreter moved since its behaviour was measured at "
        f"{measured_at}: {changed}")


def test_production_surfaces_are_untouched():
    """/mi/query, React and Teams are not in this sprint."""
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--name-only", "ea8c65b", "--",
         "mi_agent_api", "frontend", "trakt_notifications", "mi_workflows",
         "engine", "analytics_lib", "trakt_tools"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    assert changed == [], f"production surfaces changed: {changed}"


def test_the_model_and_tool_configuration_did_not_change():
    from mi_agent.interpretation_v2.opus_interpreter import (
        CONFIGURED_MODEL, INTENT_TOOL_NAME, AnthropicInterpreterClient)

    assert CONFIGURED_MODEL == "claude-opus-5"
    assert INTENT_TOOL_NAME == "emit_candidate_intent"
    client = AnthropicInterpreterClient.__init__
    defaults = inspect.signature(client).parameters
    assert defaults["temperature"].default is None, (
        "temperature must stay unset so the benchmark configuration is the one "
        "Run 6 used")
    assert AnthropicInterpreterClient.max_rounds == 6


def test_the_policy_phase_changed_only_the_interpreter():
    """What the policy phase itself did, fixed as history rather than as a live guard.

    Measured between the phase's own endpoints, so it keeps saying something true
    after later sprints edit other modules. The live boundary for whatever sprint
    is in progress belongs to that sprint's own tests — for contract
    normalisation, `test_contract_normalisation.py`.
    """
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--name-status", "28422ec", "5ae1f73", "--",
         "mi_agent/interpretation_v2"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("phase commits not reachable in this checkout")
    modified = set()
    for line in diff.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        # Evidence files are written by benchmark runs, not by the policy change.
        if status.startswith("M") and "/evidence/" not in path:
            modified.add(path.strip())
    assert modified == {"mi_agent/interpretation_v2/opus_interpreter.py"}, modified
