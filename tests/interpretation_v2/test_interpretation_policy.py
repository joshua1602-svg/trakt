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
    assert INTENT_SCHEMA_VERSION == "candidate_intent/1.0"
    schema = candidate_intent_json_schema()
    assert set(schema["properties"]) == {
        "schema_version", "capability", "operation", "population", "measures",
        "dimensions", "filters", "geography", "time", "comparison", "target",
        "outputs", "ambiguity", "evidence"}
    assert schema["additionalProperties"] is False


def test_the_plan_schema_did_not_change():
    assert PLAN_SCHEMA_VERSION == "governed_query_plan/1.0"


def test_the_metadata_tool_surface_did_not_change():
    names = [t["name"] for t in metadata_tool_schemas()]
    assert names == ["search_concepts", "get_concept_metadata",
                     "get_allowed_values", "search_capabilities",
                     "get_capability_metadata", "get_asset_metadata",
                     "get_portfolio_semantic_context"]


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


def test_the_compiler_and_contract_modules_are_untouched_by_this_phase():
    """Measured against the commit this phase started from.

    A policy sprint that edited the compiler would be a different sprint, and
    its benchmark would measure two changes at once.
    """
    import subprocess

    start = "28422ec"
    guarded = [
        "mi_agent/interpretation_v2/intent.py",
        "mi_agent/interpretation_v2/compiler.py",
        "mi_agent/interpretation_v2/plan.py",
        "mi_agent/interpretation_v2/equivalence.py",
        "mi_agent/interpretation_v2/metadata.py",
        "mi_agent/interpretation_v2/vocabulary.py",
        "mi_agent/interpretation_v2/outcomes.py",
        "mi_agent/interpretation_v2/banks/interpretation_bank_135.yaml",
        "mi_agent/interpretation_v2/banks/expected_intents.yaml",
    ]
    diff = subprocess.run(["git", "diff", "--name-only", start, "--"] + guarded,
                          cwd=_REPO_ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    assert changed == [], (
        f"Phase 2A is policy only, but these moved since {start}: {changed}")


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


def test_the_only_MODIFIED_package_module_is_the_interpreter():
    """New analysis tooling is fine; editing an existing module is not.

    The distinction matters for reading the benchmark: a probe script added
    beside the package cannot change what the model is asked or what the
    compiler decides, whereas an edit to any existing module would mean the run
    measured two changes at once.
    """
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--name-status", "28422ec", "--",
         "mi_agent/interpretation_v2"],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("start commit not reachable in this checkout")
    modified = set()
    for line in diff.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        # Evidence files are written by benchmark runs, not by the policy change.
        if status.startswith("M") and "/evidence/" not in path:
            modified.add(path.strip())
    assert modified == {"mi_agent/interpretation_v2/opus_interpreter.py"}, modified
