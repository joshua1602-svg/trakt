"""The bank is verbatim, and this package is wired into nothing.

Two claims the final report makes, measured rather than asserted:

  * every one of the 135 questions is a character-for-character copy of a
    question already frozen in this repository;
  * nothing outside ``mi_agent/interpretation_v2/`` and
    ``tests/interpretation_v2/`` imports it, so no production behaviour can have
    changed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import yaml

from mi_agent.interpretation_v2.benchmark import iter_questions, load_bank

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = _REPO_ROOT / "mi_agent" / "interpretation_v2"


@pytest.fixture(scope="module")
def bank():
    return load_bank()


def test_the_bank_is_45_canonicals_of_3(bank):
    assert bank["canonical_count"] == 45
    assert bank["question_count"] == 135
    assert len(bank["canonicals"]) == 45
    for canonical in bank["canonicals"]:
        assert len(canonical["variants"]) == 3, canonical["id"]
    assert len(list(iter_questions(bank))) == 135


def test_every_canonical_id_is_unique(bank):
    ids = [c["id"] for c in bank["canonicals"]]
    assert len(ids) == len(set(ids))


def _frozen_questions() -> dict:
    """Every question string in the four frozen source banks, by source id."""
    found: dict = {}

    bank75 = yaml.safe_load(
        (_REPO_ROOT / "migration_phase0" / "MI_FINAL_ACCEPTANCE_75.yaml")
        .read_text(encoding="utf-8"))
    for case in bank75["cases"]:
        for formulation in case["formulations"]:
            found[formulation["id"]] = formulation["q"]

    movement = yaml.safe_load(
        (_REPO_ROOT / "tests" / "fixtures" / "mi_query_stage_movement"
         / "STAGE_MOVEMENT_BANK.yaml").read_text(encoding="utf-8"))
    for case in movement["cases"]:
        for formulation in case["formulations"]:
            found[formulation["id"]] = formulation["q"]

    borrowing = yaml.safe_load(
        (_REPO_ROOT / "migration_phase0" / "BORROWING_BASE_MI_BANK.yaml")
        .read_text(encoding="utf-8"))
    for question in borrowing["questions"]:
        found[question["id"]] = question["question"]

    spec = importlib.util.spec_from_file_location(
        "_nl_bank_check",
        _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
        / "nl_bank.py")
    nl_bank = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nl_bank)
    for intent, variations in nl_bank.VARIATIONS.items():
        for index, question in enumerate(variations, start=1):
            found[f"{intent}.{index}"] = question
    for row in nl_bank.q8_variations("alderbridge"):
        found[f"Q8.{row['pair']}.{row['variation']}"] = row["question"]

    return found


def test_every_question_is_verbatim_from_a_frozen_source(bank):
    frozen = _frozen_questions()
    for canonical in bank["canonicals"]:
        for variant in canonical["variants"]:
            source_id = variant["source_id"]
            assert source_id in frozen, (
                f"{canonical['id']}{variant['variant']} cites {source_id!r}, "
                "which no frozen bank carries")
            assert variant["question"] == frozen[source_id], (
                f"{canonical['id']}{variant['variant']} does not match its "
                f"source character for character")


def test_rebuilding_the_bank_reproduces_it_exactly():
    """The bank is assembled, never authored, so it must be reproducible."""
    from mi_agent.interpretation_v2.banks import build_bank_135

    rebuilt = build_bank_135.build()
    committed = load_bank()
    assert rebuilt == committed, (
        "the committed bank differs from what the builder produces — it was "
        "edited by hand")


def test_the_expectations_cover_every_canonical(bank):
    from mi_agent.interpretation_v2.benchmark import load_expectations

    expectations = load_expectations()
    missing = [c["id"] for c in bank["canonicals"] if c["id"] not in expectations]
    assert missing == [], f"canonicals with no expected intent: {missing}"


def test_a_contested_canonical_leaves_the_contested_dimension_unstated():
    """A human_review entry must not also assert the thing under review."""
    from mi_agent.interpretation_v2.benchmark import load_expectations
    from mi_agent.interpretation_v2 import SCORED_DIMENSIONS

    for canonical_id, entry in load_expectations().items():
        if not entry.get("human_review"):
            continue
        assert entry.get("note"), (
            f"{canonical_id} is marked for human review with no note saying why")
        stated = set(entry.get("expected", {})) & set(SCORED_DIMENSIONS)
        assert len(stated) < len(SCORED_DIMENSIONS), (
            f"{canonical_id} claims to be contested but states everything")


# --------------------------------------------------------------------------- #
# shadow only
# --------------------------------------------------------------------------- #

def test_nothing_outside_the_new_boundary_imports_it():
    """SHADOW ONLY. If production imported this, production behaviour changed."""
    offenders = []
    for path in _REPO_ROOT.rglob("*.py"):
        parts = path.relative_to(_REPO_ROOT).parts
        if parts[0] in ("node_modules", "frontend", ".git"):
            continue
        if path.is_relative_to(_PACKAGE):
            continue
        if parts[:2] == ("tests", "interpretation_v2"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any("interpretation_v2" in name for name in names):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], (
        "interpretation_v2 is imported outside its own boundary: " + ", ".join(
            sorted(set(offenders))))


def test_the_package_imports_no_executor_engine_or_route():
    """The boundary ends at the plan. Nothing here can run one."""
    forbidden = ("mi_query_executor", "query_plan_execution", "mi_agent_api",
                 "trakt_tools", "analytics_lib", "engine.", "mi_workflows",
                 "streamlit", "fastapi", "flask")
    offenders = []
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name.startswith(bad) or f".{bad}" in name
                       for bad in forbidden):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == [], offenders


def test_no_environment_variable_enables_this_in_production():
    """Nothing here reads a feature flag, because there is nothing to flag on."""
    import os

    sources = "\n".join(p.read_text(encoding="utf-8")
                        for p in _PACKAGE.rglob("*.py"))
    # ANTHROPIC_API_KEY is the model credential and is the only one permitted.
    read_vars = set()
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value.isupper() and node.value in os.environ.keys() | {
                        "ANTHROPIC_API_KEY", "ENABLE_LLM_MI_AGENT",
                        "MI_AGENT_LLM_PARSER", "MI_BEARER"}:
                read_vars.add(node.value)
    assert read_vars <= {"ANTHROPIC_API_KEY"}, read_vars
    assert "MI_BEARER" not in sources
