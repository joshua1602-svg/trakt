"""Phase 8A — MI question interpretation harness tests.

Proves the deterministic baseline interpreter maps the controlled question set
onto validated MIQuerySpec v2, asks for clarification when ambiguous, that every
golden example holds, and that the evaluator catches mismatches. No external LLM
calls, no Azure/Streamlit/legacy-analytics imports, no Annex 2 changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent.interpreter import (
    InterpreterContext,
    evaluate_interpretation,
    interpret,
    load_golden,
)
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_spec_validation import validate_query_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = load_golden()


def _ctx(example) -> InterpreterContext:
    return InterpreterContext(**(example.get("context_overrides") or {}))


def _ids(prefix):
    return [f"{prefix}:{e['question'][:40]}" for e in GOLDEN]


# --------------------------------------------------------------------------- #
# 1. Golden dataset shape
# --------------------------------------------------------------------------- #


def test_golden_dataset_size():
    assert len(GOLDEN) >= 30
    valid = [e for e in GOLDEN if e["expected_valid"]]
    invalid = [e for e in GOLDEN if not e["expected_valid"]]
    assert len(valid) >= 20
    assert len(invalid) >= 10


# --------------------------------------------------------------------------- #
# 2. Every golden example holds (interpreter-graded or spec-graded)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ex", GOLDEN, ids=[e["question"][:45] for e in GOLDEN])
def test_golden_example(ex):
    if ex.get("interpreter_supported"):
        result = interpret(ex["question"], _ctx(ex))
        report = evaluate_interpretation(
            result,
            expected_spec=ex.get("expected_spec"),
            expected_valid=ex.get("expected_valid"),
            expected_issue_codes=ex.get("expected_issue_codes"),
            expected_clarification_required=ex.get(
                "expected_clarification_required", False),
        )
        assert report.passed, (ex["question"], report.mismatches)
    else:
        # Spec-graded: build the expected spec and check the validation contract.
        spec = MIQuerySpec.from_dict(ex["expected_spec"])
        vr = validate_query_spec(spec)
        assert vr.ok == ex["expected_valid"], (ex["question"], vr.codes())
        for code in (ex.get("expected_issue_codes") or []):
            assert code in vr.codes(), (ex["question"], code, vr.codes())


# --------------------------------------------------------------------------- #
# 3. Deterministic interpreter handles supported valid questions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ex", [e for e in GOLDEN
                                if e.get("interpreter_supported") and e["expected_valid"]],
                         ids=lambda e: e["question"][:45])
def test_supported_valid_questions_validate(ex):
    result = interpret(ex["question"], _ctx(ex))
    assert not result.clarification_required, ex["question"]
    assert result.ok, (ex["question"], result.issue_codes())
    # Generated spec passed through normalise + validate (never raw).
    assert result.normalized_spec is not None
    assert result.validation_result is not None and result.validation_result.ok


# --------------------------------------------------------------------------- #
# 4. Ambiguous questions require clarification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ex", [e for e in GOLDEN
                                if e.get("expected_clarification_required")],
                         ids=lambda e: e["question"][:45])
def test_ambiguous_questions_clarify(ex):
    result = interpret(ex["question"], _ctx(ex))
    assert result.clarification_required
    assert result.clarification_question
    assert not result.ok                 # a clarification is not a valid answer
    assert result.normalized_spec is None


# --------------------------------------------------------------------------- #
# 5. Invalid specs produce expected issue codes (spec-graded golden)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ex", [e for e in GOLDEN
                                if not e.get("interpreter_supported")],
                         ids=lambda e: e["question"][:45])
def test_spec_graded_invalids(ex):
    spec = MIQuerySpec.from_dict(ex["expected_spec"])
    vr = validate_query_spec(spec)
    assert vr.ok is False
    for code in ex["expected_issue_codes"]:
        assert code in vr.codes()


# --------------------------------------------------------------------------- #
# 6. Validation integration — interpreter never bypasses validation
# --------------------------------------------------------------------------- #


def test_interpreter_always_validates_supported_specs():
    for ex in GOLDEN:
        if ex.get("interpreter_supported") and ex["expected_valid"]:
            r = interpret(ex["question"], _ctx(ex))
            assert r.validation_result is not None  # validated, not raw


# --------------------------------------------------------------------------- #
# 7. Evaluator catches mismatches
# --------------------------------------------------------------------------- #


def test_evaluator_detects_spec_mismatch():
    r = interpret("show total funded", InterpreterContext())
    good = evaluate_interpretation(r, expected_spec={"state": "total_funded"},
                                   expected_valid=True)
    bad = evaluate_interpretation(r, expected_spec={"state": "total_pipeline"},
                                  expected_valid=True)
    assert good.passed and not bad.passed
    assert any("state" in m for m in bad.mismatches)


def test_evaluator_detects_unexpected_clarification():
    r = interpret("show risk", InterpreterContext())   # clarifies
    rep = evaluate_interpretation(r, expected_spec={"state": "total_funded"},
                                  expected_valid=True)
    assert not rep.passed


def test_evaluator_detects_missing_clarification():
    r = interpret("show total funded", InterpreterContext())  # does NOT clarify
    rep = evaluate_interpretation(r, expected_clarification_required=True)
    assert not rep.passed


# --------------------------------------------------------------------------- #
# 8. Guards
# --------------------------------------------------------------------------- #


def test_no_llm_or_forbidden_imports_in_interpreter():
    pkg = REPO_ROOT / "mi_agent" / "interpreter"
    banned = ("import openai", "import anthropic", "import streamlit",
              "import azure", "from azure", "from analytics ", "from analytics.")
    for py in pkg.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{py.name} contains {token!r}"
        for line in text.splitlines():
            s = line.strip()
            assert s != "import analytics" and not s.startswith("import analytics."), s


def test_no_regulatory_or_annex2_files_modified():
    import subprocess
    try:
        if subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--verify",
                           "main"], capture_output=True).returncode != 0:
            pytest.skip("no 'main' ref")
        diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--name-only",
                               "main"], capture_output=True, text=True,
                              check=True).stdout.split()
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status",
                                 "--porcelain"], capture_output=True, text=True,
                                check=True).stdout.splitlines()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"git not available: {exc}")
    changed = set(diff) | {ln[3:].strip() for ln in status if ln.strip()}
    bad_prefixes = ("config/regime/", "config/delivery/", "engine/gate_",
                    "engine/delivery_xml_agent/", "engine/projection_agent/")
    bad_substr = ("annex2", "annex_2", "annex12", "_xsd", ".xsd")
    for path in changed:
        low = path.lower()
        assert not any(low.startswith(p) for p in bad_prefixes), path
        assert not any(s in low for s in bad_substr), path
