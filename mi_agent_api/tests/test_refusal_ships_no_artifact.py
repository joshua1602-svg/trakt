"""A declined answer states itself once, and puts nothing in the workspace.

Reported defect: a query the MI Agent would not answer produced TWO renderings
of the same sentence — the refusal in the chat, and a workspace card titled
"Query Validation" restating it as a blocker.

Worse than the duplication: it made the refusal look answered. The browser
flagged an error message with ``!ok && artifacts.length === 0``, so shipping one
artifact was enough to render a declined answer in the ordinary answer styling —
no error colour, no Retry — for a question that was not answered.

The chart, table and KPI were already gated on ``refused``; the validation
artifact was not. It is now. Validation issues on a SUCCESSFUL answer still
surface exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import adapters  # noqa: E402


def _workflow(**over):
    base = {
        "ok": True,
        "question": "balance by region",
        "answer": "London has the largest balance.",
        "interpreted": {},
        "spec": {"metric": "current_outstanding_balance",
                 "dimension": "geographic_region_obligor"},
        "validation": {"ok": True, "errors": [], "warnings": []},
        "query_result": None,
        "chart_result": None,
        "warnings": [],
        "diagnostics": [],
    }
    base.update(over)
    return base


def _artifacts(workflow):
    env = adapters.adapt_workflow_result(workflow, portfolio_id="client_001")
    return env, env.get("artifacts") or []


def _types(artifacts):
    return sorted(a.get("type") for a in artifacts)


# --------------------------------------------------------------------------- #
# The defect
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("refusal", [
    {"ok": False, "error": "That dimension is not governed for this book."},
    {"ok": True, "semantic_refusal": True,
     "error": "I have not substituted a broader figure."},
    {"ok": True, "controlled_refusal": True,
     "error": "That concept is not supported for this portfolio."},
])
def test_a_refusal_ships_no_artifact_even_when_validation_objects(refusal):
    """Every refusal shape, with validation carrying the same sentence."""
    wf = _workflow(validation={"ok": False,
                               "errors": [refusal["error"]],
                               "warnings": []},
                   **refusal)
    _env, artifacts = _artifacts(wf)
    assert artifacts == [], f"a refusal shipped {_types(artifacts)}"


def test_the_refusal_would_otherwise_have_read_as_an_answer():
    """The browser flags an error with `!ok && artifacts.length === 0`, so one
    artifact was enough to dress a refusal as an answer. Pins the count the
    browser actually reads."""
    wf = _workflow(ok=False, error="Not a governed dimension for this book.",
                   validation={"ok": False,
                               "errors": ["Not a governed dimension for this book."],
                               "warnings": []})
    env, artifacts = _artifacts(wf)
    assert env["ok"] is False
    assert len(artifacts) == 0


def test_a_refusal_still_says_why():
    """Nothing is hidden — the reason travels in the answer, once."""
    wf = _workflow(ok=False, error="That dimension is not governed for this book.",
                   answer=None,
                   validation={"ok": False, "errors": ["bad dimension"], "warnings": []})
    env, _ = _artifacts(wf)
    assert "not governed" in (env["answer"] or "")
    assert env["error"] == "That dimension is not governed for this book."


def test_validation_warnings_alone_are_also_suppressed_on_a_refusal():
    wf = _workflow(ok=False, error="Refused.",
                   validation={"ok": True, "errors": [],
                               "warnings": ["region resolved loosely"]})
    _env, artifacts = _artifacts(wf)
    assert artifacts == []


# --------------------------------------------------------------------------- #
# A successful answer is untouched
# --------------------------------------------------------------------------- #
def test_a_successful_answer_still_carries_its_validation_artifact():
    wf = _workflow(validation={"ok": True, "errors": [],
                               "warnings": ["region resolved via NUTS 2024"]})
    _env, artifacts = _artifacts(wf)
    assert "validation" in _types(artifacts)


def test_a_clean_successful_answer_carries_no_validation_artifact():
    """Unchanged: nothing to report means no card, refusal or not."""
    _env, artifacts = _artifacts(_workflow())
    assert "validation" not in _types(artifacts)
