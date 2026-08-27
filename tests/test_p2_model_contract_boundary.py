"""tests/test_p2_model_contract_boundary.py — the model may not write the contract.

The free-form arm asks the model for a whole governed ``MIQuerySpec`` and
executes what comes back. Measured against the deterministic arm it chose
periods and metrics the reader never gave, dropped concepts the book cannot
express instead of refusing them, bound "lump sum" and "drawdown" to the wrong
governed fields, degraded an interest-rate BUCKET to the raw rate, turned
correct refusals into confident answers and correct answers into refusals.

That is not a prompt defect and no prompt fixes it. Every guard in the estate
reads the governed contract, so a contract the model wrote is checked against
itself. The arm is therefore withdrawn from serving rather than tuned, and its
replacement runs the other way round:

    question -> semantic proposal -> deterministic binding/merge
             -> the same governed contract -> the same guards

These tests hold that boundary shut. They are about WHO MAY WRITE THE CONTRACT,
not about whether a model is called: `concept_merge_arm` calls one, proposes
concepts in registered vocabulary, and lets the REGISTRY bind them to fields.

Run: python -m pytest tests/test_p2_model_contract_boundary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import concept_merge_arm as ARM  # noqa: E402
from mi_agent_api import datasets as D  # noqa: E402
from question_interpretation import concept_proposal as CP  # noqa: E402


# --------------------------------------------------------------------------- #
# The gate serving reads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env", [
    {},                                                     # nothing set
    {"ANTHROPIC_API_KEY": "present"},                       # F2: a key alone
    {"ANTHROPIC_API_KEY": "present", "MI_AGENT_LLM_PARSER": "on"},
    {"MI_AGENT_LLM_PARSER": "on"},                          # asked without a key
    {"ANTHROPIC_API_KEY": "present", "MI_AGENT_LLM_PARSER": "auto"},
])
def test_serving_never_enables_the_free_form_arm(monkeypatch, env):
    """No environment turns it on. F2 was that a KEY ALONE did."""
    for name in ("ANTHROPIC_API_KEY", "MI_AGENT_LLM_PARSER"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    cfg = D._mi_llm_config()
    assert cfg.enabled is False
    assert cfg.available is False


def test_an_operator_who_asked_for_it_is_told_what_happened(monkeypatch):
    """Withdrawn is not the same as ignored. A request is reported back."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "on")
    cfg = D._mi_llm_config()
    assert cfg.requested is True
    assert cfg.status == "withdrawn_unsafe_boundary"
    assert cfg.warnings and "withdrawn" in cfg.warnings[0]


def test_a_deployment_that_never_asked_reads_exactly_as_before(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    cfg = D._mi_llm_config()
    assert (cfg.enabled, cfg.available, cfg.status) == (False, False, "disabled")
    assert cfg.warnings == []


# --------------------------------------------------------------------------- #
# No second surface
# --------------------------------------------------------------------------- #
def test_no_serving_surface_selects_the_llm_parser_mode():
    """The two surfaces a person can reach: the API and the workbench.

    Everything else that runs the arm — the A/B harnesses, the calibration
    bank's `live_llm` switch — calls the parser directly and is a MEASUREMENT
    of the rejected architecture, not a request path. This asserts on the
    surfaces, which is where the distinction actually lives.
    """
    workbench = (_REPO_ROOT / "mi_agent" / "streamlit_mi_agent.py").read_text()
    assert 'use_llm, parser_mode = False, "deterministic"' in workbench
    assert '"Mode", ["Deterministic", "LLM"]' not in workbench

    import re

    service = (_REPO_ROOT / "mi_agent_api" / "mi_service.py").read_text()
    # The API takes its answer from the one gate above and from nothing else:
    # every `llm_enabled=` it passes is that config, or the local bound from it.
    sources = set(re.findall(r"llm_enabled=([A-Za-z_.]+)", service))
    assert sources <= {"llm_cfg.enabled", "llm_enabled"}, sources
    # `parser_mode` is chosen from the same gate, so it is now permanently
    # "deterministic"; what matters is that nothing sets it unconditionally.
    modes = set(re.findall(r'parser_mode=("llm"[^,\n]*)', service))
    assert modes <= {'"llm" if llm_enabled else "deterministic"'}, modes


# --------------------------------------------------------------------------- #
# What the replacement may and may not do
# --------------------------------------------------------------------------- #
def test_the_proposal_schema_has_no_slot_for_a_canonical_field():
    """2B, by construction: there is nowhere for the model to put one."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(CP.ProposedConcept)}
    assert fields == {"kind", "term", "covers", "comparator", "value"}
    # `term` is checked against the registered vocabulary, so a field name is
    # not a smuggling route: it is not a term, and an unregistered term is
    # rejected rather than approximated.
    assert "field" not in fields


def test_a_key_alone_does_not_turn_the_replacement_on_either(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.delenv("MI_AGENT_CONCEPT_MERGE", raising=False)
    assert ARM.enabled() is False
