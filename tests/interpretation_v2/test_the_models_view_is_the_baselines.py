"""Nothing reaches the model without being measured (owner, 2026-10-02).

The sign-off run on 42fc3768 — must-answer 88/88, no wrong answer, 78 of 81
held-out variants identical — measured ONE model view: the system prompt,
the governed context and catalogue, the user message around the question,
the intent tool and the call's settings. The model's readings depend on every
character of it, so any change to it is a change to the model's behaviour.

This pins the view's fingerprint to the baseline's. A change that moves it
fails here — on purpose. To move the baseline: deploy the change, run the
sign-off set (`run_production_bank.sh <principal> signoff`), and only if it
matches or beats the results in config/mi/model_view_baseline.json record the new commit, run and
fingerprint there. A change that leaves the view alone (speed, wording of
answers, restructuring) passes, and is checked by the suites alone.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mi_agent.interpretation_v2.opus_interpreter import model_view_fingerprint

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "config/mi/model_view_baseline.json"


def _baseline():
    return json.loads(_BASELINE.read_text(encoding="utf-8"))


def test_the_models_view_is_the_signed_off_one():
    baseline = _baseline()
    assert model_view_fingerprint() == baseline["model_view_fingerprint"], (
        "The text the model is shown has changed since the signed-off build "
        f"{baseline['commit'][:8]}. Re-run the sign-off set on this change and "
        "move config/mi/model_view_baseline.json only if it matches or beats the baseline.")


def test_the_fingerprint_does_not_depend_on_the_process():
    """A fingerprint that moved between processes would guard nothing."""
    code = ("from mi_agent.interpretation_v2.opus_interpreter import "
            "model_view_fingerprint as f; print(f())")
    env = dict(os.environ, PYTHONHASHSEED="12345",
               PYTHONPATH=os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == _baseline()["model_view_fingerprint"]


def test_the_baseline_records_what_was_measured():
    baseline = _baseline()
    assert baseline["results"]["must_answer"] == {"questions": 88, "answered": 88}
    assert baseline["results"]["wrong_answers"] == 0
    assert len(baseline["commit"]) == 40


def test_the_conversation_readers_view_is_the_recorded_one():
    """The reader (§39) is the one other model step. Its view is pinned the
    same way: a change to it is measured by the conversation bank
    (`run_production_bank.sh <principal> conversations`) before it is used."""
    from mi_agent.interpretation_v2.conversation_reader import (
        reader_view_fingerprint)
    recorded = _baseline()["conversation_reader"]
    assert reader_view_fingerprint() == recorded["fingerprint"], (
        "The text the conversation reader is shown has changed. Re-run the "
        "conversation bank on this change and record it in "
        "config/mi/model_view_baseline.json only if it matches or beats the "
        "results there.")
