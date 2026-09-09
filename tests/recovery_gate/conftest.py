"""Adversarial fixtures for the RECOVERY GATE.

THE GATE EXISTS BECAUSE THE OFFLINE ESTATE PASSED A BUILD THAT BROKE LIVE.

`fe7fea03` cleared every focused test, every vertical suite and the semantic
census, and then lost 11 CORRECT answers against the frozen 135-question bank.
Every one of those losses had the same shape: a fixture book that was FRIENDLIER
than the live book, so the consolidation under test was never asked the question
that breaks it.

The fixtures here are deliberately UNFRIENDLY, and each one is unfriendly in the
specific way the live book turned out to be:

  * ITL3 code columns whose values are CODES, not place names — the live book's
    codes are not in the region ladder's vocabulary, the demo book's are, and
    that difference alone decides whether the geography surface is available;
  * a governed period series with an IRREGULAR gap far beyond
    `max_snapshot_gap_days` — the live book observes 2025-11 then 2026-06, and
    every fixture book observes consecutive months;
  * an earlier snapshot that cannot carry a metric the later one carries, so
    "how many metrics could be compared" is genuinely zero.

NOTHING HERE REACHES A NETWORK, A MODEL OR A RUN ARTEFACT, and nothing here
mutates process state that outlives the test. Two defects in an earlier sprint's
fixtures were caused by exactly that — a session-scoped `os.environ` update that
restored after other files had already run, and a geography contract installed
process-wide — so the environment is set per-test and restored, and no contract
is ever bound.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

#: The governed source portfolios the live book carries (five of them), so a
#: summary answer exercises the cohort split rather than the single-cohort
#: shortcut.
PORTFOLIOS = ("alp_origination", "alp_acquired", "spv1_sponsored",
              "spv2_sponsored", "legacy_book")

#: Place names the region ladder recognises.
REGION_NAMES = ("South East", "London", "Wales")

#: REAL ITL3 CODES THE REGION LADDER DOES NOT RECOGNISE AS PLACES.
#:
#: Every one of these is a valid ITL3 code and none of them reads as a place, so
#: a chooser that asks "do these values look like geography?" says no — which is
#: precisely the live book's situation.
ITL3_CODES_UNRECOGNISED = ("TLK12", "TLI31", "TLC22")

#: ITL3 codes the ladder DOES accept, all of them equally valid codes.
#:
#: THE DIFFERENCE BETWEEN THESE TWO TUPLES IS THE WHOLE DEFECT. `TLC22` is
#: rejected and `TLD33` accepted; `TLK12` rejected and `TLK41` accepted. The
#: ladder probes a HEAD SAMPLE and returns true if ANY value looks like a place,
#: so whether a book's geography surface exists at all comes down to which
#: arbitrary areas happen to appear in the first rows of its tape.
ITL3_CODES_RECOGNISED = ("TLG31", "TLF14", "TLL52")

#: Postcodes the in-repo ITL3 master lookup cannot resolve.
POSTCODES_UNMATCHABLE = ("ZZ99 9ZZ", "ZZ99 9ZY", "ZZ99 9ZX")


@pytest.fixture(autouse=True)
def _deterministic_environment(monkeypatch):
    """Parser and auth settings for one test, restored by monkeypatch.

    Per-test, never session-scoped: a session-scoped environment fixture
    restores at the END OF THE WHOLE SESSION, long after other files have run,
    which is how `MI_AGENT_AUTH_ENABLED=false` once leaked into another file's
    auth-guard test.
    """
    warnings.simplefilter("ignore")
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("MI_AGENT_LLM_PARSER", "off")
    monkeypatch.setenv("MI_AGENT_LLM_ENABLED", "0")
    yield


@pytest.fixture(scope="session")
def semantics():
    from pathlib import Path

    from mi_agent.mi_query_validator import load_mi_semantics

    root = Path(__file__).resolve().parents[2]
    return load_mi_semantics(root / "mi_agent" / "mi_semantics_field_registry.yaml")


def funded_frame(*, rows: int, balance: float, origination: str,
                 geography: Optional[Dict[str, Sequence[Any]]] = None,
                 drop: Sequence[str] = (),
                 balance_is_null: bool = False) -> pd.DataFrame:
    """One governed funded snapshot, with exactly the geography asked for.

    `geography` maps column name to the cycle of values that column carries, so
    a test states the book's geography shape in one literal rather than
    describing it in prose.
    """
    geography = geography or {}
    records: List[Dict[str, Any]] = []
    for i in range(rows):
        record: Dict[str, Any] = {
            "loan_identifier": f"L{i:05d}",
            "current_outstanding_balance": (np.nan if balance_is_null
                                            else balance / max(rows, 1)),
            "current_loan_to_value": 0.30 + (i % 5) * 0.10,
            "current_interest_rate": 6.0,
            "youngest_borrower_age": 71.0,
            "origination_date": origination,
            "source_portfolio_id": PORTFOLIOS[i % len(PORTFOLIOS)],
            "source_portfolio_label": PORTFOLIOS[i % len(PORTFOLIOS)].upper(),
        }
        for column, values in geography.items():
            record[column] = values[i % len(values)]
        records.append(record)
    frame = pd.DataFrame(records)
    return frame.drop(columns=[c for c in drop if c in frame.columns])


def install_series(monkeypatch, series: Sequence[Dict[str, Any]]) -> None:
    """Make `evolution.funded_frames` serve exactly ``series``.

    Every consumer under gate — the summary, the movement, the bridge — reads
    its periods from this one function, so replacing it is enough to put a whole
    governed history in front of the real routes.
    """
    from mi_agent_api import evolution as evolution_mod

    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: [dict(f) for f in series])


def snapshot(run_id: str, frame: pd.DataFrame) -> Dict[str, Any]:
    return {"run_id": run_id, "reporting_date": run_id, "df": frame,
            "source": f"blob://gate/{run_id}/platform_canonical_typed.csv"}


def route(question: str, frame: pd.DataFrame, semantics) -> Optional[Dict[str, Any]]:
    """Ask the REAL routing entry point, exactly as the service does.

    `try_route` is where a claimed route's exception becomes the governed
    execution-failure envelope, so a gate that called a handler directly could
    not see the failure mode it exists to catch.
    """
    from mi_agent_api import chat_routing

    return chat_routing.try_route(
        question, portfolio_id="gate/2026-06-30", view="funded",
        output_root="blob://gate", pipeline_root=None, semantics=semantics,
        frame_resolver=lambda _view, _pid: frame,
        base_frame_resolver=lambda _view, _pid: frame)


def raised(envelope: Optional[Dict[str, Any]]) -> bool:
    """Did the claimed route fail while running?

    `metadata.executionFailure` is set by `chat_routing._execution_failure_envelope`
    and by nothing else, so it is the one honest signal that a route claimed the
    question and then raised.
    """
    if not isinstance(envelope, dict):
        return False
    return bool((envelope.get("metadata") or {}).get("executionFailure"))
