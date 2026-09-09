"""Fixtures for the semantic-recovery invariants.

The invariants are about WHO DECIDES a semantic fact, so they run the real
parser, the real interpretation builder and the real registry in process on the
platform canonical schema. Nothing here reaches a network or a run artefact.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def demo_env():
    """The demo book's environment, RESTORED when this file's tests finish.

    An earlier cut of this fixture mutated `os.environ` and left it that way,
    which changed the book that every LATER test file resolved: the estate's
    own census guard (`mi_agent/tests/test_semantic_census.py`) passed alone
    and failed when it ran after these tests, because it was censusing a
    different book. A test file that changes the meaning of the next one is
    the same defect this whole sprint is about, in the test suite.
    """
    warnings.simplefilter("ignore")
    before = dict(os.environ)
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "test")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        yield True
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture(scope="session")
def semantics(demo_env):
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


@pytest.fixture(scope="session")
def frame(demo_env):
    from mi_agent_api.data_source import get_dataframe
    return get_dataframe()


@pytest.fixture(scope="session")
def book_values(frame, semantics):
    from mi_agent_api import mi_service
    return mi_service._book_values(frame, semantics)


@pytest.fixture(scope="session")
def columns(frame):
    return set(frame.columns)


@pytest.fixture(scope="session")
def geography(frame):
    """The demo book's geography contract — RESOLVED, never INSTALLED.

    An earlier cut called `bind_geography`, which installs the contract for the
    remainder of the enclosing `geography_context`. With no enclosing context
    it stays installed for the whole PROCESS, and a session-scoped fixture
    cannot undo that in time: its teardown runs at the end of the entire
    pytest session, long after other files have run. The estate's census guard
    then censused the corpus with this book's contract in force and reported a
    movement ("by collateral region" binding `collateral_geography` instead of
    `geographic_region_collateral`) that was an artefact of the test run.

    Nothing here needs it installed: `ParsedQuestion.parse(geography=...)`
    opens its own context, and every other test passes the contract
    explicitly. Serving is unaffected either way — `mi_service` opens
    `geography_context(None)` at the outermost edge of every request, so a
    bound contract cannot outlive one.
    """
    from mi_agent_api import mi_service

    return mi_service._resolve_geography(None, None, frame)


@pytest.fixture
def parse(semantics, columns, book_values, geography):
    from mi_agent.parsed_question import ParsedQuestion

    def _parse(question: str):
        return ParsedQuestion.parse(question, semantics, geography=geography,
                                    available_columns=columns,
                                    available_values=book_values,
                                    llm_enabled=False)
    return _parse


@pytest.fixture
def two_basis_frame() -> pd.DataFrame:
    """A frame carrying BOTH geography bases, fully populated, so a chooser
    that reads column presence cannot be told apart from one that reads the
    contract — unless the contract says borrower and the chooser says
    collateral."""
    return pd.DataFrame({
        "loan_identifier": ["L1", "L2", "L3", "L4"],
        "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0],
        "collateral_geography": ["London", "London", "Wales", "Wales"],
        "geographic_region_collateral": ["TLI4", "TLI4", "TLL1", "TLL1"],
        "geographic_region_collateral_itl3": ["TLI43", "TLI43", "TLL11", "TLL11"],
        "geographic_region_obligor": ["Scotland", "Scotland", "Scotland", "London"],
        "geographic_region_obligor_itl3": ["TLM71", "TLM71", "TLM71", "TLI43"],
        "property_postcode": ["EC1A 1BB", "EC1A 1BB", "CF10 1AA", "CF10 1AA"],
    })


def frames_for(dates):
    """Ordered prepared-run frames as `evolution.funded_frames` returns them,
    on the dates given — deliberately IRREGULAR where the test needs them."""
    out = []
    for i, d in enumerate(dates):
        df = pd.DataFrame({
            "loan_identifier": [f"L{i}{j}" for j in range(3)],
            "current_outstanding_balance": [100.0 * (i + 1)] * 3,
            "collateral_geography": ["London", "Wales", "Scotland"],
        })
        out.append({"run_id": d, "reporting_date": d, "df": df, "source": f"tape_{d}"})
    return out


@pytest.fixture
def interpret(parse, semantics):
    """The deterministic `QuestionInterpretation` for a question, built by the
    ONE production construction site (`projection.from_parts`)."""
    from question_interpretation.projection import from_parts

    def _interpret(question: str):
        parsed = parse(question)
        return parsed, from_parts(question, spec=parsed.spec, facets=[],
                                  dim_terms=(), semantics=semantics)
    return _interpret


def contract_for(basis: str, frame):
    """A resolved geography contract whose primary basis is ``basis``, with the
    field decision made against ``frame`` — exactly what a request carries."""
    from mi_agent import mi_geography as G
    fields = G.resolved_basis_fields(frame=frame)
    return G.GeographyContract(
        primary_basis=basis, source=G.SOURCE_PORTFOLIO, asset_class=None,
        supported=tuple(b for b, f in fields if f), fields=fields)
