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
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "test")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    return True


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
    from mi_agent import llm_query_parser as P
    from mi_agent_api import mi_service
    return P.bind_geography(mi_service._resolve_geography(None, None, frame))


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
