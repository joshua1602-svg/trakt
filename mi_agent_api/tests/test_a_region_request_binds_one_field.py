#!/usr/bin/env python3
"""The live geography failure shapes, across every frame and lens.

THE SHAPE THIS REPRODUCES, from the production response for
"What is the funded balance in the London region" on ERE/2026-06-30:

    spec.filters {"collateral_geography": "London",
                  "canonical_region_reporting": "London"}
    validation   Canonical column 'canonical_region_reporting' not present
    error        'Region' is not available in this dataset

One concept, two physical predicates, one of them naming a column the execution
frame does not carry. The proposal below is the model's OWN, copied verbatim
from that response's `conceptMerge.proposed`, so this is the request that failed
rather than a reconstruction of it. The outbound call is stubbed: no model, no
credit, no network.

FOUR REQUIREMENTS, asserted for every frame and lens combination:

    exactly one physical region predicate
    the predicate's field exists in the execution frame
    dimension and filter resolve to the same field where both are present
    no requested geographic facet is left unaccounted

The frames are the two the estate actually has — the harmonised column present
or absent — plus both together, because which one leads is what the parse and
the arm disagreed about.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import llm_query_parser as LQ

CG = "collateral_geography"
CRR = "canonical_region_reporting"
REGION_FIELDS = (CG, CRR, "canonical_region_detail",
                 "geographic_region_collateral", "geographic_region_obligor")

#: The model's own proposal for this question, from the live response.
LONDON_PROPOSAL = (
    '{"concepts": [{"kind": "dataset", "term": "funded", "covers": "funded"},'
    ' {"kind": "measure", "term": "balance", "covers": "balance"},'
    ' {"kind": "category_value", "term": "london", "covers": "London"},'
    ' {"kind": "dimension", "term": "region", "covers": "region"}]}')

QUESTION = "What is the funded balance in the London region"


def _tape(columns, path):
    """A funded tape carrying `columns` as its region spelling(s), across both
    books so a lens genuinely narrows."""
    regions = ["London", "South West", "Scotland", "London", "South West"]
    books = ["direct_001", "acquired_001", "direct_001", "acquired_001",
             "direct_001"]
    types = ["direct", "acquired", "direct", "acquired", "direct"]
    frame = pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(len(regions))],
        "current_outstanding_balance": [100000.0, 200000.0, 300000.0,
                                        400000.0, 500000.0],
        "current_loan_to_value": [40.0, 45.0, 50.0, 55.0, 60.0],
        "source_portfolio_id": books,
        "source_portfolio_type": types,
        "reporting_date": ["2026-06-30"] * len(regions),
    })
    for column in columns:
        frame[column] = regions
    frame.to_csv(path, index=False)
    return frame


class _Arm:
    """The concept-merge arm on, with its one outbound call replaced."""

    def __init__(self, reply):
        self._reply = reply

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("MI_AGENT_CONCEPT_MERGE", "ANTHROPIC_API_KEY")}
        os.environ["MI_AGENT_CONCEPT_MERGE"] = "on"
        os.environ["ANTHROPIC_API_KEY"] = "sk-not-used-the-call-is-replaced"
        self._original = LQ._call_llm
        LQ._call_llm = lambda *a, **k: (self._reply, {}, False)
        return self

    def __exit__(self, *exc):
        LQ._call_llm = self._original
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def _ask(columns, lens=None, question=QUESTION):
    from trakt_core.context import ExecutionContext
    from mi_agent_api import data_source
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    tmp = Path(tempfile.mkdtemp()) / "tape.csv"
    frame = _tape(columns, tmp)
    os.environ["MI_AGENT_DATA_CSV"] = str(tmp)
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["TRAKT_RUNTIME_MODE"] = "test"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    data_source.reset_cache()
    with _Arm(LONDON_PROPOSAL):
        result = execute_governed_mi_query(
            MiQueryRequest(question=question, source_portfolio_lens=lens),
            context=ExecutionContext.for_internal("ERE"))
    data_source.reset_cache()
    envelope = result.result if not isinstance(result, dict) else result
    return (envelope or {}), frame


_FRAMES = {"collateral_geography only": (CG,),
           "canonical_region_reporting only": (CRR,),
           "both present": (CG, CRR)}
_LENSES = {"total": None, "direct": "direct", "acquired": "acquired"}


class TestOneRegionPredicatePerRequest(unittest.TestCase):

    def _region_filters(self, envelope):
        filters = ((envelope.get("spec") or {}).get("filters") or {})
        return {k: v for k, v in filters.items() if k in REGION_FIELDS}

    def test_every_frame_and_lens_binds_exactly_one_region_field(self):
        for frame_name, columns in _FRAMES.items():
            for lens_name, lens in _LENSES.items():
                with self.subTest(frame=frame_name, lens=lens_name):
                    envelope, frame = _ask(columns, lens)
                    region = self._region_filters(envelope)
                    self.assertEqual(
                        len(region), 1,
                        "%s / %s bound %s" % (frame_name, lens_name, region))
                    field = next(iter(region))
                    self.assertIn(field, set(frame.columns),
                                  "bound a column the frame does not carry")

    def test_the_answer_stands_rather_than_refusing_on_an_absent_column(self):
        for frame_name, columns in _FRAMES.items():
            for lens_name, lens in _LENSES.items():
                with self.subTest(frame=frame_name, lens=lens_name):
                    envelope, _ = _ask(columns, lens)
                    self.assertNotIn(
                        "is not available in this dataset",
                        str(envelope.get("error") or ""),
                        "%s / %s" % (frame_name, lens_name))

    def test_dimension_and_filter_resolve_to_the_same_field(self):
        for frame_name, columns in _FRAMES.items():
            with self.subTest(frame=frame_name):
                envelope, _ = _ask(columns)
                spec = envelope.get("spec") or {}
                dimension = spec.get("dimension")
                region = self._region_filters(envelope)
                if dimension in REGION_FIELDS and region:
                    self.assertEqual(dimension, next(iter(region)),
                                     "grouped on one region field, filtered on another")

    def test_no_geographic_facet_is_left_unaccounted(self):
        for frame_name, columns in _FRAMES.items():
            with self.subTest(frame=frame_name):
                envelope, _ = _ask(columns)
                coverage = ((envelope.get("metadata") or {})
                            .get("semanticCoverage") or {})
                self.assertEqual(coverage.get("unaccounted") or [], [])

    def test_the_lens_does_not_change_what_region_means(self):
        """A portfolio lens narrows the population; it must not move the
        concept onto a different physical field."""
        for frame_name, columns in _FRAMES.items():
            fields = set()
            for lens in _LENSES.values():
                envelope, _ = _ask(columns, lens)
                fields |= set(self._region_filters(envelope))
            with self.subTest(frame=frame_name):
                self.assertLessEqual(len(fields), 1, fields)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestTheTwoFramesThatDisagreed(unittest.TestCase):
    """THE PRODUCTION CONDITION, reproduced at the seam where it arises.

    The tests above share one frame between the parse and the arm, so both
    resolve "London" to the same field and the defect cannot appear — which is
    why they pass on the unfixed code too. They pin the invariant; they do not
    reproduce the bug.

    In production the two do NOT share a frame. The parse binds against
    `mi_service._resolve_frame(view, portfolio_id)`; the arm binds against
    `chat_routing._values_for_recognition()` -> `base_frame_resolver`, resolved
    separately. On ERE/2026-06-30 they disagreed about which region columns
    exist, so the parse bound `collateral_geography` and the arm bound
    `canonical_region_reporting` — a column the execution frame does not carry.

    Handing the arm a catalogue from one frame and the columns of another is
    exactly that condition, and it is the shape that must never write a second
    predicate.
    """

    def _semantics(self):
        from mi_agent.mi_query_validator import load_mi_semantics

        return load_mi_semantics(_REPO_ROOT / "mi_agent" /
                                 "mi_semantics_field_registry.yaml")

    def _apply_with_divergent_frames(self):
        from mi_agent_api import concept_merge_arm as arm
        from mi_agent.mi_query_spec import MIQuerySpec
        from question_interpretation import projection

        semantics = self._semantics()
        # What the PARSE produced, against the execution frame.
        spec = MIQuerySpec(intent="summary", metric="current_outstanding_balance",
                           aggregation="sum", filters={CG: "London"})
        interpretation = projection.from_parts(
            QUESTION, spec=spec, facets=[], dim_terms=[], semantics=semantics)
        with _Arm(LONDON_PROPOSAL):
            evidence = arm.apply(
                QUESTION, spec, semantics, interpretation=interpretation,
                # THE OTHER FRAME'S catalogue ...
                available_values={CRR: {"london": "London"}},
                # ... and THIS frame's columns.
                available_columns={CG, "current_outstanding_balance"})
        return spec, (evidence or {})

    def test_the_arm_does_not_write_a_second_region_predicate(self):
        spec, _ = self._apply_with_divergent_frames()
        region = {k: v for k, v in (spec.filters or {}).items()
                  if k in REGION_FIELDS}
        self.assertEqual(region, {CG: "London"},
                         "the arm added a second spelling of one concept")

    def test_it_never_writes_a_field_the_execution_frame_lacks(self):
        spec, _ = self._apply_with_divergent_frames()
        self.assertNotIn(CRR, spec.filters or {})

    def test_and_it_says_what_it_did_rather_than_going_quiet(self):
        _, evidence = self._apply_with_divergent_frames()
        self.assertTrue(evidence, "the arm reported nothing at all")
        self.assertEqual(evidence.get("applied") or [], [])
