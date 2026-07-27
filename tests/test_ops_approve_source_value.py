#!/usr/bin/env python3
"""tests/test_ops_approve_source_value.py

The operator CLI for approving a detected source-value normalisation.

``ops approve`` approves a source MAPPING and requires a ``--mapping-id``. A
source-value treatment is a different kind of statement — what one value means,
in one column, for one source, under one target contract — and has no mapping
id; fabricating one would push a bogus mapping into the registry on promotion.
``ops approve-source-value`` is that path.

It fails closed. Approving the wrong value, or the right value in the wrong
scope, is worse than not approving at all, so every ambiguity is an error with a
reason rather than a best guess.

Run: python -m pytest tests/test_ops_approve_source_value.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import decisions_bridge as DB
from apps.blob_trigger_app import ops as OPS
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.storage import Storage
from engine.onboarding_agent import source_value_rules as svr

PACK_KEY = "ERE_acquired_001_funded_ad_hoc_2026-06-30"
MI = "mi_semantics_field_registry"
A2 = "esma_annex_2"


def _candidate(**over):
    c = {
        "target_field": "borrower_1_DOB",
        "canonical_field": "borrower_1_DOB",
        "source_file": "AcquiredLoanTape.csv",
        "source_sheet": "",
        "source_column": "Borrower 1 DOB",
        "source_value": "TBC",
        "affected_row_count": 2,
        "canonical_format": "date",
        "target_contract_id": MI,
        "client_id": "ERE",
        "source_portfolio_id": "acquired_001",
        "source_schema_fingerprint": "sha256:abc",
    }
    c.update(over)
    return c


class _Ctx:
    """A state store holding one pack's persisted 28d document."""

    def __init__(self, root: Path, candidates=None, rules=None):
        self.storage = Storage(root)
        self.layout = Layout()
        self.persistence = ProductionPersistence(self.storage, self.layout)
        self.storage.write_text(
            self.layout.run_onboarding_uri(PACK_KEY, "28d_source_value_rules.json"),
            json.dumps({
                "target_contract_id": MI, "client_id": "ERE",
                "source_portfolio_id": "acquired_001",
                "source_schema_fingerprint": "sha256:abc",
                "rules": rules if rules is not None else [],
                "candidates_pending": (candidates if candidates is not None
                                       else [_candidate()]),
            }))

    def approve(self, **kw):
        params = {"canonical_field": "borrower_1_DOB",
                  "source_column": "Borrower 1 DOB",
                  "source_value": "TBC", "approved_by": "ops"}
        params.update(kw)
        return OPS.approve_source_value(self.persistence, PACK_KEY, **params)

    def approved_doc(self):
        uri = self.layout.run_onboarding_uri(
            PACK_KEY, "34_target_first_decisions_approved.yaml")
        if not self.storage.exists(uri):
            return None
        return yaml.safe_load(self.storage.read_text(uri))


class _Base(unittest.TestCase):
    def _ctx(self, **kw):
        td = tempfile.TemporaryDirectory(prefix="ops_svr_")
        self.addCleanup(td.cleanup)
        return _Ctx(Path(td.name), **kw)


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
class TestSuccessfulApproval(_Base):

    def test_exact_match_is_approved(self):
        ctx = self._ctx()
        summary = ctx.approve()
        decision = summary["decision"]
        self.assertEqual(decision["selected_action"], svr.ACTION_TREAT_AS_NULL)
        self.assertEqual(decision["status"], "approved")
        self.assertEqual(decision["source_value"], "TBC")
        self.assertEqual(decision["affected_row_count"], 2)
        self.assertEqual(decision["approved_by"], "ops")

    def test_the_approved_decision_carries_the_candidate_scope_verbatim(self):
        # Requirement: preserve the explicit scope; never re-derive it later.
        decision = self._ctx().approve()["decision"]
        self.assertEqual(decision["target_contract_id"], MI)
        self.assertEqual(decision["client_id"], "ERE")
        self.assertEqual(decision["source_portfolio_id"], "acquired_001")
        self.assertEqual(decision["source_schema_fingerprint"], "sha256:abc")
        self.assertEqual(decision["source_file"], "AcquiredLoanTape.csv")

    def test_it_is_written_to_the_pack_scoped_approved_artefact(self):
        ctx = self._ctx()
        summary = ctx.approve()
        self.assertTrue(summary["approved_decisions_uri"].endswith(
            "34_target_first_decisions_approved.yaml"))
        doc = ctx.approved_doc()
        self.assertEqual(len(doc["decisions"]), 1)
        self.assertEqual(doc["decisions"][0]["selected_action"],
                         svr.ACTION_TREAT_AS_NULL)

    def test_no_mapping_id_is_required_or_invented(self):
        decision = self._ctx().approve()["decision"]
        self.assertNotIn("mapping_id", decision)
        self.assertNotIn("suggested_mapping_id", decision)

    def test_scope_pins_that_agree_are_accepted(self):
        summary = self._ctx().approve(scope={
            "target_contract_id": MI, "client_id": "ERE",
            "source_portfolio_id": "acquired_001",
            "source_file": "AcquiredLoanTape.csv"})
        self.assertEqual(summary["decision"]["source_value"], "TBC")

    def test_value_matching_ignores_case_and_padding(self):
        summary = self._ctx().approve(source_value="  tbc ")
        # The ORIGINAL spelling from the candidate is what gets recorded.
        self.assertEqual(summary["decision"]["source_value"], "TBC")

    def test_re_approving_is_idempotent(self):
        ctx = self._ctx()
        ctx.approve()
        summary = ctx.approve()
        self.assertTrue(summary["already_approved"])
        self.assertEqual(len(ctx.approved_doc()["decisions"]), 1)

    def test_existing_mapping_approvals_are_preserved(self):
        # Must not disturb the existing approval path.
        ctx = self._ctx()
        uri = ctx.layout.run_onboarding_uri(
            PACK_KEY, "34_target_first_decisions_approved.yaml")
        ctx.storage.write_text(uri, yaml.safe_dump({
            "version": 1,
            "decisions": [{"decision_id": "DQ-001", "status": "approved",
                           "selected_action": "confirm_selected",
                           "target_field": "current_interest_rate"}]}))
        ctx.approve()
        actions = [d["selected_action"] for d in ctx.approved_doc()["decisions"]]
        self.assertEqual(actions, ["confirm_selected", svr.ACTION_TREAT_AS_NULL])

    def test_approval_is_recorded_on_the_run_record_when_one_exists(self):
        ctx = self._ctx()
        from apps.blob_trigger_app import run_records as RR
        RR.write_run_record(ctx.storage, ctx.layout, {
            "pack_key": PACK_KEY, "client_id": "ERE",
            "source_portfolio_id": "acquired_001",
            "dataset": "funded", "frequency": "ad_hoc"})
        ctx.approve()
        rec = ctx.persistence.load_run_record(PACK_KEY)
        self.assertEqual(len(rec["source_value_decisions"]), 1)
        self.assertEqual(rec["source_value_decisions"][0]["source_value"], "TBC")


# --------------------------------------------------------------------------- #
# Failing closed
# --------------------------------------------------------------------------- #
class TestFailsClosed(_Base):

    def _refuses(self, ctx, **kw):
        with self.assertRaises(svr.CandidateMatchError) as raised:
            ctx.approve(**kw)
        self.assertIsNone(ctx.approved_doc(), "nothing may be written on failure")
        return str(raised.exception)

    def test_no_matching_candidate(self):
        message = self._refuses(self._ctx(), source_value="NOT-DETECTED")
        self.assertIn("no pending candidate", message)

    def test_no_matching_column(self):
        message = self._refuses(self._ctx(), source_column="Borrower 2 DOB")
        self.assertIn("no pending candidate", message)

    def test_no_matching_canonical_field(self):
        message = self._refuses(self._ctx(), canonical_field="borrower_2_DOB")
        self.assertIn("no pending candidate", message)

    def test_ambiguous_match_across_files(self):
        ctx = self._ctx(candidates=[
            _candidate(source_file="AcquiredLoanTape.csv"),
            _candidate(source_file="AcquiredLoanTape_supplement.csv")])
        message = self._refuses(ctx)
        self.assertIn("2 candidates match", message)
        self.assertIn("--source-file", message)

    def test_an_ambiguous_match_can_be_narrowed_by_file(self):
        ctx = self._ctx(candidates=[
            _candidate(source_file="AcquiredLoanTape.csv"),
            _candidate(source_file="AcquiredLoanTape_supplement.csv")])
        summary = ctx.approve(scope={"source_file": "AcquiredLoanTape.csv"})
        self.assertEqual(summary["decision"]["source_file"], "AcquiredLoanTape.csv")

    def test_unsupported_action(self):
        message = self._refuses(self._ctx(), action="delete_the_row")
        self.assertIn("unsupported action", message)
        self.assertIn(svr.ACTION_TREAT_AS_NULL, message)

    def test_wrong_target_contract(self):
        message = self._refuses(self._ctx(), scope={"target_contract_id": A2})
        self.assertIn("scope differs", message)
        self.assertIn("target_contract_id", message)

    def test_wrong_client(self):
        message = self._refuses(self._ctx(), scope={"client_id": "OTHER"})
        self.assertIn("scope differs", message)
        self.assertIn("client_id", message)

    def test_wrong_source_portfolio(self):
        message = self._refuses(self._ctx(),
                                scope={"source_portfolio_id": "direct_001"})
        self.assertIn("scope differs", message)
        self.assertIn("source_portfolio_id", message)

    def test_wrong_schema_fingerprint(self):
        message = self._refuses(self._ctx(),
                                scope={"source_schema_fingerprint": "sha256:zzz"})
        self.assertIn("scope differs", message)

    def test_no_candidates_recorded_at_all(self):
        ctx = self._ctx(candidates=[])
        self._refuses(ctx)

    def test_pack_with_no_28d_artefact(self):
        td = tempfile.TemporaryDirectory(prefix="ops_svr_none_")
        self.addCleanup(td.cleanup)
        storage = Storage(Path(td.name))
        layout = Layout()
        persistence = ProductionPersistence(storage, layout)
        with self.assertRaises(svr.CandidateMatchError) as raised:
            OPS.approve_source_value(
                persistence, PACK_KEY, canonical_field="borrower_1_DOB",
                source_column="Borrower 1 DOB", source_value="TBC")
        self.assertIn("no source-value candidates recorded", str(raised.exception))


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
class TestCli(_Base):

    def _run(self, root, *args):
        return OPS.main(["--local-root", str(root), "approve-source-value", PACK_KEY,
                         *args])

    def test_cli_approves_and_prints_the_operator_summary(self):
        td = tempfile.TemporaryDirectory(prefix="ops_svr_cli_")
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        _Ctx(root)
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = self._run(root, "--canonical-field", "borrower_1_DOB",
                             "--source-column", "Borrower 1 DOB",
                             "--source-value", "TBC",
                             "--action", svr.ACTION_TREAT_AS_NULL,
                             "--by", "ops")
        out = buffer.getvalue()
        self.assertEqual(code, 0, out)
        self.assertIn("Approved: Borrower 1 DOB value 'TBC' -> null "
                      "(2 rows, MI only)", out)
        self.assertIn("promote", out)

    def test_cli_reports_a_failure_and_exits_non_zero(self):
        td = tempfile.TemporaryDirectory(prefix="ops_svr_cli_fail_")
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        _Ctx(root)
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = self._run(root, "--canonical-field", "borrower_1_DOB",
                             "--source-column", "Borrower 1 DOB",
                             "--source-value", "NOPE")
        self.assertEqual(code, 2)
        self.assertIn("no pending candidate", buffer.getvalue())

    def test_mapping_approval_command_is_untouched(self):
        # `approve` still requires --mapping-id; the new path did not change it.
        import argparse
        with self.assertRaises(SystemExit):
            OPS.main(["approve", "some-approval-id"])
        self.assertTrue(hasattr(argparse, "ArgumentParser"))


# --------------------------------------------------------------------------- #
# The persisted artefact reaches the next run
# --------------------------------------------------------------------------- #
class TestPersistence(_Base):

    def test_28d_is_persisted_for_the_pack(self):
        # Without this the CLI could not read candidates once the ephemeral run
        # scratch is reclaimed.
        self.assertIn("28d_source_value_rules.json", DB._PERSIST_NAMES)

    def test_approved_rules_rehydrate_into_applicable_rules(self):
        ctx = self._ctx()
        ctx.approve()
        doc = ctx.approved_doc()
        rules = svr.rules_from_decisions(doc["decisions"])
        self.assertEqual(len(rules), 1)
        rule = rules[0]
        self.assertEqual(rule.source_value, "TBC")
        self.assertEqual(rule.canonical_field, "borrower_1_DOB")
        self.assertEqual(rule.target_contract_id, MI)

    def test_the_rule_only_applies_within_the_approved_contract(self):
        ctx = self._ctx()
        ctx.approve()
        rule = svr.rules_from_decisions(ctx.approved_doc()["decisions"])[0]
        mi_ctx = svr.NormalisationContext(
            client_id="ERE", source_portfolio_id="acquired_001",
            target_contract_id=MI)
        a2_ctx = svr.NormalisationContext(
            client_id="ERE", source_portfolio_id="acquired_001",
            target_contract_id=A2)
        self.assertTrue(rule.in_scope_for_run(mi_ctx))
        self.assertFalse(rule.in_scope_for_run(a2_ctx))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
