#!/usr/bin/env python3
"""tests/test_occ_approved_mapping_consumption.py

The production consumer of an OCC-approved mapping contract.

Mapping semantics are decided once, during onboarding, by the mapping model and
a human operator. Until Gate 1 could be told about that decision, the only
client-specific channel was the alias overlay — which ranks BELOW exact and
normalised canonical-name matching, so an approved decision was silently
overridable by automated matching, and where no overlay existed the fuzzy tiers
decided unsupervised.

These tests prove the four things that has to mean:

  * an approved decision beats every automated method, including the two that
    used to outrank it;
  * a client/portfolio governed by an approved contract FAILS rather than
    quietly reverting to alias and fuzzy matching;
  * a contract for another book, or written to another schema, is refused;
  * the receipt says which method actually won, not merely that a contract was
    loaded.

Plus the acquired book's two governed fills — the youngest-borrower-age
derivation and the portfolio-scoped broker channel — which are what make the
acquired component carry the same MI fields as the direct one.

Everything here is synthetic. No production tape, canonical or client value
appears in this file.

Run: python -m pytest tests/test_occ_approved_mapping_consumption.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

sys.path.insert(0, str(_REPO / "engine" / "gate_1_alignment"))
import semantic_alignment as sa  # noqa: E402

from engine.gate_2_transform import canonical_transform as ct  # noqa: E402
from engine.orchestrator import trakt_run as tr  # noqa: E402
from engine.transformation_agent import canonical_derivations as cd  # noqa: E402

REGISTRY = _REPO / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO / "config" / "system"

#: The four source concepts the direct book carries as separate columns. Header
#: names only — no values from any real tape.
DIRECT_HEADERS = ["Loan Type", "Product Category", "Lump Sum or Drawdown",
                  "Policy Status"]

#: The approved targets, as the operator approves them in OCC. Declared here so
#: the test asserts the contract is HONOURED, never that these targets are
#: correct — that is the operator's decision, not this file's.
APPROVED_TARGETS = {
    "Loan Type": "loan_sub_type",
    "Product Category": "erm_product_type",
    "Lump Sum or Drawdown": "erm_sub_product_type",
    "Policy Status": "account_status",
}


def _contract(entries, *, client="ACME", portfolio="direct_001",
              dataset="funded", version=1, group="user_overrides",
              scope=True) -> dict:
    """A contract document in the shape operations_control writes."""
    doc = {"version": version,
           "_doc": "synthetic approved mapping contract",
           group: [
               {"source_column": col, "canonical_field": target,
                "method": "operator_approved", "confidence": 1.0,
                "rule_id": f"rul_{i:03d}", "rule_version": 2}
               for i, (col, target) in enumerate(entries.items())
           ]}
    if scope:
        doc["contract_scope"] = {"client_id": client, "portfolio_id": portfolio,
                                 "dataset": dataset}
    return doc


def _write(path: Path, doc: dict) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _fields() -> list:
    registry = sa.load_field_registry(REGISTRY)
    return sa.select_registry_fields(registry, "equity_release")


def _mapper(contract_doc=None, tmp: Path = None, **load_kwargs):
    alias_map = sa.load_aliases_from_dir(ALIASES)
    approved = None
    if contract_doc is not None:
        path = _write(tmp / "12_approved_mapping_overrides.yaml", contract_doc)
        approved = sa.load_approved_mappings([path], canonical_fields=_fields(),
                                             **load_kwargs)
    return sa.HeaderMapper(_fields(), alias_map, approved=approved)


# --------------------------------------------------------------------------- #
# 1-3. Approved decisions outrank every automated method
# --------------------------------------------------------------------------- #

class TestApprovedMappingPrecedence(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_approved_beats_exact_canonical_name_match(self):
        """A header spelled exactly like a canonical field still answers to the
        approved decision. Tier 1 used to win here and there was no way to say
        otherwise."""
        m = _mapper(_contract({"loan_sub_type": "erm_sub_product_type"}), self.tmp)
        self.assertEqual(m.map_one("loan_sub_type"),
                         ("erm_sub_product_type", "operator_approved", 1.0))
        # Without the contract the exact match wins, which is the behaviour the
        # approved decision has to be able to override.
        self.assertEqual(_mapper().map_one("loan_sub_type")[1], "exact")

    def test_approved_beats_normalised_canonical_name_match(self):
        m = _mapper(_contract({"Loan Sub Type": "erm_sub_product_type",
                               "Product Type": "erm_product_type"}), self.tmp)
        self.assertEqual(m.map_one("Loan Sub Type")[0], "erm_sub_product_type")
        self.assertEqual(m.map_one("Product Type")[0], "erm_product_type")
        self.assertEqual(_mapper().map_one("Loan Sub Type")[1], "normalized")

    def test_approved_beats_alias_and_fuzzy_matching(self):
        """'Loan Type' resolves to type_of_loan through the global alias library
        and 'Lump Sum or Drawdown' to loan_sub_type. The approved contract sends
        both somewhere else."""
        plain = _mapper()
        self.assertEqual(plain.map_one("Loan Type")[1], "alias")
        self.assertEqual(plain.map_one("Lump Sum or Drawdown")[0], "loan_sub_type")

        m = _mapper(_contract(APPROVED_TARGETS), self.tmp)
        for header, target in APPROVED_TARGETS.items():
            canon, method, conf = m.map_one(header)
            self.assertEqual((canon, method, conf), (target, "operator_approved", 1.0),
                             f"{header} did not answer to the approved decision")

    def test_operator_approved_outranks_every_method_in_duplicate_resolution(self):
        """Two headers competing for one canonical field: the approved one wins."""
        ranks = sa.METHOD_RANK
        self.assertGreater(ranks["operator_approved"], max(
            v for k, v in ranks.items() if k != "operator_approved"))

    def test_user_overrides_outrank_approved_high_confidence(self):
        doc = _contract({"Policy Status": "account_status"})
        doc["approved_high_confidence_mappings"] = [
            {"source_column": "Policy Status", "canonical_field": "property_status",
             "confidence": 0.95}]
        m = _mapper(doc, self.tmp)
        self.assertEqual(m.map_one("Policy Status")[0], "account_status")

    def test_an_approved_decision_lands_only_on_the_column_it_was_made_about(self):
        """The match key keeps case and separator tolerance and nothing else.

        The fuzzy normalisation used for header GUESSING drops stopwords and
        sorts tokens, so "Loan Type", "Type" and "Account Type" collapse together
        — tolerance that is right when guessing and wrong when applying a
        decision an operator made about one specific column.
        """
        m = _mapper(_contract({"Loan Type": "loan_sub_type"}), self.tmp)
        for spelling in ("Loan Type", "loan_type", "  LOAN   Type "):
            self.assertEqual(m.map_one(spelling)[1], "operator_approved", spelling)
        for other in ("Type", "Account Type", "Loan Sub Type"):
            self.assertNotEqual(m.map_one(other)[1], "operator_approved", other)

    def test_the_four_source_concepts_stay_separate(self):
        m = _mapper(_contract(APPROVED_TARGETS), self.tmp)
        targets = [m.map_one(h)[0] for h in DIRECT_HEADERS]
        self.assertEqual(len(set(targets)), 4,
                         f"source concepts collapsed onto shared targets: {targets}")
        self.assertNotIn(None, targets)


# --------------------------------------------------------------------------- #
# 4-5. Fail closed
# --------------------------------------------------------------------------- #

class TestFailClosed(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def _load(self, doc, **kwargs):
        path = _write(self.tmp / "12_approved_mapping_overrides.yaml", doc)
        return sa.load_approved_mappings([path], canonical_fields=_fields(), **kwargs)

    def test_absent_contract_file_is_refused(self):
        with self.assertRaises(sa.ApprovedMappingError):
            sa.load_approved_mappings([self.tmp / "nope.yaml"])

    def test_malformed_contract_is_refused(self):
        (self.tmp / "bad.yaml").write_text("- just\n- a list\n", encoding="utf-8")
        with self.assertRaises(sa.ApprovedMappingError):
            sa.load_approved_mappings([self.tmp / "bad.yaml"])

    def test_entry_without_a_target_is_refused(self):
        doc = _contract({})
        doc["user_overrides"] = [{"source_column": "Policy Status"}]
        with self.assertRaises(sa.ApprovedMappingError):
            self._load(doc)

    def test_target_outside_the_registry_scope_is_refused(self):
        with self.assertRaises(sa.ApprovedMappingError):
            self._load(_contract({"Policy Status": "not_a_canonical_field"}))

    def test_contract_for_another_portfolio_is_refused(self):
        doc = _contract(APPROVED_TARGETS, portfolio="acquired_001")
        with self.assertRaises(sa.ApprovedMappingError) as ctx:
            self._load(doc, expected_scope={"client_id": "ACME",
                                            "portfolio_id": "direct_001"})
        self.assertIn("portfolio_id", str(ctx.exception))

    def test_contract_for_another_client_is_refused(self):
        doc = _contract(APPROVED_TARGETS, client="OTHER")
        with self.assertRaises(sa.ApprovedMappingError):
            self._load(doc, expected_scope={"client_id": "ACME"})

    def test_unscoped_contract_is_refused_when_a_scope_is_expected(self):
        """A document that cannot prove which book it belongs to is exactly what
        a mis-filed one looks like."""
        with self.assertRaises(sa.ApprovedMappingError):
            self._load(_contract(APPROVED_TARGETS, scope=False),
                       expected_scope={"client_id": "ACME",
                                       "portfolio_id": "direct_001"})

    def test_unsupported_contract_version_is_refused(self):
        with self.assertRaises(sa.ApprovedMappingError) as ctx:
            self._load(_contract(APPROVED_TARGETS, version=99))
        self.assertIn("version", str(ctx.exception))

    def test_missing_version_is_refused(self):
        doc = _contract(APPROVED_TARGETS)
        doc.pop("version")
        with self.assertRaises(sa.ApprovedMappingError):
            self._load(doc)

    def test_contract_with_no_decisions_in_it_is_refused(self):
        doc = {"version": 1, "contract_scope": {"client_id": "ACME"}}
        with self.assertRaises(sa.ApprovedMappingError):
            self._load(doc)

    def test_gate1_refuses_to_run_when_a_required_contract_is_absent(self):
        """The end of the line: Gate 1 itself, as a subprocess, produces no
        canonical rather than mapping by alias and fuzzy matching."""
        src = self.tmp / "tape.csv"
        src.write_text("Loan Policy Number," + ",".join(DIRECT_HEADERS) + "\n"
                       "L0001,Initial Advance,SYNTH 500,Lump Sum,Inforce\n",
                       encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(_REPO / "engine/gate_1_alignment/semantic_alignment.py"),
             "--input", str(src), "--portfolio-type", "equity_release",
             "--registry", str(REGISTRY), "--aliases-dir", str(ALIASES),
             "--require-approved-mappings", "--output-dir", str(self.tmp)],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ApprovedMappingError", proc.stderr)
        self.assertFalse((self.tmp / "tape_canonical_full.csv").exists(),
                         "a canonical was published without the approved contract")


class TestClientScopedRequirement(unittest.TestCase):
    """Requiring a contract is a per-client configuration decision (14)."""

    def test_required_only_for_the_configured_portfolios(self):
        cfg = {"mapping": {"require_approved_contract": True,
                           "approved_contract_portfolios": ["direct_001",
                                                            "acquired_001"]}}
        self.assertTrue(tr.approved_contract_required(cfg, "direct_001"))
        self.assertTrue(tr.approved_contract_required(cfg, "acquired_001"))
        self.assertFalse(tr.approved_contract_required(cfg, "acquired_009"))

    def test_a_client_with_no_mapping_block_is_unchanged(self):
        for cfg in ({}, None, {"defaults": {}}, {"mapping": {}}):
            self.assertFalse(tr.approved_contract_required(cfg, "direct_001"))

    def test_requirement_without_a_portfolio_list_covers_every_portfolio(self):
        cfg = {"mapping": {"require_approved_contract": True}}
        self.assertTrue(tr.approved_contract_required(cfg, "anything_001"))

    def test_the_contract_reaches_gate1_as_its_own_flag_not_as_an_alias_dir(self):
        """An approved contract must never be routed through --extra-aliases-dir:
        an alias ranks below canonical-name matching, which is the whole defect."""
        import argparse
        args = argparse.Namespace(
            approved_mappings=["/tmp/12_approved_mapping_overrides.yaml"],
            approved_scope_client="ERE", approved_scope_dataset="funded",
            source_portfolio_id="acquired_001", extra_aliases_dir=[])
        emitted = tr._approved_mapping_cli_passthrough(args, True)
        self.assertIn("--approved-mappings", emitted)
        self.assertIn("--require-approved-mappings", emitted)
        self.assertNotIn("--extra-aliases-dir", emitted)
        for flag, value in (("--approved-scope-client", "ERE"),
                            ("--approved-scope-portfolio", "acquired_001"),
                            ("--approved-scope-dataset", "funded")):
            self.assertEqual(emitted[emitted.index(flag) + 1], value)

    def test_no_requirement_emits_no_requirement_flag(self):
        import argparse
        args = argparse.Namespace(approved_mappings=[], approved_scope_client="",
                                  approved_scope_dataset="", source_portfolio_id="")
        self.assertEqual(tr._approved_mapping_cli_passthrough(args, False), [])

    def test_the_shipped_client_config_requires_both_reruns(self):
        cfg = yaml.safe_load(
            (_REPO / "config/client/config_client_ERM_UK.yaml").read_text())
        self.assertTrue(tr.approved_contract_required(cfg, "direct_001"))
        self.assertTrue(tr.approved_contract_required(cfg, "acquired_001"))


# --------------------------------------------------------------------------- #
# 6. The receipt names the winning method
# --------------------------------------------------------------------------- #

class TestReceipt(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def _run_gate1(self, contract: bool = True):
        src = self.tmp / "tape.csv"
        src.write_text("Loan Policy Number," + ",".join(DIRECT_HEADERS) + "\n"
                       "L0001,Initial Advance,SYNTH 500,Lump Sum,Inforce\n",
                       encoding="utf-8")
        cmd = [sys.executable, str(_REPO / "engine/gate_1_alignment/semantic_alignment.py"),
               "--input", str(src), "--portfolio-type", "equity_release",
               "--registry", str(REGISTRY), "--aliases-dir", str(ALIASES),
               "--output-dir", str(self.tmp)]
        if contract:
            path = _write(self.tmp / "12_approved_mapping_overrides.yaml",
                          _contract(APPROVED_TARGETS))
            cmd += ["--approved-mappings", str(path), "--require-approved-mappings",
                    "--approved-scope-client", "ACME",
                    "--approved-scope-portfolio", "direct_001",
                    "--approved-scope-dataset", "funded"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return json.loads((self.tmp / "tape_header_mapping_report.json").read_text())

    def test_receipt_records_the_approved_decision_that_won(self):
        report = self._run_gate1()
        by_header = {r["raw_header"]: r for r in report["mappings"]}
        for header, target in APPROVED_TARGETS.items():
            row = by_header[header]
            self.assertEqual(row["canonical_field"], target)
            self.assertEqual(row["mapping_method"], "operator_approved")
            self.assertEqual(row["confidence"], 1.0)
            self.assertTrue(row["approved_rule_id"])
            self.assertEqual(row["approved_rule_version"], "2")
        self.assertEqual(sorted(report["columns_decided_by_operator_approval"]),
                         sorted(DIRECT_HEADERS))

    def test_receipt_distinguishes_a_loaded_contract_from_a_winning_one(self):
        """A column the contract does not govern reports the method that DID
        decide it — the old failure mode was a receipt that said a contract was
        applied while name matching quietly overruled it."""
        report = self._run_gate1()
        row = next(r for r in report["mappings"]
                   if r["raw_header"] == "Loan Policy Number")
        self.assertNotEqual(row["mapping_method"], "operator_approved")
        self.assertEqual(row["approved_rule_id"], "")
        self.assertEqual(report["approved_mapping_contract"]["approved_mapping_count"], 4)
        self.assertTrue(report["approved_mapping_required"])

    def test_receipt_without_a_contract_shows_no_approved_decisions(self):
        report = self._run_gate1(contract=False)
        self.assertEqual(report["columns_decided_by_operator_approval"], [])
        self.assertFalse(report["approved_mapping_required"])
        self.assertEqual(report["approved_mapping_contract"]["approved_mapping_count"], 0)


# --------------------------------------------------------------------------- #
# 8. Inforce becomes A
# --------------------------------------------------------------------------- #

class TestAccountStatusNormalisation(unittest.TestCase):

    def setUp(self):
        self.cfg = yaml.safe_load(
            (_REPO / "config/client/config_client_ERM_UK.yaml").read_text())

    def test_inforce_normalises_to_A(self):
        df = pd.DataFrame({"account_status": ["Inforce", "inforce", "INFORCE", "A"]})
        ct.apply_canonical_enum_normalization(
            df, ct.resolve_canonical_enum_normalization(self.cfg))
        self.assertEqual(df["account_status"].tolist(), ["A", "A", "A", "A"])

    def test_D_is_left_alone_because_its_meaning_is_not_confirmed(self):
        df = pd.DataFrame({"account_status": ["D", "Redeemed", ""]})
        report = ct.apply_canonical_enum_normalization(
            df, ct.resolve_canonical_enum_normalization(self.cfg))
        self.assertEqual(df["account_status"].tolist(), ["D", "Redeemed", ""])
        self.assertIn("D", report["canonical_enum_normalization"]["fields"]
                      ["account_status"]["unmapped_examples"])

    def test_the_rule_is_client_scoped_not_global(self):
        """The shared enum library must not learn one lender's vocabulary."""
        self.assertEqual(
            ct.resolve_canonical_enum_normalization({}).get("account_status"), None)


# --------------------------------------------------------------------------- #
# 9-10. Youngest borrower age
# --------------------------------------------------------------------------- #

class TestYoungestBorrowerAge(unittest.TestCase):

    CUT_OFF = "2026-06-30"

    def _derive(self, rows):
        df = pd.DataFrame(rows)
        df["data_cut_off_date"] = self.CUT_OFF
        result = cd.apply_selected_derivations(df, ["youngest_borrower_age"])
        return df, result["youngest_borrower_age"]

    def test_one_borrower(self):
        df, res = self._derive([{"borrower_1_DOB": "1955-01-01",
                                 "borrower_2_DOB": ""}])
        self.assertEqual(df["youngest_borrower_age"].iloc[0], 71)
        self.assertEqual(res["value_counts"]["derived"], 1)

    def test_second_borrower_alone_is_enough(self):
        df, _ = self._derive([{"borrower_1_DOB": "", "borrower_2_DOB": "1960-05-05"}])
        self.assertEqual(df["youngest_borrower_age"].iloc[0], 66)

    def test_two_borrowers_takes_the_younger(self):
        df, _ = self._derive([{"borrower_1_DOB": "1946-01-01",
                               "borrower_2_DOB": "1955-01-01"}])
        self.assertEqual(df["youngest_borrower_age"].iloc[0], 71)

    def test_birthday_boundary(self):
        """A birthday ON the cut-off has happened; one day later has not."""
        df, _ = self._derive([
            {"borrower_1_DOB": "1950-06-29", "borrower_2_DOB": ""},   # passed
            {"borrower_1_DOB": "1950-06-30", "borrower_2_DOB": ""},   # today
            {"borrower_1_DOB": "1950-07-01", "borrower_2_DOB": ""},   # tomorrow
        ])
        self.assertEqual(df["youngest_borrower_age"].tolist(), [76, 76, 75])

    def test_one_dob_missing_uses_the_other(self):
        df, res = self._derive([{"borrower_1_DOB": "TBC",
                                 "borrower_2_DOB": "1958-03-03"}])
        self.assertEqual(df["youngest_borrower_age"].iloc[0], 68)
        self.assertEqual(res["unresolved_count"], 0)

    def test_both_dobs_missing_stays_null_and_is_flagged(self):
        df, res = self._derive([{"borrower_1_DOB": "", "borrower_2_DOB": ""},
                                {"borrower_1_DOB": "TBC", "borrower_2_DOB": "n/k"}])
        self.assertTrue(df["youngest_borrower_age"].isna().all())
        self.assertEqual(res["unresolved_count"], 2)
        self.assertEqual(res["unresolved_row_positions"], [0, 1])
        self.assertTrue(res["unresolved_reason"])
        # An unanswerable row is an operator question, never a blocking failure.
        self.assertEqual(res["failure_count"], 0)

    def test_a_mapped_age_is_never_overwritten(self):
        df = pd.DataFrame({
            "borrower_1_DOB": ["1930-01-01", "1955-01-01"],
            "borrower_2_DOB": ["", ""],
            "youngest_borrower_age": [68, None],
            "data_cut_off_date": [self.CUT_OFF, self.CUT_OFF],
        })
        res = cd.apply_selected_derivations(df, ["youngest_borrower_age"])
        self.assertEqual(df["youngest_borrower_age"].iloc[0], 68)
        self.assertEqual(df["youngest_borrower_age"].iloc[1], 71)
        self.assertEqual(res["youngest_borrower_age"]["value_counts"],
                         {"derived": 1, "kept_supplied_value": 1, "unresolved_null": 0})

    def test_a_future_dob_is_unresolved_not_a_negative_age(self):
        df, res = self._derive([{"borrower_1_DOB": "2030-01-01", "borrower_2_DOB": ""}])
        self.assertTrue(pd.isna(df["youngest_borrower_age"].iloc[0]))
        self.assertEqual(res["unresolved_count"], 1)

    def test_age_is_as_at_the_governed_cut_off_not_today(self):
        df = pd.DataFrame({"borrower_1_DOB": ["1950-01-01", "1950-01-01"],
                           "borrower_2_DOB": ["", ""],
                           "data_cut_off_date": ["2026-06-30", "2020-06-30"]})
        cd.apply_selected_derivations(df, ["youngest_borrower_age"])
        self.assertEqual(df["youngest_borrower_age"].tolist(), [76, 70])

    def test_absent_second_dob_column_is_tolerated(self):
        df = pd.DataFrame({"borrower_1_DOB": ["1955-01-01"],
                           "data_cut_off_date": [self.CUT_OFF]})
        res = cd.apply_selected_derivations(df, ["youngest_borrower_age"])
        self.assertEqual(df["youngest_borrower_age"].iloc[0], 71)
        self.assertEqual(res["youngest_borrower_age"]["absent_sources"],
                         ["borrower_2_DOB"])

    def test_no_cut_off_column_derives_nothing_rather_than_guessing(self):
        df = pd.DataFrame({"borrower_1_DOB": ["1955-01-01"]})
        res = cd.apply_selected_derivations(df, ["youngest_borrower_age"])
        self.assertFalse(res["youngest_borrower_age"]["applied"])
        self.assertEqual(res["youngest_borrower_age"]["reason"], "as_at_field_absent")
        self.assertNotIn("youngest_borrower_age", df.columns)

    def test_the_rule_is_declared_in_the_governed_library(self):
        spec = cd.load_derivations()["youngest_borrower_age"]
        self.assertEqual(spec["rule"], "youngest_full_age_at")
        self.assertEqual(spec["at"], "data_cut_off_date")
        self.assertTrue(spec["fill_blank_only"])
        self.assertEqual(spec["sources"], ["borrower_1_DOB", "borrower_2_DOB"])


# --------------------------------------------------------------------------- #
# 11-12. Portfolio-scoped broker channel
# --------------------------------------------------------------------------- #

class TestPortfolioBrokerDefault(unittest.TestCase):

    def setUp(self):
        self.cfg = yaml.safe_load(
            (_REPO / "config/client/config_client_ERM_UK.yaml").read_text())

    def test_a_mapped_broker_value_beats_the_portfolio_default(self):
        df = pd.DataFrame({"broker_channel": ["Synthetic Broker Ltd", "", None]})
        report = ct.apply_portfolio_defaults(df, self.cfg, "acquired_001")
        self.assertEqual(df["broker_channel"].tolist(),
                         ["Synthetic Broker Ltd", "Acquired_001", "Acquired_001"])
        entry = report["portfolio_defaults"]["broker_channel"]
        self.assertEqual(entry["rows_filled"], 2)
        self.assertEqual(entry["rows_kept_source_value"], 1)

    def test_an_absent_broker_column_takes_the_configured_name(self):
        df = pd.DataFrame({"loan_identifier": ["L1", "L2"]})
        ct.apply_portfolio_defaults(df, self.cfg, "acquired_001")
        self.assertEqual(df["broker_channel"].tolist(),
                         ["Acquired_001", "Acquired_001"])

    def test_provenance_marks_the_value_as_a_portfolio_default(self):
        df = pd.DataFrame({"loan_identifier": ["L1"]})
        report = ct.apply_portfolio_defaults(df, self.cfg, "acquired_001")
        entry = report["portfolio_defaults"]["broker_channel"]
        self.assertEqual(entry["value_origin"], "portfolio_default")
        self.assertEqual(entry["source_portfolio_id"], "acquired_001")
        self.assertEqual(entry["value"], "Acquired_001")

    def test_the_direct_book_is_untouched_by_the_acquired_default(self):
        df = pd.DataFrame({"loan_identifier": ["L1"]})
        self.assertEqual(ct.apply_portfolio_defaults(df, self.cfg, "direct_001"), {})
        self.assertNotIn("broker_channel", df.columns)

    def test_a_second_acquired_portfolio_needs_only_configuration(self):
        """acquired_002 uses its OWN name with no code change: the value is read
        from configuration and nothing here knows any portfolio's name."""
        cfg = {"portfolio_defaults": {
            "acquired_001": {"broker_channel": "Acquired_001"},
            "acquired_002": {"broker_channel": "Acquired_002"},
            "acquired_003": {"broker_channel": "Acquired_003"}}}
        for pid, expected in (("acquired_001", "Acquired_001"),
                              ("acquired_002", "Acquired_002"),
                              ("acquired_003", "Acquired_003")):
            df = pd.DataFrame({"loan_identifier": ["L1"]})
            report = ct.apply_portfolio_defaults(df, cfg, pid)
            self.assertEqual(df["broker_channel"].iloc[0], expected)
            self.assertEqual(
                report["portfolio_defaults"]["broker_channel"]["source_portfolio_id"],
                pid)

    def test_no_portfolio_name_is_hard_coded_in_the_transform(self):
        """No portfolio name is a value the code can act on.

        Checked against the executable string constants, so the configuration
        EXAMPLE in the docstring still documents the shape without the name ever
        being something the transform could match, branch on or emit.
        """
        import ast
        for rel in ("engine/gate_2_transform/canonical_transform.py",
                    "engine/transformation_agent/gate2_adapter.py"):
            tree = ast.parse((_REPO / rel).read_text())
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node, clean=False)
                    if doc is not None:
                        docstrings.add(doc)
            live = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and n.value not in docstrings]
            for literal in live:
                self.assertNotIn("cquired_00", literal,
                                 f"{rel} carries a portfolio name in code")

    def test_an_unconfigured_portfolio_gets_nothing(self):
        df = pd.DataFrame({"loan_identifier": ["L1"]})
        self.assertEqual(ct.apply_portfolio_defaults(df, self.cfg, "acquired_999"), {})
        self.assertNotIn("broker_channel", df.columns)


# --------------------------------------------------------------------------- #
# 13. The Assembler is untouched
# --------------------------------------------------------------------------- #

class TestAssemblerUnchanged(unittest.TestCase):

    def test_platform_assembler_is_not_modified_by_this_change(self):
        """The audit established the Assembler is not defective. Its behaviour is
        a column-preserving union and this sprint must not have touched it."""
        proc = subprocess.run(
            ["git", "diff", "--name-only", "origin/main", "--",
             "engine/platform_assembler.py", "engine/assembler_agent.py"],
            cwd=_REPO, capture_output=True, text=True)
        self.assertEqual(proc.stdout.strip(), "",
                         "the Assembler was modified; it is out of scope")

    def test_the_union_still_preserves_a_column_only_one_component_carries(self):
        import tempfile
        from engine import platform_assembler as pa
        from engine import provenance as prov

        tmp = Path(tempfile.mkdtemp())
        base = {f: "x" for f in prov.PROVENANCE_FIELDS}
        direct = pd.DataFrame([{**base, "source_portfolio_id": "direct_001",
                                "loan_identifier": "D1",
                                "broker_channel": "Synthetic Broker Ltd",
                                "youngest_borrower_age": 71}])
        acquired = pd.DataFrame([{**base, "source_portfolio_id": "acquired_001",
                                  "loan_identifier": "A1",
                                  "broker_channel": "Acquired_001",
                                  "youngest_borrower_age": 68}])
        direct.to_csv(tmp / "direct_001_canonical_typed.csv", index=False)
        acquired.to_csv(tmp / "acquired_001_canonical_typed.csv", index=False)

        res = pa.assemble_platform_canonical(tmp, tmp, write=False)
        combined = res.dataframe
        self.assertEqual(len(combined), 2)
        self.assertEqual(combined["platform_loan_key"].nunique(), 2)
        # Both components carry both fields, so every combined row is populated —
        # which is the whole point of the acquired derivation and default.
        self.assertTrue(combined["broker_channel"].notna().all())
        self.assertTrue(combined["youngest_borrower_age"].notna().all())


# --------------------------------------------------------------------------- #
# End to end: Gate 1 -> Gate 2 on the two synthetic books
# --------------------------------------------------------------------------- #

class TestDirectAndAcquiredEndToEnd(unittest.TestCase):
    """Both books through the real Gate 1 and Gate 2 executables, with the real
    client configuration. Synthetic tapes; the assertions are about behaviour,
    never about any production count."""

    CLIENT_CONFIG = _REPO / "config/client/config_client_ERM_UK.yaml"
    CUT_OFF = "2026-06-30"

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def _gate1(self, name: str, header: str, rows: list, *,
               contract: dict = None, portfolio: str = "direct_001") -> Path:
        src = self.tmp / f"{name}.csv"
        src.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        cmd = [sys.executable, str(_REPO / "engine/gate_1_alignment/semantic_alignment.py"),
               "--input", str(src), "--portfolio-type", "equity_release",
               "--registry", str(REGISTRY), "--aliases-dir", str(ALIASES),
               "--output-dir", str(self.tmp)]
        if contract is not None:
            path = _write(self.tmp / f"{name}_contract.yaml", contract)
            cmd += ["--approved-mappings", str(path), "--require-approved-mappings",
                    "--approved-scope-client", "ACME",
                    "--approved-scope-portfolio", portfolio,
                    "--approved-scope-dataset", "funded"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return self.tmp / f"{name}_canonical_full.csv"

    def _gate2(self, canonical_full: Path, portfolio: str) -> pd.DataFrame:
        proc = subprocess.run(
            [sys.executable, str(_REPO / "engine/gate_2_transform/canonical_transform.py"),
             str(canonical_full), "--registry", str(REGISTRY),
             "--portfolio-type", "equity_release",
             "--config", str(self.CLIENT_CONFIG), "--output-dir", str(self.tmp),
             "--source-portfolio-id", portfolio,
             "--allow-unknown-acquisition-date"],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        stem = canonical_full.stem.replace("_canonical_full", "")
        return pd.read_csv(self.tmp / f"{stem}_canonical_typed.csv")

    def test_direct_book_four_concepts_survive_and_inforce_becomes_A(self):
        full = self._gate1(
            "direct",
            "Loan Policy Number,Cut Off Date," + ",".join(DIRECT_HEADERS),
            [f"L000{i},{self.CUT_OFF},Initial Advance,SYNTH 500,Lump Sum,Inforce"
             for i in range(1, 4)],
            contract=_contract(APPROVED_TARGETS))
        df = self._gate2(full, "direct_001")

        # Each approved target carries its own source concept, still distinct.
        self.assertEqual(set(df["loan_sub_type"]), {"Initial Advance"})
        self.assertEqual(set(df["erm_product_type"]), {"SYNTH 500"})
        self.assertEqual(set(df["erm_sub_product_type"]), {"Lump Sum"})
        # 8. Inforce is normalised to the governed active representation.
        self.assertEqual(set(df["account_status"]), {"A"})

    def test_acquired_book_derives_age_and_takes_the_portfolio_broker(self):
        header = ("Loan Policy Number,Cut Off Date,Borrower 1 DOB,Borrower 2 DOB,"
                  "Policy Status")
        rows = [
            f"A0001,{self.CUT_OFF},1946-01-01,1955-01-01,Inforce",   # two DOBs
            f"A0002,{self.CUT_OFF},1958-03-03,,Inforce",             # one DOB
            f"A0003,{self.CUT_OFF},,,Inforce",                       # neither
        ]
        full = self._gate1("acquired", header, rows, portfolio="acquired_001",
                           contract=_contract({"Policy Status": "account_status"},
                                              portfolio="acquired_001"))
        df = self._gate2(full, "acquired_001")

        # The age is calculated at the GOVERNED cut-off date, which for this
        # client's configuration is the pinned reporting date — not the folder
        # period and not today. Deriving the expectation from the same governed
        # value is the point: an answer about a given book must not move.
        governed = str(yaml.safe_load(self.CLIENT_CONFIG.read_text())
                       ["portfolio"]["static_reporting_date"])
        at = cd._parse_date(governed)
        ages = df["youngest_borrower_age"].tolist()
        self.assertEqual(ages[0], cd.full_calendar_age(cd._parse_date("1955-01-01"), at),
                         "expected the YOUNGER of the two borrowers")
        self.assertEqual(ages[1], cd.full_calendar_age(cd._parse_date("1958-03-03"), at),
                         "expected the only supplied borrower")
        self.assertTrue(pd.isna(ages[2]))             # neither usable -> null
        # 12. No source broker column at all: every row takes the configured
        # portfolio name, and nothing in the code knows that name.
        self.assertEqual(set(df["broker_channel"]), {"Acquired_001"})
        self.assertEqual(set(df["account_status"]), {"A"})

    def test_a_source_broker_value_survives_the_portfolio_default(self):
        header = ("Loan Policy Number,Cut Off Date,Broker Channel,Borrower 1 DOB")
        rows = [f"A0001,{self.CUT_OFF},Synthetic Broker Ltd,1955-01-01",
                f"A0002,{self.CUT_OFF},,1955-01-01"]
        full = self._gate1("acqbroker", header, rows, portfolio="acquired_001",
                           contract=_contract({"Broker Channel": "broker_channel"},
                                              portfolio="acquired_001"))
        df = self._gate2(full, "acquired_001")
        self.assertEqual(df["broker_channel"].tolist(),
                         ["Synthetic Broker Ltd", "Acquired_001"])

    def test_the_transform_report_records_the_default_and_the_derivation(self):
        header = "Loan Policy Number,Cut Off Date,Borrower 1 DOB"
        full = self._gate1("acqreport", header,
                           [f"A0001,{self.CUT_OFF},1955-01-01"],
                           portfolio="acquired_001",
                           contract=_contract({"Borrower 1 DOB": "borrower_1_DOB"},
                                              portfolio="acquired_001"))
        self._gate2(full, "acquired_001")
        report = json.loads((self.tmp / "acqreport_transform_report.json").read_text())
        default = report["portfolio_defaults"]["broker_channel"]
        self.assertEqual(default["value"], "Acquired_001")
        self.assertEqual(default["value_origin"], "portfolio_default")
        self.assertEqual(default["source_portfolio_id"], "acquired_001")
        derived = report["derived"]["youngest_borrower_age"]
        self.assertEqual(derived["rule"], "youngest_full_age_at")
        self.assertEqual(derived["as_at"], "data_cut_off_date")
        self.assertEqual(derived["value_origin"], "derived")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
