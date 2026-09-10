#!/usr/bin/env python3
"""The acceptance harness's pure parts, tested without a deployment.

Three of its helpers decide whether a run means anything, and all three fail
silently if they are wrong:

    the VFS path mapping — a wrong path reads exactly like a missing sink, which
    would stop the run for the wrong reason or, worse, let it "pass" with nothing;

    the publish-profile parse — a wrong host does the same;

    the adjudication — the part that has to call a defect a defect and refuse to
    score a legacy agreement as a pass.

And the scrubber, because a credential reaching an uploaded artifact is the one
failure here that cannot be undone.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HARNESS = (_REPO_ROOT / "due_diligence" / "evidence"
            / "deployed_acceptance_0399a315")
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

import run_acceptance as harness                                      # noqa: E402

MANIFEST = json.loads((_HARNESS / "acceptance_manifest.json").read_text())
BY_ID = {c["case_id"]: c for c in MANIFEST["cases"]}


def evidence_for(case, *, eligible=None, reason="", plan=None, execution=None,
                 disposition="EXECUTED", outcome="PLAN", plan_id=None):
    """An evidence record of the shape the deployed recorder writes."""
    expected = case["expected_semantics"]
    if plan is None:
        plan = {
            "capability": expected["capability"],
            "operation": expected["operation"],
            "population": {"base": expected["population_base"],
                           "lens": expected["population_lens"]},
            "period": {"form": expected["period_form"]},
            "comparison_kind": expected["comparison_kind"],
            "geography": None,
            "filters": [{"canonical_field": f[0], "comparator": f[1],
                         "value": f[2]} for f in expected["filters"]],
            "outputs": [{"measures": [{"statistic": expected["statistic"],
                                       "canonical_field":
                                           expected["measure_field"],
                                       "weight_field":
                                           expected["weight_field"]}],
                         "dimensions": [{"canonical_field": d}
                                        for d in expected["dimensions"]],
                         "filters": [], "geography": None}],
        }
    if eligible is None:
        eligible = case["expected_slice1_eligible"]
        reason = case["expected_ineligible_reason"]
    return {
        "correlation_id": "shadow_deadbeef",
        "request": {"question": case["question"], "client_id": "ERE"},
        "disposition": disposition,
        "model": {"model_id": "claude-opus-5", "raw_payload": {"x": 1}},
        # `produced_intent` is what `plan_shadow_wiring._shadow` actually writes
        # (`bool(outcome.ok)`), and the completed-interpretation rule reads it. The
        # helper omitted it before that rule existed.
        "interpretation": {"produced_intent": outcome == "PLAN",
                           "candidate_intent": {"schema_version": "x"}},
        "compiler": {"outcome": outcome, "plan": plan if outcome == "PLAN" else None,
                     "plan_id": plan_id or case["expected_frozen_plan_id"]},
        "eligibility": {"eligible": eligible, "reason": reason},
        "execution": execution if execution is not None else (
            {"attempted": False, "why_not": reason} if not eligible else None),
    }


def executed(case, **over):
    expected = case["expected_semantics"]
    body = {
        "attempted": True,
        "requested_semantics": {
            "measure_field": expected["measure_field"],
            "statistic": expected["statistic"],
            "dimensions": list(expected["dimensions"]),
            "filters": [{"field": f[0], "comparator": f[1], "value": f[2]}
                        for f in expected["filters"]],
        },
        "bound_spec": {
            "metric": expected["measure_field"],
            "aggregation": {"sum": "sum", "count": "count",
                            "average": "avg",
                            "weighted_average": "weighted_avg"}[
                expected["statistic"]],
            "dimensions": list(expected["dimensions"]),
            "filters": {f[0]: f[2] for f in expected["filters"]},
        },
        "value": 195.0,
        "receipt": {"row_count": 1, "applied_predicates": [{"field": "x"}]},
        "warnings": [],
        "error": "",
        "grouped_cells": None,
    }
    body.update(over)
    return body


LEGACY_OK = {"ok": True, "http_status": 200, "value": 195.0, "route": None,
             "transport_error": False}


class TestTheVfsPathMapping(unittest.TestCase):

    def test_a_home_path_loses_the_home_prefix(self):
        """Kudu's VFS root IS /home, so /home/LogFiles/x is LogFiles/x."""
        self.assertEqual(harness.vfs_path("/home/LogFiles/shadow/evidence.jsonl"),
                         "LogFiles/shadow/evidence.jsonl")

    def test_a_path_outside_home_is_addressed_from_the_root(self):
        self.assertEqual(harness.vfs_path("/tmp/evidence.jsonl"),
                         "tmp/evidence.jsonl")

    def test_whitespace_does_not_become_part_of_the_path(self):
        self.assertEqual(harness.vfs_path("  /home/LogFiles/e.jsonl  "),
                         "LogFiles/e.jsonl")

    def test_the_url_is_built_from_the_mapped_path(self):
        sink = harness.Sink("app.scm.azurewebsites.net", "u", "p",
                            "/home/LogFiles/e.jsonl")
        self.assertEqual(sink.url,
                         "https://app.scm.azurewebsites.net/api/vfs/"
                         "LogFiles/e.jsonl")


class TestThePublishProfileParse(unittest.TestCase):

    PROFILE = """<publishData>
      <publishProfile publishMethod="MSDeploy" publishUrl="trakt-mi-api.scm.azurewebsites.net:443"
        userName="$trakt-mi-api" userPWD="a-password"/>
      <publishProfile publishMethod="FTP" publishUrl="ftp://other/site/wwwroot"
        userName="$x" userPWD="y"/>
    </publishData>"""

    def test_the_msdeploy_profile_supplies_the_scm_credential(self):
        host, user, password = harness.publish_profile_credentials(self.PROFILE)
        self.assertEqual(host, "trakt-mi-api.scm.azurewebsites.net")
        self.assertEqual(user, "$trakt-mi-api")
        self.assertEqual(password, "a-password")

    def test_a_profile_with_no_msdeploy_entry_is_refused_not_guessed(self):
        ftp_only = """<publishData><publishProfile publishMethod="FTP"
            publishUrl="ftp://h/site" userName="u" userPWD="p"/></publishData>"""
        with self.assertRaises(ValueError):
            harness.publish_profile_credentials(ftp_only)

    def test_a_profile_missing_a_field_is_refused(self):
        partial = """<publishData><publishProfile publishMethod="MSDeploy"
            publishUrl="h:443" userName="u"/></publishData>"""
        with self.assertRaises(ValueError):
            harness.publish_profile_credentials(partial)


class TestAdjudication(unittest.TestCase):

    def test_a_matching_eligible_case_is_exact_parity(self):
        case = BY_ID["A01"]
        verdict = harness.adjudicate(
            case, evidence_for(case, execution=executed(case)), LEGACY_OK)
        self.assertEqual(verdict["classification"],
                         harness.EXACT_SEMANTIC_PARITY)
        self.assertEqual(verdict["silent_drops"], [])

    def test_a_different_plan_id_with_identical_semantics_is_equivalent(self):
        case = BY_ID["A01"]
        verdict = harness.adjudicate(
            case, evidence_for(case, execution=executed(case),
                               plan_id="plan_something_else"), LEGACY_OK)
        self.assertEqual(verdict["classification"],
                         harness.SEMANTICALLY_EQUIVALENT)

    def test_a_dropped_filter_is_an_execution_defect(self):
        case = BY_ID["A01"]
        spec = executed(case)
        spec["bound_spec"]["filters"] = {}
        verdict = harness.adjudicate(case, evidence_for(case, execution=spec),
                                     LEGACY_OK)
        self.assertEqual(verdict["classification"],
                         harness.DETERMINISTIC_EXECUTION_DEFECT)
        self.assertTrue(verdict["filter_drops"])

    def test_a_dropped_dimension_is_an_execution_defect(self):
        case = BY_ID["A04"]
        spec = executed(case)
        spec["bound_spec"]["dimensions"] = spec["bound_spec"]["dimensions"][:1]
        verdict = harness.adjudicate(case, evidence_for(case, execution=spec),
                                     LEGACY_OK)
        self.assertEqual(verdict["classification"],
                         harness.DETERMINISTIC_EXECUTION_DEFECT)
        self.assertTrue(verdict["dimension_drops"])

    def test_an_ineligible_control_that_becomes_eligible_is_a_defect(self):
        case = BY_ID["B01"]
        verdict = harness.adjudicate(
            case, evidence_for(case, eligible=True, reason="",
                               execution=executed(case)), LEGACY_OK)
        self.assertEqual(verdict["classification"], harness.ELIGIBILITY_DEFECT)

    def test_an_ineligible_control_refused_for_its_own_reason_passes(self):
        case = BY_ID["B02"]
        verdict = harness.adjudicate(case, evidence_for(case), LEGACY_OK)
        self.assertEqual(verdict["classification"], harness.JUSTIFIED_INELIGIBLE)

    def test_an_ineligible_plan_that_reached_the_adapter_is_a_defect(self):
        case = BY_ID["B03"]
        record = evidence_for(case)
        record["execution"] = {"attempted": True}
        verdict = harness.adjudicate(case, record, LEGACY_OK)
        self.assertEqual(verdict["classification"], harness.ELIGIBILITY_DEFECT)

    def test_a_missing_record_is_inconclusive_never_a_pass(self):
        verdict = harness.adjudicate(BY_ID["A01"], None, LEGACY_OK)
        self.assertEqual(verdict["classification"], harness.ASYNC_EVIDENCE_LOST)

    def test_a_missing_fixture_column_is_truth_unavailable_not_a_defect(self):
        """A05 pre-registers both branches; the error branch is a dataset fact."""
        case = BY_ID["A05"]
        spec = executed(case)
        spec["error"] = "KeyError: 'ticket_bucket'"
        verdict = harness.adjudicate(case, evidence_for(case, execution=spec),
                                     LEGACY_OK)
        self.assertEqual(verdict["classification"], harness.TRUTH_UNAVAILABLE)

    def test_an_execution_error_on_a_carried_field_is_a_defect(self):
        case = BY_ID["A01"]
        spec = executed(case)
        spec["error"] = "MIQueryExecutionError: something real"
        verdict = harness.adjudicate(case, evidence_for(case, execution=spec),
                                     LEGACY_OK)
        self.assertEqual(verdict["classification"],
                         harness.DETERMINISTIC_EXECUTION_DEFECT)

    def test_grouped_cells_must_agree_with_the_receipts_own_group_count(self):
        case = BY_ID["A04"]
        spec = executed(case, grouped_cells=[
            {"ltv_bucket": "0-30%", "age_bucket": "<70", "value": 1.0}],
            receipt={"row_count": 20, "applied_predicates": []})
        verdict = harness.adjudicate(case, evidence_for(case, execution=spec),
                                     LEGACY_OK)
        self.assertEqual(verdict["classification"],
                         harness.DETERMINISTIC_EXECUTION_DEFECT)
        self.assertIn("receipt reports", " ".join(verdict["notes"]))

    def test_a_cell_missing_a_group_key_is_a_defect(self):
        case = BY_ID["A04"]
        spec = executed(case, grouped_cells=[
            {"ltv_bucket": "0-30%", "age_bucket": "", "value": 1.0}],
            receipt={"row_count": 1, "applied_predicates": []})
        verdict = harness.adjudicate(case, evidence_for(case, execution=spec),
                                     LEGACY_OK)
        self.assertEqual(verdict["classification"],
                         harness.DETERMINISTIC_EXECUTION_DEFECT)

    def test_legacy_agreement_is_recorded_as_a_signal_not_a_verdict(self):
        case = BY_ID["A01"]
        verdict = harness.adjudicate(
            case, evidence_for(case, execution=executed(case)), LEGACY_OK)
        self.assertEqual(verdict["legacy_agreement_signal"], "AGREES")
        self.assertTrue(any("not the oracle" in n for n in verdict["notes"]))

    def test_legacy_disagreement_alone_is_not_scored_as_a_defect(self):
        """Without independent truth, a difference is a signal to adjudicate."""
        case = BY_ID["A01"]
        legacy = dict(LEGACY_OK, value=999.0)
        verdict = harness.adjudicate(
            case, evidence_for(case, execution=executed(case)), legacy)
        self.assertEqual(verdict["legacy_agreement_signal"], "DIFFERS")
        self.assertEqual(verdict["classification"],
                         harness.EXACT_SEMANTIC_PARITY)

    def test_every_eligible_case_is_pre_registered_truth_unavailable(self):
        for case in MANIFEST["cases"]:
            if case["expected_slice1_eligible"]:
                self.assertIsNone(case["independent_numerical_truth"])
                self.assertIn("TRUTH_UNAVAILABLE",
                              case["independent_numerical_truth_status"])


class TestPassIsImpossibleWithoutRealInterpretations(unittest.TestCase):
    """The fail-closed rule, exercised through the REAL `verdict_for`.

    Not a reimplementation of the branch: a hand-reproduction of it gave the wrong
    answer once already, which is why the rule is a function now.
    """

    CLIENT = "ERE"

    def _adjudicated(self, records):
        """Run the real adjudicator over (case, record) pairs, as main() does."""
        out = []
        for case, record in zip(MANIFEST["cases"], records):
            verdict = harness.adjudicate(case, record, LEGACY_OK)
            verdict["raw_evidence"] = record
            out.append(verdict)
        return out

    def _verdict(self, records):
        adjudicated = self._adjudicated(records)
        return harness.verdict_for(adjudicated, client_id=self.CLIENT,
                                   expected_cases=len(MANIFEST["cases"]))

    def _good(self, case):
        """A record for a case that completed on the required model."""
        if case["expected_slice1_eligible"]:
            return evidence_for(case, execution=executed(case))
        return evidence_for(case, disposition="INELIGIBLE")

    def _interpreter_failure(self, case):
        """Exactly what plan_shadow_wiring writes when no model answered."""
        return {
            "correlation_id": f"shadow_nokey_{case['case_id']}",
            "request": {"question": case["question"], "client_id": self.CLIENT},
            "disposition": harness.INTERPRETER_FAILURE,
            "model": {"model_id": "", "raw_payload": None,
                      "failure": {"code": "MODEL_UNAVAILABLE",
                                  "subject": "interpreter",
                                  "detail": "no ANTHROPIC_API_KEY"}},
            "interpretation": {"produced_intent": False,
                               "candidate_intent": None},
            "compiler": {"outcome": "REFUSE", "plan": None, "plan_id": None,
                         "reason_codes": ["MODEL_UNAVAILABLE"]},
            "eligibility": None,
            "execution": None,
        }

    # -- A ------------------------------------------------------------------
    def test_A_twelve_interpreter_failures_fail(self):
        records = [self._interpreter_failure(c) for c in MANIFEST["cases"]]
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.FAIL)
        self.assertEqual(tallies["successful_interpretations"], 0)
        self.assertEqual(tallies["interpreter_failures"], 12)
        self.assertIn("did not happen", tallies["fail_reason"])

    # -- B ------------------------------------------------------------------
    def test_B_one_interpreter_failure_among_eleven_successes_fails(self):
        records = [self._good(c) for c in MANIFEST["cases"]]
        records[5] = self._interpreter_failure(MANIFEST["cases"][5])
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.FAIL)
        self.assertEqual(tallies["successful_interpretations"], 11)
        self.assertEqual(tallies["interpreter_failures"], 1)

    # -- C ------------------------------------------------------------------
    def test_C_one_wrong_model_fails(self):
        records = [self._good(c) for c in MANIFEST["cases"]]
        records[2]["model"]["model_id"] = "claude-haiku-4-5-20251001"
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.FAIL)
        self.assertEqual(tallies["model_substitutions"], 1)
        self.assertIn("claude-haiku-4-5-20251001", tallies["fail_reason"])

    def test_C2_an_empty_model_identity_is_not_a_success(self):
        """Rule 4: "no model answered" must never read as "the right model did"."""
        records = [self._good(c) for c in MANIFEST["cases"]]
        records[0]["model"]["model_id"] = ""
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.FAIL)
        self.assertEqual(tallies["successful_interpretations"], 11)
        self.assertEqual(tallies["model_substitutions"], 0,
                         "an empty identity is absence, not substitution")

    # -- D ------------------------------------------------------------------
    def test_D_twelve_good_opus_interpretations_can_still_pass(self):
        records = [self._good(c) for c in MANIFEST["cases"]]
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.PASS, tallies.get("fail_reason"))
        self.assertEqual(tallies["successful_interpretations"], 12)
        self.assertEqual(tallies["models_returned"], ["claude-opus-5"])

    def test_D2_a_dated_opus_identifier_still_counts(self):
        """One notion of "is Opus 5", shared with the substitution count."""
        records = [self._good(c) for c in MANIFEST["cases"]]
        for record in records:
            record["model"]["model_id"] = "claude-opus-5-20260101"
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.PASS, tallies.get("fail_reason"))
        self.assertEqual(tallies["model_substitutions"], 0)

    # -- E ------------------------------------------------------------------
    def test_E_a_harmless_movement_on_an_ineligible_control_does_not_fail(self):
        """B03's plan moves, the gate still blocks it, nothing executes.

        The deviation is reported; it must not become a blanket failure merely for
        carrying that label, because the eligibility boundary did its job.
        """
        records = [self._good(c) for c in MANIFEST["cases"]]
        index = next(i for i, c in enumerate(MANIFEST["cases"])
                     if c["case_id"] == "B03")
        case = MANIFEST["cases"][index]
        moved = self._good(case)
        # A different population base — a real interpretation movement — while the
        # capability, and so the gate's refusal, is unchanged.
        moved["compiler"]["plan"]["population"]["base"] = "forecast"
        records[index] = moved

        adjudicated = self._adjudicated(records)
        this_case = next(c for c in adjudicated if c["case_id"] == "B03")
        self.assertEqual(this_case["classification"],
                         harness.INTERPRETATION_DEVIATION)
        self.assertFalse(this_case["silent_drops"])
        verdict, tallies = harness.verdict_for(
            adjudicated, client_id=self.CLIENT,
            expected_cases=len(MANIFEST["cases"]))
        self.assertEqual(verdict, harness.PASS, tallies.get("fail_reason"))
        self.assertEqual(tallies["classifications"]
                         [harness.INTERPRETATION_DEVIATION], 1)

    def test_E2_a_deviation_that_costs_an_eligible_case_its_filter_still_fails(self):
        """The other half of the principle: the facet checks still bite."""
        records = [self._good(c) for c in MANIFEST["cases"]]
        index = next(i for i, c in enumerate(MANIFEST["cases"])
                     if c["case_id"] == "A01")
        records[index]["execution"]["bound_spec"]["filters"] = {}
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.FAIL)
        self.assertEqual(
            tallies["classifications"][harness.DETERMINISTIC_EXECUTION_DEFECT], 1)

    # -- the rule's own guards ----------------------------------------------
    def test_an_interpretation_deviation_is_not_a_defect_by_itself(self):
        self.assertNotIn(harness.INTERPRETATION_DEVIATION, harness.DEFECTS)

    def test_a_missing_record_still_reads_as_inconclusive_not_failure(self):
        records = [self._good(c) for c in MANIFEST["cases"]]
        records[4] = None
        verdict, tallies = self._verdict(records)
        self.assertEqual(verdict, harness.INCONCLUSIVE)
        self.assertEqual(tallies["async_evidence_lost"], 1)


class TestItRunsOnABareRunner(unittest.TestCase):
    """The GitHub runner's import path, reproduced.

    No acceptance workflow in this estate installs Python dependencies, so the
    harness has to import and adjudicate with nothing but the standard library
    available. It did not: `semantics_of` imported `preregister`, which imports
    `mi_agent.plan_runtime_adapter`, which loads `mi_agent/__init__.py`, which
    imports PyYAML — and run 34520491626 died on
    `ModuleNotFoundError: No module named 'yaml'` mid-bank, after two live
    interpretations had been spent.

    So this runs in a SUBPROCESS with `yaml`, `pandas` and `mi_agent` made
    unimportable, which is a harsher environment than the runner and therefore a
    sound proxy for it.
    """

    # Placeholders are substituted with `replace`, not `format`: the script is
    # Python and full of braces, and formatting it treated them as fields.
    BLOCKER = r'''
import sys, json
BLOCKED = ("yaml", "pandas", "numpy", "mi_agent")

class Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("blocked for this test: " + name)
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, @HARNESS_DIR@)
sys.path.insert(0, @REPO_ROOT@)

import run_acceptance as h            # must import with nothing installed

case = json.loads(@CASE@)
record = json.loads(@RECORD@)
legacy = {"ok": True, "http_status": 200, "value": 195.0, "route": None,
          "transport_error": False}
verdict = h.adjudicate(case, record, legacy)
verdict["raw_evidence"] = record
overall, tallies = h.verdict_for([verdict], client_id="ERE", expected_cases=1)
print(json.dumps({"classification": verdict["classification"],
                  "completed": tallies["successful_interpretations"],
                  "overall": overall,
                  "semantics": h.semantics_of(record["compiler"]["plan"])}))
'''

    def _run_blocked(self, case, record):
        import subprocess
        script = (self.BLOCKER
                  .replace("@HARNESS_DIR@", repr(str(_HARNESS)))
                  .replace("@REPO_ROOT@", repr(str(_REPO_ROOT)))
                  .replace("@CASE@", repr(json.dumps(case)))
                  .replace("@RECORD@", repr(json.dumps(record))))
        finished = subprocess.run([sys.executable, "-c", script], text=True,
                                  capture_output=True, cwd=str(_REPO_ROOT))
        if finished.returncode != 0:
            self.fail(f"the harness cannot run on a bare runner:\n"
                      f"{finished.stderr[-3000:]}")
        return json.loads(finished.stdout.strip().splitlines()[-1])

    def test_it_imports_and_adjudicates_with_no_product_dependencies(self):
        case = BY_ID["A01"]
        record = evidence_for(case, execution=executed(case))
        out = self._run_blocked(case, record)
        self.assertEqual(out["classification"], harness.EXACT_SEMANTIC_PARITY)
        self.assertEqual(out["completed"], 1)
        self.assertEqual(out["overall"], harness.PASS)

    def test_the_projection_is_computed_without_importing_mi_agent(self):
        """The exact call that crashed the deployed run."""
        case = BY_ID["A04"]
        record = evidence_for(case, execution=executed(case))
        out = self._run_blocked(case, record)
        self.assertEqual(out["semantics"]["dimensions"],
                         list(case["expected_semantics"]["dimensions"]))

    def test_the_harness_names_no_product_module_at_module_scope(self):
        source = (_HARNESS / "run_acceptance.py").read_text()
        head = source.split("def publish_profile_credentials")[0]
        for forbidden in ("import preregister", "from mi_agent", "import mi_agent",
                          "import yaml", "import pandas"):
            self.assertNotIn(forbidden, head,
                             f"{forbidden} puts the product's dependency tree on "
                             f"the runner")


class TestTheProjectionMatchesTheFrozenManifest(unittest.TestCase):
    """The drift guard between the harness's copy and `preregister`'s.

    Two copies of the projection exist: the pure one in the harness, and
    `preregister._semantics`, which built the manifest. They are only safe while
    they agree, and the committed manifest is the arbiter — its
    `expected_semantics` is what every live plan is compared against.
    """

    FROZEN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
              / "run8_135_signoff_2b00172.json")

    def test_every_case_projects_to_its_preregistered_semantics(self):
        frozen = json.loads(self.FROZEN.read_text())
        by_question_id = {r["question_id"]: r for r in frozen["results"]}
        checked = 0
        for case in MANIFEST["cases"]:
            plan = by_question_id[case["frozen_question_id"]].get("plan")
            self.assertIsNotNone(plan, case["frozen_question_id"])
            self.assertEqual(
                harness._comparable(harness.semantics_of(plan)),
                harness._comparable(case["expected_semantics"]),
                f"{case['case_id']} ({case['frozen_question_id']}): the harness's "
                f"projection no longer matches the manifest it adjudicates against")
            checked += 1
        self.assertEqual(checked, 12)


class TestTheScrubber(unittest.TestCase):

    def test_a_secret_value_is_removed_wherever_it_appears(self):
        clean = harness.scrub({"a": {"b": ["prefix SECRETTOKEN suffix"]}},
                              ["SECRETTOKEN"])
        self.assertEqual(clean["a"]["b"], ["[REDACTED]"])

    def test_a_credential_shaped_key_is_dropped(self):
        clean = harness.scrub(
            {"Authorization": "Bearer x", "userPWD": "y", "keep": 1}, [])
        self.assertEqual(clean, {"keep": 1})

    def test_saving_refuses_if_a_secret_somehow_survives(self):
        """The backstop, for a secret the scrubber cannot see.

        `scrub` masks string VALUES and drops credential-shaped KEYS. A secret
        appearing as a key whose name reads as ordinary — `XYZZYVALUE` below —
        slips both, which is exactly why `_save` re-scans the serialised body
        before writing. A key containing `token`, `secret`, `password` and so on
        is dropped by the key filter instead, so it never reaches the scan.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.json")
            with self.assertRaises(SystemExit):
                harness._save({"XYZZYVALUE": 1}, out, ["XYZZYVALUE"])
            self.assertFalse(Path(out).exists(),
                             "the file must not be written at all")

    def test_a_credential_shaped_key_never_reaches_the_backstop(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.json")
            harness._save({"access_token": "abc", "keep": 1}, out, ["abc"])
            saved = json.loads(Path(out).read_text())
            self.assertEqual(saved, {"keep": 1})

    def test_an_ordinary_report_saves_and_round_trips(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.json")
            harness._save({"verdict": "PASS", "cases": []}, out, ["abc"])
            self.assertEqual(json.loads(Path(out).read_text())["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
