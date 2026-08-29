"""OCC go-live acceptance: Blob -> OCC -> Gates -> MI, and -> Annex 2.

These are the sprint's definition of done, executed rather than asserted about.
Every test drives the real production entrypoint with the real agents:

    raw file at the governed blob prefix
      -> Event Grid intake (apps.blob_trigger_app.occ_intake)
      -> OCC batch, workflow selection from approved configuration
      -> Gate 1 Onboarding -> operator review -> Gate 2 -> Gate 3
      -> platform canonical -> publication approval -> MI
      -> (regime_required only) Gate 4 -> Gate 5 -> XML -> XSD

Two unrelated equity release lenders are rehearsed end to end. Client A is MI
only and recurs for three months; Client B is regime-required, speaks a
different vocabulary, and carries management statuses ESMA has no code for.

The rehearsals are expensive (real onboarding over real tapes), so each one is
performed once in a module-scoped fixture and asserted many times.
"""

from __future__ import annotations

import pytest

from operations_control.contracts import (

    KIND_PUBLICATION,
    RUN_AWAITING_PUBLICATION,
    RUN_NEEDS_REVIEW,
    ST_COMPLETED,
    ST_READY,
)

from .occ_go_live_harness import (
    ALPHA_MAPPING,
    BETA_MAPPING,
    LEI_A,
    LEI_B,
    alpha_tape,
    answer_mapping_queue,
    arrive,
    beta_tape,
    blocking_validation_fields,
    bootstrap,
    central_tape,
    gate_steps,
    handoff,
    name_the_file,
    onboard,
    settle,
    validated_canonical,
)

pytestmark = pytest.mark.slow

ALPHA_STATIC = {"portfolio_id": "direct_001",
                "source_portfolio_id": "direct_001",
                "source_portfolio_type": "direct"}
BETA_STATIC = {"portfolio_id": "direct_002",
               "source_portfolio_id": "direct_002",
               "source_portfolio_type": "direct"}


# --------------------------------------------------------------------------- #
# Client A — MI only, three consecutive months
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def alpha(tmp_path_factory):
    """A brand-new MI-only lender, rehearsed from zero through three months."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("alpha")
    env = bootstrap(tmp, mp)
    out = {"env": env, "months": {}}
    try:
        out["activation"] = onboard(
            env, client_id="ALPHA", name="Alpha Equity Release",
            portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))

        from apps.blob_trigger_app import occ_intake
        out["selection"] = occ_intake.outcome_for_source(
            env["engine"], "ALPHA", "direct_001", "funded")

        # --- month 1: first contact, unknown vocabulary ------------------- #
        tape = alpha_tape(tmp / "alpha_2025-11-30.csv", "2025-11-30")
        r = arrive(env, client_id="ALPHA", portfolio_id="direct_001",
                   period="2025-11-30", tape=tape)
        out["arrival"] = r
        first = settle(env, "ALPHA", r["workflow_id"])
        # Snapshot everything about first contact NOW: the orchestrator state
        # file and the stage results are overwritten by the rerun below, so a
        # WorkflowRun object held across it would read back the later state.
        out["m1_status"] = first.status
        out["m1_steps"] = gate_steps(first)
        out["m1_handoff"] = handoff(first)
        out["m1_mapping"] = env["store"].load_result("ALPHA", first.workflow_id,
                                                     "mapping")
        out["m1_queue"] = env["store"].open_decisions("ALPHA", first.workflow_id)
        out["m1_loan_key_q"] = [d for d in out["m1_queue"]
                                if "loankey" in d["decision_id"]]
        tape = central_tape(first)
        out["m1_identity_before"] = (sorted(tape["loan_identifier"].astype(str))[:1]
                                     if tape is not None else [])
        for d in out["m1_loan_key_q"]:
            env["engine"].resolve_decision(
                client_id="ALPHA", decision_id=d["decision_id"],
                action="approve", actor="Operator", value="ACCT_REF",
                scope="portfolio", actor_is_admin=True)
        out["m1_answers"] = answer_mapping_queue(
            env, "ALPHA", first.workflow_id,
            mapping=ALPHA_MAPPING, static=ALPHA_STATIC,
            not_applicable_reason="not held in an equity release book")
        out["months"]["2025-11-30"] = settle(env, "ALPHA", first.workflow_id)
        env["engine"].approve_publication(
            client_id="ALPHA", workflow_id=first.workflow_id, actor="Operator")

        # --- months 2 and 3: same schema, no developer -------------------- #
        for period in ("2025-12-31", "2026-01-31"):
            t = alpha_tape(tmp / f"alpha_{period}.csv", period)
            rr = arrive(env, client_id="ALPHA", portfolio_id="direct_001",
                        period=period, tape=t)
            run = settle(env, "ALPHA", rr["workflow_id"])
            out["months"][period] = run
            out.setdefault("recurring_questions", {})[period] = [
                d for d in env["store"].open_decisions("ALPHA", run.workflow_id)
                if d["kind"] != KIND_PUBLICATION]
        yield out
    finally:
        mp.undo()


class TestClientAOnboarding:
    def test_a_new_client_is_created_without_bespoke_python(self, alpha):
        """Definition of done #1."""
        registry = alpha["env"]["registry"]
        records = registry.records_for_dataset("ALPHA", "direct_001", "funded")
        assert records, "activation registered no source for the new client"
        assert "ALPHA" in alpha["env"]["store"].known_clients()

    def test_a_blob_arrival_creates_the_right_workflow(self, alpha):
        """Definition of done #2 and #9: the path decides, nothing guesses."""
        assert alpha["selection"] == "mi"      # no ESMA product -> MI only
        assert alpha["arrival"]["registered"] is True
        assert alpha["arrival"]["workflow_id"]

    def test_gate_two_refuses_an_unready_gate_one_package(self, alpha):
        """The handoff contract does the refusing — OCC does not re-implement it."""
        assert alpha["m1_steps"]["onboard"] == "done"
        assert alpha["m1_steps"]["transform"] == "halted"
        assert alpha["m1_handoff"]["ready_for_transformation_validation"] is False

    def test_occ_reports_what_gate_one_found(self, alpha):
        """Definition of done #4. Not a fixed sentence about approved rules."""
        assert alpha["m1_status"] == RUN_NEEDS_REVIEW
        gar = alpha["m1_mapping"]
        assert gar.status == "needs_review"
        assert "All fields were matched" not in gar.summary
        assert gar.decisions_required, "the queue was hidden from the operator"
        # The numbers shown are Gate 1's own, not a second opinion.
        table = [e for e in gar.evidence if e.get("label") == "What Gate 1 found"]
        assert table, "no coverage evidence surfaced"
        assert (table[0]["data"]["Of those, blocking"]
                == alpha["m1_handoff"]["blocking_decision_count"])

    def test_the_queue_is_answerable_by_a_person(self, alpha):
        """Every question offers actions the engine can actually execute."""
        assert alpha["m1_queue"], "no decisions were raised"
        for d in alpha["m1_queue"]:
            values = {o["value"] for o in d["options"]}
            assert values, f"{d['decision_id']} offered no options"
            assert "map_source_column" not in values, (
                "offers an action name the apply step does not support")
            if (d.get("subject") or {}).get("artefact") == "target_first_decisions":
                # "I need to ask the lender" is always available on a question
                # about how to treat a field. (The loan-key question offers
                # column names rather than actions, so it has no defer option.)
                assert "defer" in values
        assert alpha["m1_answers"]["mapped"] >= 10


class TestClientAReachesGovernedMI:
    def test_all_three_gates_ran(self, alpha):
        """Definition of done #3."""
        steps = gate_steps(alpha["months"]["2025-11-30"])
        assert steps["onboard"] == "done"
        assert steps["transform"] == "done"      # Gate 2
        assert steps["validate"] == "done"       # Gate 3
        assert steps["stamp"] == "done"

    def test_the_operators_answers_reached_the_canonical(self, alpha):
        """An approval that applies to nothing is not an approval."""
        tape = central_tape(alpha["months"]["2025-11-30"])
        assert tape is not None and len(tape) == 30
        for field in ("account_status", "current_outstanding_balance",
                      "current_interest_rate", "original_principal_balance",
                      "origination_date", "current_valuation_amount",
                      "youngest_borrower_age"):
            assert field in tape.columns, f"{field} never reached the tape"
            assert tape[field].notna().all()

    def test_the_run_reaches_publication(self, alpha):
        run = alpha["months"]["2025-11-30"]
        assert run.status == RUN_AWAITING_PUBLICATION
        assert run.stage_status("validation") == ST_COMPLETED
        assert run.stage_status("publication") == ST_READY


class TestClientARecurs:
    """Definition of done #5 and #13: months 2 and 3 need no developer."""

    @pytest.mark.parametrize("period", ["2025-12-31", "2026-01-31"])
    def test_a_later_month_asks_nothing_and_publishes(self, alpha, period):
        run = alpha["months"][period]
        assert alpha["recurring_questions"][period] == []
        assert run.status == RUN_AWAITING_PUBLICATION
        steps = gate_steps(run)
        assert [steps[k] for k in ("onboard", "transform", "validate", "stamp")] \
            == ["done"] * 4

    def test_the_standing_contract_is_scoped_to_the_source(self, alpha):
        """It is reachable for this client's portfolio and nowhere else."""
        env = alpha["env"]
        layout, storage = env["store"].layout, env["storage"]
        mine = layout.approved_decisions_uri(
            "ALPHA", "direct_001", "funded",
            "34_target_first_decisions_approved.yaml")
        assert storage.exists(mine)
        for other in (("ALPHA", "direct_999"), ("OTHER", "direct_001")):
            assert not storage.exists(layout.approved_decisions_uri(
                other[0], other[1], "funded",
                "34_target_first_decisions_approved.yaml"))


# --------------------------------------------------------------------------- #
# Client B — a second, unrelated lender: MI + Annex 2
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def beta(tmp_path_factory):
    """A regime-required lender whose file is named after its own product."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("beta")
    env = bootstrap(tmp, mp)
    out = {"env": env}
    try:
        onboard(env, client_id="BETA", name="Beta Lifetime Mortgages",
                portfolio_id="direct_002", lei=LEI_B,
                products=("mi_reporting", "esma_annex2"))

        from apps.blob_trigger_app import occ_intake
        out["selection"] = occ_intake.outcome_for_source(
            env["engine"], "BETA", "direct_002", "funded")

        tape = beta_tape(tmp / "beta_2026-01-31.csv", "2026-01-31")
        r = arrive(env, client_id="BETA", portfolio_id="direct_002",
                   period="2026-01-31", tape=tape, filename="PolicyExtract.csv")
        out["arrival"] = r
        out["batch_before_naming"] = env["engine"].intake.load_batch(
            "BETA", r["batch_id"])
        out["role_questions"] = name_the_file(env, "BETA")
        batch = env["engine"].intake.load_batch("BETA", r["batch_id"])
        out["batch_after_naming"] = batch

        run = settle(env, "BETA", batch["workflow_id"])
        out["first"] = run
        out["queue"] = env["store"].open_decisions("BETA", run.workflow_id)
        answer_mapping_queue(env, "BETA", run.workflow_id,
                             mapping=BETA_MAPPING, static=BETA_STATIC)
        out["run"] = settle(env, "BETA", run.workflow_id)
        yield out
    finally:
        mp.undo()


class TestClientBIsRegimeRequired:
    def test_regime_required_selects_the_annex2_workflow(self, beta):
        """Definition of done #9 — from approved configuration, not a guess."""
        assert beta["selection"] == "mi_annex2"
        records = beta["env"]["registry"].records_for_dataset(
            "BETA", "direct_002", "funded")
        assert any(r.regime_required for r in records)

    def test_an_unrecognised_file_is_asked_about_not_ignored(self, beta):
        """A lender's tape is called whatever the lender calls it."""
        assert beta["batch_before_naming"]["status"] == "review_required"
        assert beta["role_questions"] == 1
        assert beta["batch_after_naming"]["missing_input_roles"] == []
        assert beta["batch_after_naming"]["workflow_id"]

    def test_the_full_gate_chain_ran_for_the_regime_book_too(self, beta):
        steps = gate_steps(beta["run"])
        assert [steps[k] for k in ("onboard", "transform", "validate", "stamp")] \
            == ["done"] * 4


class TestManagementEnumsSurvive:
    """The sprint's non-negotiable principle 4.

    "Probate - awaiting sale" is a real servicing state. ESMA has no code for
    it; management information depends on it. It must reach MI as written.
    """

    def test_the_lenders_own_words_reach_mi_unchanged(self, beta):
        tape = central_tape(beta["run"])
        assert tape is not None
        seen = set(tape["account_status"].astype(str))
        assert "Probate - awaiting sale" in seen
        assert {"Live", "Redeemed", "In possession", "Moved to LTC"} <= seen

    def test_an_unknown_enum_does_not_null_the_value(self, beta):
        """Gate 2 leaves it alone; Gate 3 warns rather than rejecting."""
        canonical = validated_canonical(beta["run"])
        assert canonical is not None
        assert canonical["account_status"].notna().all()
        assert "Probate - awaiting sale" in set(
            canonical["account_status"].astype(str))

    def test_an_unknown_enum_is_not_a_blocking_validation_error(self, beta):
        assert "account_status" not in blocking_validation_fields(beta["run"])


class TestClientIsolation:
    """Definition of done #12: two lenders, one codebase, no shared state."""

    def test_each_client_is_governed_by_its_own_configuration(self, alpha, beta):
        from operations_control.configuration.resolver import (
            EffectiveConfigResolver)
        import yaml
        a = EffectiveConfigResolver(alpha["env"]["store"],
                                    alpha["env"]["engine"].rules)
        b = EffectiveConfigResolver(beta["env"]["store"],
                                    beta["env"]["engine"].rules)
        cfg_a = yaml.safe_load(a.client_config_for("ALPHA").read_text("utf-8"))
        cfg_b = yaml.safe_load(b.client_config_for("BETA").read_text("utf-8"))
        assert cfg_a != cfg_b
        # Neither is the repository default standing in for both.
        assert LEI_A in str(cfg_a) and LEI_A not in str(cfg_b)
        assert LEI_B in str(cfg_b) and LEI_B not in str(cfg_a)

    def test_neither_client_can_see_the_others_rules(self, alpha, beta):
        a_rules = alpha["env"]["engine"].rules.applicable(
            client_id="ALPHA", portfolio_id="direct_001")
        assert a_rules, "client A approved rules but none are applicable"
        assert not alpha["env"]["engine"].rules.applicable(
            client_id="BETA", portfolio_id="direct_002")
        assert not beta["env"]["engine"].rules.applicable(
            client_id="ALPHA", portfolio_id="direct_001")

    def test_the_two_books_map_the_same_meaning_from_different_words(
            self, alpha, beta):
        """Same canonical field, different lender column, no shared alias."""
        assert ALPHA_MAPPING["current_outstanding_balance"] == "BAL_OS"
        assert BETA_MAPPING["current_outstanding_balance"] == "Loan Balance"
        for tape in (central_tape(alpha["months"]["2025-11-30"]),
                     central_tape(beta["run"])):
            assert tape["current_outstanding_balance"].notna().all()


# --------------------------------------------------------------------------- #
# Negative paths — every one ends in acceptance, review, a preserved warning,
# or a fail-closed block. None ends in a silent guess.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def blank(tmp_path, monkeypatch):
    """An empty Trakt: no clients, nothing onboarded."""
    return bootstrap(tmp_path, monkeypatch)


class TestUngovernedClientsAreRefused:
    def test_a_delivery_for_an_unknown_client_is_blocked(self, blank, tmp_path):
        """Not processed under the repository's default configuration."""
        tape = alpha_tape(tmp_path / "stranger.csv")
        r = arrive(blank, client_id="STRANGER", portfolio_id="direct_001",
                   period="2025-11-30", tape=tape)
        assert not r.get("workflow_id")
        batch = blank["engine"].intake.load_batch("STRANGER", r["batch_id"])
        assert batch["status"] == "blocked"
        assert batch["status_reason"] == \
            blank["engine"].UNACTIVATED_SENTENCE

    def test_the_same_client_runs_once_it_is_activated(self, blank, tmp_path):
        """The block is about governance, not about the files."""
        onboard(blank, client_id="LATE", name="Late Onboarding",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        tape = alpha_tape(tmp_path / "late.csv")
        r = arrive(blank, client_id="LATE", portfolio_id="direct_001",
                   period="2025-11-30", tape=tape)
        assert r["workflow_id"]


class TestCoreCanonicalCannotBeBypassed:
    """Definition of done #6: an incomplete core canonical never reaches MI."""

    def _run_without(self, env, tmp_path, *, drop: str, client: str):
        import pandas as pd
        onboard(env, client_id=client, name=f"{client} Lending",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        tape = alpha_tape(tmp_path / f"{client}.csv")
        df = pd.read_csv(tape)
        df = df.drop(columns=[drop]) if drop in df.columns else df
        df.to_csv(tape, index=False)
        r = arrive(env, client_id=client, portfolio_id="direct_001",
                   period="2025-11-30", tape=tape)
        run = settle(env, client, r["workflow_id"])
        mapping = {k: v for k, v in ALPHA_MAPPING.items() if v != drop}
        answer_mapping_queue(env, client, run.workflow_id,
                             mapping=mapping, static=ALPHA_STATIC)
        return settle(env, client, run.workflow_id)

    def test_a_missing_balance_column_blocks_before_publication(
            self, blank, tmp_path):
        """No balance, no report. Which gate refuses is not the point."""
        run = self._run_without(blank, tmp_path, drop="BAL_OS", client="NOBAL")
        assert run.stage_status("publication") != ST_READY
        assert run.status != RUN_AWAITING_PUBLICATION
        # Whichever gate stopped it, the canonical must not have been assembled.
        assert run.stage_status("assembly") != ST_COMPLETED
        # Gate 2 refuses an unready package before Gate 3 sees it, so a core
        # violation is reported only when the run got that far.
        blocked = blocking_validation_fields(run)
        if blocked:
            assert any("balance" in f for f in blocked), blocked

    def test_a_blank_status_column_blocks_before_publication(
            self, blank, tmp_path):
        """Present but empty is not present: CORE002, not a silent pass."""
        import pandas as pd
        onboard(blank, client_id="NOSTAT", name="No Status Lending",
                portfolio_id="direct_001", lei=LEI_A, products=("mi_reporting",))
        tape = alpha_tape(tmp_path / "nostat.csv")
        df = pd.read_csv(tape)
        df["STATUS_CD"] = ""
        df.to_csv(tape, index=False)
        r = arrive(blank, client_id="NOSTAT", portfolio_id="direct_001",
                   period="2025-11-30", tape=tape)
        run = settle(blank, "NOSTAT", r["workflow_id"])
        answer_mapping_queue(blank, "NOSTAT", run.workflow_id,
                             mapping=ALPHA_MAPPING, static=ALPHA_STATIC)
        run = settle(blank, "NOSTAT", run.workflow_id)
        assert run.status != RUN_AWAITING_PUBLICATION
        assert run.stage_status("publication") != ST_READY
        assert run.stage_status("assembly") != ST_COMPLETED
        blocked = blocking_validation_fields(run)
        if blocked:
            assert "account_status" in blocked, blocked



class TestTheLoanKeyIsAskedAboutNotGuessed:
    """Phase I. Three unique columns; only one of them is the loan.

    Keying on the customer merges a borrower's separate loans and changes every
    figure in the report. Gate 1 has no way to know which is which, so the one
    unacceptable outcome is choosing silently.
    """

    def test_the_question_is_raised_with_every_candidate(self, alpha):
        assert len(alpha["m1_loan_key_q"]) == 1
        d = alpha["m1_loan_key_q"][0]
        assert {o["value"] for o in d["options"]} == {"ACCT_REF", "ROLL_NO",
                                                      "CUST_ID"}
        assert d["blocking"] is False        # it does not stop the run
        assert d["recommendation"]["value"]  # but it does say what it used

    def test_the_operators_answer_governs_the_identity(self, alpha):
        """Before: the customer. After: the account."""
        assert alpha["m1_identity_before"], "no tape was built on first contact"
        assert alpha["m1_identity_before"][0].startswith(("9", "C9")), \
            alpha["m1_identity_before"]
        tape = central_tape(alpha["months"]["2025-11-30"])
        after = sorted(tape["loan_identifier"].astype(str))
        assert after[0].startswith(("100000", "ACC100000")), after[:3]
        assert len(set(after)) == len(tape)

    def test_the_choice_is_durable(self, alpha):
        """Later months are keyed the same way without being asked again."""
        for period in ("2025-12-31", "2026-01-31"):
            tape = central_tape(alpha["months"][period])
            assert sorted(tape["loan_identifier"].astype(str))[0].startswith(
                ("100000", "ACC100000"))
