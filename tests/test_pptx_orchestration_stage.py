"""Orchestration tests for the investor-PPTX final stage.

These are ORCHESTRATION tests only — the generator itself is mocked (no
matplotlib / python-pptx execution, no slide-content assertions). They verify
that a successful Azure blob-triggered run wires the existing PPTX generator in
as its final artifact, writes the deck beneath ``<run_dir>/reports``, updates
the run manifest, records failures without failing the run, and is idempotent on
replay.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from apps.blob_trigger_app import orchestrator_invoke as OI
from apps.blob_trigger_app import pptx_stage as S


def _fake_generator(create=True):
    """Return a generator stub that writes a dummy deck to --output."""
    def _gen(argv):
        out = argv[argv.index("--output") + 1]
        if create:
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PK\x03\x04 dummy pptx")
        return 0
    return _gen


def _seed_run(run_dir: Path, status="done", client_id="ERE"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(json.dumps({
        "run_id": run_dir.name, "client_id": client_id, "status": status,
        "central_canonical_path": "out_platform/platform_canonical_typed.csv",
    }, indent=2))


# --------------------------------------------------------------------------- #
# Stage-level behaviour.
# --------------------------------------------------------------------------- #
class TestPptxStage(unittest.TestCase):
    def test_success_writes_deck_and_updates_manifest(self):
        with TemporaryDirectory() as td:
            run_dir = Path(td) / "orun_1"
            _seed_run(run_dir)
            with mock.patch.object(S, "_invoke_generator",
                                   side_effect=_fake_generator()):
                art = S.generate_investor_pptx(run_dir, client_name="ERE",
                                               as_of_date="2026-01-31")
            # Deck written beneath run_dir/reports.
            deck = run_dir / "reports" / "investor_pack.pptx"
            self.assertTrue(deck.exists())
            self.assertEqual(art["status"], "available")
            self.assertEqual(art["path"], "reports/investor_pack.pptx")
            self.assertEqual(art["generator"], "mi_agent_pptx")
            # Manifest updated with the artifact (existing keys preserved).
            man = json.loads((run_dir / "run_state.json").read_text())
            self.assertEqual(man["client_id"], "ERE")
            self.assertEqual(
                man["artifacts"]["investor_pack_pptx"]["status"], "available")

    def test_client_name_and_as_of_passed_to_generator(self):
        captured = {}

        def _capture(argv):
            captured["argv"] = argv
            out = argv[argv.index("--output") + 1]
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x")
            return 0

        with TemporaryDirectory() as td:
            run_dir = Path(td) / "orun_2"
            _seed_run(run_dir)
            with mock.patch.object(S, "_invoke_generator", side_effect=_capture):
                S.generate_investor_pptx(run_dir, client_name="Aurora",
                                         as_of_date="2026-01-31")
        argv = captured["argv"]
        self.assertIn("--client-name", argv)
        self.assertEqual(argv[argv.index("--client-name") + 1], "Aurora")
        self.assertEqual(argv[argv.index("--as-of-date") + 1], "2026-01-31")
        self.assertEqual(argv[argv.index("--run-dir") + 1], str(run_dir))

    def test_failure_records_failed_artifact_without_raising(self):
        with TemporaryDirectory() as td:
            run_dir = Path(td) / "orun_3"
            _seed_run(run_dir)
            with mock.patch.object(S, "_invoke_generator",
                                   side_effect=RuntimeError("boom")):
                art = S.generate_investor_pptx(run_dir, client_name="ERE",
                                               mandatory=False)
            self.assertEqual(art["status"], "failed")
            self.assertIn("boom", art["error"])
            man = json.loads((run_dir / "run_state.json").read_text())
            self.assertEqual(
                man["artifacts"]["investor_pack_pptx"]["status"], "failed")

    def test_missing_output_is_treated_as_failure(self):
        with TemporaryDirectory() as td:
            run_dir = Path(td) / "orun_4"
            _seed_run(run_dir)
            # Generator returns rc=0 but writes nothing.
            with mock.patch.object(S, "_invoke_generator",
                                   side_effect=_fake_generator(create=False)):
                art = S.generate_investor_pptx(run_dir, client_name="ERE")
            self.assertEqual(art["status"], "failed")

    def test_mandatory_failure_raises(self):
        with TemporaryDirectory() as td:
            run_dir = Path(td) / "orun_5"
            _seed_run(run_dir)
            with mock.patch.object(S, "_invoke_generator",
                                   side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    S.generate_investor_pptx(run_dir, client_name="ERE",
                                             mandatory=True)

    def test_replay_overwrites_single_deck_and_refreshes_timestamp(self):
        with TemporaryDirectory() as td:
            run_dir = Path(td) / "orun_6"
            _seed_run(run_dir)
            stamps = iter(["2026-07-02T10:00:00+00:00",
                           "2026-07-02T11:00:00+00:00"])
            with mock.patch.object(S, "_invoke_generator",
                                   side_effect=_fake_generator()), \
                 mock.patch.object(S, "_now_iso", side_effect=lambda: next(stamps)):
                a1 = S.generate_investor_pptx(run_dir, client_name="ERE")
                a2 = S.generate_investor_pptx(run_dir, client_name="ERE")
            # Only one deck file, timestamp refreshed, single manifest artifact.
            reports = list((run_dir / "reports").glob("*.pptx"))
            self.assertEqual(len(reports), 1)
            self.assertNotEqual(a1["generated_at"], a2["generated_at"])
            man = json.loads((run_dir / "run_state.json").read_text())
            self.assertEqual(
                man["artifacts"]["investor_pack_pptx"]["generated_at"],
                a2["generated_at"])

    def test_disabled_flag_skips_generation(self):
        with mock.patch.dict("os.environ",
                             {"TRAKT_INVESTOR_PPTX_ENABLED": "false"}):
            self.assertFalse(S.pptx_enabled())
        self.assertTrue(S.pptx_enabled())  # default on


# --------------------------------------------------------------------------- #
# Invoker wiring — the Azure orchestration boundary.
# --------------------------------------------------------------------------- #
class TestInvokerWiring(unittest.TestCase):
    def _fake_state(self, out_root: str, status="done", client_id="ERE"):
        run_id = "orun_wire"
        state = SimpleNamespace(
            run_id=run_id, status=status, client_id=client_id,
            out_root=out_root, central_canonical_path="x.csv", blockers=[],
        )
        state.state_path = lambda: Path(out_root) / run_id / "run_state.json"
        return state

    def _invoke(self, out_root, status="done"):
        """Call default_orchestrator_invoker with the engine fully stubbed."""
        state = self._fake_state(out_root, status=status)
        _seed_run(Path(out_root) / state.run_id, status=status)

        with mock.patch("engine.orchestrator_agent.run_orchestration",
                        return_value=state), \
             mock.patch("engine.orchestrator_agent.orchestrator."
                        "onboarding_mode_for_target", return_value="mi_only"), \
             mock.patch("engine.orchestrator_agent.adapters.RealAgentAdapters",
                        return_value=mock.MagicMock()), \
             mock.patch("apps.blob_trigger_app.llm_recommendations."
                        "resolve_llm_policy", return_value={"enabled": False}), \
             mock.patch.object(S, "_invoke_generator",
                               side_effect=_fake_generator()):
            return OI.default_orchestrator_invoker(
                processing_mode="deterministic", client_id="ERE",
                source_portfolio_id="direct_001", source_portfolio_type="direct",
                dataset="funded", frequency="monthly",
                reporting_period="2026-01-31", input_path="/tmp/x.xlsx",
                target="mi", run_regime=False, mapping_config_path=None,
                out_dir=out_root)

    def test_done_run_invokes_pptx_generation(self):
        with TemporaryDirectory() as td:
            result = self._invoke(td, status="done")
            self.assertEqual(result["status"], "done")
            art = result.get("investor_pack_pptx")
            self.assertIsNotNone(art)
            self.assertEqual(art["status"], "available")
            deck = Path(td) / "orun_wire" / "reports" / "investor_pack.pptx"
            self.assertTrue(deck.exists())

    def test_halted_run_does_not_generate_pptx(self):
        with TemporaryDirectory() as td:
            result = self._invoke(td, status="halted")
            self.assertEqual(result["status"], "halted")
            self.assertNotIn("investor_pack_pptx", result)
            deck = Path(td) / "orun_wire" / "reports" / "investor_pack.pptx"
            self.assertFalse(deck.exists())

    def test_generation_failure_keeps_run_successful(self):
        with TemporaryDirectory() as td:
            state = self._fake_state(td, status="done")
            _seed_run(Path(td) / state.run_id)
            with mock.patch("engine.orchestrator_agent.run_orchestration",
                            return_value=state), \
                 mock.patch("engine.orchestrator_agent.orchestrator."
                            "onboarding_mode_for_target", return_value="mi_only"), \
                 mock.patch("engine.orchestrator_agent.adapters."
                            "RealAgentAdapters", return_value=mock.MagicMock()), \
                 mock.patch("apps.blob_trigger_app.llm_recommendations."
                            "resolve_llm_policy", return_value={"enabled": False}), \
                 mock.patch.object(S, "_invoke_generator",
                                   side_effect=RuntimeError("render failed")):
                result = OI.default_orchestrator_invoker(
                    processing_mode="deterministic", client_id="ERE",
                    source_portfolio_id="direct_001",
                    source_portfolio_type="direct", dataset="funded",
                    frequency="monthly", reporting_period="2026-01-31",
                    input_path="/tmp/x.xlsx", target="mi", run_regime=False,
                    mapping_config_path=None, out_dir=td)
            # Run stays successful; the failed artifact is recorded.
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["investor_pack_pptx"]["status"], "failed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
