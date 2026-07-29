"""Rule persistence: scope precedence (file > portfolio > client > global),
versioning/supersession, retirement, and projection into the existing
client-memory artefact."""

from __future__ import annotations

from pathlib import Path

from operations_control.rules import (
    RuleRecord,
    RuleStore,
    project_rules_to_client_memory,
)


def _rule(scope: str, value: str, *, client="client_a", portfolio="",
          file_ref="", kind="field_mapping", source="Prop_Val_Idx") -> RuleRecord:
    return RuleRecord(
        rule_id="", version=0, kind=kind, scope=scope,
        client_id=(client if scope != "global" else ""),
        portfolio_id=portfolio, file_ref=file_ref,
        payload={"source_column": source, "canonical_field": value},
        approved_by="alice")


class TestPrecedence:
    def test_more_specific_scope_wins(self, store):
        rules = RuleStore(store)
        rules.approve(_rule("global", "global_target"))
        rules.approve(_rule("client", "client_target"))
        rules.approve(_rule("portfolio", "portfolio_target", portfolio="pf1"))
        rules.approve(_rule("file", "file_target", portfolio="pf1",
                            file_ref="sha256:f1"))

        # File context: the file rule wins.
        got = rules.applicable(client_id="client_a", portfolio_id="pf1",
                               file_ref="sha256:f1")
        assert len(got) == 1
        assert got[0].payload["canonical_field"] == "file_target"

        # Same portfolio, different file: portfolio rule wins.
        got = rules.applicable(client_id="client_a", portfolio_id="pf1",
                               file_ref="sha256:other")
        assert got[0].payload["canonical_field"] == "portfolio_target"

        # Different portfolio: client rule wins.
        got = rules.applicable(client_id="client_a", portfolio_id="pf2")
        assert got[0].payload["canonical_field"] == "client_target"

    def test_global_applies_to_every_client(self, store):
        rules = RuleStore(store)
        rules.approve(_rule("global", "global_target"))
        got = rules.applicable(client_id="client_b")
        assert got and got[0].payload["canonical_field"] == "global_target"


class TestVersioning:
    def test_reapproval_supersedes_same_subject_same_scope(self, store):
        rules = RuleStore(store)
        r1 = rules.approve(_rule("client", "first_target"))
        r2 = rules.approve(_rule("client", "second_target"))
        assert r2.rule_id == r1.rule_id
        assert r2.version == r1.version + 1

        history = rules.history("client_a", r1.rule_id)
        assert [h.version for h in history] == [1, 2]
        assert history[0].status == "superseded"
        assert history[0].superseded_by == 2
        assert history[1].status == "active"

        # Only one current rule for the subject.
        current = [r for r in rules.list_current("client_a")
                   if r.subject_key() == r1.subject_key()]
        assert len(current) == 1 and current[0].version == 2

    def test_different_subjects_do_not_supersede(self, store):
        rules = RuleStore(store)
        a = rules.approve(_rule("client", "t1", source="Col_A"))
        b = rules.approve(_rule("client", "t2", source="Col_B"))
        assert a.rule_id != b.rule_id
        assert a.version == b.version == 1

    def test_retire(self, store):
        rules = RuleStore(store)
        r = rules.approve(_rule("client", "t1"))
        retired = rules.retire("client_a", r.rule_id, by="alice",
                               reason="no longer applies")
        assert retired.status == "retired"
        assert rules.applicable(client_id="client_a") == []


class TestProjection:
    def test_rules_project_to_existing_client_memory_artifact(self, ops_env, store):
        from engine.onboarding_agent.mapping_memory import MappingMemoryStore
        rules = RuleStore(store)
        r = rules.approve(_rule("client", "current_valuation_indexed"))
        mdir = Path(ops_env["tmp"]) / "memory" / "client_a" / "client_memory"
        n = project_rules_to_client_memory([r], "client_a", memory_dir=mdir)
        assert n == 1
        mem = MappingMemoryStore(mdir, client_id="client_a")
        entries = mem.entries
        assert len(entries) == 1
        assert entries[0].canonical_field == "current_valuation_indexed"
        assert entries[0].approved_by == "alice"

        # Idempotent: projecting again does not duplicate.
        project_rules_to_client_memory([r], "client_a", memory_dir=mdir)
        assert len(MappingMemoryStore(mdir, client_id="client_a").entries) == 1
