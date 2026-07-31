"""The line between the OCC Agent and the Onboarding Agent.

The OCC Agent coordinates the onboarding **operating process**. The existing
Onboarding Agent owns technical file ingestion: source inspection, mapping,
transformation and validation. Where that line falls has to be checkable, not
merely stated, because the failure mode is quiet — a second implementation of
header mapping that works slightly differently from the first is worse than no
second implementation at all.

The rule these tests enforce:

* the OCC Agent's **live** path may open a governed input pack, place files
  through the platform's own intake, and start the Onboarding Agent;
* it may **not** inspect a schema, build a mapping, transform loan data or
  validate a tape itself.

The rehearsal is the subtlety. ``execution.py`` *does* profile, map, transform
and validate — by calling the platform's own components over an isolated
sandbox. That is reuse, not duplication, and the distinction is exactly what
these tests pin down: the rehearsal imports the real implementations and adds
none of its own.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from operations_control.occ_agent import adapters as _adapters

PACKAGE = Path(_adapters.__file__).parent

#: Modules that make up the OCC Agent's own coordination logic. ``execution``
#: and ``artefacts`` are deliberately absent: they are the rehearsal harness,
#: and they are covered by their own rule below.
COORDINATION_MODULES = (
    "adapters", "classification", "client_form", "communication",
    "derive", "extraction", "interpretation", "pack", "planning",
    "readiness", "review", "service", "states", "store", "api",
)

#: What "processing a file" looks like in code. A coordination module that did
#: any of these would be doing the Onboarding Agent's job.
FORBIDDEN_IN_COORDINATION = (
    "read_csv", "DataFrame", "profile_file", "HeaderMapper",
    "apply_types", "run_rules", "aggregate_validation_results",
    "semantic_alignment", "canonical_transform", "validate_business_rules",
)


def _source(module: str) -> str:
    return (PACKAGE / f"{module}.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The OCC Agent does not process files
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("module", COORDINATION_MODULES)
def test_no_coordination_module_processes_a_file(module):
    """Coordination decides WHAT should happen, never how a tape is read."""
    text = _source(module)
    for token in FORBIDDEN_IN_COORDINATION:
        assert token not in text, (
            f"{module}.py mentions {token!r} — file processing belongs to the "
            "Onboarding Agent")


def test_the_live_adapter_only_sequences_existing_calls():
    """Four governed calls, and nothing that could be a fifth pipeline."""
    source = inspect.getsource(_adapters.LiveExecutionAdapter)
    for call in ("self.onboarding.activate", "self.engine.create_batch",
                 "self.engine.upload_batch_files", "self.engine.start_batch"):
        assert call in source, call
    for forbidden in ("write_bytes", "BlobServiceClient", "run_orchestration",
                      "requests.", "subprocess", "open("):
        assert forbidden not in source, forbidden


def test_the_live_adapter_makes_no_other_outbound_call():
    """Every attribute call on an injected collaborator, enumerated.

    A new call added to the live path shows up here, which is the point: the
    boundary is a list somebody has to change deliberately.
    """
    tree = ast.parse(inspect.getsource(_adapters.LiveExecutionAdapter))
    calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        value = func.value
        if isinstance(value, ast.Attribute) and isinstance(value.value,
                                                           ast.Name) \
                and value.value.id == "self":
            calls.add(f"self.{value.attr}.{func.attr}")
    assert calls == {
        "self.onboarding.activate",
        "self.engine.create_batch",
        "self.engine.upload_batch_files",
        "self.engine.start_batch",
    }, calls


def test_the_agent_never_builds_a_mapping():
    """Mappings are the Onboarding Agent's, approved through its own path.

    Asserted on what would ASSIGN one. A coordination module may name a
    canonical field — the review package reports the mapping report, and a
    resolved decision carries one — but nothing may derive one.
    """
    for module in COORDINATION_MODULES:
        tree = ast.parse(_source(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in ("map_headers", "align_headers", "best_alias",
                                "score_alias", "infer_mapping"), (
                f"{module}.py derives a mapping via {name}()")


# --------------------------------------------------------------------------- #
# The rehearsal REUSES the real components rather than reimplementing them
# --------------------------------------------------------------------------- #

def test_the_rehearsal_imports_the_real_components():
    """``execution.py`` may profile and map — by calling the real thing."""
    text = _source("execution")
    for real in ("file_profiler", "semantic_alignment", "canonical_transform",
                 "validate_business_rules"):
        assert real in text, (
            f"the rehearsal no longer uses {real}; if it grew its own "
            "implementation that is the duplication this forbids")


def test_the_rehearsal_defines_no_mapping_or_validation_of_its_own():
    """It subclasses the platform's adapter seam and overrides where it must."""
    text = _source("execution")
    tree = ast.parse(text)
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)}
    # Names that would indicate a home-grown implementation rather than a call.
    for suspicious in ("_map_headers", "_infer_mapping", "_apply_types",
                       "_validate_rules", "_fuzzy_match", "_score_alias"):
        assert suspicious not in defined, suspicious


# --------------------------------------------------------------------------- #
# What the OCC Agent MAY do
# --------------------------------------------------------------------------- #

def test_the_agent_tells_the_client_which_files_are_required(service):
    from .conftest import ACTOR, TENANT_A
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    pack = service.build_pack(case)
    assert pack.artefacts.required, "the pack names no required file"
    # And the list is the governed input requirements', not this feature's.
    from operations_control.occ_agent.input_roles import artefact_vocabulary
    assert {r["role"] for r in pack.artefacts.required} == set(
        artefact_vocabulary().required_roles(service.facts(case).outcome))


def test_the_agent_generates_delivery_instructions_with_production_paths(
        service):
    from apps.blob_trigger_app.path_parser import parse_blob_path

    from .conftest import ACTOR, TENANT_A
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    case.run.reporting_period = "2026-06-30"
    pack = service.build_pack(case)
    from operations_control.manual_intake import raw_container
    assert pack.delivery.locations, "no delivery location was generated"
    for location in pack.delivery.locations:
        # A location the LIVE path parser accepts, derived and not invented.
        path = location["location"] + "file.csv"
        assert parse_blob_path(path, raw_container()) is not None


def test_the_agent_verifies_required_artefacts_are_present_without_reading_them(
        service):
    """Presence and role, from the governed requirements. Not content."""
    from .conftest import ACTOR, TENANT_A
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    readiness = service.artefacts.readiness(case.run,
                                            service.facts(case).outcome)
    assert readiness.ready is False
    assert readiness.to_dict()["missing_roles"], "nothing reported as missing"
    # Reported by ROLE, from the governed input requirements — the agent has
    # not opened anything to work this out.
    assert readiness.to_dict()["required_roles"]
