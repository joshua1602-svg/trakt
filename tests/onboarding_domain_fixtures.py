"""
Shared helpers for the Onboarding Agent Phase-1 domain tests.

Not a test module (no ``test_`` prefix) so pytest does not collect it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import storage_paths
from engine.onboarding_agent.answer_ingestion import ingest_answers
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_onboarding_pack_domain_based"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"

SCENARIO_A = "scenario_a_combined"
SCENARIO_B = "scenario_b_split"


def build_run(
    scenario: str = SCENARIO_A,
    mode: str = "regulatory_mi",
    ingest: bool = True,
    drop_precedence: bool = False,
    storage_backend: str = "local",
    input_uri: str = "",
    output_uri: str = "",
    regulatory_reporting_enabled: bool = False,
    input_dir: str | None = None,
):
    """Run onboarding (and optionally answer ingestion) into a temp project dir.

    Returns ``(project, project_dir, run_paths)``.
    """
    tmp = Path(tempfile.mkdtemp(prefix="onb_domain_"))
    project_dir = tmp / "run"
    in_dir = Path(input_dir) if input_dir else (PACK / scenario)
    run_paths = storage_paths.resolve_run_paths(
        project_dir=str(project_dir),
        input_dir=str(in_dir),
        output_root=str(project_dir / "output"),
        client_id="client_x",
        run_id="run_001",
        storage_backend=storage_backend,
        input_uri=input_uri,
        output_uri=output_uri,
    )
    project = run_onboarding(
        input_dir=str(in_dir),
        client_name="CLIENT_X",
        output_dir=str(project_dir),
        registry_path=str(REGISTRY),
        aliases_dir=str(ALIASES),
        mode=mode,
        client_id="client_x",
        run_id="run_001",
        storage_backend=storage_backend,
        input_uri=input_uri,
        output_uri=output_uri,
        regulatory_reporting_enabled=regulatory_reporting_enabled,
    )
    if ingest:
        ingest_answers(str(project_dir), str(project_dir / "example_answers.yaml"), confirm=True)
    if drop_precedence:
        pp = project_dir / "13_source_precedence_rules.yaml"
        if pp.exists():
            pp.unlink()
    return project, project_dir, run_paths
