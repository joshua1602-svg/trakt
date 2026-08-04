"""simulation.pipeline — drives the REAL production pathway.

Nothing analytical happens here. This module only invokes production entry
points and reports what they produced:

    engine/orchestrator/trakt_run.py --mode mi          Gates 1 → 3b
    engine/platform_assembler.py                        platform canonical
    snapshot.adapters.LocalFsSnapshotStore              historical integration
    engine/orchestrator/trakt_run.py --mode regulatory  Gates 4 → 5

The gates run as subprocesses because that is exactly how the managed service
and the blob trigger run them (``function_app.py``,
``demo_platform.orchestration``); calling the modules in-process would test a
path production does not use.

No Azure resource is required: the ``blob://`` URIs the MI layer resolves are
mapped onto the run directory through the existing filesystem storage backend
(``TRAKT_STORAGE_BACKEND=file`` + ``TRAKT_LOCAL_BLOB_ROOT``), which is the
supported local substitute the repository already ships.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAKT_RUN = REPO_ROOT / "engine" / "orchestrator" / "trakt_run.py"
PLATFORM_ASSEMBLER = REPO_ROOT / "engine" / "platform_assembler.py"
CONFIG_ROOT = REPO_ROOT / "config"
ALIASES_ROOT = Path(__file__).resolve().parent / "aliases"

PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"
PLATFORM_MANIFEST_NAME = "platform_canonical_manifest.json"

#: Production modules this driver exercises, recorded on every run summary so a
#: reader can see the framework really used the platform rather than a copy.
PRODUCTION_MODULES = (
    "engine.orchestrator.trakt_run",
    "engine.gate_1_alignment.semantic_alignment",
    "engine.gate_2_transform.canonical_transform",
    "engine.gate_2_transform.lineage_tracker",
    "engine.gate_3_validation.validate_canonical",
    "engine.gate_3_validation.validate_business_rules",
    "engine.gate_3_validation.aggregate_validation_results",
    "engine.provenance",
    "engine.platform_assembler",
    "snapshot.adapters.local_fs.LocalFsSnapshotStore",
)


class PipelineError(RuntimeError):
    """A production invocation failed. Carries the captured output verbatim."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "",
                 returncode: Optional[int] = None):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@dataclass
class GateRun:
    """The outcome of one ``trakt_run.py`` invocation."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    canonical_path: Optional[Path] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    validation_dashboard: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    command: List[str] = field(default_factory=list)

    @property
    def gate_lines(self) -> List[str]:
        return [ln.strip() for ln in self.stdout.splitlines()
                if ln.strip().startswith(("[Gate", "[MI]", "[Transform]",
                                          "[Loan Engine]"))]

    def gate_line(self, prefix: str) -> str:
        return next((ln for ln in self.gate_lines if ln.startswith(prefix)), "")

    @property
    def canonical_validation_failed(self) -> bool:
        """True when Gate 2 REPORTED a failure.

        The orchestrator prints the Gate 2 verdict but does not stop on it: a
        run whose canonical validation fails still produces a canonical and
        still exits 0. Reading the reported verdict — rather than the exit code
        alone — is what lets the framework classify a missing mandatory field at
        the boundary that actually detected it.
        """
        return "FAIL" in self.gate_line("[Gate 2]")

    @property
    def canonical_row_count(self) -> Optional[int]:
        import re
        match = re.search(r"([\d,]+) loans", self.gate_line("[Gate 2]"))
        return int(match.group(1).replace(",", "")) if match else None


def _run(cmd: Sequence[Any], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=timeout,
                          env=env)


# --------------------------------------------------------------------------- #
# Gates 1-3: funded ingestion → canonical
# --------------------------------------------------------------------------- #
def run_mi_gates(source: Path, case, *, reporting_date: str, out_dir: Path,
                 validation_dir: Path,
                 master_config: Optional[Path] = None) -> GateRun:
    """Run the real funded-ingestion pathway for one source file.

    This is ``trakt_run.py --mode mi``: Gate 1 semantic alignment (with this
    asset class's approved alias contract layered on), the canonical transform,
    Gate 2 canonical validation, Gate 2.5 lineage, Gate 3 business rules and
    Gate 3b aggregation — the same command the managed service runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    overlay = ALIASES_ROOT / case.asset_class
    cmd: List[Any] = [
        sys.executable, TRAKT_RUN,
        "--mode", "mi",
        "--input", source,
        "--portfolio-type", case.portfolio_type,
        "--extra-aliases-dir", overlay,
        "--source-portfolio-id", case.portfolio_id,
        "--source-portfolio-type", case.source_portfolio_type,
        "--source-portfolio-label", f"{case.client_id} {case.portfolio_id}",
        "--reporting-date", reporting_date,
        "--currency", case.currency,
        "--out-dir", out_dir,
        "--validation-out-dir", validation_dir,
    ]
    cmd += _acquisition_cli(case)
    if master_config:
        cmd += ["--master-config", master_config]

    started = time.monotonic()
    proc = _run(cmd)
    duration = time.monotonic() - started

    canonical = _find_canonical(out_dir, source)
    manifest_path = out_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    dashboard = next(iter(sorted(validation_dir.glob("*_dashboard.json"))), None)

    run = GateRun(
        ok=proc.returncode == 0 and canonical is not None,
        returncode=proc.returncode, stdout=proc.stdout or "",
        stderr=proc.stderr or "", canonical_path=canonical, manifest=manifest,
        validation_dashboard=_read_json(dashboard) if dashboard else {},
        duration_seconds=duration, command=[str(c) for c in cmd])
    # A reported gate failure is a failed ingestion, whatever the exit code says.
    if run.ok and (run.canonical_validation_failed or run.canonical_row_count == 0):
        run.ok = False
    return run


def _acquisition_cli(case) -> List[str]:
    """Acquisition provenance for an acquired book.

    ``engine.provenance`` fails closed on an acquired portfolio with no
    acquisition date rather than onboarding a back book with an unknown one, so
    the manifest carries it and it is passed through here.
    """
    out: List[str] = []
    if getattr(case, "acquisition_date", None):
        out += ["--acquisition-date", str(case.acquisition_date)]
    if getattr(case, "seller_name", None):
        out += ["--seller-name", str(case.seller_name)]
    return out


def _find_canonical(out_dir: Path, source: Path) -> Optional[Path]:
    expected = out_dir / f"{source.stem}_canonical_typed.csv"
    if expected.exists():
        return expected
    candidates = sorted(out_dir.glob("*_canonical_typed.csv"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- #
# Platform consolidation
# --------------------------------------------------------------------------- #
def assemble_platform(canonicals: Sequence[Path], out_dir: Path) -> Dict[str, Any]:
    """Run the production platform assembler over one period's canonicals."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = _run([sys.executable, PLATFORM_ASSEMBLER, "--inputs", *canonicals,
                 "--out-dir", out_dir])
    if proc.returncode != 0:
        raise PipelineError("platform_assembler failed",
                            stdout=proc.stdout or "", stderr=proc.stderr or "",
                            returncode=proc.returncode)
    manifest = _read_json(out_dir / PLATFORM_MANIFEST_NAME)
    if not (out_dir / PLATFORM_CANONICAL_NAME).exists():
        raise PipelineError(
            f"platform_assembler produced no {PLATFORM_CANONICAL_NAME}",
            stdout=proc.stdout or "", stderr=proc.stderr or "")
    return {"canonical": out_dir / PLATFORM_CANONICAL_NAME, "manifest": manifest}


# --------------------------------------------------------------------------- #
# Historical funded-snapshot integration
# --------------------------------------------------------------------------- #
def snapshot_store(root: Path):
    """The production ``SnapshotStore`` used for historical integration."""
    from snapshot.adapters import LocalFsSnapshotStore
    return LocalFsSnapshotStore(root=str(root))


def register_period(store, canonical: Path, case, *, reporting_date: str,
                    route: str = "mi", cadence: str = "monthly",
                    source_file_id: Optional[str] = None) -> Dict[str, Any]:
    """Register one reporting period through the production snapshot store.

    ``reporting_date`` is supplied EXPLICITLY and never inferred from an upload
    timestamp — that separation is the snapshot layer's central rule and the
    framework must not be the thing that breaks it.
    """
    import pandas as pd

    from snapshot.model import SnapshotHeader

    frame = pd.read_csv(canonical, low_memory=False)
    header = SnapshotHeader(
        client_id=case.client_id, route=route, reporting_date=reporting_date,
        source_file_id=source_file_id or f"{case.case_id}:{reporting_date}",
        cadence=cadence, cut_off_date=reporting_date,
        source_file_name=Path(canonical).name,
        metadata={"case_id": case.case_id, "seed": case.seed,
                  "asset_class": case.asset_class,
                  "asset_subtype": case.asset_subtype,
                  "portfolio_id": case.portfolio_id, "spv_id": case.spv_id,
                  "simulation_version": case.simulation_version})
    result = store.register_snapshot(header, frame)
    return {"snapshot_id": result.snapshot_id, "created": result.created,
            "idempotent": result.idempotent, "issues": result.issues,
            "row_count": int(len(frame)), "reporting_date": reporting_date}


# --------------------------------------------------------------------------- #
# Regime projection + regulatory artefact
# --------------------------------------------------------------------------- #
def run_regulatory_gates(source: Path, case, *, regime: str,
                         reporting_date: str, out_dir: Path,
                         validation_dir: Path,
                         master_config: Optional[Path] = None) -> GateRun:
    """Run ``trakt_run.py --mode regulatory`` for one regime.

    Gate 4 projects canonical truth into the regime schema; Gate 5 builds the
    regulatory artefact. For any regime other than Annex 2 the repository has no
    committed delivery template, so Gate 5 refuses with an explicit reason — the
    framework asserts that refusal rather than fabricating an artefact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    overlay = ALIASES_ROOT / case.asset_class
    cmd: List[Any] = [
        sys.executable, TRAKT_RUN,
        "--mode", "regulatory",
        "--input", source,
        "--regime", regime,
        # The simulation client configs keep ESMA off so an MI run stops at
        # Gate 3b; the regulatory run turns it on explicitly, which is the
        # production control for exactly this.
        "--esma-enabled",
        "--portfolio-type", case.portfolio_type,
        "--extra-aliases-dir", overlay,
        "--source-portfolio-id", case.portfolio_id,
        "--source-portfolio-type", case.source_portfolio_type,
        "--source-portfolio-label", f"{case.client_id} {case.portfolio_id}",
        "--reporting-date", reporting_date,
        "--currency", case.currency,
        "--out-dir", out_dir,
        "--validation-out-dir", validation_dir,
    ]
    cmd += _acquisition_cli(case)
    if master_config:
        cmd += ["--master-config", master_config]
    overlay = build_delivery_rules(case, out_dir, master_config)
    if overlay is not None:
        cmd += ["--annex2-delivery-rules", overlay]
    started = time.monotonic()
    proc = _run(cmd)
    return GateRun(
        ok=proc.returncode == 0, returncode=proc.returncode,
        stdout=proc.stdout or "", stderr=proc.stderr or "",
        canonical_path=_find_canonical(out_dir, source),
        manifest=_read_json(out_dir / "run_manifest.json"),
        duration_seconds=time.monotonic() - started,
        command=[str(c) for c in cmd])


def build_delivery_rules(case, out_dir: Path,
                         master_config: Optional[Path] = None) -> Optional[Path]:
    """A per-client Annex 2 delivery-rules overlay for this run.

    ``config/regime/annex2_delivery_rules.yaml`` enumerates the ACCEPTED value
    of ``RREL83`` (the originator LEI) in a per-field ``enum_map``. That map
    currently contains one client's LEI, so any other originator's LEI is
    rejected as an unmapped enum and Gate 4b blocks delivery — an identifier is
    being validated as if it were a controlled vocabulary. See the framework
    report: the rule should check the LEI's SHAPE, not membership of a list.

    Rather than edit the shared regulatory contract, this writes a run-scoped
    overlay adding the simulated originator's own LEI, and passes it through the
    existing ``--annex2-delivery-rules`` CLI option — the configuration point
    the pipeline already provides for exactly this.
    """
    import yaml

    rules_path = CONFIG_ROOT / "regime" / "annex2_delivery_rules.yaml"
    if not rules_path.exists():
        return None
    lei = _originator_lei(master_config)
    if not lei:
        return None
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    field_rules = rules.get("field_rules") or {}
    rrel83 = field_rules.get("RREL83")
    if isinstance(rrel83, dict):
        enum_map = ((rrel83.get("transform") or {}).get("enum_map") or {})
        enum_map[lei] = lei
        rrel83.setdefault("transform", {})["enum_map"] = enum_map
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "delivery_rules_overlay.yaml"
    path.write_text(yaml.safe_dump(rules, sort_keys=False), encoding="utf-8")
    return path


def _originator_lei(master_config: Optional[Path]) -> Optional[str]:
    import yaml

    if not master_config or not Path(master_config).exists():
        return None
    cfg = yaml.safe_load(Path(master_config).read_text(encoding="utf-8")) or {}
    lei = ((cfg.get("defaults") or {})
           .get("originator_legal_entity_identifier"))
    return str(lei).strip() if lei else None


def projected_regime_csv(out_dir: Path, regime: str) -> Optional[Path]:
    """The Gate 4 projection for *regime*, whichever naming it landed under."""
    short = regime.replace("ESMA_", "").lower()
    direct = out_dir / f"{short}_projected.csv"
    if direct.exists():
        return direct
    candidates = sorted(out_dir.glob(f"*{regime}*projected*.csv"))
    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------- #
# Governed MI / Agent environment
# --------------------------------------------------------------------------- #
def mi_environment(run_dir: Path, case, *, reporting_date: str) -> Dict[str, str]:
    """Environment pointing the governed MI capability at this run's platform.

    Mirrors a production deployment exactly, resolved through the FILESYSTEM
    storage backend: ``blob://`` URIs map under ``TRAKT_LOCAL_BLOB_ROOT``. No
    Azure resource is contacted, and no production code path is bypassed.
    """
    store_root = run_dir / "store"
    prefix = f"blob://processed/platform/{case.client_id}"
    return {
        "TRAKT_STORAGE_BACKEND": "file",
        "TRAKT_LOCAL_BLOB_ROOT": str(store_root),
        "MI_AGENT_PLATFORM_URI": f"{prefix}/latest",
        "MI_AGENT_ONBOARDING_OUTPUT_ROOT": prefix,
        "MI_AGENT_CLIENT_ID": case.client_id,
        "MI_AGENT_RUN_ID": reporting_date,
        "MI_AGENT_REPORTING_DATE": reporting_date,
        "MI_AGENT_SCRATCH": str(run_dir / "scratch"),
        # Deterministic by construction: the governed MI Agent must answer from
        # the data, never from a model call.
        "MI_AGENT_LLM_PARSER": "off",
        "MI_AGENT_LLM_ENABLED": "0",
        "MI_AGENT_AUTH_ENABLED": "false",
    }


def publish_platform(run_dir: Path, case, *, reporting_date: str,
                     canonical: Path, manifest: Optional[Path] = None,
                     latest: bool = False) -> Path:
    """Publish one period's platform canonical into the dated production layout.

    ``processed/platform/{client}/{YYYY-MM-DD}/platform_canonical_typed.csv`` is
    exactly where the MI data source looks, so publishing here is what makes the
    period visible to the governed MI capability — no MI code is adapted.
    """
    import shutil

    base = run_dir / "store" / "processed" / "platform" / case.client_id
    dated = base / reporting_date
    dated.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, dated / PLATFORM_CANONICAL_NAME)
    if manifest and Path(manifest).exists():
        shutil.copyfile(manifest, dated / PLATFORM_MANIFEST_NAME)
    if latest:
        pointer_dir = base / "latest"
        pointer_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical, pointer_dir / PLATFORM_CANONICAL_NAME)
        if manifest and Path(manifest).exists():
            shutil.copyfile(manifest, pointer_dir / PLATFORM_MANIFEST_NAME)
        (pointer_dir / "latest_platform_canonical.json").write_text(
            json.dumps({"client_id": case.client_id,
                        "reporting_date": reporting_date,
                        "case_id": case.case_id,
                        "platform_canonical": PLATFORM_CANONICAL_NAME},
                       indent=2), encoding="utf-8")
    return dated


__all__ = ["ALIASES_ROOT", "GateRun", "PLATFORM_CANONICAL_NAME",
           "PLATFORM_MANIFEST_NAME", "PRODUCTION_MODULES", "PipelineError",
           "REPO_ROOT", "assemble_platform", "mi_environment",
           "projected_regime_csv", "publish_platform", "register_period",
           "run_mi_gates", "run_regulatory_gates", "snapshot_store"]
