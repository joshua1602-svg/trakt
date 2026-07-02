"""mi_agent_pptx.artifact_loader — discover & load MI Agent run artifacts.

Consumes a completed MI Agent pipeline run directory (``out/runs/<run_id>`` or
any directory that carries the canonical artifacts) and exposes the artifacts
the deck needs. It mirrors the discovery conventions already used by
``mi_agent_api/data_source.py`` and the orchestrator run layout
(``engine/orchestrator_agent/state.py``), but adds no dependency on the FastAPI
app, Streamlit, or any live service.

Discovery is convention-based and *degrades gracefully*: every artifact is
optional. When an artifact is absent the loader records the gap (surfaced later
in the deck's appendix coverage notes) rather than raising. The one hard
requirement to produce a meaningful deck is a canonical typed tape CSV; without
it the loader still returns a container (``has_tape == False``) so callers can
emit a fully-branded "data unavailable" deck.

Everything on disk in an MI Agent run is CSV + JSON (there is no parquet), so
this loader only needs ``pandas`` and ``json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Discovery conventions (ordered by preference).
# --------------------------------------------------------------------------- #

# Canonical typed tape (funded book / central canonical). First match wins.
TAPE_CANDIDATES: List[str] = [
    "out_platform/platform_canonical_typed.csv",
    "platform_canonical_typed.csv",
    "**/platform_canonical_typed.csv",
    "**/output/central/18_central_lender_tape.csv",
    "**/18_central_lender_tape.csv",
    "**/stamped/*_canonical_typed.csv",
    "**/*_canonical_typed.csv",
    "**/*canonical*typed*.csv",
]

# Pipeline (pre-funded) tape.
PIPELINE_CANDIDATES: List[str] = [
    "**/20_prepared_pipeline_mi.csv",
    "**/18a_central_pipeline_tape.csv",
    "**/*pipeline*tape*.csv",
    "**/*prepared_pipeline*.csv",
]

# Optional route-response / analytics artifacts (JSON). Keyed by logical name;
# each value is a list of glob candidates.
JSON_ARTIFACTS: Dict[str, List[str]] = {
    "run_state": ["run_state.json"],
    "analytics": ["**/*analytics*.json", "**/mi_analytics*.json"],
    "metrics": ["**/*metric*registry*.json", "**/*metrics*.json"],
    "charts": ["**/*chart*registry*.json", "**/*charts*.json"],
    "validation": [
        "**/40_validation_manifest.json",
        "**/*validation*summary*.json",
        "**/*validation*.json",
    ],
    "risk_monitor": [
        "**/*risk_limits*.json",
        "**/*risk_monitor*.json",
        "**/*risk*.json",
    ],
    "pipeline_snapshot": ["**/*pipeline*snapshot*.json"],
    "forecast_bridge": ["**/*forecast*bridge*.json", "**/*forecast*.json"],
    "scenario": ["**/*scenario*.json"],
    # LLM-supplied straplines/insights (see insight_resolver).
    "straplines": [
        "**/*strapline*.json",
        "**/*insight*.json",
        "**/pptx_straplines*.json",
    ],
}


@dataclass
class RunArtifacts:
    """Container for the artifacts discovered under a run directory."""

    run_dir: Path
    tape: Optional[pd.DataFrame] = None
    tape_path: Optional[Path] = None
    tape_kind: Optional[str] = None
    pipeline_tape: Optional[pd.DataFrame] = None
    pipeline_tape_path: Optional[Path] = None
    json_artifacts: Dict[str, Any] = field(default_factory=dict)
    json_paths: Dict[str, Path] = field(default_factory=dict)
    coverage_notes: List[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    @property
    def has_tape(self) -> bool:
        return self.tape is not None and not self.tape.empty

    @property
    def has_pipeline(self) -> bool:
        return self.pipeline_tape is not None and not self.pipeline_tape.empty

    def artifact(self, name: str) -> Any:
        return self.json_artifacts.get(name)

    def has_artifact(self, name: str) -> bool:
        return name in self.json_artifacts and self.json_artifacts[name] is not None

    @property
    def run_state(self) -> Dict[str, Any]:
        rs = self.json_artifacts.get("run_state")
        return rs if isinstance(rs, dict) else {}

    def note(self, message: str) -> None:
        if message not in self.coverage_notes:
            self.coverage_notes.append(message)


def _first_match(run_dir: Path, patterns: List[str]) -> Optional[Path]:
    """Return the first existing file matching any glob in *patterns*."""
    for pat in patterns:
        # Direct path (no glob magic) first for speed / determinism.
        if not any(ch in pat for ch in "*?[]"):
            direct = run_dir / pat
            if direct.is_file():
                return direct
            continue
        matches = sorted(run_dir.glob(pat))
        matches = [m for m in matches if m.is_file()]
        if matches:
            return matches[0]
    return None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:  # pragma: no cover - defensive
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return None


def load_run_artifacts(run_dir: str | Path) -> RunArtifacts:
    """Discover and load all deck-relevant artifacts under *run_dir*."""
    run_path = Path(run_dir)
    artifacts = RunArtifacts(run_dir=run_path)

    if not run_path.exists():
        artifacts.note(f"Run directory not found: {run_path}")
        return artifacts

    # --- canonical typed tape (funded / central) ------------------------
    tape_path = _first_match(run_path, TAPE_CANDIDATES)
    if tape_path is not None:
        df = _read_csv(tape_path)
        if df is not None and not df.empty:
            artifacts.tape = df
            artifacts.tape_path = tape_path
            artifacts.tape_kind = _classify_tape(tape_path)
            artifacts.note(
                f"Canonical typed tape: {tape_path.name} "
                f"({len(df)} rows, {len(df.columns)} columns)."
            )
        else:
            artifacts.note(f"Canonical tape present but unreadable: {tape_path}")
    else:
        artifacts.note("No canonical typed tape found in run directory.")

    # --- pipeline tape ---------------------------------------------------
    pipe_path = _first_match(run_path, PIPELINE_CANDIDATES)
    if pipe_path is not None:
        pdf = _read_csv(pipe_path)
        if pdf is not None and not pdf.empty:
            artifacts.pipeline_tape = pdf
            artifacts.pipeline_tape_path = pipe_path
            artifacts.note(f"Pipeline tape: {pipe_path.name} ({len(pdf)} rows).")
    else:
        artifacts.note("No pipeline tape found (pipeline lens will be limited).")

    # --- optional JSON artifacts ----------------------------------------
    for name, patterns in JSON_ARTIFACTS.items():
        path = _first_match(run_path, patterns)
        if path is None:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        artifacts.json_artifacts[name] = payload
        artifacts.json_paths[name] = path
        if name != "run_state":
            artifacts.note(f"Loaded {name} artifact: {path.name}.")

    return artifacts


def _classify_tape(path: Path) -> str:
    name = path.name.lower()
    if "platform_canonical" in name:
        return "platform_canonical"
    if "central_lender_tape" in name or "18_central" in name:
        return "central_tape"
    if "canonical_typed" in name:
        return "canonical_typed"
    return "explicit_csv"
