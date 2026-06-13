"""
storage_paths.py
================

PART 3 / PART 4 — Azure-ready run-folder contract and lightweight storage / path
abstraction for the Onboarding Agent.

The engine runs locally (or inside an Azure job / container) and reads input
files from a configured run folder. This module creates the consistent local
folder layout and, when supplied, mirrors it with Azure-Blob-compatible URI
references so a downstream Azure trigger can consume the same manifest.

Nothing here performs real Azure SDK calls. ``storage_backend`` only controls
whether manifests carry Azure-style URIs in addition to local paths — files are
always written locally.

Logical layout (mirrors how blob would be mounted / synced)::

    runs/{client_id}/onboarding/{run_id}/
      input/uploaded/
      working/
      review/        (numbered review artefacts 01..09 live here = project_dir)
      approved/      (approved artefacts 10..15 live here   = project_dir)
      output/
        central/     18_central_lender_tape.csv ...
        lineage/     18b_central_tape_lineage.csv ...
        gaps/        18c_central_tape_gaps.csv ...
        manifests/   19..23 promotion / handoff / readiness / trigger
      logs/

To keep backwards compatibility with the existing flat onboarding output dir,
``review_dir`` and ``approved_dir`` resolve to ``project_dir`` itself (the
numbered 01..15 artefacts stay where every other module already expects them),
while the new consolidated outputs land under ``output_root`` sub-folders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

STORAGE_LOCAL = "local"
STORAGE_AZURE = "azure_blob_compatible"
VALID_STORAGE_BACKENDS = (STORAGE_LOCAL, STORAGE_AZURE)


# ---------------------------------------------------------------------------


@dataclass
class OnboardingRunPaths:
    """Resolved local paths + optional Azure-compatible URI references."""

    client_id: str = ""
    run_id: str = ""
    storage_backend: str = STORAGE_LOCAL

    project_dir: str = ""
    input_dir: str = ""
    output_root: str = ""

    working_dir: str = ""
    review_dir: str = ""
    approved_dir: str = ""
    central_dir: str = ""
    lineage_dir: str = ""
    gaps_dir: str = ""
    manifests_dir: str = ""
    logs_dir: str = ""

    # Optional Azure-style URIs for the input pack and the output root. When
    # absent, manifest URIs fall back to ``null`` / local relative paths.
    input_uri: Optional[str] = None
    output_uri: Optional[str] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- manifest helpers ----------------------------------------------
    def to_manifest_path(self, local_path: str | Path) -> str:
        """Return a stable, blob-compatible *local* path for a manifest.

        Paths inside ``project_dir`` are emitted relative to it (so manifests are
        portable when the run folder is mounted / synced elsewhere); anything
        else is returned as an absolute POSIX string.
        """
        p = Path(local_path).resolve()
        project = Path(self.project_dir).resolve()
        try:
            return p.relative_to(project).as_posix()
        except ValueError:
            return p.as_posix()

    def to_manifest_uri(self, local_path: str | Path) -> Optional[str]:
        """Return an Azure-style URI for ``local_path`` when one can be derived.

        * If the path is under ``output_root`` and ``output_uri`` is configured,
          join ``output_uri`` with the output-relative sub-path.
        * If the path is under ``input_dir`` and ``input_uri`` is configured,
          join ``input_uri`` with the input-relative sub-path.
        * Otherwise return ``None`` (manifest URI may legitimately be null).
        """
        p = Path(local_path).resolve()
        out_root = Path(self.output_root).resolve()
        in_dir = Path(self.input_dir).resolve()

        if self.output_uri:
            try:
                rel = p.relative_to(out_root).as_posix()
                return _join_uri(self.output_uri, rel)
            except ValueError:
                pass
        if self.input_uri:
            try:
                rel = p.relative_to(in_dir).as_posix()
                return _join_uri(self.input_uri, rel)
            except ValueError:
                pass
        return None

    def manifest_ref(self, local_path: str | Path) -> Dict[str, Optional[str]]:
        """Convenience: both the local manifest path and (optional) URI."""
        return {
            "path": self.to_manifest_path(local_path),
            "uri": self.to_manifest_uri(local_path),
        }

    def guard(self, path: str | Path) -> Path:
        """Assert ``path`` stays inside ``project_dir`` or ``output_root``."""
        p = Path(path).resolve()
        roots = [Path(self.project_dir).resolve(), Path(self.output_root).resolve()]
        for root in roots:
            if _is_within(p, root):
                return p
        raise ValueError(
            f"Refusing to write '{p}': outside project_dir/output_root "
            f"({roots[0]} | {roots[1]})."
        )


# ---------------------------------------------------------------------------
# Module functions
# ---------------------------------------------------------------------------


def _join_uri(base: str, rel: str) -> str:
    if not rel:
        return base
    return base.rstrip("/") + "/" + rel.lstrip("/")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_within_project(path: str | Path, project_dir: str | Path) -> Path:
    """Raise ``ValueError`` if ``path`` would write outside ``project_dir``.

    Returns the resolved path on success so callers can use it directly.
    """
    p = Path(path).resolve()
    root = Path(project_dir).resolve()
    if not _is_within(p, root):
        raise ValueError(
            f"Refusing to write '{p}': outside project root '{root}'."
        )
    return p


def resolve_run_paths(
    project_dir: str | Path,
    input_dir: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    client_id: str = "",
    run_id: str = "",
    storage_backend: str = STORAGE_LOCAL,
    input_uri: Optional[str] = None,
    output_uri: Optional[str] = None,
    create: bool = True,
) -> OnboardingRunPaths:
    """Resolve (and optionally create) the local run-folder layout.

    ``project_dir`` holds the numbered review/approved artefacts (flat, as the
    rest of the agent expects). ``output_root`` (default ``project_dir/output``)
    holds the consolidated central / lineage / gaps / manifests outputs.
    """
    project = Path(project_dir).resolve()
    out_root = Path(output_root).resolve() if output_root else (project / "output")
    in_dir = Path(input_dir).resolve() if input_dir else (project / "input" / "uploaded")

    backend = storage_backend or STORAGE_LOCAL
    if backend not in VALID_STORAGE_BACKENDS:
        backend = STORAGE_LOCAL

    paths = OnboardingRunPaths(
        client_id=client_id,
        run_id=run_id,
        storage_backend=backend,
        project_dir=str(project),
        input_dir=str(in_dir),
        output_root=str(out_root),
        working_dir=str(out_root / "working"),
        # Review / approved artefacts stay flat in project_dir (existing layout).
        review_dir=str(project),
        approved_dir=str(project),
        central_dir=str(out_root / "central"),
        lineage_dir=str(out_root / "lineage"),
        gaps_dir=str(out_root / "gaps"),
        manifests_dir=str(out_root / "manifests"),
        logs_dir=str(out_root / "logs"),
        input_uri=input_uri or None,
        output_uri=output_uri or None,
    )

    if create:
        for d in (
            project, out_root,
            Path(paths.working_dir), Path(paths.central_dir), Path(paths.lineage_dir),
            Path(paths.gaps_dir), Path(paths.manifests_dir), Path(paths.logs_dir),
        ):
            d.mkdir(parents=True, exist_ok=True)

    return paths
