"""engine.platform_assembler — Platform Portfolio Assembler.

Orchestration-only enhancement. A managed-service client onboards several
portfolios over time (e.g. ``direct_001``, ``acquired_001``, ``acquired_002``),
each producing its own canonical typed output exactly as today. This module
assembles the **latest accepted canonical output per ``source_portfolio_id``**
into one combined *platform canonical* dataset representing the current managed
portfolio.

It does NOT re-run onboarding, re-transform, or touch Regime/MI/validation
logic. It only **reads existing ``*_canonical_typed.csv`` outputs** (the single
source of truth) and concatenates the latest snapshot per portfolio, preserving
all provenance fields unchanged.

Contract:
  * latest snapshot only — for each ``source_portfolio_id`` the output contains
    exactly one (most recent) canonical output;
  * provenance preserved — every row keeps its six provenance fields untouched;
  * composite uniqueness — a loan is identified by
    ``source_portfolio_id + loan_identifier`` *inside the platform dataset*;
    duplicate composite keys are rejected;
  * non-destructive — individual portfolio canonicals are never overwritten;
    only ``platform_canonical_typed.csv`` (+ manifest) is written.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

try:
    from engine import provenance as _provenance
except ModuleNotFoundError:  # pragma: no cover - path bootstrap
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from engine import provenance as _provenance

PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"
PLATFORM_MANIFEST_NAME = "platform_canonical_manifest.json"

# Pool-level date used to pick the latest snapshot per portfolio (first present).
SNAPSHOT_DATE_FIELDS = ("reporting_date", "data_cut_off_date", "cut_off_date")
# Loan-level identifier used for the composite platform key (first present).
LOAN_KEY_FIELDS = ("loan_identifier", "unique_identifier")
# Additive column carrying the platform composite identity.
PLATFORM_KEY_COLUMN = "platform_loan_key"


class PlatformAssemblyError(ValueError):
    """Raised when the platform assembly inputs are invalid or inconsistent."""


@dataclass
class PortfolioSnapshot:
    """A single discovered canonical output for one portfolio."""

    source_portfolio_id: str
    path: Path
    snapshot_date: Optional[str]
    row_count: int
    mtime: float


# --------------------------------------------------------------------------- #
# Discovery + per-file metadata
# --------------------------------------------------------------------------- #

def _norm_date(series: pd.Series) -> Optional[str]:
    vals = pd.to_datetime(series, errors="coerce").dropna()
    if len(vals) == 0:
        return None
    return vals.max().date().isoformat()


def _snapshot_date(df: pd.DataFrame) -> Optional[str]:
    for f in SNAPSHOT_DATE_FIELDS:
        if f in df.columns:
            d = _norm_date(df[f])
            if d is not None:
                return d
    return None


def _loan_key_field(df: pd.DataFrame) -> Optional[str]:
    for f in LOAN_KEY_FIELDS:
        if f in df.columns:
            return f
    return None


def read_portfolio_snapshot(path: Union[str, Path]) -> PortfolioSnapshot:
    """Read one canonical output's identifying metadata (not for combination).

    Fails closed if the file is not a single-portfolio provenance-stamped
    canonical (missing or multiple ``source_portfolio_id`` values).
    """
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)
    if "source_portfolio_id" not in df.columns:
        raise PlatformAssemblyError(
            f"{path.name}: no source_portfolio_id column — not a provenance-"
            f"stamped canonical. Re-onboard with --source-portfolio-id."
        )
    ids = sorted({
        str(v).strip() for v in df["source_portfolio_id"].dropna().tolist()
        if str(v).strip() not in ("", "nan", "None")
    })
    if not ids:
        raise PlatformAssemblyError(f"{path.name}: source_portfolio_id is blank on all rows.")
    if len(ids) > 1:
        raise PlatformAssemblyError(
            f"{path.name}: contains multiple source_portfolio_ids {ids} — the "
            f"assembler consumes single-portfolio canonicals, not pre-combined files."
        )
    return PortfolioSnapshot(
        source_portfolio_id=ids[0],
        path=path,
        snapshot_date=_snapshot_date(df),
        row_count=int(len(df)),
        mtime=path.stat().st_mtime,
    )


def discover_canonical_outputs(root: Union[str, Path]) -> List[Path]:
    """Find candidate ``*_canonical_typed.csv`` outputs under ``root``.

    Excludes any previously written platform canonical so re-runs are idempotent.
    """
    root = Path(root)
    out: List[Path] = []
    for p in sorted(root.glob("**/*_canonical_typed.csv")):
        if p.name == PLATFORM_CANONICAL_NAME:
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Latest-snapshot selection
# --------------------------------------------------------------------------- #

def select_latest_per_portfolio(
    snapshots: Sequence[PortfolioSnapshot],
) -> Dict[str, PortfolioSnapshot]:
    """Pick exactly one (latest) snapshot per ``source_portfolio_id``.

    Latest is by ``snapshot_date``. Two snapshots for the same portfolio sharing
    the latest date are ambiguous and rejected (duplicate snapshot). Multiple
    snapshots with no date and no way to order are also rejected.
    """
    by_pid: Dict[str, List[PortfolioSnapshot]] = {}
    for s in snapshots:
        by_pid.setdefault(s.source_portfolio_id, []).append(s)

    chosen: Dict[str, PortfolioSnapshot] = {}
    for pid, snaps in by_pid.items():
        if len(snaps) == 1:
            chosen[pid] = snaps[0]
            continue
        undated = [s for s in snaps if not s.snapshot_date]
        if undated:
            raise PlatformAssemblyError(
                f"Portfolio {pid!r} has {len(snaps)} snapshots but "
                f"{len(undated)} lack a snapshot date ({SNAPSHOT_DATE_FIELDS}); "
                f"cannot determine the latest. Files: {[s.path.name for s in snaps]}"
            )
        latest_date = max(s.snapshot_date for s in snaps)
        at_latest = [s for s in snaps if s.snapshot_date == latest_date]
        if len(at_latest) > 1:
            raise PlatformAssemblyError(
                f"Portfolio {pid!r} has {len(at_latest)} snapshots at the same "
                f"latest date {latest_date} — duplicate portfolio snapshot. "
                f"Files: {[s.path.name for s in at_latest]}"
            )
        chosen[pid] = at_latest[0]
    return chosen


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

@dataclass
class AssemblyResult:
    dataframe: pd.DataFrame
    manifest: Dict[str, Any]
    output_csv: Optional[Path]
    output_manifest: Optional[Path]


def _composite_keys(df: pd.DataFrame, loan_field: str) -> pd.Series:
    sid = df["source_portfolio_id"].astype(str).str.strip()
    loan = df[loan_field].astype(str).str.strip()
    return sid + "/" + loan


def assemble_platform_canonical(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    write: bool = True,
) -> AssemblyResult:
    """Assemble the latest canonical per portfolio into one platform dataset.

    ``inputs`` may be a directory (globbed for ``*_canonical_typed.csv``) or an
    explicit list of canonical output paths.
    """
    # 1. Resolve input file list (reads only; never re-runs onboarding).
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        paths = discover_canonical_outputs(inputs)
    elif isinstance(inputs, (str, Path)):
        paths = [Path(inputs)]
    else:
        paths = [Path(p) for p in inputs]
    if not paths:
        raise PlatformAssemblyError("No canonical outputs found to assemble.")

    # 2. Read per-file metadata and pick the latest snapshot per portfolio.
    snapshots = [read_portfolio_snapshot(p) for p in paths]
    chosen = select_latest_per_portfolio(snapshots)

    # Validation: each source_portfolio_id appears exactly once in the assembly.
    if len(chosen) != len({s.source_portfolio_id for s in chosen.values()}):  # pragma: no cover
        raise PlatformAssemblyError("Internal: duplicate portfolio after latest-selection.")

    # 3. Read the chosen full canonicals and combine (no transformation).
    frames: List[pd.DataFrame] = []
    per_portfolio: List[Dict[str, Any]] = []
    for pid in sorted(chosen):
        snap = chosen[pid]
        df = pd.read_csv(snap.path, low_memory=False)

        # Provenance must be present and preserved unchanged.
        missing = [f for f in _provenance.PROVENANCE_FIELDS if f not in df.columns]
        if missing:
            raise PlatformAssemblyError(
                f"{snap.path.name}: missing provenance fields {missing}; "
                f"cannot assemble without full provenance."
            )
        loan_field = _loan_key_field(df)
        if loan_field is None:
            raise PlatformAssemblyError(
                f"{snap.path.name}: no loan identifier column ({LOAN_KEY_FIELDS})."
            )
        df = df.copy()
        df[PLATFORM_KEY_COLUMN] = _composite_keys(df, loan_field)
        frames.append(df)
        per_portfolio.append({
            "source_portfolio_id": pid,
            "source_portfolio_type": str(df["source_portfolio_type"].iloc[0]) if len(df) else None,
            "source_portfolio_label": (str(df["source_portfolio_label"].iloc[0])
                                       if len(df) and pd.notna(df["source_portfolio_label"].iloc[0]) else None),
            "snapshot_date": snap.snapshot_date,
            "source_file": str(snap.path),
            "loan_key_field": loan_field,
            "row_count": int(len(df)),
        })

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # 4. Composite uniqueness: (source_portfolio_id + loan_identifier) is unique.
    dup_mask = combined[PLATFORM_KEY_COLUMN].duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(combined.loc[dup_mask, PLATFORM_KEY_COLUMN].unique().tolist())
        raise PlatformAssemblyError(
            f"Duplicate composite keys (source_portfolio_id + loan_identifier) "
            f"in platform assembly: {dups[:20]}"
            f"{' …' if len(dups) > 20 else ''}"
        )

    manifest: Dict[str, Any] = {
        "artifact": "platform_canonical",
        "note": "Combined latest-per-portfolio canonical. Source of truth for "
                "downstream MI; individual portfolio canonicals are unchanged.",
        "composite_key": "source_portfolio_id + loan_identifier",
        "composite_key_column": PLATFORM_KEY_COLUMN,
        "portfolio_count": len(per_portfolio),
        "total_rows": int(len(combined)),
        "portfolios": per_portfolio,
        "candidate_files_scanned": [str(p) for p in paths],
    }

    output_csv = output_manifest = None
    if write:
        if out_dir is None:
            raise PlatformAssemblyError("out_dir is required when write=True.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / PLATFORM_CANONICAL_NAME
        output_manifest = out_dir / PLATFORM_MANIFEST_NAME
        combined.to_csv(output_csv, index=False)
        manifest["output_csv"] = str(output_csv)
        output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AssemblyResult(combined, manifest, output_csv, output_manifest)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Platform Portfolio Assembler — combine the latest accepted "
                    "canonical output per source_portfolio_id into one platform "
                    "canonical dataset. Reads canonical outputs only; never "
                    "re-runs onboarding."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", help="Directory to scan for *_canonical_typed.csv outputs.")
    g.add_argument("--inputs", nargs="+", help="Explicit canonical output paths.")
    ap.add_argument("--out-dir", required=True, help="Where to write the platform canonical.")
    args = ap.parse_args(argv)

    try:
        res = assemble_platform_canonical(
            args.root if args.root else args.inputs, args.out_dir, write=True,
        )
    except PlatformAssemblyError as exc:
        print(f"[platform-assembler] ERROR: {exc}")
        return 2

    print("=" * 64)
    print("Platform canonical assembled")
    print(f"  portfolios: {res.manifest['portfolio_count']}  rows: {res.manifest['total_rows']}")
    for p in res.manifest["portfolios"]:
        print(f"  - {p['source_portfolio_id']:14} {p['snapshot_date']}  "
              f"({p['row_count']} loans)  <- {Path(p['source_file']).name}")
    print(f"  output: {res.output_csv}")
    print("=" * 64)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
