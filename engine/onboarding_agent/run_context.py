"""
run_context.py
==============

Deterministic, auditable extraction of **run / source context fields** that are
portfolio-level (one value for the whole tape) rather than per-loan — most
importantly the reporting cut-off date ``data_cut_off_date`` (ESMA Annex 2
``RREL6`` / ``CutOffDt``).

The Onboarding Agent already profiles each source file and surfaces a
``detected_reporting_date`` onto ``01_file_inventory.csv`` (from columns flagged
``likely_reporting_date``). This module turns that, plus a deterministic
fallback chain, into a single resolved ``data_cut_off_date`` with full source
evidence so it can be carried through the handoff package and materialised by the
Transformation Agent.

Resolution priority (deterministic, never invented):

    1. explicit ``--override-reporting-date`` CLI value (operator override);
    2. source column values (the canonical ``data_cut_off_date`` column / the
       profiler's ``detected_reporting_date``);
    3. a date embedded in the source file name (e.g. ``..._012026.csv``);
    4. a configured static reporting date in asset/regime config;
    5. a plain ``--reporting-date`` CLI fallback (only when nothing above).

If several candidates within the chosen tier disagree, the conflict is surfaced
(never silently resolved). If nothing is found, it is surfaced as missing —
no date is fabricated, so downstream Validation can still fail.
"""

from __future__ import annotations

import calendar
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Source labels (controlled vocabulary, recorded in lineage / manifest).
SRC_CLI_OVERRIDE = "cli_override"
SRC_SOURCE_COLUMN = "source_column"
SRC_FILENAME = "filename"
SRC_CONFIG = "config"
SRC_CLI_FALLBACK = "cli_fallback"

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def _month_end_iso(year: int, month: int) -> Optional[str]:
    if not (1 <= month <= 12) or not (1900 <= year <= 2100):
        return None
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last:02d}"


def normalize_to_iso(raw: Any, *, dayfirst: bool = True) -> Optional[str]:
    """Parse a single date-ish value to ISO ``YYYY-MM-DD`` deterministically.

    Handles ISO, D/M/Y (day-first), Y/M/D, and "Month YYYY" (-> month end).
    Returns ``None`` when it cannot be parsed without guessing.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "nat", "<na>"):
        return None

    # Already ISO.
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return f"{y:04d}-{mo:02d}-{d:02d}" if _valid_ymd(y, mo, d) else None
        except ValueError:
            return None

    # D/M/Y or M/D/Y with separators.
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", s)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = c if c > 99 else (2000 + c)
        day, month = (a, b) if dayfirst else (b, a)
        # If the day-first interpretation is impossible but the other works, swap.
        if not _valid_ymd(year, month, day) and _valid_ymd(year, a, b):
            month, day = a, b
        return f"{year:04d}-{month:02d}-{day:02d}" if _valid_ymd(year, month, day) else None

    # Y/M/D with separators.
    m = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}" if _valid_ymd(y, mo, d) else None

    # "Month YYYY" / "YYYY Month" -> month end.
    tokens = re.split(r"[\s,_\-]+", s.lower())
    yr = next((int(t) for t in tokens if re.fullmatch(r"(19|20)\d{2}", t)), None)
    mon = next((_MONTHS[t] for t in tokens if t in _MONTHS), None)
    if yr and mon:
        return _month_end_iso(yr, mon)

    return None


def _valid_ymd(y: int, mo: int, d: int) -> bool:
    if not (1900 <= y <= 2100 and 1 <= mo <= 12):
        return False
    try:
        return 1 <= d <= calendar.monthrange(y, mo)[1]
    except (ValueError, IndexError):
        return False


def dates_from_filename(name: str) -> List[str]:
    """Extract candidate ISO dates embedded in a file name (deterministic)."""
    out: List[str] = []
    stem = Path(str(name)).stem

    # Full ISO / D-M-Y patterns.
    for m in re.finditer(r"(\d{4}[\-_./]\d{1,2}[\-_./]\d{1,2}|\d{1,2}[\-_./]\d{1,2}[\-_./]\d{4})", stem):
        iso = normalize_to_iso(m.group(1).replace("_", "-").replace(".", "-"))
        if iso:
            out.append(iso)

    # "Month YYYY" / "YYYY Month".
    low = stem.lower().replace("_", " ").replace("-", " ")
    yr = re.search(r"(19|20)\d{2}", low)
    mon = next((v for k, v in _MONTHS.items() if re.search(rf"\b{k}\b", low)), None)
    if yr and mon:
        me = _month_end_iso(int(yr.group(0)), mon)
        if me:
            out.append(me)

    # 6-digit MMYYYY / YYYYMM tokens (e.g. 012026 -> Jan 2026, 202601 -> Jan 2026).
    for tok in re.findall(r"(?<!\d)(\d{6})(?!\d)", stem):
        head4, tail4 = tok[:4], tok[2:]
        if re.fullmatch(r"(19|20)\d{2}", head4):          # YYYYMM
            me = _month_end_iso(int(head4), int(tok[4:6]))
        elif re.fullmatch(r"(19|20)\d{2}", tail4):        # MMYYYY
            me = _month_end_iso(int(tail4), int(tok[:2]))
        else:
            me = None
        if me:
            out.append(me)

    # de-dup preserving order
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# Candidate gathering
# --------------------------------------------------------------------------- #

def _read_inventory(project_dir: Path) -> List[Dict[str, str]]:
    p = project_dir / "01_file_inventory.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _central_column_values(central_tape: Path, column: str = "data_cut_off_date") -> List[str]:
    if not central_tape.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(central_tape, dtype=str, usecols=lambda c: c == column)
    except (ValueError, ImportError):
        return []
    if column not in df.columns:
        return []
    vals: List[str] = []
    for v in df[column].tolist():
        iso = normalize_to_iso(v)
        if iso:
            vals.append(iso)
    return vals


def _config_static_date(*config_paths: str) -> Optional[Dict[str, str]]:
    for cp in config_paths:
        if not cp:
            continue
        p = Path(cp)
        if not p.exists():
            continue
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for block_key in ("portfolio", "run_context", "defaults", None):
            block = cfg if block_key is None else (cfg.get(block_key) or {})
            if not isinstance(block, dict):
                continue
            for key in ("static_reporting_date", "reporting_date",
                        "data_cut_off_date", "cut_off_date"):
                iso = normalize_to_iso(block.get(key))
                if iso:
                    return {"value": iso, "source_file": p.name, "key": key}
    return None


def extract_data_cut_off_date(
    project_dir: str | Path,
    central_tape: str | Path,
    *,
    asset_config_path: str = "",
    regime_config_path: str = "",
    cli_reporting_date: str = "",
    override_reporting_date: bool = False,
) -> Dict[str, Any]:
    """Resolve a single portfolio-level ``data_cut_off_date`` with full evidence.

    Returns a dict with the resolved ``value`` (ISO or ""), ``source`` label,
    ``source_file``, ``source_location``, ``confidence``, the full ``candidates``
    evidence list, and ``conflict`` / ``missing`` booleans.
    """
    project_dir = Path(project_dir)
    central_tape = Path(central_tape)
    candidates: List[Dict[str, Any]] = []

    # Tier 2: source columns (canonical column + profiler detected_reporting_date)
    source_vals: List[Dict[str, Any]] = []
    for iso in _central_column_values(central_tape):
        source_vals.append({"value": iso, "source": SRC_SOURCE_COLUMN,
                            "source_file": central_tape.name,
                            "source_location": "column:data_cut_off_date",
                            "confidence": 0.97})
    inventory = _read_inventory(project_dir)
    for item in inventory:
        iso = normalize_to_iso(item.get("detected_reporting_date"))
        if iso:
            source_vals.append({"value": iso, "source": SRC_SOURCE_COLUMN,
                                "source_file": item.get("file_name", ""),
                                "source_location": "profiler:detected_reporting_date",
                                "confidence": 0.95})

    # Tier 3: filename patterns
    filename_vals: List[Dict[str, Any]] = []
    seen_files = set()
    for item in inventory:
        fname = item.get("file_name", "")
        if not fname or fname in seen_files:
            continue
        seen_files.add(fname)
        for iso in dates_from_filename(fname):
            filename_vals.append({"value": iso, "source": SRC_FILENAME,
                                  "source_file": fname,
                                  "source_location": "filename", "confidence": 0.6})

    # Tier 4: config static date
    config_vals: List[Dict[str, Any]] = []
    cfg_hit = _config_static_date(asset_config_path, regime_config_path)
    if cfg_hit:
        config_vals.append({"value": cfg_hit["value"], "source": SRC_CONFIG,
                            "source_file": cfg_hit["source_file"],
                            "source_location": f"config:{cfg_hit['key']}",
                            "confidence": 0.9})

    # CLI value (override or fallback)
    cli_iso = normalize_to_iso(cli_reporting_date) if cli_reporting_date else None

    candidates = source_vals + filename_vals + config_vals
    if cli_iso:
        candidates = [{"value": cli_iso,
                       "source": SRC_CLI_OVERRIDE if override_reporting_date else SRC_CLI_FALLBACK,
                       "source_file": "", "source_location": "cli", "confidence": 1.0}] + candidates

    result: Dict[str, Any] = {
        "value": "", "source": "", "source_file": "", "source_location": "",
        "confidence": 0.0, "candidates": candidates, "conflict": False,
        "conflict_detail": "", "missing": False,
    }

    def _accept(c: Dict[str, Any]) -> None:
        result.update({"value": c["value"], "source": c["source"],
                       "source_file": c.get("source_file", ""),
                       "source_location": c.get("source_location", ""),
                       "confidence": c.get("confidence", 0.0)})

    # Explicit operator override wins outright (records source-derived for audit).
    if cli_iso and override_reporting_date:
        _accept({"value": cli_iso, "source": SRC_CLI_OVERRIDE, "source_file": "",
                 "source_location": "cli", "confidence": 1.0})
        # note if it disagreed with a source-derived value
        sd = {v["value"] for v in source_vals + filename_vals + config_vals}
        if sd and cli_iso not in sd:
            result["conflict_detail"] = (
                f"cli override {cli_iso} differs from source-derived {sorted(sd)}")
        return result

    # Walk the deterministic tiers; surface intra-tier conflicts.
    for tier in (source_vals, filename_vals, config_vals):
        distinct = sorted({c["value"] for c in tier})
        if len(distinct) == 1:
            _accept(next(c for c in tier if c["value"] == distinct[0]))
            return result
        if len(distinct) > 1:
            result["conflict"] = True
            result["conflict_detail"] = (
                f"conflicting {tier[0]['source']} candidates: {distinct}")
            return result

    # No source-derived value: allow a plain CLI fallback if supplied.
    if cli_iso:
        _accept({"value": cli_iso, "source": SRC_CLI_FALLBACK, "source_file": "",
                 "source_location": "cli", "confidence": 0.8})
        return result

    result["missing"] = True
    return result
