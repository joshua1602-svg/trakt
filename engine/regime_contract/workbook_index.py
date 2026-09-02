"""The Annex 2 mapping workbook's XML facts, cached as a build artefact.

The workbook is the authority for a code's XML path, its multiplicity and which
performance branch it belongs to. Reading it costs about twelve seconds per
performance mode, which is fine once inside Gate 5 and unaffordable in every
stage that needs to know whether a field is mandatory.

So the workbook's facts are extracted once into
``config/generated/annex2_workbook_index.json`` and the file records the SHA-256
of the workbook it came from. Any consumer that finds the hash stale rebuilds
from the workbook itself rather than trusting the cache: the workbook always
wins, the cache only makes it cheap. Regenerate with
``python -m engine.regime_contract.workbook_index``.

Nothing here decides anything. It is the workbook, in a format that can be read
in milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]

WORKBOOK = _REPO / ("DRAFT1auth.099.001.04_non-ABCP Underlying Exposure "
                   "Report_Version_1.3.1.xlsx")
SHEET = "DRAFT1auth.099.001.04"
INDEX_PATH = _REPO / "config" / "generated" / "annex2_workbook_index.json"

#: Performance branches the workbook distinguishes.
MODES = ("PRF", "NPRF")

_log = logging.getLogger(__name__)
_cache: Optional[Dict[str, Any]] = None


def workbook_sha256(path: Path = WORKBOOK) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def build_index(workbook: Path = WORKBOOK, sheet: str = SHEET) -> Dict[str, Any]:
    """Read the workbook and reduce it to the facts the contract needs."""
    from engine.gate_5_delivery.xml_builder_annex2 import (
        load_mapping_specs, select_specs_for_value, _parse_multiplicity,
        RECORD_ANCHOR, ND_TAGS)

    codes: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        for code, specs in load_mapping_specs(str(workbook), sheet, mode).items():
            # Annex 2 is the residential real-estate template. The workbook
            # carries every annex on one sheet; the rest belong to other
            # templates and are not this contract's business.
            if not code.startswith(("RREL", "RREC")):
                continue
            entry = codes.setdefault(code, {
                "modes": [], "min_occ": {}, "blank_min_occ": {},
                "value_path": "", "record_anchored": False,
            })
            if mode not in entry["modes"]:
                entry["modes"].append(mode)
            entry["min_occ"][mode] = min(
                _parse_multiplicity(s.multiplicity)[0] for s in specs)
            # Whether the BUILDER will refuse a blank here. It picks a branch
            # first and then reads that branch's own multiplicity, so a code
            # with an optional branch somewhere can still be refused when the
            # branch chosen for an empty value is [1..1]. This is the definition
            # Gate 4b must share, or the two gates disagree about the same frame.
            record_specs = [x for x in specs if RECORD_ANCHOR in x.path]
            chosen = select_specs_for_value(record_specs or specs, "")
            entry["blank_min_occ"][mode] = (
                _parse_multiplicity(chosen[0].multiplicity)[0] if chosen
                else min(_parse_multiplicity(x.multiplicity)[0]
                         for x in (record_specs or specs)))
            for s in specs:
                # "/Cxl" is the cancellation message branch, not a value the
                # report carries. The builder skips it too.
                if (s.tag.upper() not in ND_TAGS and not entry["value_path"]
                        and "/Cxl" not in s.path):
                    entry["value_path"] = s.path
                if RECORD_ANCHOR in s.path:
                    entry["record_anchored"] = True
    return {
        "source": workbook.name,
        "sheet": sheet,
        "source_sha256": workbook_sha256(workbook),
        "note": ("Generated from the mapping workbook by "
                 "engine/regime_contract/workbook_index.py. Do not edit. "
                 "A stale source_sha256 makes every consumer rebuild from the "
                 "workbook instead of reading this file."),
        "codes": codes,
    }


def load_index(*, allow_rebuild: bool = True) -> Dict[str, Any]:
    """The workbook's facts, from the cache when it is current."""
    global _cache
    if _cache is not None:
        return _cache
    current = workbook_sha256()
    if INDEX_PATH.exists():
        try:
            doc = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            doc = {}
        if doc.get("codes") and (not current or doc.get("source_sha256") == current):
            _cache = doc
            return _cache
        _log.warning("Annex 2 workbook index is stale (%s); rebuilding from %s",
                     INDEX_PATH.name, WORKBOOK.name)
    if not allow_rebuild:
        raise FileNotFoundError(f"no current Annex 2 workbook index at {INDEX_PATH}")
    _cache = build_index()
    return _cache


def codes() -> Dict[str, Dict[str, Any]]:
    return load_index().get("codes", {})


def min_occurrence(code: str, mode: Optional[str] = None,
                   *, key: str = "blank_min_occ") -> int:
    """The ``minOccurs`` that decides whether a BLANK is refused.

    Gate 5 selects an XML branch for the value in hand and then reads that
    branch's own multiplicity. For an empty value it selects the first non-
    no-data branch, so the number that matters is that branch's ``minOccurs`` —
    not the minimum across every branch, which can be lower. Mandatory-ness is
    therefore defined as "the builder would refuse a blank here", which is the
    only definition on which Gate 4b and Gate 5 can agree.

    With no mode the answer is the lowest across both performance branches: a
    field is only optional if some branch that could be selected permits it to
    be absent.
    """
    entry = codes().get(code) or {}
    occ = entry.get(key) or entry.get("min_occ") or {}
    vals = [occ[m] for m in ([mode] if mode else MODES) if m in occ]
    return min(vals) if vals else 0


def is_mandatory(code: str, mode: Optional[str] = None) -> bool:
    return min_occurrence(code, mode) >= 1


def value_path(code: str) -> str:
    """The first non-no-data XML path, or '' where the code has no element."""
    return str((codes().get(code) or {}).get("value_path") or "")


def has_value_path(code: str) -> bool:
    """Does the workbook give this code a value-carrying XML path?

    Whether that path is an ELEMENT is the schema's answer, not the workbook's —
    see ``annex2_contract``, which resolves the path against the XSD. A code with
    a workbook path the schema does not define as an element is carried as an
    attribute of the value it qualifies (a currency on an amount): disclosed, but
    producing no node of its own.
    """
    return bool(value_path(code))


def main() -> None:
    doc = build_index()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(f"wrote {INDEX_PATH} — {len(doc['codes'])} Annex 2 codes")


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.path.insert(0, str(_REPO))
    main()
