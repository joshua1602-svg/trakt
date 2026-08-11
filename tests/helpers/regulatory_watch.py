"""tests.helpers.regulatory_watch — fixtures for the ESMA Annex 2 watch tests.

Loads the committed slice of the REAL ESMA artefacts
(``tests/fixtures/regulatory_watch/``) and provides deterministic mutation
helpers. The mutations operate on the *raw workbook rows*, so every scenario
exercises the real normalizer as well as the comparator — a synthetic delta
that the parser cannot see is not a proof of anything.

No randomness, no timestamps, no network. Every helper is a pure function of
its inputs so a scenario replays byte-identically.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from regulatory_watch import PARSER_VERSION
from regulatory_watch.annex2_spec import RawRow, normalize
from regulatory_watch.contracts import NormalizedSpec, SourceSnapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "regulatory_watch"

ROWS_FIXTURE = FIXTURE_DIR / "annex2_workbook_rows.json"
CODELISTS_FIXTURE = FIXTURE_DIR / "annex2_codelists.json"

WORKBOOK_ARTEFACT = "esma_annex2_message_workbook"
SCHEMA_ARTEFACT = "esma_annex2_xml_schema"

#: The vendored authoritative artefacts, for the integration-scale tests.
REAL_WORKBOOK = (REPO_ROOT / "DRAFT1auth.099.001.04_non-ABCP Underlying "
                             "Exposure Report_Version_1.3.1.xlsx")
REAL_SCHEMA = REPO_ROOT / "DRAFT1auth.099.001.04_1.3.0.xsd"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_rows() -> List[RawRow]:
    data = json.loads(ROWS_FIXTURE.read_text(encoding="utf-8"))
    return [RawRow.from_dict(r) for r in data["rows"]]


def load_codelists() -> Dict[str, List[str]]:
    data = json.loads(CODELISTS_FIXTURE.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in data["codelists"].items()}


def build_spec(rows: Optional[Sequence[RawRow]] = None,
               codelists: Optional[Dict[str, List[str]]] = None,
               *, spec_version: str = "fixture-baseline",
               snapshots: Optional[Sequence[SourceSnapshot]] = None,
               ) -> NormalizedSpec:
    """Normalize rows + code lists exactly as the production path does."""
    return normalize(
        list(rows if rows is not None else load_rows()),
        dict(codelists if codelists is not None else load_codelists()),
        artefact_id=WORKBOOK_ARTEFACT, schema_artefact_id=SCHEMA_ARTEFACT,
        spec_version=spec_version, snapshots=snapshots,
        scope={"templates": ["ALL", "RRE"], "performance": "PRF",
               "fixture": "committed slice of the ESMA v1.3.1 workbook"})


def snapshot(sha256: str, artefact_id: str = WORKBOOK_ARTEFACT,
             *, byte_size: int = 1024,
             retrieved_at: str = "2026-01-01T00:00:00Z") -> SourceSnapshot:
    """A deterministic OK snapshot with a caller-chosen digest."""
    return SourceSnapshot(
        artefact_id=artefact_id, sha256=sha256, byte_size=byte_size,
        retrieved_at=retrieved_at, snapshot_path=f"/snapshots/{sha256[:8]}",
        parser_version=PARSER_VERSION, external_version="test",
        retrieval_status="OK")


def failed_snapshot(artefact_id: str = WORKBOOK_ARTEFACT,
                    detail: str = "source unavailable") -> SourceSnapshot:
    from regulatory_watch.contracts import SOURCE_CHECK_FAILED
    return SourceSnapshot(
        artefact_id=artefact_id, sha256="", byte_size=0,
        retrieved_at="2026-01-01T00:00:00Z", snapshot_path="",
        parser_version=PARSER_VERSION,
        retrieval_status=SOURCE_CHECK_FAILED, retrieval_detail=detail)


# --------------------------------------------------------------------------- #
# Row-level mutations (deterministic)
# --------------------------------------------------------------------------- #

def _replace(row: RawRow, **changes: Any) -> RawRow:
    return dataclasses.replace(row, **changes)


def rows_for(rows: Sequence[RawRow], code: str) -> List[RawRow]:
    return [r for r in rows if code in r.codes]


def drop_code(rows: Sequence[RawRow], code: str) -> List[RawRow]:
    """Remove a field entirely (rows naming only that code are deleted)."""
    out: List[RawRow] = []
    for row in rows:
        if code not in row.codes:
            out.append(row)
            continue
        remaining = tuple(c for c in row.codes if c != code)
        if remaining:
            out.append(_replace(row, codes=remaining))
    return out


def add_code(rows: Sequence[RawRow], template_code: str, new_code: str,
             *, label: str, tag: str) -> List[RawRow]:
    """Introduce a new field by cloning an existing one under a new tag.

    The clone keeps the source structure (value leaf, ND branch) so the new
    field normalizes exactly like a real one.
    """
    template_rows = rows_for(rows, template_code)
    if not template_rows:
        raise KeyError(template_code)
    base_element = min((r.path for r in template_rows), key=len)
    parent = base_element.rsplit("/", 1)[0]
    new_element = f"{parent}/{tag}"
    max_row = max(r.row_number for r in rows)

    clones: List[RawRow] = []
    for offset, row in enumerate(sorted(template_rows,
                                        key=lambda r: r.row_number), start=1):
        clones.append(_replace(
            row,
            row_number=max_row + offset,
            codes=(new_code,),
            rts_field_name=label,
            path=row.path.replace(base_element, new_element, 1),
            xml_tag=(tag if row.path == base_element else row.xml_tag),
        ))
    return list(rows) + clones


def edit_rows(rows: Sequence[RawRow], code: str,
              predicate, **changes: Any) -> List[RawRow]:
    """Apply ``changes`` to the rows of ``code`` matching ``predicate``."""
    out: List[RawRow] = []
    hit = False
    for row in rows:
        if code in row.codes and predicate(row):
            hit = True
            out.append(_replace(row, **changes))
        else:
            out.append(row)
    if not hit:
        raise AssertionError(f"no row matched for {code}")
    return out


def element_row(rows: Sequence[RawRow], code: str) -> RawRow:
    """The shallowest non-ND row for a code (its element node, in the slice)."""
    candidates = [r for r in rows_for(rows, code) if "NoDataOptn" not in r.path]
    return min(candidates, key=lambda r: (len(r.path.split("/")),
                                          r.row_number))


def value_leaf_row(rows: Sequence[RawRow], code: str) -> RawRow:
    """The first typed value leaf for a code."""
    candidates = [r for r in rows_for(rows, code)
                  if "NoDataOptn" not in r.path and r.data_type]
    return min(candidates, key=lambda r: r.row_number)


def nd_leaf_rows(rows: Sequence[RawRow], code: str) -> List[RawRow]:
    return [r for r in rows_for(rows, code)
            if r.path.endswith("/NoData") and r.data_type]


def remove_nd_branch(rows: Sequence[RawRow], code: str) -> List[RawRow]:
    """Withdraw the ND option for a field (its NoDataOptn branch disappears)."""
    return [r for r in rows
            if not (code in r.codes and "NoDataOptn" in r.path.split("/"))]


def move_code_path(rows: Sequence[RawRow], code: str,
                   old_prefix: str, new_prefix: str) -> List[RawRow]:
    """Relocate a field's element (and everything under it) to a new XML path."""
    return [
        _replace(row, path=row.path.replace(old_prefix, new_prefix, 1))
        if code in row.codes and row.path.startswith(old_prefix) else row
        for row in rows
    ]


def reorder_code(rows: Sequence[RawRow], code: str,
                 new_row_number: int) -> List[RawRow]:
    """Move a field's rows in the sheet sequence without changing anything else."""
    base = min(r.row_number for r in rows_for(rows, code))
    out: List[RawRow] = []
    for row in rows:
        if code in row.codes:
            out.append(_replace(row,
                                row_number=new_row_number
                                + (row.row_number - base)))
        else:
            out.append(row)
    return out


def mutate_codelist(codelists: Dict[str, List[str]], type_name: str,
                    *, add: Sequence[str] = (), remove: Sequence[str] = (),
                    ) -> Dict[str, List[str]]:
    out = {k: list(v) for k, v in codelists.items()}
    values = [v for v in out.get(type_name, []) if v not in set(remove)]
    values.extend(v for v in add if v not in values)
    out[type_name] = values
    return out
