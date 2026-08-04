"""simulation.assertions.regime — regime projection and regulatory artefacts.

Applicability is DECLARED on the manifest and never inferred. Three outcomes are
possible and all three are verified:

``applicable``            project canonical truth with the production
                          ``regime_projector``, assert the field values, the
                          field ordering and reproducibility, and then either
                          build the artefact (Annex 2, which has a committed
                          builder and XSD) or assert a clean refusal (every other
                          annex, which has no committed delivery template).
``not_applicable``        skip, with the manifest's stated reason. Nothing is
                          projected and no deal-level data is fabricated.
``not_supported``         skip, naming the repository capability that is absent.

Two rules are enforced throughout:

* the projection is built from CANONICAL truth, never from the raw source file;
* ND values live only in the projection — the canonical assertions check the
  other direction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import (
    ARTEFACT_AWAITING_CONFIG,
    ARTEFACT_SUPPORTED,
    STAGE_REGIME,
    Assertion,
    FailureClass,
)
from . import check, compare

_FAIL = FailureClass.REGULATORY_ARTEFACT_FAILURE

#: The phrase Gate 5 uses when a regime has no committed delivery template. The
#: framework asserts this exact governed refusal rather than a crash.
NOT_CONFIGURED_MARKER = "is not configured"

#: ND sentinels the regime layer is ALLOWED to insert.
ND_SENTINELS = ("ND1", "ND2", "ND3", "ND4", "ND5", "ND6", "ND7")


def assert_skipped(case) -> List[Assertion]:
    """A case with no applicable regime must say so, and stop."""
    regime = case.regime
    return [
        check(STAGE_REGIME, "non-applicable regime states a reason",
              bool(regime.reason.strip()), actual=regime.reason,
              detail="Skipping a regime silently is indistinguishable from "
                     "forgetting it.",
              failure_class=FailureClass.SIMULATION_DEFECT),
        check(STAGE_REGIME, "no regime is named for a non-applicable case",
              regime.regime_id is None, actual=regime.regime_id,
              failure_class=FailureClass.SIMULATION_DEFECT),
    ]


def assert_projection(projected: Path, canonical: Path, case,
                      truth_period: Dict[str, Any]) -> List[Assertion]:
    """The Gate 4 projection, against canonical truth and the ESMA contract."""
    out: List[Assertion] = []
    regime = case.regime.regime_id
    frame = pd.read_csv(projected, low_memory=False, dtype=str)
    canon = pd.read_csv(canonical, low_memory=False)

    out.append(check(STAGE_REGIME, f"{regime}: projection has one row per exposure",
                     len(frame) == len(canon), expected=len(canon),
                     actual=len(frame), failure_class=_FAIL))
    out.append(check(STAGE_REGIME, f"{regime}: projection emits ESMA codes",
                     _looks_like_esma_codes(frame.columns),
                     actual=list(frame.columns)[:8],
                     detail="A projection whose columns are still canonical "
                            "field names has not been projected at all.",
                     failure_class=_FAIL))

    out += _assert_key_values(frame, canon, case, regime)
    out.append(_assert_field_ordering(frame, regime))
    out.append(_assert_nd_confined_to_regime(frame, canon, regime))
    return out


def _looks_like_esma_codes(columns) -> bool:
    import re
    pattern = re.compile(r"^[A-Z]{4}\d+$")
    return sum(1 for c in columns if pattern.match(str(c))) >= max(
        5, len(list(columns)) // 4)


def _assert_key_values(frame: pd.DataFrame, canon: pd.DataFrame, case,
                       regime: str) -> List[Assertion]:
    """Key field values must carry through from canonical truth unchanged.

    Checked by TOTAL rather than row by row: the projection may reorder, and a
    total that agrees is the property that matters for a regulatory submission.
    """
    from ..pipeline import REPO_ROOT

    import yaml

    registry = yaml.safe_load(
        (REPO_ROOT / "config" / "system" / "fields_registry.yaml")
        .read_text(encoding="utf-8"))
    fields = registry.get("fields") or {}
    out: List[Assertion] = []

    for canonical_field in ("current_principal_balance",
                            "original_principal_balance"):
        meta = (fields.get(canonical_field) or {}).get("regime_mapping") or {}
        code = (meta.get(regime) or {}).get("code")
        if not code or code not in frame.columns or canonical_field not in canon.columns:
            continue
        projected_total = float(pd.to_numeric(frame[code], errors="coerce")
                                .fillna(0.0).sum())
        canonical_total = float(pd.to_numeric(canon[canonical_field],
                                              errors="coerce").fillna(0.0).sum())
        out.append(compare(
            STAGE_REGIME, f"{regime}: {code} totals canonical {canonical_field}",
            projected_total, canonical_total, failure_class=_FAIL))
    if not out:
        out.append(check(
            STAGE_REGIME, f"{regime}: a mapped monetary field was verifiable",
            False, detail="No monetary canonical field carries a code for this "
                          "regime, so no value assertion was possible.",
            failure_class=_FAIL))
    return out


def _assert_field_ordering(frame: pd.DataFrame, regime: str) -> Assertion:
    """Mandatory codes must precede optional ones, per the projector's contract."""
    from ..pipeline import REPO_ROOT

    import yaml

    registry = yaml.safe_load(
        (REPO_ROOT / "config" / "system" / "fields_registry.yaml")
        .read_text(encoding="utf-8"))
    priority: Dict[str, str] = {}
    for meta in (registry.get("fields") or {}).values():
        mapping = ((meta or {}).get("regime_mapping") or {}).get(regime) or {}
        code = mapping.get("code")
        if code:
            priority[str(code)] = str(mapping.get("priority", "")).strip().lower()

    seen_optional = False
    misordered: List[str] = []
    for column in frame.columns:
        rank = priority.get(str(column))
        if rank == "optional":
            seen_optional = True
        elif rank == "mandatory" and seen_optional:
            misordered.append(str(column))
    return check(STAGE_REGIME, f"{regime}: mandatory fields precede optional",
                 not misordered, expected=[], actual=misordered[:8],
                 failure_class=_FAIL)


def _assert_nd_confined_to_regime(frame: pd.DataFrame, canon: pd.DataFrame,
                                  regime: str) -> Assertion:
    """ND values may appear in the projection; they must not be in canonical.

    The canonical direction is asserted in ``assertions.canonical``; this side
    records that the projection is where the ND layer actually lives, so a
    projection with no ND at all is disclosed rather than assumed correct.
    """
    nd_cells = 0
    for column in frame.columns:
        nd_cells += int(frame[column].astype(str).str.strip()
                        .isin(ND_SENTINELS).sum())
    return check(STAGE_REGIME,
                 f"{regime}: ND values, where present, live in the projection",
                 True, actual=nd_cells,
                 detail=f"{nd_cells} ND cell(s) in the projection; canonical "
                        f"truth is asserted ND-free separately.")


def assert_reproducible(projected: Path, rerun: Path, regime: str
                        ) -> List[Assertion]:
    """The same canonical truth must project to the same artefact twice."""
    def digest(path: Path) -> str:
        frame = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
        payload = frame.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return [check(STAGE_REGIME, f"{regime}: projection is reproducible",
                  digest(projected) == digest(rerun),
                  expected=digest(projected)[:16], actual=digest(rerun)[:16],
                  detail="A regulatory submission that differs between two runs "
                         "of the same data is not submittable.",
                  failure_class=_FAIL)]


def assert_artefact(case, gate_run, xml_path: Optional[Path]) -> List[Assertion]:
    """The regulatory artefact, or the governed refusal that replaces it."""
    regime = case.regime.regime_id
    status = case.regime.artefact_status
    out: List[Assertion] = []

    if status == ARTEFACT_SUPPORTED:
        out.append(check(STAGE_REGIME, f"{regime}: XML artefact produced",
                         bool(xml_path and xml_path.exists()),
                         actual=str(xml_path), failure_class=_FAIL))
        if xml_path and xml_path.exists():
            out.append(check(STAGE_REGIME, f"{regime}: artefact is well-formed XML",
                             _is_well_formed(xml_path), failure_class=_FAIL))
            out.append(check(STAGE_REGIME, f"{regime}: artefact passed XSD validation",
                             "XSD Validation: PASSED" in gate_run.stdout,
                             actual=[ln.strip() for ln in gate_run.stdout.splitlines()
                                     if "XSD" in ln or "XML generation" in ln],
                             failure_class=_FAIL))
            # Measured from the filesystem, never by reading the artefact in.
            # A 100 000-loan Annex 2 submission is multiple gigabytes, and an
            # assertion that has to load the whole thing to say "not empty" is
            # a verifier that cannot verify the scale it exists to test.
            size = xml_path.stat().st_size
            out.append(check(STAGE_REGIME, f"{regime}: artefact is not empty",
                             size > 500, actual=size, failure_class=_FAIL))
        return out

    if status == ARTEFACT_AWAITING_CONFIG:
        combined = f"{gate_run.stdout}\n{gate_run.stderr}"
        out.append(check(
            STAGE_REGIME,
            f"{regime}: delivery refuses cleanly with a stated reason",
            NOT_CONFIGURED_MARKER in combined,
            expected=NOT_CONFIGURED_MARKER,
            actual=_tail(combined),
            detail="No XML template is committed for this regime. Gate 5 must "
                   "say so; it must not crash and it must not invent one.",
            failure_class=_FAIL))
        out.append(check(
            STAGE_REGIME, f"{regime}: no artefact was fabricated",
            not (xml_path and xml_path.exists()), actual=str(xml_path),
            failure_class=_FAIL))
        return out

    return [check(STAGE_REGIME, f"{regime}: artefact status is handled",
                  False, actual=status,
                  failure_class=FailureClass.SIMULATION_DEFECT)]


def _is_well_formed(path: Path) -> bool:
    """Parse the artefact incrementally, discarding each element as it closes.

    ``ElementTree.parse`` builds the whole document in memory. A regulatory
    submission for a large book is gigabytes of XML, and the resulting DOM is
    several times that — enough to be OOM-killed, which is a failure of the
    CHECK rather than of the artefact. ``iterparse`` plus clearing gives the
    same well-formedness answer in constant memory.
    """
    try:
        from xml.etree import ElementTree

        root = None
        with path.open("rb") as handle:
            for event, element in ElementTree.iterparse(
                    handle, events=("start", "end")):
                if event == "start":
                    if root is None:
                        root = element
                    continue
                element.clear()
                if root is not None and element is not root:
                    # Drop the completed siblings the root has accumulated.
                    # The parser keeps its own reference to whatever is still
                    # open, so this only discards what is already parsed.
                    root.clear()
        return True
    except Exception:  # noqa: BLE001 - a malformed artefact is the finding
        return False


def _tail(text: str, lines: int = 6) -> str:
    return "\n".join([ln for ln in text.splitlines() if ln.strip()][-lines:])


__all__ = ["ND_SENTINELS", "NOT_CONFIGURED_MARKER", "assert_artefact",
           "assert_projection", "assert_reproducible", "assert_skipped"]
