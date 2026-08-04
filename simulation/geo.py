"""simulation.geo — UK geography drawn from the repository's own ITL lookup.

A generated loan's postcode must genuinely sit inside the region it claims,
otherwise the canonical transform's postcode → ITL3 enrichment and the readable
region label would disagree and the geography assertions would be testing the
simulator rather than the platform. Districts are therefore read from
``uk_itl_master_lookup_v2.csv``, the same file
``engine/gate_2_transform/canonical_transform.py`` enriches against.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
ITL_LOOKUP = REPO_ROOT / "uk_itl_master_lookup_v2.csv"

#: Clean ITL1 display label → the name as it appears in the lookup.
ITL1_LOOKUP_NAMES: Dict[str, str] = {
    "South East": "South East (England)",
    "London": "London",
    "South West": "South West (England)",
    "East of England": "East (England)",
    "North West": "North West (England)",
    "West Midlands": "West Midlands (England)",
    "Yorkshire and The Humber": "Yorkshire and The Humber",
    "East Midlands": "East Midlands (England)",
    "Scotland": "Scotland",
    "North East": "North East (England)",
    "Wales": "Wales",
}

REGIONS: List[str] = list(ITL1_LOOKUP_NAMES)

#: Regional value index relative to the national average. Drives collateral
#: values so LTVs reconcile regionally rather than being sampled independently.
REGION_VALUE_INDEX: Dict[str, float] = {
    "South East": 1.34, "London": 1.92, "South West": 1.12,
    "East of England": 1.18, "North West": 0.78, "West Midlands": 0.84,
    "Yorkshire and The Humber": 0.74, "East Midlands": 0.82,
    "Scotland": 0.72, "North East": 0.62, "Wales": 0.72,
}

_POOL: Optional[Dict[str, List[str]]] = None

#: Letters used in the postcode inward code. ``C``, ``I``, ``K``, ``M``, ``O``
#: and ``V`` are excluded, as they are in real UK postcodes.
_INWARD_LETTERS = "ABDEFGHJLNPQRSTUWXYZ"


def postcode_pool() -> Dict[str, List[str]]:
    """``{clean ITL1 label: [postcode districts]}`` from the ITL master lookup."""
    global _POOL
    if _POOL is not None:
        return _POOL
    by_region: Dict[str, set] = {k: set() for k in ITL1_LOOKUP_NAMES}
    reverse = {v: k for k, v in ITL1_LOOKUP_NAMES.items()}
    with ITL_LOOKUP.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            clean = reverse.get(str(row.get("itl1_name", "")).strip())
            prefix = str(row.get("postcode_prefix", "")).strip().upper()
            if clean and prefix:
                by_region[clean].add(prefix)
    pool = {k: sorted(v) for k, v in by_region.items()}
    missing = [k for k, v in pool.items() if not v]
    if missing:  # pragma: no cover - would be a lookup-file regression
        raise ValueError(f"no postcode districts for ITL1 regions {missing}; "
                         f"check {ITL_LOOKUP.name}")
    _POOL = pool
    return pool


def postcode(rng, region: str) -> str:
    """A plausible full UK postcode genuinely inside *region*."""
    districts = postcode_pool()[region]
    district = districts[int(rng.integers(0, len(districts)))]
    sector = int(rng.integers(1, 10))
    a = _INWARD_LETTERS[int(rng.integers(0, len(_INWARD_LETTERS)))]
    b = _INWARD_LETTERS[int(rng.integers(0, len(_INWARD_LETTERS)))]
    return f"{district} {sector}{a}{b}"


def weighted_region(rng, weights: Dict[str, float]) -> str:
    """Pick a region from an explicit weight map (weights need not sum to 1)."""
    names = list(weights)
    return names[weighted_index(rng, [weights[n] for n in names])]


def weighted_index(rng, weights) -> int:
    """Index into *weights*, drawn proportionally. Weights need not sum to 1."""
    values = [float(w) for w in weights]
    total = sum(values)
    if total <= 0:  # pragma: no cover - configuration error
        raise ValueError("weighted_index needs at least one positive weight")
    draw = float(rng.random()) * total
    cumulative = 0.0
    for i, w in enumerate(values):
        cumulative += w
        if draw <= cumulative:
            return i
    return len(values) - 1


__all__ = ["ITL1_LOOKUP_NAMES", "REGIONS", "REGION_VALUE_INDEX", "postcode",
           "postcode_pool", "weighted_index", "weighted_region"]
