"""tests/analytical/client_pack.py — the whole client-readiness fixture, one call.

``build(root)`` writes, for both books:

* thirteen monthly funded snapshots, 2025-06-30 to 2026-06-30 (``client_funded``)
* fifty-two weekly pipeline extracts ending 2026-06-26 (``client_fixture``)

and returns a truth manifest an independent reconciliation checks against.

One entry point rather than three call sites, because every Tranche E/F/G
measurement has to run against the SAME bytes: a baseline taken from a
differently-parameterised build is not a baseline.

The nominated stress book
-------------------------
``kestrelmoor`` carries the declining-KFI profile over the final thirteen weeks;
``alderbridge`` stays stationary as the control. Two reasons, both about
measurability: kestrelmoor has the larger pipeline (61 new cases/week against
42), so a 65% fall in intake still leaves a usable sample in the final weeks;
and alderbridge is the book the V1 evidence and the demonstration platform are
anchored to, so leaving it stationary keeps a control comparable with every
existing measurement.

The decline is applied to NEW KFI intake only. A lender's existing pipeline does
not evaporate when origination slows, and the resulting shape is the point of the
stress case: intake falls 61 -> 21/week and the live pipeline 368 -> 203, but the
fall is top-weighted (KFI stock 218 -> 87 while OFFER holds 59 -> 50) and
completions lag rather than collapse. A forecast extrapolating an older run-rate
over-states; a maturity treatment that reads the thin recent KFI cohort as a wave
of non-conversions under-states.

Nothing here is hashed against the V1 evidence set. The V1 packs and tapes are
produced by ``pipeline_fixture`` and the default path of ``second_book``, neither
of which this touches.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List

from . import client_fixture, client_funded
from .client_fixture import ALDERBRIDGE, KESTRELMOOR, DeclineProfile

#: Last published weekly pipeline extract. A Friday, four days before the last
#: funded cut, so the pipeline is legitimately AHEAD of the funded book — which
#: is the operational reality a forecast has to reconcile across.
LAST_PIPELINE_WEEK = date(2026, 6, 26)
PIPELINE_WEEKS = 52

#: The book carrying the declining-KFI stress case, and the profile applied.
STRESS_BOOK = KESTRELMOOR.client
STRESS_PROFILE = DeclineProfile(weeks=13, final_fraction=0.35)


def build(root: Path) -> Dict[str, object]:
    """Write both books' funded history and pipeline pack under ``root``."""
    root = Path(root)
    funded = client_funded.build(root)
    pipelines: Dict[str, List[str]] = {}
    for profile in (ALDERBRIDGE, KESTRELMOOR):
        decline = STRESS_PROFILE if profile.client == STRESS_BOOK else None
        paths = client_fixture.write_pack(
            root, profile, weeks=PIPELINE_WEEKS,
            last_week=LAST_PIPELINE_WEEK, decline=decline)
        pipelines[profile.client] = [str(p) for p in paths]
    return {
        "funded": funded,
        "pipeline": pipelines,
        "reporting_dates": list(client_funded.REPORTING_DATES),
        "last_pipeline_week": LAST_PIPELINE_WEEK.isoformat(),
        "stress_book": STRESS_BOOK,
        "control_book": ALDERBRIDGE.client,
    }


def mi_env(root: Path, client_id: str, reporting_date: str = None) -> Dict[str, str]:
    """The environment pointing the MI read path at one book of this pack."""
    env = client_funded.mi_env(root, client_id,
                               reporting_date or client_funded.REPORTING_DATES[-1])
    env["MI_AGENT_PIPELINE_ROOT"] = str(Path(root) / client_id / "pipeline")
    return env


__all__ = ["build", "mi_env", "LAST_PIPELINE_WEEK", "PIPELINE_WEEKS",
           "STRESS_BOOK", "STRESS_PROFILE"]
