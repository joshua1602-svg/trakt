"""analytics_lib.migration — snapshot-to-snapshot migration (STUB ONLY).

Phase 1 intentionally does NOT implement recurring snapshot-to-snapshot
migration (risk-grade / PD / IFRS 9 transition matrices, deterioration flags).
That work depends on the snapshot & history layer (build-plan Phase 2) which
does not exist yet. This module is a placeholder that documents the intended
surface so later phases have a stable import location.

When implemented, these functions will operate on two (or N) loan-level frames
joined on a stable ``loan_id`` and return transition matrices / deterioration
flags as plain DataFrames — still pure, still UI-free.
"""

from __future__ import annotations

from typing import Any

_DEFERRED = (
    "Snapshot-to-snapshot migration is deferred to a later phase (requires the "
    "Phase 2 snapshot/history layer). Phase 1 ships point-in-time analytics "
    "only; see analytics_lib.cohort for point-in-time cohort/vintage."
)


def transition_matrix(*_args: Any, **_kwargs: Any) -> None:
    """Not implemented in Phase 1 — see module docstring."""
    raise NotImplementedError(_DEFERRED)


def deterioration_flags(*_args: Any, **_kwargs: Any) -> None:
    """Not implemented in Phase 1 — see module docstring."""
    raise NotImplementedError(_DEFERRED)
