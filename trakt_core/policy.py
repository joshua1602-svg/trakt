"""trakt_core.policy — the production data-source approval rule.

One place decides whether a resolved dataset may answer a production question.
Before this module the rule lived only in the Copilot adapter
(``copilot_actions._BLOCKED_SOURCE_KINDS``), so the React route, the deck build
and any direct Python caller could answer from unapproved data. It now sits
below every channel, inside the governed capability.

The decision is made on the dataset's **resolution base** — *how* the dataset
was located — rather than on its display ``kind``. That matters: pointing
``MI_AGENT_DATA_CSV`` at the bundled synthetic demo CSV yields the display kind
``explicit_csv``, which the old adapter-level block did not catch. The base is
unambiguous about provenance.

Approved for production:

  * ``platform_canonical`` — the assembled central canonical published by the
    pipeline (``…/platform/{client}/latest/platform_canonical_typed.csv``);
  * ``central_tape`` — a promoted onboarding run's central lender tape.

Everything else (an explicitly pointed CSV, a pre-prepared analytics dataset,
the bundled synthetic demo, or nothing at all) is a fixture: usable in
``development``/``test``, refused in ``production``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

from .errors import ErrorCode
from .runtime import MODE_PRODUCTION, runtime_mode

# --------------------------------------------------------------------------- #
# Resolution bases (mirrors mi_agent_api.data_source.resolve_data_source).
# A drift test in tests/test_governance_source_policy.py keeps these aligned.
# --------------------------------------------------------------------------- #
BASE_PLATFORM_CANONICAL = "platform_canonical"
BASE_CENTRAL_TAPE = "central_tape"
BASE_PREPARED_EXPLICIT = "prepared_explicit"
BASE_EXPLICIT_CSV = "explicit_csv"
BASE_SYNTHETIC_DEMO = "synthetic_demo"
BASE_UNAVAILABLE = "unavailable"

#: The only bases a production caller may be answered from.
PRODUCTION_APPROVED_SOURCE_BASES: FrozenSet[str] = frozenset({
    BASE_PLATFORM_CANONICAL,
    BASE_CENTRAL_TAPE,
})

#: Bases that are fixtures: permitted in development/test, never in production.
FIXTURE_SOURCE_BASES: FrozenSet[str] = frozenset({
    BASE_PREPARED_EXPLICIT,
    BASE_EXPLICIT_CSV,
    BASE_SYNTHETIC_DEMO,
})

#: Bases that are synthetic demonstration data. Refused in production under a
#: distinct message so the operator sees *why*.
SYNTHETIC_SOURCE_BASES: FrozenSet[str] = frozenset({BASE_SYNTHETIC_DEMO})


@dataclass(frozen=True)
class SourceApproval:
    """The approval decision for one resolved dataset."""

    approved: bool
    #: ``approved`` | ``fixture_permitted`` | the refusing :class:`ErrorCode`.
    state: str
    reason: str
    error_code: Optional[str] = None
    #: True when a non-production fixture was allowed by the runtime mode.
    fixture: bool = False

    def raise_if_blocked(self, *, request_id: Optional[str] = None) -> None:
        if self.approved:
            return
        from .errors import TraktError
        raise TraktError(self.error_code or ErrorCode.DATA_SOURCE_NOT_APPROVED,
                         self.reason, request_id=request_id)


def evaluate_source_approval(
    source_base: Optional[str],
    *,
    mode: Optional[str] = None,
    dataset_available: bool = True,
) -> SourceApproval:
    """Decide whether ``source_base`` may answer a governed question.

    ``mode`` defaults to the process runtime mode; pass it explicitly only in
    tests that exercise the matrix. ``dataset_available`` lets a caller report a
    dataset that failed to load even though its base looked approved.
    """
    mode = mode or runtime_mode()
    base = (source_base or BASE_UNAVAILABLE).strip() or BASE_UNAVAILABLE

    if not dataset_available or base == BASE_UNAVAILABLE:
        return SourceApproval(
            approved=False,
            state=ErrorCode.DATA_SOURCE_UNAVAILABLE,
            reason=("No governed data source is available for this deployment; "
                    "Trakt cannot answer portfolio questions right now."),
            error_code=ErrorCode.DATA_SOURCE_UNAVAILABLE,
        )

    if base in PRODUCTION_APPROVED_SOURCE_BASES:
        return SourceApproval(
            approved=True, state="approved",
            reason=f"Governed production data source ({base}).")

    # Everything below is a fixture of some kind.
    if mode == MODE_PRODUCTION:
        if base in SYNTHETIC_SOURCE_BASES:
            reason = ("The active data source is the synthetic demonstration "
                      "dataset. Trakt does not answer production questions from "
                      "synthetic data.")
        else:
            reason = (f"The active data source ({base}) is not an approved "
                      "production dataset. Trakt answers only from the governed "
                      "platform canonical or a promoted central lender tape.")
        return SourceApproval(
            approved=False,
            state=ErrorCode.DATA_SOURCE_NOT_APPROVED,
            reason=reason,
            error_code=ErrorCode.DATA_SOURCE_NOT_APPROVED,
        )

    return SourceApproval(
        approved=True, state="fixture_permitted", fixture=True,
        reason=(f"Fixture data source ({base}) permitted by runtime mode "
                f"{mode!r}. Not valid for production answers."))
