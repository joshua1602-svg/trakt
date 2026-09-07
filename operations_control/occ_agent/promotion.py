"""operations_control.occ_agent.promotion — the one crossing into live state.

The OCC Agent authors a case in its own isolated container. ``store.
assert_isolated`` refuses to let either of its stores point at the governed
operations container, and that refusal is what makes the Agent safe to leave
switched on in production: nothing it does while a conversation is in progress
can reach live operational state.

Activation is the one moment that must reach it. This module is that crossing,
and it is deliberately the only one.

What crosses, and what does not
-------------------------------
Only the **approved case** — the answers a human approved. Not the generated
configuration.

That distinction is the whole safety argument. ``OnboardingService.activate``
derives every artefact from the case's answers and writes them through its own
store, so handing the governed side the answers lets it build its own
configuration in its own container, exactly as a manual onboarding would. The
alternative — copying finished artefacts across — would mean the live
configuration was a *copy* of something built elsewhere, and a copy can be
stale, partial, or silently divergent. Here there is nothing to diverge from:
production generates what production uses.

Crossing exactly once
---------------------
A second crossing would produce a second configuration version for the same
approved answers, and the reporting period would then depend on which one a
reader happened to resolve. :func:`promote` therefore refuses a case the
governed side has already activated, and :func:`record_activation` writes the
activation back to the practice case so the ordinary ``already_activated``
precondition sees it too. Two independent guards, because this is the one
irreversible step in the workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..engine import OpsError
from ..onboarding.case import APPROVED, OnboardingCase

logger = logging.getLogger("operations_control.occ_agent.promotion")


def promote(*, source: Any, target: Any, case_ref: str,
            actor: str) -> OnboardingCase:
    """Copy one approved case from the practice store into the governed one.

    ``source`` and ``target`` are both ``OnboardingService``s differing only in
    the store they were constructed with. Returns the case as the governed side
    now holds it.

    Refuses unless the case is approved, and refuses a case the governed side
    has already activated.
    """
    case = source.load_case(case_ref)
    if case.status != APPROVED:
        raise OpsError(
            "OPS_ONBOARDING_NOT_APPROVED",
            "This onboarding has not been approved, so there is nothing to "
            "activate.", 409)

    existing: Optional[OnboardingCase] = target.cases.load_case(case_ref)
    if existing is not None and existing.activated_version:
        raise OpsError(
            "OPS_ALREADY_ACTIVATED",
            "This configuration has already been activated as version "
            f"{existing.activated_version}. Activating again would create a "
            "second configuration for the same approved answers.", 409)

    target.cases.save_case(case)
    logger.info("occ agent case %s promoted to the governed store by %s",
                case_ref, actor)
    return case


def record_activation(*, source: Any, activated: OnboardingCase) -> None:
    """Write the governed activation back to the practice case.

    Without this the practice case still reads as merely approved: the
    activation stamp was written on the governed side. The ordinary
    ``already_activated`` precondition reads the practice case, so this is what
    makes a second confirmation refuse through the normal gate rather than
    relying on :func:`promote` alone.

    Never raises. The activation has already happened and is recorded where it
    counts; failing here must not turn a successful activation into an error.
    """
    try:
        source.cases.save_case(activated)
    except Exception:  # noqa: BLE001 — see above
        logger.warning("case %s activated on the governed side but the "
                       "practice copy could not be updated",
                       getattr(activated, "case_id", "?"), exc_info=True)
