"""demo_platform — the isolated synthetic product-demonstration environment.

Everything in this package exists to produce ONE thing: a reproducible,
entirely synthetic demonstration of the real Trakt operating model for a
fictional UK specialist lender ("Alderbridge Lending Platform"), from raw
source extracts through onboarding contracts, recurring orchestration,
platform assembly, MI, investor artefacts and the Remotion film.

It is additive and isolated:

  * it NEVER contains, derives from, or approximates live-client data;
  * it does not modify production business logic — it *drives* the production
    pipeline (``engine.orchestrator.trakt_run``, ``engine.platform_assembler``,
    ``mi_agent_api``) exactly as a managed-service run would;
  * every number shown in the film is exported from the same deterministic
    execution the product itself uses (see ``demo_platform.metrics``).

Entry point::

    python -m demo_platform.run_demo --all
"""

from __future__ import annotations

__all__ = ["config"]
