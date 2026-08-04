"""simulation — deterministic, seeded asset-class hardening framework.

Generates internally consistent funded portfolio histories for equity release,
bridge lending and asset finance, renders the SAME economic truth into several
lender source dialects, and drives them through the real Trakt production
pathway (ingestion → canonical → validation → historical integration → MI →
risk → governed MI Agent → regime projection → regulatory artefact).

The framework never re-implements a production calculation. Its only
independent arithmetic is :mod:`simulation.reference_truth`, which is
deliberately small and auditable so an MI assertion is not the production MI
checking itself.

Two components are kept strictly apart:

* :mod:`simulation.generator` / :mod:`simulation.history` produce ONE economic
  truth per case (loan states and month-by-month movements);
* :mod:`simulation.dialects` render that truth into materially different source
  formats.

That separation is what makes canonical equivalence across dialects a
meaningful test rather than a tautology.
"""

from __future__ import annotations

#: Bumped whenever the generator's economics change in a way that would alter a
#: previously generated case for the same seed. Recorded in every manifest and
#: every run summary so a stored fixture can be matched to the code that made it.
SIMULATION_VERSION = 1

__all__ = ["SIMULATION_VERSION"]
