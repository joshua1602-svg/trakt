"""mi_agent.concentration_tests — governed concentration-test capability.

One capability, four concerns, one truth:

* ``library``    — the shared, versioned metric REGISTRY
                   (``config/risk/concentration_test_library.yaml``): what a
                   concentration test *is*, its parameters, aliases, required
                   canonical fields and implementation status. Client thresholds
                   never live here.
* ``matching``   — deterministic extraction of client-supplied tests and their
                   mapping onto the library (matched / composed / ambiguous /
                   unsupported), with concern codes and targeted follow-up
                   questions. A model may *propose* through the assist seam; it
                   never decides.
* ``store``      — tenant-scoped persistence: proposals, immutable activated
                   configuration versions and a rebuildable ``current`` pointer,
                   over the same ``Storage`` abstraction every other governed
                   store uses.
* ``evaluation`` — the single deterministic evaluation service: active approved
                   configuration × funded canonical frame (current and prior)
                   → per-test value, threshold, status, headroom, movement and a
                   reconciling drill-through. The dashboard, MI Query and
                   Copilot all consume THIS; nothing recalculates.

The language model may classify, extract and propose mappings. It is never the
authority that calculates compliance — every activated test resolves to a
registered deterministic metric implementation with explicit parameters, and
nothing activates without an explicit human approval.
"""

from .models import (  # noqa: F401
    SCHEMA_VERSION,
    MATCH_MATCHED,
    MATCH_COMPOSED,
    MATCH_AMBIGUOUS,
    MATCH_UNSUPPORTED,
    PROPOSAL_STATUSES,
    TEST_STATUSES,
    ConcernCode,
    TestProposal,
    ApprovalRecord,
    ActiveTest,
    ActiveConfiguration,
)
from .library import ConcentrationLibrary, MetricDefinition, load_library  # noqa: F401
