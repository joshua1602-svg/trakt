"""operations_control.onboarding — the governed Client Onboarding capability.

Client Onboarding creates and maintains a client's STANDING configuration —
the facts that stay true between deliveries. It sits alongside Operations; it
does not replace any part of the delivery workflow.

The wizard asks business questions and generates the artefacts Trakt already
reads:

  * ``config/client/config_client_{CLIENT}.yaml``   (client configuration)
  * ``config/client/config_client_annex12.yaml``    (investor reporting overlay)
  * ``config/client/portfolio_registry.yaml``       (portfolio metadata)
  * the durable source registry                     (source registrations)

No parallel configuration is introduced: the generated artefacts use the
existing formats, sit at the existing layers, and are consumed by the existing
loaders. What onboarding adds is authorship, versioning and an audit trail.
"""

from .model import (                                       # noqa: F401
    ClientOnboardingProfile,
    ClientStanding,
    PortfolioStanding,
    ReportingStanding,
    SourceRegistrationStanding,
    StaticReportingStanding,
    governed_vocabularies,
    standing_field_spec,
)
from .store import OnboardingStore, ProfileVersion          # noqa: F401
from .service import OnboardingService                      # noqa: F401
