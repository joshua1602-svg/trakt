"""operations_control.onboarding — the governed Client Onboarding capability.

Client Onboarding brings a client Trakt has never met into Trakt: it asks the
business questions, records what the client still owes, validates the answers,
and — only on approval and activation — generates the configuration every later
delivery resolves against.

It starts BLANK. No existing client configuration is required, read or implied.
The three entry points share one model:

    start_new_client()   the product: a blank governed case
    start_migration()    secondary: the same model, pre-populated from a legacy
                         client's files, for review and adoption
    start_amendment()    for an active client: the same model, starting from the
                         version currently in force

Activation generates the artefacts Trakt already reads — the client
configuration, the investor report overlay, portfolio metadata, the client index
and the source registrations — so no parallel configuration stack exists.
"""

from .artefacts import Artefact                             # noqa: F401
from .case import (                                         # noqa: F401
    ACTIVATED,
    APPROVED,
    AWAITING_CLIENT,
    CHANGES_REQUIRED,
    DRAFT,
    INFORMATION_REQUESTED,
    IN_REVIEW,
    KIND_AMENDMENT,
    KIND_MIGRATION,
    KIND_NEW_CLIENT,
    READY_FOR_APPROVAL,
    STATUSES,
    WITHDRAWN,
    InformationRequest,
    OnboardingCase,
)
# NOTE: the `catalogue()` accessor is deliberately NOT re-exported here — the
# name would shadow the `catalogue` SUBMODULE for anyone writing
# `from operations_control.onboarding import catalogue`. Import it from
# `.catalogue` directly.
from .catalogue import Catalogue, Field, Section          # noqa: F401
from .service import OnboardingService                       # noqa: F401
from .store import ConfigurationVersion, OnboardingStore     # noqa: F401
from .validation import Problem, Validator                   # noqa: F401
