"""operations_control.access_admin — administering who may reach what.

The OCC-side lifecycle for the Stage 1-3 access model: propose a structured
change, validate it against the real runtime registries, put it in front of a
human, and activate it as an immutable, versioned configuration.

Strictly an administration layer. Runtime authorisation is unchanged and does
not import anything here: a request is still decided by ``ExecutionContext`` →
``EntitlementStore`` → ``ResourceCatalogue`` → ``authorise_resource_access``,
reading published configuration. The OCC Agent proposes; it never approves, and
it is never asked whether an access is allowed.
"""

from __future__ import annotations

from .changeset import (
    ACTIONS,
    ADD_GRANT,
    ADD_MICROSOFT_TENANT,
    ADD_ORGANISATION,
    ADD_PRINCIPAL,
    ADD_RESOURCE,
    DISABLE_GRANT,
    DISABLE_ORGANISATION,
    DISABLE_PRINCIPAL,
    ENABLE_GRANT,
    ENABLE_ORGANISATION,
    ENABLE_PRINCIPAL,
    PROPOSAL_CONFIRMED,
    PROPOSAL_DRAFT,
    PROPOSAL_REJECTED,
    REVOKE_GRANT,
    SET_GRANT_CAPABILITIES,
    SOURCE_OCC_AGENT,
    SOURCE_OPERATOR,
    UPDATE_RESOURCE,
    AccessChange,
    AccessChangeSet,
    ChangeSetError,
    apply_change_set,
)
from .document import ACCESS_FILES, AccessDocument
from .service import (
    AccessAdminService,
    ConfirmationError,
    Proposal,
)
from .validation import validate_access_files, validate_document

__all__ = [
    "AccessAdminService", "Proposal", "ConfirmationError",
    "AccessChange", "AccessChangeSet", "ChangeSetError", "apply_change_set",
    "AccessDocument", "ACCESS_FILES",
    "validate_document", "validate_access_files",
    "ACTIONS",
    "ADD_ORGANISATION", "DISABLE_ORGANISATION", "ENABLE_ORGANISATION",
    "ADD_MICROSOFT_TENANT",
    "ADD_PRINCIPAL", "DISABLE_PRINCIPAL", "ENABLE_PRINCIPAL",
    "ADD_RESOURCE", "UPDATE_RESOURCE",
    "ADD_GRANT", "REVOKE_GRANT", "SET_GRANT_CAPABILITIES",
    "DISABLE_GRANT", "ENABLE_GRANT",
    "SOURCE_OPERATOR", "SOURCE_OCC_AGENT",
    "PROPOSAL_DRAFT", "PROPOSAL_CONFIRMED", "PROPOSAL_REJECTED",
]
