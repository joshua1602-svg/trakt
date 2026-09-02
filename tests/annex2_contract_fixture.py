"""Test helper — the effective Annex 2 contract, as a document and as a path.

The Annex 2 contract used to be a file (``config/regime/annex2_delivery_rules.yaml``)
that tests could read and pass around. It is now derived from the field
universe, the fields registry, the mapping workbook and the XSD, so tests that
need the document ask for it and tests that need a path get one materialised.

Nothing here decides anything: it is the same contract the pipeline builds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def contract_rules() -> Dict[str, Any]:
    """The contract as a delivery-rules document (``{field_rules: {...}}``)."""
    from engine.regime_contract.annex2_contract import as_delivery_rules
    return as_delivery_rules()


def contract_field_rules() -> Dict[str, Any]:
    return contract_rules().get("field_rules", {}) or {}


def contract_path() -> str:
    """A file path to the contract, for callers that can only take a path."""
    from engine.regime_contract.annex2_contract import materialised_contract_path
    return materialised_contract_path()
