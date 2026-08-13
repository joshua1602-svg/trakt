#!/usr/bin/env bash
# Development environment for the Trakt agent tool API (Sprint 1).
#
#   source scripts/agent_dev_env.sh
#   uvicorn mi_agent_api.app:app --port 8000
#   python scripts/agent_reference_client.py --portfolio ERE/source_portfolio/direct_001
#
# Sets ONLY the governance wiring and the local-development identity. It does not
# point at any dataset: the data source is whatever the deployment already
# resolves, and the source-approval policy still decides whether it may answer.
#
# TRAKT_AGENT_AUTH_MODE=disabled is refused outright when the runtime mode is
# production, and it still requires a directory — an unattributed machine
# resolves no organisation, therefore no entitlements, therefore nothing.

# --- the five governance registries (see config/dev/README.md) --------------
export TRAKT_TENANCY_CONFIG=config/dev/tenancy.yaml
export TRAKT_ORGANISATIONS_CONFIG=config/dev/organisations.yaml
export TRAKT_RESOURCES_CONFIG=config/dev/resources.yaml
export TRAKT_ENTITLEMENTS_CONFIG=config/dev/entitlements.yaml
export TRAKT_PRINCIPALS_CONFIG=config/dev/principals.yaml

# --- whose data this deployment serves --------------------------------------
export MI_AGENT_CLIENT_ID=ERE

# --- the agent surface is off unless switched on ----------------------------
export TRAKT_AGENT_API_ENABLED=true

# --- local development identity: acts as organisation `a2a_test_agent` -------
export TRAKT_AGENT_AUTH_MODE=disabled
export TRAKT_AGENT_DEV_DIRECTORY=00000000-0000-4000-8000-00000000a2a7

echo "Trakt agent API development environment set."
echo "  tenant        : ${MI_AGENT_CLIENT_ID}"
echo "  agent acts as : a2a_test_agent (directory ${TRAKT_AGENT_DEV_DIRECTORY})"
echo "  granted       : risk:read on ERE/source_portfolio/direct_001 — nothing else"
echo
echo "For a real deployment: TRAKT_AGENT_AUTH_MODE=entra,"
echo "TRAKT_AGENT_ENTRA_AUDIENCE=api://<app-id>, and assign the Trakt.Agent app role."
