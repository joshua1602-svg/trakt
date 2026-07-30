#!/usr/bin/env bash
# Kudu deployment-status classification. Source this; do not execute it.
#
# THE ENUM, AND WHY IT IS WRITTEN DOWN HERE
# Kudu's DeployStatus (Kudu.Contracts) is:
#
#     Pending   = 0
#     Building  = 1
#     Deploying = 2
#     Failed    = 3
#     Success   = 4
#
# Confirmed against a real deployment of this app (run 2026-07-30T07:54Z):
#
#     status=0 complete=False   "Receiving changes." / "Fetching changes."
#     status=1 complete=False   "Building and Deploying 'ef5d9230-…'."
#     status=4 complete=True    "Deployment successful. deployer = Push-Deployer
#                                deploymentPath = ZipDeploy. Extract zip.
#                                Remote build."   Errors (0)  Warnings (0)
#
# An earlier version of the deploy script used an off-by-one mapping
# (4=Failed, 5=Success) and therefore reported that successful deployment as a
# hard failure. The classification lives in this one function so it is asserted
# by tests instead of being restated at each call site.
#
# classify_deploy_status <status> <complete>
#   echoes exactly one of:
#     success      terminal, deployment succeeded
#     failed       terminal, deployment failed
#     unexpected   terminal with a status this enum does not cover -> fail closed
#     pending | building | deploying | in-progress   not terminal, keep polling
classify_deploy_status() {
  local status="${1:-}" complete="${2:-}"

  case "$complete" in
    True|true|TRUE)
      case "$status" in
        4) echo "success" ;;
        3) echo "failed" ;;
        # Anything else arriving as terminal is not something this script can
        # interpret. Fail closed rather than guessing it is a success.
        *) echo "unexpected" ;;
      esac
      ;;
    *)
      case "$status" in
        0) echo "pending" ;;
        1) echo "building" ;;
        2) echo "deploying" ;;
        3) echo "failed" ;;
        4) echo "success" ;;
        *) echo "in-progress" ;;
      esac
      ;;
  esac
}

#: Human label for a raw status code, for log lines.
deploy_status_label() {
  case "${1:-}" in
    0) echo "pending" ;;
    1) echo "building" ;;
    2) echo "deploying" ;;
    3) echo "failed" ;;
    4) echo "success" ;;
    "") echo "unknown" ;;
    *) echo "unrecognised($1)" ;;
  esac
}
