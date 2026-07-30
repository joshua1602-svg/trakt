#!/usr/bin/env bash
# Hostname discovery for trakt-ops-api. Source this; do not execute it.
#
# WHY THIS EXISTS
# App Service hostnames must never be constructed from the app name. With the
# "secure unique default hostname" feature enabled, a site's hostnames are:
#
#     <app>-<hash>.<region>.azurewebsites.net           (public)
#     <app>-<hash>.scm.<region>.azurewebsites.net       (SCM / Kudu)
#
# not the legacy:
#
#     <app>.azurewebsites.net
#     <app>.scm.azurewebsites.net
#
# The legacy guess does not resolve in DNS, which surfaces as
# `curl: (6) Could not resolve host: ...` and HTTP 000 — a submission failure
# that looks nothing like the deployment problem it gets mistaken for. Both
# forms are returned by the site resource itself, so ask Azure instead of
# guessing. The functions below work unchanged for legacy and secure-unique
# naming because they never assemble a hostname.
#
# Usage:
#   . deploy/trakt-ops-api/azure_hosts.sh
#   SCM_HOST="$(discover_scm_host "$RG" "$APP")" || exit 1
#   APP_HOST="$(discover_app_host "$RG" "$APP")" || exit 1
#
# All diagnostics go to stderr so the discovered hostname is the only thing on
# stdout and command substitution stays clean. No secret or token is ever
# printed by these functions.

# Reject anything that is not a plain App Service hostname. The value is
# interpolated into a URL that receives a bearer token, so a scheme, path,
# credential, whitespace or query fragment arriving from a malformed query
# result must not be accepted.
validate_scm_host() {
  local host="$1"
  case "$host" in
    "")                       echo "empty hostname" >&2; return 1 ;;
    *[[:space:]]*)            echo "hostname contains whitespace: '$host'" >&2; return 1 ;;
    */*)                      echo "hostname contains a path or scheme: '$host'" >&2; return 1 ;;
    *@*)                      echo "hostname contains credentials: '$host'" >&2; return 1 ;;
    *\?*)                     echo "hostname contains a query string: '$host'" >&2; return 1 ;;
  esac
  case "$host" in
    *.scm.*) ;;
    *) echo "hostname is not an SCM endpoint (no '.scm.'): '$host'" >&2; return 1 ;;
  esac
  case "$host" in
    *.azurewebsites.net) ;;
    *) echo "hostname does not end in .azurewebsites.net: '$host'" >&2; return 1 ;;
  esac
  return 0
}

validate_app_host() {
  local host="$1"
  case "$host" in
    "")                       echo "empty hostname" >&2; return 1 ;;
    *[[:space:]]*)            echo "hostname contains whitespace: '$host'" >&2; return 1 ;;
    */*)                      echo "hostname contains a path or scheme: '$host'" >&2; return 1 ;;
    *@*)                      echo "hostname contains credentials: '$host'" >&2; return 1 ;;
  esac
  case "$host" in
    *.azurewebsites.net) ;;
    *) echo "hostname does not end in .azurewebsites.net: '$host'" >&2; return 1 ;;
  esac
  # The public hostname must NOT be the SCM one — those are different endpoints
  # and mixing them up sends application traffic to Kudu.
  case "$host" in
    *.scm.*) echo "expected the public hostname, got the SCM one: '$host'" >&2; return 1 ;;
  esac
  return 0
}

#: The SCM (Kudu) hostname of the site, from the site resource.
discover_scm_host() {
  local rg="$1" app="$2" host
  host="$(az webapp show \
            --resource-group "$rg" \
            --name "$app" \
            --query "enabledHostNames[?contains(@, '.scm.')]|[0]" \
            --output tsv 2>/dev/null | tr -d '\r')"

  if [ -z "$host" ] || [ "$host" = "None" ]; then
    {
      echo "FAIL: could not discover the SCM hostname for '$app' in '$rg'."
      echo "      enabledHostNames contained no '.scm.' entry. Check that:"
      echo "        * the app name and resource group are correct;"
      echo "        * the deploying identity can read the site resource;"
      echo "        * the site exists in this subscription."
      echo "      Inspect manually with:"
      echo "        az webapp show -g $rg -n $app --query enabledHostNames -o json"
      echo "      The hostname is NEVER constructed from the app name: a site"
      echo "      using secure unique default hostnames is reachable only at"
      echo "      <app>-<hash>.scm.<region>.azurewebsites.net."
    } >&2
    return 1
  fi

  validate_scm_host "$host" || return 1
  printf '%s\n' "$host"
}

#: The public (default) hostname of the site, from the site resource.
discover_app_host() {
  local rg="$1" app="$2" host
  host="$(az webapp show \
            --resource-group "$rg" \
            --name "$app" \
            --query "defaultHostName" \
            --output tsv 2>/dev/null | tr -d '\r')"

  if [ -z "$host" ] || [ "$host" = "None" ]; then
    {
      echo "FAIL: could not discover the public hostname for '$app' in '$rg'."
      echo "      Inspect manually with:"
      echo "        az webapp show -g $rg -n $app --query defaultHostName -o tsv"
    } >&2
    return 1
  fi

  validate_app_host "$host" || return 1
  printf '%s\n' "$host"
}
