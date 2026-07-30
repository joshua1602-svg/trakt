#!/usr/bin/env bash
# Shell-level tests for App Service hostname discovery (azure_hosts.sh).
#
# Runs with NO Azure access: a stub `az` on PATH returns fixture hostnames, so
# both the legacy and the secure-unique naming schemes are exercised, along with
# the rejection cases that protect the URL a bearer token is sent to.
#
#   bash deploy/trakt-ops-api/tests/test_host_discovery.sh
#
# Exits non-zero on the first failing assertion.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HERE/../azure_hosts.sh"
STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT

PASS=0
FAIL=0

ok()   { PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; }
no()   { FAIL=$((FAIL+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }

check_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then ok "$label"; else no "$label" "$expected" "$actual"; fi
}

# ------------------------------------------------------------------------- #
# A stub `az` that answers the two queries azure_hosts.sh makes, driven by
# fixture files. It mimics `--output tsv` behaviour, including the literal
# "None" the CLI prints for a null JMESPath result.
# ------------------------------------------------------------------------- #
cat > "$STUB_DIR/az" <<'STUB'
#!/usr/bin/env bash
# Fixture-driven stub. FIXTURE_SCM / FIXTURE_DEFAULT drive the answers.
args="$*"
case "$args" in
  *"contains(@, '.scm.')"*) printf '%s\n' "${FIXTURE_SCM-None}" ;;
  *defaultHostName*)        printf '%s\n' "${FIXTURE_DEFAULT-None}" ;;
  *)                        printf 'None\n' ;;
esac
STUB
chmod +x "$STUB_DIR/az"
PATH="$STUB_DIR:$PATH"

# shellcheck source=../azure_hosts.sh
. "$LIB"

echo "=== 1. SCM discovery: legacy hostname ==="
check_eq "legacy SCM host is returned unchanged" \
  "trakt-ops-api.scm.azurewebsites.net" \
  "$(FIXTURE_SCM='trakt-ops-api.scm.azurewebsites.net' discover_scm_host rg app 2>/dev/null)"

echo "=== 2. SCM discovery: secure unique default hostname ==="
check_eq "secure-unique SCM host is returned unchanged" \
  "trakt-ops-api-a1b2c3d4.scm.westeurope-01.azurewebsites.net" \
  "$(FIXTURE_SCM='trakt-ops-api-a1b2c3d4.scm.westeurope-01.azurewebsites.net' discover_scm_host rg app 2>/dev/null)"

echo "=== 3. SCM discovery: other regions / hash shapes ==="
for host in \
  "trakt-ops-api-hash.scm.uksouth-01.azurewebsites.net" \
  "trakt-ops-api-e5f6a7b8c9.scm.eastus2-02.azurewebsites.net" \
  "app.scm.azurewebsites.net"
do
  check_eq "accepted: $host" "$host" \
    "$(FIXTURE_SCM="$host" discover_scm_host rg app 2>/dev/null)"
done

echo "=== 4. SCM discovery failure cases (must fail, must not emit a host) ==="
for bad_label_and_value in \
  "empty result:" \
  "CLI null:None" \
  "public hostname without .scm.:trakt-ops-api-hash.westeurope-01.azurewebsites.net" \
  "wrong suffix:trakt-ops-api.scm.example.com" \
  "scheme injected:https://trakt-ops-api.scm.azurewebsites.net" \
  "path injected:trakt-ops-api.scm.azurewebsites.net/api/zipdeploy" \
  "credentials injected:user@trakt-ops-api.scm.azurewebsites.net" \
  "query injected:trakt-ops-api.scm.azurewebsites.net?x=1" \
  "whitespace:bad host.scm.azurewebsites.net"
do
  label="${bad_label_and_value%%:*}"
  value="${bad_label_and_value#*:}"
  out="$(FIXTURE_SCM="$value" discover_scm_host rg app 2>/dev/null)"
  rc=$?
  if [ "$rc" -ne 0 ] && [ -z "$out" ]; then
    ok "rejected ($label)"
  else
    no "rejected ($label)" "non-zero exit and no output" "rc=$rc out='$out'"
  fi
done

echo "=== 5. Public hostname discovery ==="
check_eq "legacy public host" \
  "trakt-ops-api.azurewebsites.net" \
  "$(FIXTURE_DEFAULT='trakt-ops-api.azurewebsites.net' discover_app_host rg app 2>/dev/null)"
check_eq "secure-unique public host" \
  "trakt-ops-api-a1b2c3d4.westeurope-01.azurewebsites.net" \
  "$(FIXTURE_DEFAULT='trakt-ops-api-a1b2c3d4.westeurope-01.azurewebsites.net' discover_app_host rg app 2>/dev/null)"

echo "=== 6. Public hostname failure cases ==="
for bad_label_and_value in \
  "empty result:" \
  "CLI null:None" \
  "SCM host mistaken for public:trakt-ops-api-hash.scm.westeurope-01.azurewebsites.net" \
  "wrong suffix:trakt-ops-api.example.com" \
  "scheme injected:https://trakt-ops-api.azurewebsites.net"
do
  label="${bad_label_and_value%%:*}"
  value="${bad_label_and_value#*:}"
  out="$(FIXTURE_DEFAULT="$value" discover_app_host rg app 2>/dev/null)"
  rc=$?
  if [ "$rc" -ne 0 ] && [ -z "$out" ]; then
    ok "rejected ($label)"
  else
    no "rejected ($label)" "non-zero exit and no output" "rc=$rc out='$out'"
  fi
done

echo "=== 7. Carriage returns are stripped (CLI on some shells emits CRLF) ==="
check_eq "CR stripped from SCM host" \
  "trakt-ops-api-hash.scm.westeurope-01.azurewebsites.net" \
  "$(FIXTURE_SCM="$(printf 'trakt-ops-api-hash.scm.westeurope-01.azurewebsites.net\r')" discover_scm_host rg app 2>/dev/null)"

echo "=== 8. No hostname is ever assembled from the app name ==="
# The regression that caused the DNS failure: constructing the host from APP.
# Assert the sources contain no such construction.
ASSEMBLY_HITS="$(grep -rnE '\$\{?(APP|APP_NAME|app)\}?\.(scm\.)?azurewebsites\.net' \
  "$HERE/../azure_hosts.sh" \
  "$HERE/../deploy_async.sh" \
  "$HERE/../collect_diagnostics.sh" \
  "$HERE/../preflight_checks.sh" \
  "$HERE/../provision.sh" 2>/dev/null || true)"
if [ -z "$ASSEMBLY_HITS" ]; then
  ok "no script builds a hostname from the app name"
else
  no "no script builds a hostname from the app name" "no matches" "$ASSEMBLY_HITS"
fi

echo
echo "-------------------------------------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "All hostname discovery tests passed."
