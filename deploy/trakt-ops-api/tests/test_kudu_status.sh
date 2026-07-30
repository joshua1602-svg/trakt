#!/usr/bin/env bash
# Tests for Kudu deployment-status classification (kudu_status.sh).
#
# These exist because an off-by-one enum (4=Failed instead of 4=Success) made the
# deploy script report a SUCCESSFUL deployment as a hard failure. The fixtures in
# section 1 are the literal status/complete pairs observed in the real run of
# 2026-07-30T07:54Z, so the mapping is pinned to evidence rather than to memory.
#
#   bash deploy/trakt-ops-api/tests/test_kudu_status.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../kudu_status.sh
. "$HERE/../kudu_status.sh"

PASS=0
FAIL=0
check() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    PASS=$((PASS+1)); printf '  PASS  %s\n' "$label"
  else
    FAIL=$((FAIL+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' \
      "$label" "$expected" "$actual"
  fi
}

echo "=== 1. The real deployment observed on 2026-07-30 (run ef5d9230) ==="
# 07:54:28  status=0 complete=False  "Receiving changes."
check "status=0 complete=False -> pending (not terminal)" \
  "pending" "$(classify_deploy_status 0 False)"
# 07:54:45  status=1 complete=False  "Building and Deploying 'ef5d9230-…'."
check "status=1 complete=False -> building (not terminal)" \
  "building" "$(classify_deploy_status 1 False)"
# 08:02:09  status=4 complete=True   "Deployment successful. … Remote build."
check "status=4 complete=True  -> success  (THE REGRESSION: was read as failed)" \
  "success" "$(classify_deploy_status 4 True)"

echo "=== 2. Terminal states ==="
check "status=3 complete=True -> failed" "failed" "$(classify_deploy_status 3 True)"
check "status=4 complete=true (lowercase) -> success" "success" "$(classify_deploy_status 4 true)"
check "status=3 complete=true (lowercase) -> failed"  "failed"  "$(classify_deploy_status 3 true)"
check "status=4 complete=TRUE -> success" "success" "$(classify_deploy_status 4 TRUE)"

echo "=== 3. Terminal with an unrecognised status fails closed ==="
for bad in 5 9 99 "" "abc"; do
  check "status='$bad' complete=True -> unexpected" \
    "unexpected" "$(classify_deploy_status "$bad" True)"
done

echo "=== 4. Non-terminal states keep polling ==="
check "status=2 complete=False -> deploying" "deploying" "$(classify_deploy_status 2 False)"
check "status='' complete=False -> in-progress" "in-progress" "$(classify_deploy_status "" False)"
check "unparseable poll (both empty) -> in-progress" \
  "in-progress" "$(classify_deploy_status "" "")"

echo "=== 5. A failure seen before 'complete' flips is still a failure ==="
# Kudu can report status=3 while complete is not yet true; that must not be
# mistaken for "still building".
check "status=3 complete=False -> failed" "failed" "$(classify_deploy_status 3 False)"

echo "=== 6. Labels used in the poll log ==="
check "label 0" "pending"   "$(deploy_status_label 0)"
check "label 1" "building"  "$(deploy_status_label 1)"
check "label 2" "deploying" "$(deploy_status_label 2)"
check "label 3" "failed"    "$(deploy_status_label 3)"
check "label 4" "success"   "$(deploy_status_label 4)"
check "label empty" "unknown" "$(deploy_status_label "")"
check "label unknown code" "unrecognised(7)" "$(deploy_status_label 7)"

echo "=== 7. No call site re-implements the enum ==="
# The mapping must exist in exactly one place. A literal status comparison in the
# deploy or diagnostics script is how the off-by-one survived review last time.
HITS="$(grep -nE '\[ *"?\$\{?STATUS_CODE\}?"? *(=|==) *"?[0-9]' \
  "$HERE/../deploy_async.sh" "$HERE/../collect_diagnostics.sh" 2>/dev/null || true)"
if [ -z "$HITS" ]; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "no raw status-code comparison outside kudu_status.sh"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n     found: %s\n' \
    "no raw status-code comparison outside kudu_status.sh" "$HITS"
fi

echo
echo "-------------------------------------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "All Kudu status classification tests passed."
