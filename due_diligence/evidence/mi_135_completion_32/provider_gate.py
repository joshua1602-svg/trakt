"""Stop the tranche the moment the provider runs out again.

WHY THIS EXISTS RATHER THAN A COLLECTOR CHANGE. The frozen collector aborts on
consecutive 401/403 and on a misconfigured canary. Neither sees a provider quota
exhaustion: that arrives as HTTP 200 carrying a legacy fallback answer, which is
exactly how thirty questions were spent against a dead upstream on 2026-09-13.
The collector is pinned for this task, so the tranche is asked in batches and
this runs between them. It reads evidence and exits non-zero; it changes nothing.

IT ALSO REFUSES TO CALL A PROVIDER FAILURE A PRODUCT RESULT. A record whose model
failure is MODEL_UNAVAILABLE is INFRASTRUCTURE, whatever the envelope served.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROVIDER_CODES = ("MODEL_UNAVAILABLE",)
QUOTA_MARKERS = ("credit balance is too low", "quota", "rate_limit_error",
                 "insufficient_quota", "billing")


def provider_failures(records):
    out = []
    for record in records:
        rec = record.get("record") or {}
        failure = (rec.get("model") or {}).get("failure") or {}
        code = failure.get("code") or ""
        detail = str(failure.get("detail") or "")
        if code in PROVIDER_CODES:
            quota = any(marker in detail.lower() for marker in QUOTA_MARKERS)
            out.append({"question_id": record.get("question_id"), "code": code,
                        "quota": quota, "detail": detail[:400]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--require-canary", action="store_true",
                        help="also assert the canary principal matched. The "
                             "frozen collector skips its own canary guard when "
                             "`--only` is set (`index == abort_after and not "
                             "wanted`), so a targeted tranche would not notice a "
                             "wrong MI_AGENT_PLAN_SERVE. Checked here instead, "
                             "after the first batch, from evidence.")
    args = parser.parse_args()

    records, seen = [], []
    for path in args.batches:
        p = Path(path)
        if not p.exists():
            print(f"  {path}: not written")
            continue
        rows = json.loads(p.read_text(encoding="utf-8")).get("records") or []
        records.extend(rows)
        seen.append(f"{p.name}:{len(rows)}")
    print(f"batches read: {', '.join(seen) or 'none'}  ({len(records)} records)")

    if args.require_canary:
        matched = sum(1 for r in records
                      if ((r.get("record") or {}).get("serving") or {})
                      .get("principal_matched") is True)
        print(f"canary principal matched on {matched} of {len(records)} records")
        if records and matched == 0:
            print("::error::no record matched the canary principal — "
                  "MI_AGENT_PLAN_SERVE / MI_AGENT_PLAN_SERVE_PRINCIPALS are not "
                  "in the state this tranche requires. Stopped after one batch.")
            return 1

    failures = provider_failures(records)
    if not failures:
        print("PROVIDER_FAILURES = 0 — the tranche may continue")
        return 0

    print(f"PROVIDER_FAILURES = {len(failures)}")
    for f in failures[:5]:
        print(f"  {f['question_id']}  {f['code']}  quota={f['quota']}")
        print(f"    {f['detail'][:200]}")
    print("::error::the provider failed again; the tranche stops here rather "
          "than spending the remaining cases. These are INFRASTRUCTURE, not "
          "product refusals.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
