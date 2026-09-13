#!/usr/bin/env python3
"""Diagnose N01's HTTP 500. READ ONLY.

WHAT IT DOES AND DOES NOT DO. It asks no question, calls no model, touches no MI
endpoint, changes no configuration and writes nothing to the service. It reads
App Service log files over the Kudu VFS — the same read-only door
`run_acceptance.Sink` already opens for the evidence sink — and prints the lines
around the failing request.

WHY THE LOGS AND NOT THE SINK. The sink was already polled for 120 seconds during
the run and no record for N01 was ever written, which is itself the finding: the
request failed before `plan_serving_canary` recorded anything. Whatever happened
is therefore only in the application log.

THE WINDOW IS THE REQUEST'S OWN. N01 was asked at 14:23:26Z and the envelope came
back after 33.5 seconds, so the fault is inside 14:23:26-14:24:05Z. A wider
window would drag in the other four questions and invite reading their traces as
this one's.

REDACTION. Anything that could be loan-level is dropped, and the output is
rescanned for both credentials before it is written. Log lines are truncated:
a traceback's shape and exception type are the diagnosis, and a full frame dump
is where borrower data would hide if it hid anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from due_diligence.evidence.deployed_acceptance_0399a315 import (  # noqa: E402
    run_acceptance as ra)

#: Tokens that would mark a line as carrying loan-level detail.
_ROW_BEARING = re.compile(
    r"loan_identifier|borrower|customer_name|account_number|postcode|"
    r"\bIBAN\b|sort_code", re.I)

#: A line longer than this is truncated: the exception and its frame are the
#: diagnosis, and a dumped payload is not.
_MAX_LINE = 400


def _client(profile: str, path: str):
    scm, user, password = ra.publish_profile_credentials(profile)
    return ra.Sink(scm, user, password, path), password


def listing(profile: str, directory: str):
    """One Kudu VFS directory listing. `(entries, detail)`."""
    sink, _ = _client(profile, directory.rstrip("/") + "/")
    body, detail = sink.read()
    if body is None:
        return None, detail
    try:
        return json.loads(body), detail
    except json.JSONDecodeError:
        return None, f"not a directory listing ({detail})"


def read_lines(profile: str, path: str, *, cap_bytes: int):
    sink, _ = _client(profile, path)
    body, detail = sink.read()
    if body is None:
        return None, detail
    if len(body) > cap_bytes:
        body = body[-cap_bytes:]
        detail += f" (tail {cap_bytes} bytes of a larger file)"
    return body.splitlines(), detail


def in_window(line: str, start: str, end: str) -> bool:
    stamp = re.match(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line.strip())
    if not stamp:
        return False
    return start <= stamp.group(1).replace(" ", "T") <= end


def redact(line: str) -> str:
    if _ROW_BEARING.search(line):
        return "<redacted: a line naming a row-level field>"
    return line if len(line) <= _MAX_LINE else line[:_MAX_LINE] + " …"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/home/LogFiles")
    parser.add_argument("--start", default="2026-09-13T14:23:20")
    parser.add_argument("--end", default="2026-09-13T14:24:10")
    parser.add_argument("--cap-bytes", type=int, default=4_000_000)
    parser.add_argument("--max-lines", type=int, default=200)
    parser.add_argument("--out", default=str(HERE / "n01_diagnostic.json"))
    args = parser.parse_args()

    profile = (os.environ.get("AZURE_MI_API_PUBLISH_PROFILE") or "").strip()
    if not profile:
        print("::error::AZURE_MI_API_PUBLISH_PROFILE is not set")
        return 2
    _, password = _client(profile, "/home/")

    out = {"recorded_at": datetime.now(timezone.utc).isoformat(),
           "reads_only": True, "live_model_calls": 0, "live_mi_query_calls": 0,
           "window": [args.start, args.end], "directory": args.dir}

    entries, detail = listing(profile, args.dir)
    out["listing_detail"] = detail
    if entries is None:
        out["error"] = f"could not list {args.dir}: {detail}"
        print(json.dumps(out, indent=2))
        return 2

    out["entries"] = [{"name": e.get("name"), "size": e.get("size"),
                       "mtime": e.get("mtime"), "mime": e.get("mime")}
                      for e in entries]
    print(f"--- {args.dir} ---")
    for entry in out["entries"]:
        print(f"  {str(entry['name'])[:60]:60} {entry['size']:>12} "
              f"{entry['mtime']}")

    # Files that could carry the request: a log, changed on the day of the run.
    candidates = [e for e in entries
                  if str(e.get("mime") or "").startswith("text")
                  or str(e.get("name") or "").endswith((".log", ".txt"))]
    out["files_read"] = {}
    for entry in candidates:
        name = str(entry.get("name"))
        path = f"{args.dir.rstrip('/')}/{name}"
        lines, read_detail = read_lines(profile, path, cap_bytes=args.cap_bytes)
        if lines is None:
            out["files_read"][name] = {"detail": read_detail, "matched": None}
            continue
        matched = [redact(line) for line in lines
                   if in_window(line, args.start, args.end)]
        out["files_read"][name] = {"detail": read_detail,
                                   "lines_scanned": len(lines),
                                   "matched": matched[:args.max_lines]}
        print(f"\n--- {name}: {len(matched)} line(s) in window "
              f"({read_detail}) ---")
        for line in matched[:args.max_lines]:
            print(line)

    body = json.dumps(out, indent=2, sort_keys=True, default=str)
    for secret in (profile, password or ""):
        if secret and secret in body:
            print("::error::a credential survived into the projection")
            return 2
    Path(args.out).write_text(body)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
