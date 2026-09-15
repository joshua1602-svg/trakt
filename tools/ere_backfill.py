#!/usr/bin/env python3
"""Load a client's historical tapes from SFTP into Trakt, one delivery at a time.

AN OPERATOR TOOL, NOT PART OF THE PLATFORM. It runs on your machine, holds no
credentials of its own, and uses only the governed HTTP route an operator uses
by hand:

    POST /ops/batches                  create the input pack
    POST /ops/batches/{id}/upload      send the file CONTENT

The destination is never named here. It is derived server-side from the pack's
own client, portfolio, book, frequency and reporting period, and re-parsed with
the production path parser — so a path this produces is, by construction, a path
the automated intake route would accept. That is the whole reason to go through
the API rather than writing blobs directly.

WHAT IT DOES NOT DO

* It does not decide the reporting period from a calendar. Periods come from the
  filename through ``PERIOD_RULES`` below, which encode what THIS client's names
  mean — a funded tape stamped 2026_05_01 is April's book, not May's. Getting
  that wrong puts real data in the wrong period, which is worse than not loading
  it, so an unrecognised name stops the run rather than being guessed at.
* It does not continue past a refusal. A 4xx means Trakt declined the delivery
  and the next 63 will almost certainly be declined the same way.
* It does not convert anything. Trakt reads .xlsx natively; only the password is
  removed, in memory.

CREDENTIALS come from the environment and are never written, logged or echoed:

    TRAKT_OPS_URL         https://trakt-ops-api.azurewebsites.net
    TRAKT_OPS_TOKEN       your operator token
    SFTP_HOST             sftp.example.com
    SFTP_PORT             22 (default)
    SFTP_USER             your username
    SFTP_PASSWORD         or SFTP_KEY_PATH for a private key
    SFTP_KEY_PASSPHRASE   if the key is encrypted
    FILE_PASSWORDS        path to a JSON file: {"<filename or *>": "<password>"}

Requires, locally only::

    pip install paramiko msoffcrypto-tool requests

Usage::

    python tools/ere_backfill.py --remote-dir /exports/ERE --client ERE \\
        --portfolio direct_001 --dry-run
    python tools/ere_backfill.py --remote-dir /exports/ERE --client ERE \\
        --portfolio direct_001
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# What this client's filenames mean
# --------------------------------------------------------------------------- #

#: A date stamped anywhere in the name: 2026_05_01, 2026-05-01, 20260501.
_DATE = re.compile(r"(20\d{2})[_-]?(\d{2})[_-]?(\d{2})")


def _stamped_date(name: str) -> Optional[_dt.date]:
    m = _DATE.search(name)
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _prior_month(name: str) -> Optional[str]:
    """A tape stamped the 1st of a month reports the month BEFORE it.

    ``LoanExtract One - OMNI 2026_05_01.xlsx`` is April 2026's funded book. The
    stamp is when the extract was taken, not the period it describes.
    """
    day = _stamped_date(name)
    if day is None:
        return None
    first = day.replace(day=1)
    prior = first - _dt.timedelta(days=1)
    return f"{prior.year:04d}-{prior.month:02d}"


def _as_of_date(name: str) -> Optional[str]:
    """A pipeline tape is the pipeline AS OF the stamped date.

    Its own date is the period. Declared ``adhoc`` rather than weekly because
    these arrive every few days: one batch per snapshot, no two snapshots
    colliding in one period, and the cadence stops being a fiction.
    """
    day = _stamped_date(name)
    return day.isoformat() if day else None


@dataclass(frozen=True)
class Rule:
    """How one kind of file is recognised and where its delivery belongs."""
    label: str
    matches: Callable[[str], bool]
    dataset: str
    frequency: str
    workflow: str
    period_of: Callable[[str], Optional[str]]


def _has(name: str, *needles: str) -> bool:
    low = name.lower()
    return all(n in low for n in needles)


#: Ordered — first match wins. Pipeline is tested first because "KFI and
#: Pipeline" must never fall through to a funded rule.
PERIOD_RULES: Tuple[Rule, ...] = (
    Rule(label="pipeline tape",
         matches=lambda n: _has(n, "pipeline"),
         dataset="pipeline", frequency="adhoc", workflow="mi",
         period_of=_as_of_date),
    Rule(label="funded loan tape",
         matches=lambda n: _has(n, "loanextract") or _has(n, "loan", "extract"),
         dataset="funded", frequency="monthly", workflow="mi_annex2",
         period_of=_prior_month),
    Rule(label="property / collateral tape",
         matches=lambda n: (_has(n, "propertyextract")
                            or _has(n, "property", "extract")),
         dataset="funded", frequency="monthly", workflow="mi_annex2",
         period_of=_prior_month),
)

DATA_EXTS = (".xlsx", ".xls", ".xlsm", ".csv")


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Delivery:
    filename: str
    rule: Rule
    period: str

    def describe(self) -> str:
        return (f"{self.filename}\n      -> {self.rule.label} | "
                f"{self.rule.dataset}/{self.rule.frequency} | "
                f"period {self.period} | {self.rule.workflow}")


class Refused(RuntimeError):
    """Trakt declined, or a filename could not be read. Stop."""


def plan(filenames: List[str]) -> List[Delivery]:
    """Turn filenames into deliveries, or refuse the whole run.

    Refusing everything on one unreadable name is deliberate. A partial load is
    the worst outcome: the gaps are invisible afterwards, and the fix is to work
    out which of sixty-four files are missing.
    """
    out: List[Delivery] = []
    unreadable: List[str] = []
    for name in sorted(filenames):
        if not name.lower().endswith(DATA_EXTS):
            continue
        rule = next((r for r in PERIOD_RULES if r.matches(name)), None)
        if rule is None:
            unreadable.append(f"{name}: no rule recognises this name")
            continue
        period = rule.period_of(name)
        if not period:
            unreadable.append(f"{name}: no date could be read from the name")
            continue
        out.append(Delivery(filename=name, rule=rule, period=period))
    if unreadable:
        raise Refused(
            "These files could not be placed, so nothing was loaded:\n  - "
            + "\n  - ".join(unreadable)
            + "\n\nAdd a rule in PERIOD_RULES, or correct the name. Guessing a "
              "period puts real data in the wrong one.")
    return out


# --------------------------------------------------------------------------- #
# SFTP and decryption — both local, both in memory
# --------------------------------------------------------------------------- #

def _sftp():
    import paramiko
    host = _need("SFTP_HOST")
    user = _need("SFTP_USER")
    port = int(os.environ.get("SFTP_PORT", "22"))
    key_path = os.environ.get("SFTP_KEY_PATH", "")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Refuse an unknown host rather than trusting it: this is a client's data.
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    if key_path:
        client.connect(host, port=port, username=user, key_filename=key_path,
                       passphrase=os.environ.get("SFTP_KEY_PASSPHRASE") or None)
    else:
        client.connect(host, port=port, username=user,
                       password=_need("SFTP_PASSWORD"))
    return client, client.open_sftp()


def _passwords() -> Dict[str, str]:
    path = os.environ.get("FILE_PASSWORDS", "")
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _decrypt(name: str, data: bytes, passwords: Dict[str, str]) -> bytes:
    """Return the workbook unencrypted, or unchanged if it never was.

    A wrong or missing password raises rather than uploading the encrypted
    bytes: Trakt would take the file and fail to read it later, which reads as
    an unreadable tape rather than a locked one.
    """
    import msoffcrypto
    buf = io.BytesIO(data)
    try:
        office = msoffcrypto.OfficeFile(buf)
    except Exception:
        return data                      # not an Office container; pass through
    if not office.is_encrypted():
        return data
    secret = passwords.get(name) or passwords.get("*")
    if not secret:
        raise Refused(f"{name} is password-protected and no password was given "
                      "for it in FILE_PASSWORDS.")
    out = io.BytesIO()
    try:
        office.load_key(password=secret)
        office.decrypt(out)
    except Exception as exc:             # noqa: BLE001 — the password is wrong
        raise Refused(f"{name} could not be decrypted "
                      f"({type(exc).__name__}). Check its password.") from None
    return out.getvalue()


# --------------------------------------------------------------------------- #
# Trakt
# --------------------------------------------------------------------------- #

def _need(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise Refused(f"{name} is not set. See the module docstring.")
    return value


class Trakt:
    def __init__(self, base: str, token: str, client_id: str):
        import requests
        self._requests = requests
        self.base = base.rstrip("/")
        self.client_id = client_id
        self.session = requests.Session()
        self.session.headers["X-Operator-Token"] = token

    def _check(self, response, what: str):
        if response.status_code >= 400:
            body = response.text[:400]
            raise Refused(f"{what} was refused ({response.status_code}): {body}")
        return response.json()

    def create_batch(self, d: Delivery, portfolio: str) -> str:
        body = {"client_id": self.client_id, "portfolio_id": portfolio,
                "reporting_date": d.period, "workflow_type": d.rule.workflow,
                "dataset": d.rule.dataset, "frequency": d.rule.frequency,
                "auto_start_when_ready": True}
        got = self._check(self.session.post(f"{self.base}/ops/batches",
                                            json=body, timeout=120),
                          f"creating the pack for {d.filename}")
        return got["batch"]["batch_id"]

    def upload(self, batch_id: str, filename: str, data: bytes) -> dict:
        got = self._check(
            self.session.post(
                f"{self.base}/ops/batches/{batch_id}/upload",
                params={"client": self.client_id},
                files={"files": (filename, data)}, timeout=600),
            f"uploading {filename}")
        return got["batch"]


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote-dir", required=True,
                    help="SFTP directory holding the tapes")
    ap.add_argument("--client", required=True, help="Trakt client id, e.g. ERE")
    ap.add_argument("--portfolio", required=True, help="e.g. direct_001")
    ap.add_argument("--dry-run", action="store_true",
                    help="Read the names and print the plan. Touches nothing.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Load at most N deliveries. Use 1 first.")
    args = ap.parse_args(argv)

    try:
        # A dry run still connects: the plan is built from the real filenames,
        # and a plan built from imagined ones would prove nothing. It reads the
        # directory listing and stops there.
        ssh, sftp = _sftp()
        names = sftp.listdir(args.remote_dir)
        deliveries = plan(names)
        if args.limit:
            deliveries = deliveries[:args.limit]

        print(f"{len(deliveries)} deliveries planned "
              f"for {args.client}/{args.portfolio}:\n")
        for d in deliveries:
            print("  •", d.describe())
        if args.dry_run:
            print("\nDry run. Nothing was created and nothing was uploaded.")
            return 0

        passwords = _passwords()
        trakt = Trakt(_need("TRAKT_OPS_URL"), _need("TRAKT_OPS_TOKEN"),
                      args.client)
        print()
        for index, d in enumerate(deliveries, start=1):
            head = f"[{index}/{len(deliveries)}] {d.filename}"
            with sftp.open(f"{args.remote_dir}/{d.filename}", "rb") as handle:
                handle.prefetch()
                raw = handle.read()
            data = _decrypt(d.filename, raw, passwords)
            batch_id = trakt.create_batch(d, args.portfolio)
            batch = trakt.upload(batch_id, d.filename, data)
            print(f"{head}\n    {batch_id} -> {batch.get('status')}")
        print(f"\n{len(deliveries)} delivered.")
        return 0
    except Refused as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            if ssh is not None:
                ssh.close()
        except Exception:                # noqa: BLE001 — closing, not working
            pass


if __name__ == "__main__":
    raise SystemExit(main())
