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
        --portfolio direct_001 --probe-regime
    python tools/ere_backfill.py --remote-dir /exports/ERE --client ERE \\
        --portfolio direct_001

PROVE ONE PERIOD FIRST. ``--probe-regime`` delivers a single period's funded
pack twice — once as management information, once as the regulatory return —
and prints what the second asks for that the first does not. Those decisions
are resolved once and carry forward, so finding them on one period costs an
afternoon and finding them on ninety costs the load.
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
# Probing one period before the rest are loaded
# --------------------------------------------------------------------------- #

#: The two workflows one funded pack can be delivered under. The probe sends
#: the SAME files through both, which is the only way to see what the
#: regulatory return asks for that management information does not — before
#: ninety-odd files are loaded on the assumption that it asks for nothing more.
PROBE_WORKFLOWS: Tuple[str, str] = ("mi", "mi_annex2")

#: Only a funded pack is probed. A pipeline tape is a snapshot and is never
#: routed to the regulatory return (see ``PERIOD_RULES``), so including one
#: would be asking the probe to prove something the plan forbids.
PROBE_DATASET = "funded"


def probe_period(deliveries: List[Delivery],
                 period: str = "") -> Tuple[str, List[Delivery]]:
    """Which period to probe, and the files that make up its pack.

    Defaults to the most recent funded period, because a recent tape is the
    one whose shape the client is still producing. An explicit period that
    holds no funded files is refused rather than quietly probing a different
    one — the whole value of a probe is knowing which period it proved.
    """
    funded = [d for d in deliveries if d.rule.dataset == PROBE_DATASET]
    if not funded:
        raise Refused(
            "No funded tape was planned, so there is nothing to probe. The "
            "regulatory return is prepared from the funded book; a pipeline "
            "snapshot is never routed to it.")
    periods = sorted({d.period for d in funded})
    chosen = period or periods[-1]
    pack = [d for d in funded if d.period == chosen]
    if not pack:
        raise Refused(
            f"No funded tape was planned for {chosen}. Periods that were: "
            + ", ".join(periods))
    return chosen, pack


def describe_probe(workflow: str, batch: dict, reviews: List[dict]) -> dict:
    """One probe run, reduced to what the comparison turns on."""
    blocking = [r for r in reviews if r.get("blocking")]
    return {
        "workflow": workflow,
        "batch_id": batch.get("batch_id", ""),
        "status": batch.get("status", ""),
        "status_sentence": batch.get("status_sentence", ""),
        "missing_roles": list(batch.get("missing_roles") or []),
        "configuration_ready": bool(batch.get("configuration_ready")),
        "blocking": [r.get("question") or r.get("decision_id", "")
                     for r in blocking],
        "advisory": len(reviews) - len(blocking),
    }


def compare_probes(runs: List[dict]) -> List[str]:
    """What the regulatory run asks for that the MI run does not.

    Printed as the answer to one question: if the MI load goes ahead now, what
    is still outstanding when the regime return is run over the same data?
    """
    by_workflow = {r["workflow"]: r for r in runs}
    mi = by_workflow.get("mi") or {}
    regime = by_workflow.get("mi_annex2") or {}
    extra_roles = [r for r in (regime.get("missing_roles") or [])
                   if r not in (mi.get("missing_roles") or [])]
    extra_decisions = [q for q in (regime.get("blocking") or [])
                       if q not in (mi.get("blocking") or [])]

    lines = ["", "What the regulatory return asks for that MI does not:"]
    if extra_roles:
        lines.append("  Files still needed: " + ", ".join(extra_roles))
    if extra_decisions:
        lines.append(f"  Decisions to resolve: {len(extra_decisions)}")
        lines.extend(f"    - {q}" for q in extra_decisions)
    if not extra_roles and not extra_decisions:
        lines.append("  Nothing. The same files and the same decisions serve "
                     "both, so the remaining periods can be loaded.")
    else:
        lines.append("")
        lines.append("  Resolve these ONCE, on this period, before loading the "
                     "rest. They carry forward.")
    return lines


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

    def create_batch(self, d: Delivery, portfolio: str,
                     workflow: str = "") -> str:
        # ``workflow`` is overridden only by the probe, which delivers one
        # period under both workflows on purpose. Everything else takes the
        # workflow the plan decided from the filename.
        body = {"client_id": self.client_id, "portfolio_id": portfolio,
                "reporting_date": d.period,
                "workflow_type": workflow or d.rule.workflow,
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

    def batch(self, batch_id: str) -> dict:
        got = self._check(
            self.session.get(f"{self.base}/ops/batches/{batch_id}",
                             params={"client": self.client_id}, timeout=120),
            f"reading pack {batch_id}")
        return got["batch"]

    def open_decisions(self, workflow_id: str) -> List[dict]:
        if not workflow_id:
            return []
        got = self._check(
            self.session.get(f"{self.base}/ops/reviews",
                             params={"client": self.client_id,
                                     "workflow_id": workflow_id}, timeout=120),
            f"reading the decisions on {workflow_id}")
        return list(got.get("reviews") or [])


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
    ap.add_argument("--probe-regime", action="store_true",
                    help="Deliver ONE period's funded pack under both the MI "
                         "and the regulatory workflow, and print what the "
                         "second asks for that the first does not. Nothing "
                         "else is loaded.")
    ap.add_argument("--period", default="",
                    help="Which period to probe, spelled as the plan prints "
                         "it (YYYY-MM for a monthly funded pack). Defaults to "
                         "the most recent funded period in the plan.")
    args = ap.parse_args(argv)

    try:
        # A dry run still connects: the plan is built from the real filenames,
        # and a plan built from imagined ones would prove nothing. It reads the
        # directory listing and stops there.
        ssh, sftp = _sftp()
        names = sftp.listdir(args.remote_dir)
        deliveries = plan(names)

        if args.probe_regime:
            period, deliveries = probe_period(deliveries, args.period)
            print(f"Probing {period} for {args.client}/{args.portfolio}, "
                  f"{len(deliveries)} file(s), under "
                  f"{' and '.join(PROBE_WORKFLOWS)}:\n")
        elif args.limit:
            deliveries = deliveries[:args.limit]

        if not args.probe_regime:
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

        if args.probe_regime:
            # One period, delivered twice. Reading the files once and sending
            # the same bytes to both packs is deliberate: a difference between
            # the two runs is then a difference in what Trakt asks of the
            # data, never a difference in the data.
            content = {}
            for d in deliveries:
                with sftp.open(f"{args.remote_dir}/{d.filename}",
                               "rb") as handle:
                    handle.prefetch()
                    content[d.filename] = _decrypt(d.filename, handle.read(),
                                                   passwords)
            runs = []
            for workflow in PROBE_WORKFLOWS:
                batch_id = trakt.create_batch(deliveries[0], args.portfolio,
                                              workflow=workflow)
                for d in deliveries:
                    trakt.upload(batch_id, d.filename, content[d.filename])
                batch = trakt.batch(batch_id)
                reviews = trakt.open_decisions(batch.get("workflow_id", ""))
                run = describe_probe(workflow, batch, reviews)
                runs.append(run)
                print(f"  {workflow}: {batch_id} -> {run['status']}")
                if run["missing_roles"]:
                    print("    still needed: "
                          + ", ".join(run["missing_roles"]))
                print(f"    decisions: {len(run['blocking'])} blocking, "
                      f"{run['advisory']} advisory")
            for line in compare_probes(runs):
                print(line)
            return 0
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
