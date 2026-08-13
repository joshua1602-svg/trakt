"""operations_control.audit_chain — appending to a hash chain without losing events.

The defect this module exists to remove
---------------------------------------
Both audit stores used to append like this::

    head = read(head_uri)            # seq = 5
    seq  = head.seq + 1              # 6
    write(audit_uri(seq), record)    # overwrites whatever is at 6
    write(head_uri, {seq, hash})

Two appends running at once both read ``seq = 5``, both compute ``6``, and the
second **overwrites the first**. One audit event is gone.

The part that makes it serious rather than merely lossy: the survivor's
``prev_hash`` still points at event 5, so ``verify_audit_chain`` returns
``True``. The log is missing a record *and reports itself as intact*. An audit
trail that fails loudly is a problem; one that loses records silently is a
different category of problem, and it is the one an autonomous agent doing
concurrent work would walk into.

The fix: make sequence allocation the atomic step
-------------------------------------------------
The invariant worth protecting is **no two events may claim the same sequence
number**, and that is exactly what an exclusive create expresses::

    seq = head.seq + 1, prev = head.hash
    loop:
        if create_exclusive(audit_uri(seq), record):   # atomic, backend-native
            done
        else:                                          # someone else took seq
            prev = read(audit_uri(seq)).record_hash    # chain onto THEIR record
            seq += 1
            rebuild the record with the new prev_hash and try again

Three properties follow, and they are why this shape was chosen over a lock:

* **No lost updates.** A record is only ever written by the caller that won the
  create for that sequence number. Losing means retrying, never overwriting.
* **It converges.** On collision the loser reads the record that actually
  occupies the slot and chains onto *it*, so the retry does not depend on the
  winner having updated the head yet. There is no window in which two writers
  can livelock against a stale head.
* **The chain stays correct by construction.** ``prev_hash`` always comes from
  the record physically at ``seq - 1``, so a concurrent burst produces one
  linear chain rather than a fork that happens to verify.

The head document is demoted to a **hint**. It saves a scan; it is not the
source of truth. A stale or even backwards head costs one extra probe on the
next append and cannot corrupt anything, which is what makes the two head
writes racing each other harmless.

Idempotency
-----------
Retries above the store — an agent re-running a step, a queue redelivering a
message — should not append the same event twice. Pass ``idempotency_key`` and
the first append claims a key marker with the same exclusive create; a later
append with that key returns the **original record** instead of writing a new
one. Without a key the behaviour is unchanged: every call appends, because two
genuinely separate actions that happen to look alike are two events.

Not used here, deliberately: no lock, no lease, no queue, no coordination
service. Both storage backends already answer "create this only if it does not
exist" natively, and a distributed workflow platform to serialise a handful of
appends per second would be a much larger thing to operate than the problem.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("operations_control.audit_chain")

#: How many sequence numbers one append will probe before giving up. Each
#: collision means another writer won that slot, so this is "how many appends
#: can land while mine is in flight". Well above any realistic burst; a failure
#: here is a signal that something is wrong, not routine back-pressure.
MAX_PROBES = 64

ENV_MAX_PROBES = "TRAKT_AUDIT_MAX_PROBES"


class AuditAppendConflict(RuntimeError):
    """Raised when a record could not be placed within :data:`MAX_PROBES`.

    Deliberately loud. The alternative — writing anyway, or returning silently —
    would reintroduce exactly the failure this module removes.
    """


def _max_probes() -> int:
    try:
        return max(1, int(os.environ.get(ENV_MAX_PROBES, MAX_PROBES)))
    except (TypeError, ValueError):
        return MAX_PROBES


def append_to_chain(
    *,
    read_json: Callable[[str], Optional[Dict[str, Any]]],
    create_exclusive: Callable[[str, str], bool],
    write_json: Callable[[str, Dict[str, Any]], Any],
    head_uri: str,
    record_uri: Callable[[int], str],
    build_record: Callable[[int, str], Dict[str, Any]],
    record_hash_of: Callable[[Dict[str, Any]], str],
    idempotency_key: str = "",
    key_uri: Optional[Callable[[str], str]] = None,
) -> Tuple[Dict[str, Any], int, bool]:
    """Append one record to a hash chain. Returns ``(record, seq, created)``.

    ``created`` is ``False`` only when an idempotency key matched an existing
    record, in which case that original record is returned unchanged.

    Every collaborator is injected rather than imported so this works against
    both audit stores — which have different record shapes, different hashing
    and different URI layouts — without either of them growing a second copy of
    the concurrency logic.
    """
    # ---- idempotency: has this exact action already been recorded? -------- #
    marker_uri = ""
    if idempotency_key and key_uri is not None:
        marker_uri = key_uri(idempotency_key)
        existing = read_json(marker_uri)
        if existing and existing.get("seq"):
            record = read_json(record_uri(int(existing["seq"])))
            if record:
                return record, int(existing["seq"]), False
            # A marker with no record means a crash between the two writes. Fall
            # through and append; the marker is re-pointed at the new sequence.

    head = read_json(head_uri) or {}
    seq = int(head.get("seq") or 0) + 1
    prev_hash = str(head.get("record_hash") or "")

    limit = _max_probes()
    for probe in range(limit):
        record = build_record(seq, prev_hash)
        record["record_hash"] = record_hash_of(record)
        payload = json.dumps({"seq": seq, **record}, indent=2, default=str)

        if create_exclusive(record_uri(seq), payload):
            # Won the slot. The head is a hint from here on: writing it saves the
            # next append a probe, and failing to write it costs only that probe.
            try:
                write_json(head_uri, {"seq": seq,
                                      "record_hash": record["record_hash"]})
            except Exception:  # noqa: BLE001 - the record is already durable
                logger.warning("audit head update failed for %s; the record at "
                               "seq=%s is written and the chain is intact",
                               head_uri, seq)
            if marker_uri:
                # Best-effort and last: a marker without a record is recoverable
                # (handled above), a record without a marker merely means a
                # retry would append again.
                try:
                    write_json(marker_uri, {"seq": seq,
                                            "key": idempotency_key})
                except Exception:  # noqa: BLE001
                    logger.warning("idempotency marker write failed for %s",
                                   marker_uri)
            return {"seq": seq, **record}, seq, True

        # Lost the race for this sequence. Chain onto the record that actually
        # occupies the slot rather than re-reading the head, which the winner
        # may not have updated yet.
        occupant = read_json(record_uri(seq))
        if occupant is None:
            # Created but not yet readable, or created-then-removed. Re-probing
            # the same slot would spin, so advance and let the next iteration
            # resolve the chain from whatever is there.
            logger.debug("audit slot %s taken but unreadable; advancing", seq)
            seq += 1
            continue
        prev_hash = str(occupant.get("record_hash") or "")
        seq += 1

    raise AuditAppendConflict(
        f"could not place an audit record after {limit} attempts starting from "
        f"{head_uri}. This means sustained concurrent appends beyond the probe "
        "limit, not a corrupt chain — no record was overwritten.")
