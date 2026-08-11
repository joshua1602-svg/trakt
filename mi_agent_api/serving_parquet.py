"""mi_agent_api.serving_parquet — a columnar SERVING copy of the canonical tape.

    CSV is authoritative. This is a derived cache, and it is always disposable.

The problem
-----------
``pd.read_csv`` re-parses and re-infers every column on every cold load. At a
million rows that is tens of seconds of CPU spent producing a frame the previous
load already produced, and it happens on the first request after any restart,
any scale-out event and any dataset change.

The shape of the fix
--------------------
On the first read of a canonical CSV, write a Parquet copy beside it, keyed on
the CSV's **content identity** (path, mtime, size). Subsequent reads load the
Parquet. A changed CSV produces a different key, so the stale copy is simply
never asked for — there is no invalidation to get wrong, and no TTL to tune.

Three properties this deliberately has
--------------------------------------
**The CSV stays the record.** No pipeline stage writes Parquet, no regulatory
artefact becomes Parquet, and deleting the entire cache directory costs one slow
load and nothing else. ``tests/test_serving_parquet.py`` asserts the frames are
equal, because a serving copy that differs from the source is not a cache, it is
a second dataset.

**It degrades to exactly today's behaviour.** No pyarrow, an unwritable
directory, a column Parquet cannot represent, a corrupt file — every one of them
falls back to ``read_csv`` and logs. A serving optimisation must never be able to
make the platform unavailable.

**Dtypes come out of the round trip, not out of re-inference.** The Parquet copy
is written FROM the csv-derived frame, so reading it back reproduces that frame's
dtypes exactly. This is why the copy is more faithful than a second CSV parse,
not less: ``read_csv`` re-infers, and re-inference is where "1.0" and "1" and
"01" stop agreeing.

Kill switch: ``TRAKT_SERVING_PARQUET=0``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("mi_agent_api.serving_parquet")

ENABLED_ENV = "TRAKT_SERVING_PARQUET"
CACHE_DIR_ENV = "TRAKT_SERVING_PARQUET_DIR"

#: Bumped when the copy's meaning changes (a different compression, a different
#: key derivation). An old-version file is never read, only orphaned.
SERVING_FORMAT_VERSION = "1"

#: Where copies live when the source directory is not writable. Deliberately not
#: /tmp by default on a server, but a sensible default is better than a failure.
DEFAULT_CACHE_DIRNAME = ".trakt-serving"

#: Below this a CSV parse is already fast and the copy is not worth the disk.
#: Overridable, because "fast enough" depends on the disk and on how often the
#: process restarts — a deployment that cold-starts per request wants it at 0.
MIN_BYTES_TO_CACHE = 2 * 1024 * 1024      # 2 MB
MIN_BYTES_ENV = "TRAKT_SERVING_PARQUET_MIN_BYTES"


def min_bytes_to_cache() -> int:
    try:
        return max(0, int(os.environ.get(MIN_BYTES_ENV, MIN_BYTES_TO_CACHE)))
    except (TypeError, ValueError):
        return MIN_BYTES_TO_CACHE

#: Counters, for the telemetry the tools publish and for tests.
_STATS: Dict[str, int] = {"hit": 0, "miss": 0, "write": 0, "write_failed": 0,
                          "read_failed": 0, "skipped": 0, "disabled": 0}


def enabled() -> bool:
    return str(os.environ.get(ENABLED_ENV, "1")).strip().lower() not in {
        "0", "false", "no", "off"}


def stats() -> Dict[str, int]:
    return dict(_STATS)


def reset_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


# --------------------------------------------------------------------------- #
# Keying
# --------------------------------------------------------------------------- #
def source_identity(path: Path) -> Optional[str]:
    """Content identity of the source CSV: path, mtime and size.

    The same derivation ``trakt_core.config_cache`` uses, and for the same
    reason: it is cheap, it needs no read of the file, and a changed file cannot
    collide with the unchanged one. It is NOT a content hash — hashing a
    one-gigabyte CSV to decide whether to avoid parsing a one-gigabyte CSV would
    spend most of the saving.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


def _cache_dir(source: Path) -> Optional[Path]:
    """Where the copy goes. Beside the source, or under an explicit override."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override)
    return source.parent / DEFAULT_CACHE_DIRNAME


def serving_path(source: Path) -> Optional[Path]:
    """The copy's path for this exact source revision, or ``None``."""
    identity = source_identity(source)
    directory = _cache_dir(source)
    if identity is None or directory is None:
        return None
    return directory / f"{source.stem}__v{SERVING_FORMAT_VERSION}__{identity}.parquet"


# --------------------------------------------------------------------------- #
# The read
# --------------------------------------------------------------------------- #
def read_canonical(path: str | os.PathLike, *, read_csv=None) -> Tuple[Any, str]:
    """Load a canonical CSV, via its Parquet serving copy where one is usable.

    Returns ``(frame, outcome)`` where outcome is one of ``hit``, ``miss``,
    ``skipped``, ``disabled`` — published as cache telemetry rather than kept as
    an internal detail, because "why was this request slow" should be answerable
    from the response.

    ``read_csv`` is injectable so a caller keeps its own parse options; it
    defaults to ``pd.read_csv(path, low_memory=False)``, matching every existing
    call site.
    """
    import pandas as pd

    source = Path(path)

    def _csv():
        if read_csv is not None:
            return read_csv(source)
        return pd.read_csv(source, low_memory=False)

    if not enabled():
        _STATS["disabled"] += 1
        return _csv(), "disabled"

    target = serving_path(source)
    if target is None:
        _STATS["skipped"] += 1
        return _csv(), "skipped"

    # ---- read the copy ---------------------------------------------------- #
    if target.exists():
        try:
            frame = pd.read_parquet(target)
            _STATS["hit"] += 1
            return frame, "hit"
        except Exception as exc:  # noqa: BLE001 - a bad copy is never fatal
            _STATS["read_failed"] += 1
            logger.warning("serving copy unreadable, falling back to CSV: %s (%s)",
                           target, exc)
            try:
                target.unlink()          # so the next load rewrites it
            except OSError:
                pass

    # ---- parse the source, then write the copy ---------------------------- #
    _STATS["miss"] += 1
    frame = _csv()

    try:
        if source.stat().st_size < min_bytes_to_cache():
            # A small tape parses fast enough that the copy is pure overhead.
            _STATS["skipped"] += 1
            return frame, "skipped"
    except OSError:
        return frame, "miss"

    _write_copy(frame, target)
    return frame, "miss"


def _write_copy(frame: Any, target: Path) -> bool:
    """Write the copy atomically. Every failure is a warning, never an error.

    Atomic because two workers starting together will both miss and both write,
    and a half-written Parquet read by a third is the one failure mode that would
    be worse than no cache at all. ``os.replace`` on the same filesystem makes
    the last writer win with no reader ever seeing a partial file.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=str(target.parent),
                                            prefix=".partial-", suffix=".parquet")
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            frame.to_parquet(tmp, index=False, compression="snappy")
            os.replace(tmp, target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    except Exception as exc:  # noqa: BLE001 - an unwritable cache is not an outage
        _STATS["write_failed"] += 1
        logger.warning("could not write the serving copy (serving continues from "
                       "CSV): %s (%s)", target, exc)
        return False
    _STATS["write"] += 1
    logger.info("wrote serving copy %s", target)
    return True


def purge(source: str | os.PathLike, *, all_revisions: bool = False) -> int:
    """Delete serving copies for one source. Returns how many were removed.

    Always safe: the copies are derived, and the next load rebuilds them.
    """
    src = Path(source)
    directory = _cache_dir(src)
    if directory is None or not directory.exists():
        return 0
    if all_revisions:
        pattern = f"{src.stem}__v*__*.parquet"
    else:
        target = serving_path(src)
        pattern = target.name if target else ""
    removed = 0
    for candidate in directory.glob(pattern) if pattern else ():
        try:
            candidate.unlink()
            removed += 1
        except OSError:
            pass
    return removed
