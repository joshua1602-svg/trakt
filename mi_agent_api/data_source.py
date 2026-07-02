"""Resolve the canonical dataframe and semantics registry for the API.

The same API serves either a synthetic demo portfolio (default) or a **promoted
funded central lender tape** from an onboarding run — and, for the funded path, it
runs the existing MI data-preparation layer (derive bucket source fields + the
canonical ``analytics_lib.buckets`` engine over ``config/mi/buckets.yaml``) so the
React dashboard gets an **analytics-ready funded MI dataset** (LTV / rate / ticket
/ time-on-book / vintage dimensions), consistent with Streamlit. Selection is
environment-driven (no code change), resolved in priority order:

    1. MI_AGENT_ANALYTICS_DATASET    - explicit path to an already MI-prepared CSV.
    2. MI_AGENT_CENTRAL_TAPE  /  MI_AGENT_ONBOARDING_OUTPUT_ROOT + MI_AGENT_CLIENT_ID
       + MI_AGENT_RUN_ID            - a promoted ``18_central_lender_tape.csv``;
                                      MI preparation is applied (unless
                                      MI_AGENT_DISABLE_PREP=1, which serves the raw
                                      thin tape for KPI-only mode).
    3. MI_AGENT_DATA_CSV             - explicit canonical_typed.csv.
    4. synthetic_demo/**/*canonical_typed.csv  (default demo).

    MI_AGENT_SEMANTICS               - path to mi_semantics_field_registry.yaml.

Because the funded tape is period-scoped, the dashboard inherently shows the
funded universe (e.g. 33 / 73 loans), never the old 2,196-row universe and never
pipeline rows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .funded_prep import CORE_FUNDED_DIMENSIONS, prepare_funded_mi_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

_CENTRAL_TAPE_NAME = "18_central_lender_tape.csv"
_PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"

# Source kinds (surfaced on /health).
KIND_PREPARED = "funded_mi_prepared_dataset"
KIND_FUNDED_RAW = "funded_central_lender_tape_raw"
KIND_EXPLICIT_CSV = "explicit_csv"
KIND_SYNTHETIC_DEMO = "synthetic_demo"
KIND_PLATFORM_CANONICAL = "platform_canonical"
KIND_UNAVAILABLE = "unavailable"


def _resolve_platform_uri(uri: str) -> Optional[Path]:
    """Resolve MI_AGENT_PLATFORM_URI via the storage abstraction.

    A ``blob://`` URI may point at the file or its containing ``latest/`` dir; a
    Blob source is downloaded to a local scratch dir. Returns ``None`` if absent.
    """
    try:
        from apps.blob_trigger_app.storage import open_storage
    except Exception:  # noqa: BLE001 — storage module unavailable → skip
        return None
    storage = open_storage()
    candidate = uri if uri.endswith(".csv") else f"{uri.rstrip('/')}/{_PLATFORM_CANONICAL_NAME}"
    if not storage.exists(candidate):
        return None
    local = storage._local_path(candidate)
    if Path(str(local)).exists():
        return Path(str(local))
    scratch_root = os.environ.get("MI_AGENT_SCRATCH", "/tmp/trakt/mi_platform")
    dest = Path(scratch_root) / _PLATFORM_CANONICAL_NAME
    return storage.download_file(candidate, dest)


def _resolve_platform_canonical() -> Optional[Path]:
    """Resolve the combined platform canonical, if one has been assembled.

    The platform canonical (``engine.platform_assembler``) is the current
    managed-portfolio view across all onboarded books. Resolved by, in order:
      1. MI_AGENT_PLATFORM_CANONICAL  — explicit file path;
      2. MI_AGENT_PLATFORM_DIR/platform_canonical_typed.csv;
      3. the conventional ``out_platform/platform_canonical_typed.csv`` next to
         the repo root or the current working directory.
    Returns ``None`` when no platform canonical exists, so the MI Agent falls
    back to exactly today's resolution.
    """
    # Durable persisted platform canonical (production). MI_AGENT_PLATFORM_URI may
    # be a blob:// URI or local path; resolved via the storage abstraction and
    # (for Blob) downloaded to a local scratch. Off unless the env is set, so
    # local/test resolution below is unchanged.
    uri = os.environ.get("MI_AGENT_PLATFORM_URI")
    if uri:
        resolved = _resolve_platform_uri(uri)
        if resolved is not None:
            return resolved
    explicit = os.environ.get("MI_AGENT_PLATFORM_CANONICAL")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    pdir = os.environ.get("MI_AGENT_PLATFORM_DIR")
    if pdir:
        p = Path(pdir) / _PLATFORM_CANONICAL_NAME
        return p if p.exists() else None
    for base in (_REPO_ROOT, Path.cwd()):
        p = base / "out_platform" / _PLATFORM_CANONICAL_NAME
        if p.exists():
            return p
    return None


def _prep_disabled() -> bool:
    return str(os.environ.get("MI_AGENT_DISABLE_PREP", "")).strip().lower() in ("1", "true", "yes")


def _resolve_central_tape() -> Optional[Path]:
    """Resolve a promoted run's central lender tape generically by client/run."""
    explicit = os.environ.get("MI_AGENT_CENTRAL_TAPE")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    root = os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
    if not root:
        return None
    root_path = Path(root)
    client_id = os.environ.get("MI_AGENT_CLIENT_ID", "")
    run_id = os.environ.get("MI_AGENT_RUN_ID", "")
    candidates = [
        root_path / "central" / _CENTRAL_TAPE_NAME,
        root_path / client_id / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
        root_path / "runs" / client_id / "onboarding" / run_id / "central" / _CENTRAL_TAPE_NAME,
        root_path / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    if run_id:
        hits = sorted(root_path.glob(f"**/{run_id}/**/{_CENTRAL_TAPE_NAME}"))
        if hits:
            return hits[0]
    hits = sorted(root_path.glob(f"**/{_CENTRAL_TAPE_NAME}"))
    return hits[0] if hits else None


def _find_demo_csv() -> Optional[Path]:
    """The explicit/demo CSV (MI_AGENT_DATA_CSV or the bundled synthetic demo)."""
    override = os.environ.get("MI_AGENT_DATA_CSV")
    if override:
        p = Path(override)
        return p if p.exists() else None
    candidates = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
    return candidates[0] if candidates else None


def resolve_data_source() -> Tuple[Optional[Path], str]:
    """``(path, base_kind)`` for the active data source, by priority.

    ``base_kind`` ∈ ``prepared_explicit`` | ``central_tape`` | ``explicit_csv`` |
    ``synthetic_demo`` | ``unavailable``.
    """
    ds = os.environ.get("MI_AGENT_ANALYTICS_DATASET")
    if ds:
        p = Path(ds)
        if p.exists():
            return p, "prepared_explicit"
    # The combined platform canonical (latest per portfolio) is the default
    # managed-portfolio view when present; absent it, behaviour is unchanged.
    platform = _resolve_platform_canonical()
    if platform is not None:
        return platform, "platform_canonical"
    tape = _resolve_central_tape()
    if tape is not None:
        return tape, "central_tape"
    if os.environ.get("MI_AGENT_DATA_CSV"):
        p = _find_demo_csv()
        return p, ("explicit_csv" if p else "unavailable")
    demo = _find_demo_csv()
    return demo, ("synthetic_demo" if demo else "unavailable")


def semantics_path() -> Path:
    override = os.environ.get("MI_AGENT_SEMANTICS")
    return Path(override) if override else DEFAULT_SEMANTICS


def _materialise_mi_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add registry-named bucket dimensions using the EXISTING bucketing engine.

    Used for the demo / pre-typed CSV paths. The funded path uses the richer
    ``prepare_funded_mi_dataset`` (derive sources + same bucket engine).
    """
    try:
        from analytics_lib.buckets import load_bucket_config, materialise_buckets

        out, _issues, _applied = materialise_buckets(
            df, load_bucket_config(), target="semantic_field"
        )
        return out
    except Exception:  # noqa: BLE001 - bucketing is additive; never block a query
        return df


def _present_dimensions(
    df: pd.DataFrame, missing_reason: str = "not_in_dataset",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """``(available_names, missing_dicts)`` for a non-prepared dataframe."""
    cols = set(df.columns)
    available = sorted([d for d in CORE_FUNDED_DIMENSIONS if d in cols]
                       + [c for c in cols if c.endswith("_bucket")
                          and c not in CORE_FUNDED_DIMENSIONS])
    missing = [{"dimension": d, "reason": missing_reason,
                "detail": f"{d!r} not present in this data source"}
               for d in CORE_FUNDED_DIMENSIONS if d not in cols]
    return available, missing


def _platform_uri_etag(uri: str) -> Optional[str]:
    """Cheap change-token for a ``blob://`` (or local) platform URI — the blob
    ETag via a properties HEAD, NOT a download. Lets the cache detect a
    re-published canonical without re-reading it every request."""
    try:
        from apps.blob_trigger_app.storage import open_storage
    except Exception:  # noqa: BLE001
        return None
    try:
        storage = open_storage()
        candidate = uri if uri.endswith(".csv") else f"{uri.rstrip('/')}/{_PLATFORM_CANONICAL_NAME}"
        return storage.etag(candidate)
    except Exception:  # noqa: BLE001
        return None


def _source_signature() -> str:
    """A cheap identity of the ACTIVE data source so the cache reloads when a new
    run republishes it. For a ``blob://`` platform URI this is the blob ETag (no
    download); for local sources it is ``path:mtime:size``."""
    uri = os.environ.get("MI_AGENT_PLATFORM_URI")
    if uri:
        et = _platform_uri_etag(uri)
        if et:
            return f"platform_uri:{uri}:{et}"
    try:
        path, base = resolve_data_source()
    except Exception:  # noqa: BLE001
        return "unresolved"
    if path is None:
        return "none"
    try:
        st = Path(path).stat()
        return f"{base}:{path}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return f"{base}:{path}"


def _cache_ttl() -> float:
    """Seconds a loaded dataset is served before the source signature is re-checked
    (``MI_AGENT_DATA_CACHE_TTL``; default 30s). 0 ⇒ check every request."""
    try:
        return max(0.0, float(os.environ.get("MI_AGENT_DATA_CACHE_TTL", "30")))
    except (TypeError, ValueError):
        return 30.0


#: Signature-keyed cache: (signature -> (df, info)) with a TTL re-check so a new
#: run is picked up on the next request without an API restart.
_ACTIVE_CACHE: Dict[str, Any] = {"sig": None, "value": None, "checked_at": 0.0}


def _active() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return the active ``(df, info)``, reloading when the source changed.

    Within the TTL the cached frame is served directly; after the TTL the source
    signature (blob ETag / file mtime) is re-checked and the dataset is reloaded
    ONLY when it actually changed — so a re-published platform canonical renders
    on the next request, but an unchanged source is not re-read.
    """
    import time
    now = time.monotonic()
    cached = _ACTIVE_CACHE.get("value")
    if cached is not None and (now - _ACTIVE_CACHE["checked_at"]) < _cache_ttl():
        return cached
    sig = _source_signature()
    if cached is not None and sig == _ACTIVE_CACHE.get("sig"):
        _ACTIVE_CACHE["checked_at"] = now
        return cached
    value = _load_active()
    _ACTIVE_CACHE.update(sig=sig, value=value, checked_at=now)
    return value


def _load_active() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load + prepare the active dataset; return ``(df, info)``."""
    path, base = resolve_data_source()
    if path is None:
        raise FileNotFoundError(
            "No data source found. Set MI_AGENT_ANALYTICS_DATASET, "
            "MI_AGENT_CENTRAL_TAPE, MI_AGENT_ONBOARDING_OUTPUT_ROOT + "
            "MI_AGENT_CLIENT_ID + MI_AGENT_RUN_ID, MI_AGENT_DATA_CSV, or add a "
            "synthetic_demo/**/*canonical_typed.csv file."
        )
    raw = pd.read_csv(path, low_memory=False)
    info: Dict[str, Any] = {"label": path.name, "path": str(path)}

    if base == "central_tape" and not _prep_disabled():
        try:
            df, report = prepare_funded_mi_dataset(raw)
            info.update(kind=KIND_PREPARED, **report)
        except Exception as exc:  # never block: serve the raw thin tape
            df = raw
            avail, missing = _present_dimensions(raw, "not_consumed_by_mi_prep")
            info.update(kind=KIND_FUNDED_RAW, preparation_applied=False,
                        preparation_error=str(exc), derived_fields=[],
                        dimensions_available=avail, missing_dimensions=missing)
    elif base == "central_tape":  # prep explicitly disabled -> thin KPI mode
        df = raw
        avail, missing = _present_dimensions(raw, "not_consumed_by_mi_prep")
        info.update(kind=KIND_FUNDED_RAW, preparation_applied=False,
                    derived_fields=[], dimensions_available=avail,
                    missing_dimensions=missing)
    elif base == "prepared_explicit":
        df = _materialise_mi_buckets(raw)
        avail, missing = _present_dimensions(df)
        info.update(kind=KIND_PREPARED, preparation_applied=True, derived_fields=[],
                    dimensions_available=avail, missing_dimensions=missing)
    elif base == "platform_canonical":
        # Combined latest-per-portfolio canonical; treated like a typed canonical
        # (bucket materialisation only, no funded-tape prep).
        df = _materialise_mi_buckets(raw)
        avail, missing = _present_dimensions(df)
        info.update(kind=KIND_PLATFORM_CANONICAL, preparation_applied=False,
                    derived_fields=[], dimensions_available=avail,
                    missing_dimensions=missing)
    else:  # explicit_csv | synthetic_demo
        df = _materialise_mi_buckets(raw)
        avail, missing = _present_dimensions(df)
        info.update(kind=(KIND_EXPLICIT_CSV if base == "explicit_csv" else KIND_SYNTHETIC_DEMO),
                    preparation_applied=False, derived_fields=[],
                    dimensions_available=avail, missing_dimensions=missing)

    if base == "central_tape":
        info["client_id"] = os.environ.get("MI_AGENT_CLIENT_ID", "")
        info["run_id"] = os.environ.get("MI_AGENT_RUN_ID", "")

    # Single dataset contract (per-field metadata + display hints), built from the
    # one dataset profile. /health and the review generator read THIS — never a
    # separate inference. Dimensions reflect actual non-null prepared values.
    try:
        from .mi_dataset_contract import build_dataset_contract
        from mi_agent.mi_query_validator import load_mi_semantics
        semantics = load_mi_semantics(semantics_path())
        prep_report = info if info.get("preparation_applied") else None
        contract = build_dataset_contract(df, semantics, prep_report)
        info["dataset_contract"] = contract
        info["display_hints"] = contract["display_hints"]
        # Funded prep already supplies reason-coded dimensions; for any other path
        # surface the contract's (non-null-based) availability so /health is honest.
        if not info.get("dimensions_available"):
            info["dimensions_available"] = contract["dimensions_available"]
            info["missing_dimensions"] = contract["dimensions_missing"]
    except Exception as exc:  # contract is additive; never block data serving
        info.setdefault("dataset_contract", {"fields": [], "error": str(exc)})
    return df, info


def reset_cache() -> None:
    """Clear the cached active dataset (used by tests, and by an explicit refresh
    after a new run publishes). The next request reloads from the current source."""
    _ACTIVE_CACHE.update(sig=None, value=None, checked_at=0.0)


def get_dataframe() -> pd.DataFrame:
    """The active analytics dataframe (funded MI-prepared, demo, or explicit)."""
    return _active()[0]


# Back-compat: tests historically cleared the cache via get_dataframe.cache_clear().
get_dataframe.cache_clear = reset_cache  # type: ignore[attr-defined]


def data_source_label() -> str:
    try:
        return _active()[1].get("label", KIND_UNAVAILABLE)
    except FileNotFoundError:
        return KIND_UNAVAILABLE


def data_source_kind() -> str:
    try:
        return _active()[1].get("kind", KIND_UNAVAILABLE)
    except FileNotFoundError:
        return KIND_UNAVAILABLE


def data_source_info() -> Dict[str, Any]:
    """Full data-source descriptor for /health (kind, preparation, dimensions)."""
    try:
        _df, info = _active()
        return dict(info)
    except FileNotFoundError:
        return {"kind": KIND_UNAVAILABLE, "label": KIND_UNAVAILABLE, "path": "",
                "preparation_applied": False, "dimensions_available": [],
                "missing_dimensions": [{"dimension": d, "reason": "unavailable",
                                        "detail": "no data source"}
                                       for d in CORE_FUNDED_DIMENSIONS]}
