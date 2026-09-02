"""Currency resolution + money formatting for MI answers.

The reporting currency is a GOVERNED CLIENT DECISION, captured at onboarding
(``client.reporting_currency``) and persisted as ``portfolio.base_currency`` in
the client configuration. That configuration is the authority: it outranks the
tape, because a statistical mode over a data column is an inference and the
client's approved reporting currency is not. The tape remains the fallback for
books that predate a declared currency, and GBP is the platform default.

The MI tape is pan-European, so the currency is never hardcoded at a call site;
every surface — dashboard, MI Query Agent, Copilot/Teams, decks — reads the one
request-scoped code resolved here.

Resolution is request-scoped via a :class:`~contextvars.ContextVar` so the many
money formatters don't each need a currency argument, and so concurrent requests
for different clients never cross currencies (FastAPI copies the context into
its threadpool workers, and each request runs in its own context). The default
is GBP, so behaviour is unchanged for a GBP book until a request resolves a
different currency from its data.
"""
from __future__ import annotations

import contextvars
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_SYMBOLS = {
    "GBP": "£", "EUR": "€", "USD": "$", "JPY": "¥",
    "CHF": "CHF ", "AUD": "A$", "CAD": "C$", "NZD": "NZ$", "SEK": "kr ",
}
_DEFAULT_CODE = "GBP"
# Tape columns that may carry the exposure currency, most-canonical first.
_CURRENCY_FIELDS = ("exposure_currency_denomination", "currency_denomination",
                    "collateral_currency")

_CURRENCY_CODE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mi_currency_code", default=_DEFAULT_CODE)


def symbol_for(code: Optional[str]) -> str:
    """Display symbol for an ISO currency code (``GBP`` -> ``£``). An unknown
    code falls back to the code itself plus a space (e.g. ``PLN`` -> ``PLN ``)."""
    if not code:
        return _SYMBOLS[_DEFAULT_CODE]
    key = str(code).strip().upper()
    return _SYMBOLS.get(key, f"{key} ")


def current_code() -> str:
    return _CURRENCY_CODE.get()


def current_symbol() -> str:
    return symbol_for(_CURRENCY_CODE.get())


def set_currency(code: Optional[str]) -> None:
    if code:
        _CURRENCY_CODE.set(str(code).strip().upper())


#: Where the governed client configuration lives. A deployment points this at
#: the configuration onboarding GENERATED for the client (the same artefact
#: ``operations_control.configuration.resolver`` treats as the client layer).
#: There is deliberately NO default file: a client either has a governed
#: configuration or it does not, and one lender's configuration must never
#: answer for another. Mirrors the existing ``TRAKT_PORTFOLIO_REGISTRY``
#: convention, so every governed config in this estate is located the same way
#: and none depends on the working directory.
ENV_CLIENT_CONFIG = "TRAKT_MI_CLIENT_CONFIG"
_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "client"

#: Governed keys carrying the client's base / reporting currency, in precedence
#: order. ``portfolio.base_currency`` is the required OCC standing field;
#: ``client.reporting_currency`` is what OCC extraction captures from onboarding.
_CONFIG_CURRENCY_PATHS = (
    ("portfolio", "base_currency"),
    ("client", "reporting_currency"),
    ("currency",),
    ("currency_code",),
)


def client_config_path(client_id: Optional[str] = None) -> Optional[str]:
    """The governed client configuration in force, or ``None`` when none is.

    A client must be IDENTIFIED before its configuration can speak: resolving
    without a client and then reading some client's config would apply one
    tenant's governed decision to another. With no client and no explicit
    override there is no governed answer, and the caller falls through to the
    tape — the pre-existing behaviour.
    """
    configured = os.environ.get(ENV_CLIENT_CONFIG)
    if configured:
        return configured
    if not client_id:
        return None
    per_client = _CONFIG_ROOT / f"config_client_{client_id}.yaml"
    if per_client.exists():
        return str(per_client)
    # A client with no governed configuration of its own has no governed answer.
    # There used to be a fallback to the incumbent lender's file here, which
    # meant a second client silently reported under ERE Funding's configured
    # currency. Returning None sends the caller to the tape and the platform
    # default, which is what "not configured" should look like.
    return None


@lru_cache(maxsize=8)
def _load_client_config(location: str) -> Dict[str, Any]:
    """Parse the governed client configuration. Never raises: an unreadable or
    malformed config degrades to "no governed currency", so the tape and the
    platform default still answer."""
    try:
        if "://" in location:
            from apps.blob_trigger_app.storage import open_storage
            storage = open_storage()
            raw = storage.read_text(location) if storage.exists(location) else None
        else:
            path = Path(location)
            raw = path.read_text(encoding="utf-8") if path.exists() else None
        if not raw or not raw.strip():
            return {}
        import yaml
        doc = yaml.safe_load(raw) or {}
        return doc if isinstance(doc, dict) else {}
    except Exception as exc:  # noqa: BLE001 - configuration must never break MI
        logger.info("client configuration unavailable at %s: %s", location, exc)
        return {}


def _code_from_config(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """The governed base/reporting currency declared by a client config."""
    if not config:
        return None
    for keys in _CONFIG_CURRENCY_PATHS:
        node: Any = config
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str) and node.strip():
            return node.strip().upper()
    return None


def governed_currency_code(client_id: Optional[str] = None,
                           *, client_config: Optional[dict] = None
                           ) -> Optional[str]:
    """The client's approved base/reporting currency, or ``None`` if unset.

    Checks an already-loaded configuration first (a caller that has resolved the
    approved OCC configuration passes it), then the governed client config file.
    """
    code = _code_from_config(client_config)
    if code:
        return code
    location = client_config_path(client_id)
    return _code_from_config(_load_client_config(location)) if location else None


def _code_from_tape(df: Any) -> Optional[str]:
    """The most common non-null currency code on the tape, if it carries one."""
    if df is None:
        return None
    try:
        columns = getattr(df, "columns", [])
        for field in _CURRENCY_FIELDS:
            if field in columns:
                s = df[field].dropna().astype(str).str.strip()
                s = s[s.str.upper() != ""]
                s = s[~s.str.upper().isin(("NAN", "NONE", "NULL"))]
                if not s.empty:
                    return str(s.mode().iloc[0]).strip().upper()
    except Exception:  # noqa: BLE001 - currency is presentational; never break a query
        return None
    return None


def resolve_currency_code(df: Any = None, *, client_config: Optional[dict] = None,
                          client_id: Optional[str] = None,
                          default: str = _DEFAULT_CODE) -> str:
    """Resolve the reporting currency for this request.

    Precedence — governed configuration OUTRANKS the data:

      1. the approved OCC / client configuration (``portfolio.base_currency``);
      2. the tape, inferred from its own currency column;
      3. the platform default (GBP).

    Configuration comes first deliberately: the client's approved reporting
    currency is a governed decision, and a statistical mode over a tape column
    is an inference. Where the client has declared one, the inference is a
    fallback for tapes that predate it, not a competing authority.

    Never raises — each source falls through to the next.
    """
    governed = governed_currency_code(client_id, client_config=client_config)
    if governed:
        return governed
    return _code_from_tape(df) or default


def resolve_and_set(df: Any = None, *, client_config: Optional[dict] = None,
                    client_id: Optional[str] = None,
                    default: str = _DEFAULT_CODE) -> str:
    """Resolve and store the request-scoped currency; returns the code."""
    code = resolve_currency_code(df, client_config=client_config,
                                 client_id=client_id, default=default)
    set_currency(code)
    return code


def format_money(value: Optional[float], symbol: Optional[str] = None, *,
                 signed: bool = False,
                 suffixes: Tuple[str, str, str] = ("bn", "m", "k")) -> str:
    """Format a monetary amount with magnitude suffixes.

    ``symbol`` defaults to the request-scoped currency symbol; pass an explicit
    one to override. ``suffixes`` are the (billions, millions, thousands) labels
    — chat answers use lowercase ``(bn, m, k)``; KPI tiles use ``(BN, MM, K)``.
    """
    if value is None:
        return "—"
    sym = symbol if symbol is not None else current_symbol()
    value = float(value)
    sign = "+" if (signed and value >= 0) else ("-" if signed and value < 0 else "")
    v = abs(value) if signed else value
    bn, mm, k = suffixes
    if abs(v) >= 1e9:
        body = f"{sym}{v / 1e9:.2f}{bn}"
    elif abs(v) >= 1e6:
        body = f"{sym}{v / 1e6:.1f}{mm}"
    elif abs(v) >= 1e3:
        body = f"{sym}{v / 1e3:.0f}{k}"
    else:
        body = f"{sym}{v:,.0f}"
    return f"{sign}{body}"
