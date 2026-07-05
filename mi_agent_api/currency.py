"""Currency resolution + money formatting for MI answers.

The MI tape is pan-European, so the display currency must come from the DATA
(``exposure_currency_denomination``) where present, falling back to a client /
config default rather than a hardcoded GBP.

Resolution is request-scoped via a :class:`~contextvars.ContextVar` so the many
money formatters don't each need a currency argument, and so concurrent requests
for different clients never cross currencies (FastAPI copies the context into
its threadpool workers, and each request runs in its own context). The default
is GBP, so behaviour is unchanged for a GBP book until a request resolves a
different currency from its data.
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional, Tuple

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


def resolve_currency_code(df: Any = None, *, client_config: Optional[dict] = None,
                          default: str = _DEFAULT_CODE) -> str:
    """Resolve the display currency: the tape first, then client config, then
    the default. Never raises — a bad column falls through to the next source."""
    # 1. From the tape (most common non-null value of a currency column).
    if df is not None:
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
            pass
    # 2. From client config.
    if client_config:
        code = client_config.get("currency") or client_config.get("currency_code")
        if code:
            return str(code).strip().upper()
    # 3. Default.
    return default


def resolve_and_set(df: Any = None, *, client_config: Optional[dict] = None,
                    default: str = _DEFAULT_CODE) -> str:
    """Resolve and store the request-scoped currency; returns the code."""
    code = resolve_currency_code(df, client_config=client_config, default=default)
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
