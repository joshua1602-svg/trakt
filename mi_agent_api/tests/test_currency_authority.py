"""Commercial go-live sprint — governed configuration owns the reporting currency.

Client 1 is a mono-currency GBP lender. The invariant this pins is not "GBP" —
it is WHERE GBP comes from: the approved OCC / client configuration
(``portfolio.base_currency``), not a statistical mode over a tape column and not
a hardcoded literal in a browser component.

Deliberately NOT tested here, because it is deliberately NOT built: FX
conversion, base-currency translated columns, or currency as an analytical
dimension. This sprint establishes ownership for a single-currency client only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import currency as currency_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_config_cache():
    currency_mod._load_client_config.cache_clear()
    yield
    currency_mod._load_client_config.cache_clear()


def _config(tmp_path: Path, body: str) -> str:
    path = tmp_path / "config_client_TEST.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# The governed configuration is the authority
# --------------------------------------------------------------------------- #
def test_client_1_gbp_comes_from_the_governed_client_configuration():
    """The shipped client configuration declares it; nothing infers it."""
    location = currency_mod.client_config_path("ERE")
    assert location and location.endswith(".yaml")
    doc = currency_mod._load_client_config(location)
    assert doc.get("portfolio", {}).get("base_currency") == "GBP"
    assert currency_mod.governed_currency_code("ERE") == "GBP"


def test_governed_configuration_outranks_the_tape(tmp_path, monkeypatch):
    """A tape whose column says otherwise does NOT override an approved
    reporting currency — that is the precedence correction this sprint made."""
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG,
                       _config(tmp_path, "portfolio:\n  base_currency: GBP\n"))
    tape = pd.DataFrame({"exposure_currency_denomination": ["EUR"] * 10})
    assert currency_mod.resolve_currency_code(tape) == "GBP"


def test_tape_still_answers_when_no_currency_is_configured(tmp_path, monkeypatch):
    """Books that predate a declared currency keep working — the inference is a
    fallback, not a competing authority."""
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG,
                       _config(tmp_path, "portfolio:\n  static_reporting_date: '2026-07-31'\n"))
    tape = pd.DataFrame({"exposure_currency_denomination": ["EUR"] * 10})
    assert currency_mod.resolve_currency_code(tape) == "EUR"


def test_platform_default_is_the_last_resort(tmp_path, monkeypatch):
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG, str(tmp_path / "absent.yaml"))
    assert currency_mod.resolve_currency_code(pd.DataFrame()) == "GBP"


def test_an_unidentified_client_never_inherits_another_clients_currency(monkeypatch):
    """Resolving with no client must not read some client's configuration —
    that would apply one tenant's governed decision to another. With no client
    identified the tape answers, exactly as it did before this sprint."""
    monkeypatch.delenv(currency_mod.ENV_CLIENT_CONFIG, raising=False)
    assert currency_mod.client_config_path(None) is None
    tape = pd.DataFrame({"exposure_currency_denomination": ["EUR"] * 5})
    assert currency_mod.resolve_currency_code(tape) == "EUR"


@pytest.mark.parametrize("body,expected", [
    ("portfolio:\n  base_currency: gbp\n", "GBP"),
    ("client:\n  reporting_currency: NOK\n", "NOK"),
    ("currency: SEK\n", "SEK"),
    ("currency_code: DKK\n", "DKK"),
])
def test_every_governed_currency_key_is_honoured(tmp_path, monkeypatch, body, expected):
    """OCC extraction writes ``client.reporting_currency``; the standing field is
    ``portfolio.base_currency``. Both, and the legacy shorthands, resolve."""
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG, _config(tmp_path, body))
    assert currency_mod.resolve_currency_code() == expected


def test_base_currency_wins_over_reporting_currency(tmp_path, monkeypatch):
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG, _config(
        tmp_path, "portfolio:\n  base_currency: GBP\nclient:\n  reporting_currency: EUR\n"))
    assert currency_mod.resolve_currency_code() == "GBP"


def test_an_unreadable_configuration_never_breaks_mi(tmp_path, monkeypatch):
    bad = tmp_path / "broken.yaml"
    bad.write_text("portfolio: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG, str(bad))
    tape = pd.DataFrame({"exposure_currency_denomination": ["EUR"] * 3})
    assert currency_mod.resolve_currency_code(tape) == "EUR"


# --------------------------------------------------------------------------- #
# One code, consumed by every surface
# --------------------------------------------------------------------------- #
def test_every_runtime_call_site_passes_the_client(tmp_path):
    """A call site that resolves currency without a client can only see the
    tape, so the governed configuration would be unreachable — which is exactly
    the defect this sprint fixed. Pin it so it cannot reappear."""
    api = _REPO_ROOT / "mi_agent_api"
    offenders = []
    for path in api.glob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            call = line.strip()
            if not call.startswith(("currency_mod.resolve_and_set(",
                                    "currency_mod.resolve_currency_code(")):
                continue
            if "client_id" not in call and "client_config" not in call:
                offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], f"currency resolved without a client at {offenders}"


def test_client_1_resolves_gbp_for_the_whole_request_scope():
    """The request-scoped code is what every formatter reads, so dashboard,
    MI Query Agent, Copilot and decks all render the same symbol."""
    code = currency_mod.resolve_and_set(pd.DataFrame(), client_id="ERE")
    assert code == "GBP"
    assert currency_mod.current_code() == "GBP"
    assert currency_mod.current_symbol() == "£"
    assert currency_mod.format_money(1_234_567.0) == "£1.2m"
