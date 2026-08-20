"""Configuration is fail-closed, and never leaks the secret it holds."""

from __future__ import annotations

import pytest

from trakt_mail.config import (
    ENABLED_ENV, MailNotConfigured, load, outbound_enabled,
)

from .conftest import CLIENT_SECRET, MAILBOX, mail_env


def test_the_kill_switch_alone_decides_whether_mail_is_on():
    assert outbound_enabled({ENABLED_ENV: "true"}) is True
    assert outbound_enabled({ENABLED_ENV: "1"}) is True
    assert outbound_enabled({}) is False
    # Anything unrecognised is OFF. A typo must not enable sending.
    assert outbound_enabled({ENABLED_ENV: "maybe"}) is False
    assert outbound_enabled({ENABLED_ENV: "false"}) is False


def test_a_complete_environment_loads():
    config = load(mail_env())
    assert config.mailbox == MAILBOX
    assert config.client_secret == CLIENT_SECRET


def test_credentials_present_but_switch_off_still_refuses():
    """The switch is independent of the identity, so an administrator can stop
    sending without removing the credential."""
    env = mail_env()
    env["TRAKT_MAIL_OUTBOUND_ENABLED"] = "false"
    with pytest.raises(MailNotConfigured) as caught:
        load(env)
    assert caught.value.disabled is True


@pytest.mark.parametrize("absent", [
    "TRAKT_MAIL_TENANT_ID", "TRAKT_MAIL_CLIENT_ID",
    "TRAKT_MAIL_CLIENT_SECRET", "TRAKT_MAIL_MAILBOX",
])
def test_any_missing_setting_refuses_and_names_itself(absent):
    env = mail_env()
    env[absent] = ""
    with pytest.raises(MailNotConfigured) as caught:
        load(env)
    assert absent in caught.value.missing


def test_a_refusal_never_carries_the_secret_value():
    env = mail_env()
    env["TRAKT_MAIL_MAILBOX"] = ""
    with pytest.raises(MailNotConfigured) as caught:
        load(env)
    assert CLIENT_SECRET not in str(caught.value)
    assert CLIENT_SECRET not in "".join(caught.value.missing)


def test_a_mailbox_that_is_not_an_address_is_refused():
    with pytest.raises(MailNotConfigured):
        load(mail_env(TRAKT_MAIL_MAILBOX="admin"))


def test_the_redacted_view_reports_the_secret_only_as_present():
    view = load(mail_env()).redacted()
    assert view["client_secret"] == "set"
    assert CLIENT_SECRET not in str(view)


def test_an_empty_allow_list_permits_every_recipient():
    config = load(mail_env())
    assert config.permits("anyone@example.com") is True


def test_an_allow_list_permits_exact_addresses_and_domains():
    config = load(mail_env(
        TRAKT_MAIL_RECIPIENT_ALLOWLIST="me@example.com, @traktinfra.io"))
    assert config.permits("me@example.com") is True
    assert config.permits("ME@Example.com") is True
    assert config.permits("someone@traktinfra.io") is True
    assert config.permits("client@realclient.com") is False
