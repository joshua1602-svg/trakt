"""Reading a client's reply, and knowing whose it is.

The correlation tests are the important ones. Writing a client's answers onto
the wrong case is a data incident rather than a bug, so the property under test
throughout is not "does it find the case" but "does it REFUSE when the evidence
does not establish one".
"""

from __future__ import annotations

import base64

import pytest

from trakt_mail import config as _config
from trakt_mail import inbound as _inbound
from trakt_mail.graph_client import GraphMailClient, GraphMailError

from .conftest import CLIENT_ID, CLIENT_SECRET, TENANT_ID, FakeGraph, inbound_env

MAILBOX = "onboarding@traktinfra.io"
CONVERSATION = "AAQkAGI2-conversation-id"
SENT_ID = "<0001@traktinfra.io>"


def reader(graph: FakeGraph, **kwargs) -> GraphMailClient:
    return GraphMailClient(tenant_id=TENANT_ID, client_id=CLIENT_ID,
                           client_secret=CLIENT_SECRET, mailbox=MAILBOX,
                           transport=graph, **kwargs)


def message(**overrides):
    """A Graph inbox entry, in the shape Graph actually returns one."""
    payload = {
        "id": "AAMkAGI2-reply-1",
        "internetMessageId": "<reply-1@erm.example>",
        "conversationId": CONVERSATION,
        "subject": "RE: Onboarding information request (ONB-2026-0001)",
        "from": {"emailAddress": {"name": "Dana Okafor",
                                  "address": "ops@erm.example"}},
        "receivedDateTime": "2026-08-21T08:14:00Z",
        "hasAttachments": False,
        "isRead": False,
        "body": {"contentType": "html",
                 "content": "<html><body><p>Our LEI is 5493001KJTIIGC8Y1R12."
                            "</p></body></html>"},
        "uniqueBody": {"contentType": "html",
                       "content": "<p>Our LEI is 5493001KJTIIGC8Y1R12.</p>"},
        "internetMessageHeaders": [
            {"name": "In-Reply-To", "value": SENT_ID},
            {"name": "References", "value": f"{SENT_ID} <older@x.example>"},
        ],
    }
    payload.update(overrides)
    return payload


def target(**overrides) -> _inbound.CaseTarget:
    fields = {"case_ref": "ONB-2026-0001", "tenant": "client_a",
              "conversation_id": CONVERSATION,
              "internet_message_id": SENT_ID,
              "contacts": ("ops@erm.example",)}
    fields.update(overrides)
    return _inbound.CaseTarget(**fields)


# --------------------------------------------------------------------------- #
# Parsing what arrived
# --------------------------------------------------------------------------- #

def test_a_graph_message_becomes_a_message_this_package_can_work_with():
    parsed = _inbound.parse_message(message())
    assert parsed.graph_id == "AAMkAGI2-reply-1"
    assert parsed.sender == "ops@erm.example"
    assert parsed.sender_name == "Dana Okafor"
    assert parsed.conversation_id == CONVERSATION
    assert parsed.in_reply_to == [SENT_ID]
    assert parsed.case_refs == ["ONB-2026-0001"]


def test_the_body_is_read_as_text_not_as_markup():
    parsed = _inbound.parse_message(message())
    assert parsed.body_text == "Our LEI is 5493001KJTIIGC8Y1R12."
    assert "<p>" not in parsed.body_text


def test_the_reply_is_read_without_trakts_own_words_quoted_back():
    """``uniqueBody`` is what Graph gives for exactly this, and it is preferred.

    Without it the client's two-line answer would arrive with the whole pack
    quoted underneath, and every downstream reading of "what did they say"
    would be reading Trakt's own questions back."""
    parsed = _inbound.parse_message(message())
    assert "Onboarding checklist" not in parsed.body_text
    assert len(parsed.body_text) < 200


def test_a_message_without_a_unique_body_has_its_quoting_stripped_locally():
    payload = message()
    payload.pop("uniqueBody")
    payload["body"] = {
        "contentType": "text",
        "content": ("Answers below.\n"
                    "LEI: 5493001KJTIIGC8Y1R12\n"
                    "\n"
                    "On 20 August 2026, Trakt Operations wrote:\n"
                    "> Please complete the attached checklist.\n"),
    }
    parsed = _inbound.parse_message(payload)
    assert "Please complete" not in parsed.body_text
    assert "LEI: 5493001KJTIIGC8Y1R12" in parsed.body_text


def test_script_and_style_never_become_part_of_what_the_client_said():
    text = _inbound.html_to_text(
        "<style>p{color:red}</style><p>Hello</p><script>alert(1)</script>")
    assert text == "Hello"


def test_a_message_graph_returned_half_of_does_not_raise():
    """A mailbox is not a contract. One unusual message must not stop a read."""
    parsed = _inbound.parse_message({"id": "x"})
    assert parsed.graph_id == "x"
    assert parsed.sender == ""
    assert parsed.body_text == ""


# --------------------------------------------------------------------------- #
# Whose reply is it
# --------------------------------------------------------------------------- #

def test_a_reply_in_the_same_conversation_is_matched():
    found = _inbound.correlate(_inbound.parse_message(message()), [target()])
    assert found.matched
    assert found.case_ref == "ONB-2026-0001"
    assert _inbound.BASIS_CONVERSATION in found.bases


def test_a_reply_that_names_the_message_it_answers_is_matched():
    payload = message(conversationId="a-different-conversation")
    found = _inbound.correlate(_inbound.parse_message(payload), [target()])
    assert found.matched
    assert found.bases == [_inbound.BASIS_IN_REPLY_TO, _inbound.BASIS_CASE_REF,
                           _inbound.BASIS_SENDER]


def test_the_case_reference_in_a_subject_line_is_enough_on_its_own():
    payload = message(conversationId="other", internetMessageHeaders=[])
    found = _inbound.correlate(
        _inbound.parse_message(payload),
        [target(conversation_id="", internet_message_id="", contacts=())])
    assert found.matched
    assert found.bases == [_inbound.BASIS_CASE_REF]


def test_a_known_sender_alone_is_never_enough():
    """The defect this whole module exists to prevent.

    One contact can be on several onboardings, and a shared address is one
    mailbox for a whole firm. Matching on it would write a client's answers
    onto whichever case happened to be found first.
    """
    payload = message(conversationId="other", subject="Some files for you",
                      internetMessageHeaders=[])
    found = _inbound.correlate(_inbound.parse_message(payload), [target()])
    assert not found.matched
    assert found.case_ref == ""
    assert found.candidates == ["ONB-2026-0001"]
    assert "not enough" in found.note


def test_two_cases_with_an_equal_claim_are_refused_rather_than_split():
    payload = message(conversationId="other", subject="Files",
                      internetMessageHeaders=[])
    both = [target(case_ref="ONB-2026-0001", conversation_id="",
                   internet_message_id=""),
            target(case_ref="ONB-2026-0002", conversation_id="",
                   internet_message_id="")]
    found = _inbound.correlate(_inbound.parse_message(payload), both)
    assert not found.matched
    assert found.candidates == ["ONB-2026-0001", "ONB-2026-0002"]


def test_a_message_about_nothing_we_sent_is_matched_to_nothing():
    payload = message(conversationId="other", subject="Lunch?",
                      internetMessageHeaders=[],
                      **{"from": {"emailAddress": {"address": "spam@x.example"}}})
    found = _inbound.correlate(_inbound.parse_message(payload), [target()])
    assert not found.matched
    assert found.candidates == []


def test_the_strongest_evidence_decides_between_two_cases():
    """A conversation beats a subject line. A client who replies to the right
    thread while quoting the wrong reference goes to the right case."""
    payload = message(subject="RE: Onboarding (ONB-2026-0002)")
    both = [target(case_ref="ONB-2026-0001"),
            target(case_ref="ONB-2026-0002", conversation_id="",
                   internet_message_id="", contacts=())]
    found = _inbound.correlate(_inbound.parse_message(payload), both)
    assert found.case_ref == "ONB-2026-0001"


# --------------------------------------------------------------------------- #
# Reading the folder
# --------------------------------------------------------------------------- #

def test_the_reader_asks_graph_for_the_configured_folder():
    graph = FakeGraph(inbox=[message()])
    config = _config.load_inbound(inbound_env(
        TRAKT_MAIL_INBOUND_FOLDER="ClientReplies"))
    _inbound.read_folder(reader(graph), config)
    read = [r for r in graph.requests if "/mailFolders/" in r.full_url]
    assert read, "the folder was never listed"
    assert "ClientReplies" in read[0].full_url
    assert "isRead+eq+false" in read[0].full_url


def test_the_reader_never_leaves_the_graph_host():
    graph = FakeGraph(inbox=[message()])
    config = _config.load_inbound(inbound_env())
    _inbound.read_folder(reader(graph), config)
    for request in graph.requests:
        assert request.full_url.startswith(
            ("https://graph.microsoft.com/", "https://login.microsoftonline.com/"))


def test_an_attachment_arrives_as_bytes():
    graph = FakeGraph(
        inbox=[message(hasAttachments=True)],
        attachments={"AAMkAGI2-reply-1": [{
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": "loan_tape.csv", "contentType": "text/csv", "size": 11,
            "contentBytes": base64.b64encode(b"id,balance\n").decode("ascii"),
        }]})
    config = _config.load_inbound(inbound_env())
    found = _inbound.read_folder(reader(graph), config)
    assert found[0].attachments[0].name == "loan_tape.csv"
    assert found[0].attachments[0].data == b"id,balance\n"


def test_an_attachment_that_is_not_a_file_is_not_treated_as_one():
    """An item attachment is a forwarded message and a reference attachment is
    a link. Neither carries bytes here, and putting either in the sandbox as a
    client data file would store something that is not what it claims to be."""
    graph = FakeGraph(
        inbox=[message(hasAttachments=True)],
        attachments={"AAMkAGI2-reply-1": [
            {"@odata.type": "#microsoft.graph.itemAttachment",
             "name": "Forwarded.msg", "size": 900},
            {"@odata.type": "#microsoft.graph.referenceAttachment",
             "name": "OneDrive link", "size": 0},
        ]})
    config = _config.load_inbound(inbound_env())
    found = _inbound.read_folder(reader(graph), config)
    assert found[0].attachments == []


def test_an_oversize_attachment_is_reported_rather_than_downloaded():
    graph = FakeGraph(
        inbox=[message(hasAttachments=True)],
        attachments={"AAMkAGI2-reply-1": [{
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": "huge.csv", "contentType": "text/csv", "size": 999_999,
            "contentBytes": base64.b64encode(b"x" * 10).decode("ascii"),
        }]})
    config = _config.load_inbound(inbound_env(
        TRAKT_MAIL_INBOUND_MAX_ATTACHMENT_BYTES="1000"))
    found = _inbound.read_folder(reader(graph), config)
    attachment = found[0].attachments[0]
    assert attachment.oversize is True
    assert attachment.data == b""


def test_a_sender_outside_the_allow_list_is_dropped_before_its_files_are_read():
    graph = FakeGraph(inbox=[message(hasAttachments=True)],
                      attachments={"AAMkAGI2-reply-1": [{
                          "@odata.type": "#microsoft.graph.fileAttachment",
                          "name": "x.csv", "size": 1, "contentBytes": "eA=="}]})
    config = _config.load_inbound(inbound_env(
        TRAKT_MAIL_SENDER_ALLOWLIST="@trusted.example"))
    assert _inbound.read_folder(reader(graph), config) == []
    assert not any("/attachments" in r.full_url for r in graph.requests)


def test_a_domain_suffix_on_the_sender_allow_list_admits_the_whole_domain():
    graph = FakeGraph(inbox=[message()])
    config = _config.load_inbound(inbound_env(
        TRAKT_MAIL_SENDER_ALLOWLIST="@erm.example"))
    assert len(_inbound.read_folder(reader(graph), config)) == 1


def test_attachments_that_cannot_be_read_do_not_lose_the_message():
    """A client replied. That is worth reporting even when Trakt could not read
    what they attached — "they sent something unreadable" is actionable and
    silence is not."""
    class Failing(FakeGraph):
        def __call__(self, request, timeout):
            if "/attachments" in request.full_url:
                raise GraphMailError("no", retryable=False, status=403)
            return super().__call__(request, timeout)

    graph = Failing(inbox=[message(hasAttachments=True)])
    config = _config.load_inbound(inbound_env())
    found = _inbound.read_folder(reader(graph), config)
    assert len(found) == 1
    assert found[0].attachments == []


def test_marking_a_message_read_never_deletes_or_moves_it():
    graph = FakeGraph(inbox=[message()])
    assert reader(graph).mark_read("AAMkAGI2-reply-1") is True
    assert graph.marked_read == ["AAMkAGI2-reply-1"]
    assert not any(r.get_method() in ("DELETE",) for r in graph.requests)


def test_a_mailbox_that_cannot_be_read_says_so_rather_than_reporting_empty():
    graph = FakeGraph(inbox=[message()], inbox_error=403)
    config = _config.load_inbound(inbound_env())
    with pytest.raises(GraphMailError):
        _inbound.read_folder(reader(graph), config)


# --------------------------------------------------------------------------- #
# Fail closed
# --------------------------------------------------------------------------- #

def test_reading_is_off_unless_it_is_explicitly_turned_on():
    with pytest.raises(_config.MailNotConfigured) as excinfo:
        _config.load_inbound({})
    assert excinfo.value.disabled


def test_the_sending_switch_does_not_turn_reading_on():
    """Two capabilities, two switches. A deployment that sends packs does not
    thereby acquire the ability to read a mailbox."""
    from .conftest import mail_env
    with pytest.raises(_config.MailNotConfigured):
        _config.load_inbound(mail_env())


def test_the_reading_switch_does_not_turn_sending_on():
    with pytest.raises(_config.MailNotConfigured):
        _config.load(inbound_env())


def test_reading_with_no_credentials_names_what_is_missing():
    with pytest.raises(_config.MailNotConfigured) as excinfo:
        _config.load_inbound({"TRAKT_MAIL_INBOUND_ENABLED": "true"})
    assert set(excinfo.value.missing) == {
        "TRAKT_MAIL_TENANT_ID", "TRAKT_MAIL_CLIENT_ID",
        "TRAKT_MAIL_CLIENT_SECRET", "TRAKT_MAIL_MAILBOX"}


def test_the_secret_is_never_in_what_diagnostics_print():
    config = _config.load_inbound(inbound_env())
    printed = repr(config.redacted())
    assert CLIENT_SECRET not in printed
    assert "'client_secret': 'set'" in printed
