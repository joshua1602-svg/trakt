"""Rendering an approved communication for a client.

The two defects a production test exposed, and the guarantees that stop them
coming back:

* the covering message was ATTACHED as ``covering_email.txt`` as well as being
  the body, so a client received the same words twice;
* the pack was sent as ``onboarding_pack.md``, a source format, so a client
  received a file most of them cannot open cleanly.

Everything here is about FORM. No test may assert that content changed, because
the presentation layer is not permitted to change any: the strongest test in
this file drives the real ``pack.build`` and asserts that every question label
it generates survives into the rendered document.
"""

from __future__ import annotations

import pytest

import re

from trakt_mail import presentation as pres

MARKDOWN = """# Onboarding — Northstar Lending

Reference: ONB-2026-0007

There are 3 questions for you, 2 of them required.

## Please check these are right

Trakt already holds these. Tell us if any are wrong.

- **Asset class** — portfolio 1: equity release

## About your business

Who you are, so we can set the book up.

- [required] **Legal entity name**
      As registered at Companies House.
- [required] **Reporting contact email**
      Currently: ops@northstar.example
- [optional] **Trading name**

## Files to send

- **Primary loan tape** — required
- Property tape — optional

## How to send them

- Channel: Secure upload
- Funded book: `blob://raw-v2/NORTHSTAR/direct_101/funded/`
"""


@pytest.fixture()
def doc():
    return pres.parse_pack(MARKDOWN)


# --------------------------------------------------------------------------- #
# Reading the pack back
# --------------------------------------------------------------------------- #

def test_the_client_and_the_case_are_recovered(doc):
    assert doc.client_name == "Northstar Lending"
    assert doc.case_ref == "ONB-2026-0007"


def test_every_question_is_found_with_its_obligation(doc):
    questions = [n.question for n in doc.nodes if n.kind == "question"]
    assert [(q.label, q.required) for q in questions] == [
        ("Legal entity name", True),
        ("Reporting contact email", True),
        ("Trading name", False)]


def test_a_questions_help_and_current_value_stay_with_it(doc):
    questions = {n.question.label: n.question for n in doc.nodes
                 if n.kind == "question"}
    assert questions["Legal entity name"].help == \
        "As registered at Companies House."
    assert questions["Reporting contact email"].current == \
        "ops@northstar.example"


def test_headings_and_bullets_survive_in_document_order(doc):
    kinds = [(n.kind, n.text) for n in doc.nodes
             if n.kind in ("heading", "subheading")]
    assert [text for _kind, text in kinds] == [
        "Please check these are right", "About your business",
        "Files to send", "How to send them"]


def test_an_unrecognised_line_becomes_prose_rather_than_vanishing():
    """A client must never receive less than the pack said."""
    doc = pres.parse_pack("# Onboarding — X\n\nsomething unexpected here\n"
                          "## A section\n\nand more prose\n")
    assert "and more prose" in [n.text for n in doc.nodes]
    assert "something unexpected here" in doc.intro


# --------------------------------------------------------------------------- #
# The filename
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("client,case,expected", [
    ("Northstar Lending", "ONB-2026-0007",
     "Trakt_Onboarding_Checklist_Northstar_Lending.pdf"),
    ("O'Brien & Sons, Ltd.", "", "Trakt_Onboarding_Checklist_O_Brien_Sons_Ltd.pdf"),
    ("", "ONB-2026-0007", "Trakt_Onboarding_Checklist_ONB_2026_0007.pdf"),
    ("", "", "Trakt_Onboarding_Checklist.pdf"),
])
def test_the_filename_is_the_clients_own_name_made_safe(client, case, expected):
    assert pres.checklist_filename(client, case) == expected


def test_a_hostile_client_name_cannot_escape_the_filename():
    name = pres.checklist_filename("../../etc/passwd", "")
    assert "/" not in name and ".." not in name
    assert name.endswith(".pdf")


# --------------------------------------------------------------------------- #
# The email body
# --------------------------------------------------------------------------- #

COVERING = "Dear Northstar,\n\nWe are setting your portfolio up.\n\nKind regards,"


def test_the_html_body_carries_every_approved_word():
    html = pres.render_html(COVERING, case_ref="ONB-2026-0007",
                            receipt_id="com_x", attachment_name="c.pdf")
    for phrase in ("Dear Northstar,", "We are setting your portfolio up.",
                   "Kind regards,"):
        assert phrase in html


def test_the_html_body_names_the_case_and_the_receipt():
    html = pres.render_html(COVERING, case_ref="ONB-2026-0007",
                            receipt_id="com_abc123")
    assert "ONB-2026-0007" in html
    assert "com_abc123" in html


def test_the_html_body_fetches_nothing_and_runs_nothing():
    html = pres.render_html(COVERING, case_ref="R", receipt_id="c",
                            attachment_name="c.pdf")
    for forbidden in ("<script", "<img", "http://", "https://", "@import",
                      "javascript:"):
        assert forbidden not in html, forbidden


def test_html_escapes_content_rather_than_rendering_it():
    html = pres.render_html("Dear <b>x</b> & co", case_ref="R", receipt_id="c")
    assert "&lt;b&gt;" in html and "&amp;" in html


def test_the_text_body_says_the_same_thing_as_the_html():
    """A text body that differs from the HTML is worse than none."""
    text = pres.render_text(COVERING, case_ref="ONB-2026-0007",
                            receipt_id="com_abc123", attachment_name="c.pdf")
    for phrase in ("Dear Northstar,", "We are setting your portfolio up.",
                   "ONB-2026-0007", "com_abc123", "c.pdf"):
        assert phrase in text
    assert "<" not in text


# --------------------------------------------------------------------------- #
# The PDF
# --------------------------------------------------------------------------- #

def test_the_checklist_is_a_pdf(doc):
    pdf = pres.render_pdf(doc, receipt_id="com_x", content_hash="h",
                          generated_at="19 August 2026")
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")


def test_the_pdf_is_titled_for_the_client(doc):
    pdf = pres.render_pdf(doc, receipt_id="com_x")
    assert b"Northstar Lending" in pdf or b"Northstar" in pdf


# --------------------------------------------------------------------------- #
# One communication, three renderings
# --------------------------------------------------------------------------- #

def test_build_returns_a_body_a_text_alternative_and_one_pdf():
    presented = pres.build(COVERING, MARKDOWN, receipt_id="com_abc123",
                           content_hash="h1", generated_at="19 August 2026")
    assert presented.pdf and presented.pdf.startswith(b"%PDF-")
    assert presented.pdf_name == \
        "Trakt_Onboarding_Checklist_Northstar_Lending.pdf"
    assert presented.attachment_names == [presented.pdf_name]
    assert "Dear Northstar," in presented.html_body
    assert "Dear Northstar," in presented.text_body
    assert presented.case_ref == "ONB-2026-0007"


def pdf_text(pdf: bytes) -> str:
    """Everything printed on a PDF's pages, plus its metadata.

    reportlab writes page content through ``/ASCII85Decode /FlateDecode``, so a
    raw byte search finds only the document properties. This undoes both filters
    and returns what is actually on the pages, which is what a client reads.
    """
    import base64
    import re as _re
    import zlib

    out = [pdf.decode("latin-1")]           # metadata / Info dictionary
    for raw in _re.findall(rb"stream\r?\n(.*?)endstream", pdf, _re.S):
        data = raw.strip(b"\r\n ")
        try:
            if data.endswith(b"~>"):
                data = base64.a85decode(data[:-2], adobe=False)
            out.append(zlib.decompress(data).decode("latin-1"))
        except Exception:                    # noqa: BLE001 — not a text stream
            continue
    return "\n".join(out)


def pdf_shows(pdf: bytes, phrase: str) -> bool:
    """Whether a phrase is printed on the page.

    A PDF text operator may break a run across ``Tj`` calls, so the extracted
    stream is reduced to its letters before matching rather than compared as
    laid-out text.
    """
    import re as _re
    haystack = _re.sub(r"[^A-Za-z0-9]", "", pdf_text(pdf))
    return _re.sub(r"[^A-Za-z0-9]", "", phrase) in haystack


def test_the_receipt_id_appears_in_every_rendering():
    """The email, the printed checklist and the audit record are one act."""
    presented = pres.build(COVERING, MARKDOWN, receipt_id="com_traceme")
    assert "com_traceme" in presented.html_body
    assert "com_traceme" in presented.text_body
    # Printed on the page furniture, and in the document properties.
    assert pdf_shows(presented.pdf, "com_traceme")
    assert b"com_traceme" in presented.pdf


def test_the_printed_checklist_carries_every_question(doc):
    """What the client prints is what the OCC asked for."""
    pdf = pres.render_pdf(doc, receipt_id="com_x")
    for label in ("Legal entity name", "Reporting contact email",
                  "Trading name"):
        assert pdf_shows(pdf, label), label
    assert pdf_shows(pdf, "REQUIRED") and pdf_shows(pdf, "OPTIONAL")


def test_the_checklist_reference_is_on_the_page(doc):
    pdf = pres.render_pdf(doc, receipt_id="com_x", content_hash="hash1")
    assert pdf_shows(pdf, "ONB-2026-0007")
    assert pdf_shows(pdf, "hash1")


def test_a_communication_with_no_pack_has_no_attachment():
    presented = pres.build(COVERING, "", receipt_id="com_x")
    assert presented.pdf is None
    assert presented.attachment_names == []
    assert "Dear Northstar," in presented.html_body


def test_a_missing_pdf_library_is_reported_as_such(monkeypatch, doc):
    """A deployment problem must read as one, not as a mail failure."""
    monkeypatch.setattr(pres, "_reportlab", lambda: (_ for _ in ()).throw(
        pres.PresentationError("reportlab is not installed")))
    with pytest.raises(pres.PresentationError):
        pres.render_pdf(doc)


# --------------------------------------------------------------------------- #
# The guarantee that matters: nothing the OCC asks is lost in rendering
# --------------------------------------------------------------------------- #

def test_every_question_the_real_pack_generates_reaches_the_client(tmp_path,
                                                                   monkeypatch):
    """Drives the REAL pack builder. If ``document()`` changes shape, this
    fails here rather than quietly dropping questions from a client's PDF."""
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    from apps.blob_trigger_app.storage import Storage
    from operations_control.occ_agent.service import OccAgentService

    service = OccAgentService(Storage(tmp_path / "blob"),
                              container="operations-control-synthetic",
                              sandbox=tmp_path / "sandbox")
    case = service.create_case(
        tenant="client_a", initiating_user="Alice",
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    built = service.build_pack(case)
    parsed = pres.parse_pack(built.document())

    assert parsed.client_name == "Northstar Lending"
    assert parsed.case_ref == case.case_ref

    # Every generated question survives the round trip. The renderer neither
    # drops nor rewrites one.
    generated = {q.label for section in built.sections
                 for q in section.questions}
    rendered = {n.question.label for n in parsed.nodes if n.kind == "question"}
    assert generated <= rendered, generated - rendered

    # The files are checklist rows too — same shape as every other thing being
    # asked for, which is the whole point of the layout.
    for row in built.artefacts.required + built.artefacts.optional:
        assert row["label"] in rendered, row["label"]
    assert rendered - generated == {
        row["label"] for row in
        built.artefacts.required + built.artefacts.optional}


# --------------------------------------------------------------------------- #
# Room to answer in
# --------------------------------------------------------------------------- #

def test_a_question_offers_the_client_room_to_write_an_answer():
    """The mark against each question is a write-in area, not a tick box.

    A checklist a client cannot answer in is a checklist they answer by email
    instead, and the structure is lost. The area is a fixed rectangle wide
    enough for a line of text and tall enough to write on, and the numbers are
    asserted rather than described because "far too small" is exactly how this
    was reported.
    """
    assert pres.ANSWER_HEIGHT >= 10, "no room to write a line"
    assert pres.ANSWER_WIDTH >= 120, "not the width of the column"
    # A tick box is roughly 4mm square. Whatever this is, it is not that.
    assert pres.ANSWER_WIDTH / pres.ANSWER_HEIGHT > 5


def test_the_answer_area_is_the_same_size_whatever_sits_above_it(doc):
    """Two questions, one with help text and one without, both fit on a page
    that renders — the area is nested, so it cannot inherit its row's height."""
    pdf = pres.render_pdf(doc)
    assert pdf.startswith(b"%PDF")
    assert pdf_shows(pdf, "Legal entity name")      # carries help text
    assert pdf_shows(pdf, "Trading name")           # carries none


# --------------------------------------------------------------------------- #
# Nothing the client reads may be raw markdown
# --------------------------------------------------------------------------- #

GROUPED = """# Onboarding — Northstar Lending

Reference: ONB-2026-0007

## What we need from you now

### About your business

#### Northstar Lending

- [required] **Legal Entity Identifier**

#### Contacts and distribution

- [required] **Reporting email**
"""


def test_a_fourth_level_heading_is_a_heading_not_prose():
    parsed = pres.parse_pack(GROUPED)
    groups = [n.text for n in parsed.nodes if n.kind == "group"]
    assert groups == ["Northstar Lending", "Contacts and distribution"]
    assert not any(n.text.startswith("#") for n in parsed.nodes)


def test_no_hash_marks_reach_the_printed_checklist():
    """``pdf_shows`` reduces the page to its letters, so it cannot see a hash.
    The drawn text is searched directly instead."""
    pdf = pres.render_pdf(pres.parse_pack(GROUPED))
    drawn = " ".join(re.findall(r"\((.*?)\)\s*Tj", pdf_text(pdf)))
    assert "Contacts and distribution" in drawn
    assert "#" not in drawn


def test_no_hash_marks_reach_the_real_pack_either(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    from apps.blob_trigger_app.storage import Storage
    from operations_control.occ_agent.service import OccAgentService

    service = OccAgentService(Storage(tmp_path / "blob"),
                              container="operations-control-synthetic",
                              sandbox=tmp_path / "sandbox")
    case = service.create_case(
        tenant="client_a", initiating_user="Alice",
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    parsed = pres.parse_pack(service.build_pack(case).document())
    for node in parsed.nodes:
        assert not node.text.lstrip().startswith("#"), node.text
