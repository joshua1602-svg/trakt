"""trakt_mail.presentation — how an approved pack is put in front of a client.

Presentation only. Nothing here decides what is asked, who it goes to, or
whether it may be sent: it receives an already-approved covering message and an
already-generated pack, and chooses the *format* the client receives them in.
Every question, every label and every ordering comes from
``operations_control.occ_agent.pack.OnboardingPack.document()`` and is
reproduced, never edited.

Three renderings of one approved communication:

* **HTML body** — the covering message, and only the covering message. It used
  also to be attached as ``covering_email.txt``, so a client received the same
  words twice, once as an attachment they had to open to discover it said what
  the email already said.
* **plain-text body** — the same covering message as text, for a client that
  cannot render HTML. See the note on Graph's body model below.
* **PDF checklist** — the pack, as a document a client can read on a phone,
  print, and tick through. Markdown is a source format, not a client-facing
  one; ``onboarding_pack.md`` remains in the case store as the auditable
  original and is simply no longer what gets sent.

All three carry the same case reference and the same communication receipt id,
so the email in the client's inbox, the PDF they print and the record in the
audit chain are provably one act.

A note on the plain-text body
-----------------------------
Microsoft Graph's ``message`` resource carries exactly ONE body, with one
``contentType``. It has no representation for ``multipart/alternative``, so a
message cannot be given both an HTML and a text part through the JSON API — that
requires uploading a MIME message instead, which would mean changing the send
call itself. :func:`render_text` is therefore produced and returned on every
render, is what a caller sends when HTML is not wanted, and is stored as the
textual record of what was communicated. It is deliberately not claimed to be a
MIME alternative part, because it is not one.

Why the markdown is parsed rather than the structured pack read
---------------------------------------------------------------
The communication adapter receives the pack as a stored artefact reference, not
as an ``OnboardingPack``. Parsing is confined to the constructs
``document()`` itself emits — it is this repository reading its own generator's
output, not a general markdown implementation — and a test drives the real
``pack.build`` to assert every question label survives into the PDF, so a change
to ``document()`` fails loudly here rather than quietly dropping questions from
a client's checklist.
"""

from __future__ import annotations

import html as _html
import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# The Operations Control Centre's own design tokens.
#
# Taken from frontend/operations-control-ui (index.css + the screens), so a
# client's email and checklist look like the product the operator is looking at
# rather than like a second brand: Inter over a system stack, the warm stone
# neutrals the dashboard is built on, and blue-600 as the one accent.
#
# No images and no web fonts. A remote logo renders as a broken box in most mail
# clients until the reader clicks "download pictures", and a @font-face is
# stripped outright — both look less professional than restraint. Inter is named
# first and degrades to the same system stack the dashboard falls back to.
# --------------------------------------------------------------------------- #
FONT_STACK = ('Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", '
              "Roboto, Helvetica, Arial, sans-serif")

INK = "#1c1917"          # stone-900 — the dashboard's body colour
STRONG = "#292524"       # stone-800
BODY = "#44403c"         # stone-700
MUTED = "#78716c"        # stone-500
RULE = "#e7e5e4"         # stone-200 — every border in the dashboard
SOFT_RULE = "#d6d3d1"    # stone-300
PALE = "#fafaf9"         # stone-50 — the dashboard background
ACCENT = "#2563eb"       # blue-600 — the primary action colour
ACCENT_PALE = "#eff6ff"  # blue-50

#: Corner radius. The dashboard uses rounded-xl throughout.
RADIUS = "12px"

WORDMARK = "TRAKT"
SENDER_NAME = "Trakt Operations"

#: The pack artefact that becomes the client's PDF. Any ``.md`` artefact is
#: treated this way; the name is the one the OCC writes today.
PACK_MARKDOWN = "onboarding_pack.md"

#: Attached, historically, alongside a body containing the same words. It is the
#: body; it is not a document. Never attached.
COVERING_TEXT = "covering_email.txt"

MAX_FILENAME_STEM = 60

#: Side of the answer box, and the height of the REQUIRED/OPTIONAL tag beside
#: it, in millimetres. One number for both so a checklist row always presents
#: the same two marks at the same size however much text sits to their right.
BOX_SIZE = 4.4


class PresentationError(Exception):
    """A rendering could not be produced. The caller must not send half of one."""


# --------------------------------------------------------------------------- #
# The parsed pack
# --------------------------------------------------------------------------- #

@dataclass
class Question:
    """One thing the client is being asked for."""

    label: str
    required: bool = False
    help: str = ""
    current: str = ""


@dataclass
class Node:
    """One element of the pack, in document order.

    ``kind`` is one of ``heading``, ``subheading``, ``paragraph``, ``question``
    or ``bullet``.
    """

    kind: str
    text: str = ""
    note: str = ""
    question: Optional[Question] = None


@dataclass
class PackDocument:
    """``onboarding_pack.md``, read back into the shape it was written from."""

    client_name: str = ""
    case_ref: str = ""
    intro: List[str] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)

    @property
    def question_count(self) -> int:
        return len([n for n in self.nodes if n.kind == "question"])


_H1 = re.compile(r"^#\s+(.*)$")
_H2 = re.compile(r"^##\s+(.*)$")
_H3 = re.compile(r"^###\s+(.*)$")
_MARKED = re.compile(r"^-\s+\[(required|optional)\]\s+(.*)$")
_BULLET = re.compile(r"^-\s+(.*)$")
_REFERENCE = re.compile(r"^Reference:\s*(.*)$")
_TITLE = re.compile(r"^Onboarding\s+[—-]\s*(.*)$")
#: ``document()`` indents a question's help and current value by six spaces.
_CONTINUATION = re.compile(r"^ {4,}(.*)$")
_STRONG = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")


def _strip_marks(text: str) -> str:
    """Bold and code markers removed, for a context that cannot show them."""
    return _CODE.sub(r"\1", _STRONG.sub(r"\1", text)).strip()


def parse_pack(markdown: str) -> PackDocument:
    """Read ``pack.document()`` output back into its parts.

    Anything unrecognised becomes a paragraph rather than being dropped: a
    client must never silently receive less than the pack said, and an
    unexpected line is better shown plainly than discarded.
    """
    doc = PackDocument()
    lines = (markdown or "").replace("\r\n", "\n").split("\n")
    seen_heading = False
    last_question: Optional[Question] = None

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        continuation = _CONTINUATION.match(raw)
        if continuation and last_question is not None:
            text = _strip_marks(continuation.group(1))
            if text.lower().startswith("currently:"):
                last_question.current = text.split(":", 1)[1].strip()
            elif last_question.help:
                last_question.help = f"{last_question.help} {text}"
            else:
                last_question.help = text
            continue
        last_question = None

        h1 = _H1.match(line)
        if h1:
            title = _strip_marks(h1.group(1))
            named = _TITLE.match(title)
            doc.client_name = named.group(1).strip() if named else title
            continue

        h3 = _H3.match(line)
        if h3:
            doc.nodes.append(Node("subheading", _strip_marks(h3.group(1))))
            seen_heading = True
            continue

        h2 = _H2.match(line)
        if h2:
            doc.nodes.append(Node("heading", _strip_marks(h2.group(1))))
            seen_heading = True
            continue

        reference = _REFERENCE.match(line)
        if reference and not doc.case_ref:
            doc.case_ref = reference.group(1).strip()
            continue

        marked = _MARKED.match(line)
        if marked:
            question = Question(label=_strip_marks(marked.group(2)),
                                required=marked.group(1) == "required")
            doc.nodes.append(Node("question", question=question))
            last_question = question
            continue

        bullet = _BULLET.match(line)
        if bullet:
            doc.nodes.append(_bullet_node(bullet.group(1)))
            continue

        # Prose. Before the first heading it is the pack's opening statement.
        (doc.nodes if seen_heading else doc.intro).append(
            Node("paragraph", _strip_marks(line)) if seen_heading
            else _strip_marks(line))
    return doc


def _bullet_node(body: str) -> Node:
    """A bullet, split into its emphasised part and the rest.

    ``document()`` writes three shapes — ``**Label** — required``,
    ``**Label**: value`` and ``Key: value`` — and all three read best as a term
    and a detail rather than as one run of text.
    """
    strong = _STRONG.search(body)
    if strong:
        label = strong.group(1).strip()
        rest = (body[:strong.start()] + body[strong.end():]).strip()
        rest = rest.lstrip("—-:").strip()
        return Node("bullet", _strip_marks(label), _strip_marks(rest))
    if ":" in body:
        label, rest = body.split(":", 1)
        return Node("bullet", _strip_marks(label), _strip_marks(rest))
    return Node("bullet", _strip_marks(body))


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #

def checklist_filename(client_name: str = "", case_ref: str = "") -> str:
    """``Trakt_Onboarding_Checklist_Northstar_Lending.pdf``.

    The client's own name, reduced to characters that survive every mail client,
    file system and download manager between here and them. A pack with no
    client name recorded falls back to the case reference rather than producing
    a file called ``Trakt_Onboarding_Checklist_.pdf``.
    """
    source = (client_name or "").strip() or (case_ref or "").strip()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")[:MAX_FILENAME_STEM]
    return f"Trakt_Onboarding_Checklist_{stem}.pdf" if stem \
        else "Trakt_Onboarding_Checklist.pdf"


# --------------------------------------------------------------------------- #
# The email body
# --------------------------------------------------------------------------- #

def _paragraphs(text: str) -> List[str]:
    """The covering message split on blank lines, as written."""
    blocks: List[str] = []
    current: List[str] = []
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def render_html(covering: str, *, case_ref: str = "", receipt_id: str = "",
                attachment_name: str = "") -> str:
    """The covering message as an HTML email body.

    Built to look like the Operations Control Centre the operator is sitting in
    front of: Inter over the same system stack, the stone neutrals, blue-600 for
    the one accent, and the dashboard's rounded-xl card. Tables and inline
    styles throughout, because that is what Outlook's Word rendering engine
    supports; a 600px column, because that is what survives a reading pane.
    """
    body = "".join(
        f'<tr><td style="padding:0 0 18px 0;font-family:{FONT_STACK};'
        f'font-size:15px;line-height:1.65;color:{BODY};">'
        f'{_html.escape(block)}</td></tr>'
        for block in _paragraphs(covering))

    attachment_row = ""
    if attachment_name:
        attachment_row = (
            f'<tr><td style="padding:6px 0 4px 0;">'
            f'<table role="presentation" cellpadding="0" cellspacing="0" '
            f'border="0" width="100%" style="background:{ACCENT_PALE};'
            f'border:1px solid #dbeafe;border-radius:{RADIUS};">'
            f'<tr><td style="padding:14px 18px;font-family:{FONT_STACK};'
            f'font-size:14px;line-height:1.5;color:{STRONG};">'
            f'<span style="color:{ACCENT};font-weight:600;">Attached</span>'
            f'<span style="color:{MUTED};">&nbsp;&#183;&nbsp;</span>'
            f'{_html.escape(attachment_name)}'
            f'</td></tr></table></td></tr>')

    footer_bits = [b for b in (f"Reference {_html.escape(case_ref)}"
                               if case_ref else "",
                               f"Communication {_html.escape(receipt_id)}"
                               if receipt_id else "") if b]
    footer = " &nbsp;&#183;&nbsp; ".join(footer_bits)

    return f"""<!-- Trakt onboarding communication -->
<table role="presentation" cellpadding="0" cellspacing="0" border="0" \
width="100%" style="background:{PALE};margin:0;padding:0;">
 <tr><td align="center" style="padding:28px 12px;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" \
width="600" style="width:100%;max-width:600px;background:#ffffff;\
border:1px solid {RULE};border-radius:{RADIUS};">
   <tr><td style="padding:26px 32px 0 32px;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" \
width="100%">
     <tr>
      <td style="font-family:{FONT_STACK};font-size:13px;font-weight:700;\
letter-spacing:2.2px;color:{INK};">{WORDMARK}</td>
      <td align="right" style="font-family:{FONT_STACK};font-size:12px;\
color:{MUTED};">Onboarding</td>
     </tr>
    </table>
   </td></tr>
   <tr><td style="padding:16px 32px 0 32px;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" \
width="100%"><tr><td style="border-top:1px solid {RULE};font-size:0;\
line-height:0;">&nbsp;</td></tr></table>
   </td></tr>
   <tr><td style="padding:22px 32px 0 32px;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" \
width="100%">
     {body}
     {attachment_row}
    </table>
   </td></tr>
   <tr><td style="padding:22px 32px 0 32px;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" \
width="100%"><tr><td style="border-top:1px solid {RULE};font-size:0;\
line-height:0;">&nbsp;</td></tr></table>
   </td></tr>
   <tr><td style="padding:14px 32px 26px 32px;font-family:{FONT_STACK};\
font-size:12px;line-height:1.7;color:{MUTED};">{footer}<br />Sent by \
{SENDER_NAME}. Reply to this message and your answer stays with this \
onboarding.</td></tr>
  </table>
 </td></tr>
</table>"""


def render_text(covering: str, *, case_ref: str = "", receipt_id: str = "",
                attachment_name: str = "") -> str:
    """The same covering message as plain text.

    Sent when HTML is not wanted, and kept as the textual record of what was
    communicated. Deliberately the same words in the same order as the HTML —
    a text body that says something different from the HTML body is worse than
    having no text body at all.
    """
    lines = [WORDMARK, ""]
    lines += ["\n".join(_paragraphs(covering))]
    if attachment_name:
        lines += ["", f"Attached: {attachment_name}"]
    trailer = [b for b in (f"Reference {case_ref}" if case_ref else "",
                           f"Communication {receipt_id}" if receipt_id else "")
               if b]
    lines += ["", "-" * 56]
    if trailer:
        lines.append("  ·  ".join(trailer))
    lines.append(f"Sent by {SENDER_NAME}. Reply to this message and your "
                 "answer stays with this onboarding.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The PDF checklist
# --------------------------------------------------------------------------- #

def _reportlab():
    """Imported on use, and reported clearly when absent.

    A missing PDF library must not read as a mail failure: it is a deployment
    problem with a named fix, and the caller turns this into a refusal that
    leaves the pack approved and re-sendable.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as _canvas
        from reportlab.platypus import (
            HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
            Spacer, Table, TableStyle,
        )
    except Exception as exc:  # noqa: BLE001
        raise PresentationError(
            "the PDF renderer (reportlab) is not installed in this "
            f"environment ({exc})") from None
    return {
        "colors": colors, "TA_LEFT": TA_LEFT, "A4": A4,
        "ParagraphStyle": ParagraphStyle, "mm": mm, "canvas": _canvas,
        "HRFlowable": HRFlowable, "KeepTogether": KeepTogether,
        "PageBreak": PageBreak, "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate, "Spacer": Spacer,
        "Table": Table, "TableStyle": TableStyle,
    }


def _rich(text: str) -> str:
    """Plain text as reportlab's mini-markup: escaped, then re-emphasised."""
    escaped = _html.escape(text or "", quote=False)
    escaped = _STRONG.sub(r"<b>\1</b>", escaped)
    return _CODE.sub(r'<font face="Courier">\1</font>', escaped)


def render_pdf(doc: PackDocument, *, receipt_id: str = "",
               content_hash: str = "", generated_at: str = "") -> bytes:
    """The pack as a checklist the client can read, print and work through.

    Every page carries the case reference and the communication receipt, so a
    printed page separated from its email is still traceable to one OCC act.
    """
    rl = _reportlab()
    mm, colors = rl["mm"], rl["colors"]
    Paragraph, Spacer, Table = rl["Paragraph"], rl["Spacer"], rl["Table"]

    # The dashboard's tokens again, so a printed checklist and the screen an
    # operator is looking at are recognisably the same product.
    navy = colors.HexColor(INK)          # headings — stone-900
    slate = colors.HexColor(BODY)        # secondary text — stone-700
    accent = colors.HexColor(ACCENT)     # the one accent — blue-600
    rule, pale = colors.HexColor(RULE), colors.HexColor(PALE)
    ink, muted = colors.HexColor(INK), colors.HexColor(MUTED)

    def style(name: str, **kw):
        base = {"fontName": "Helvetica", "fontSize": 10, "leading": 14,
                "textColor": ink}
        base.update(kw)
        return rl["ParagraphStyle"](name, **base)

    s_title = style("title", fontName="Helvetica-Bold", fontSize=19,
                    leading=23, textColor=navy, spaceAfter=2)
    s_client = style("client", fontSize=12, leading=16, textColor=slate,
                     spaceAfter=2)
    s_meta = style("meta", fontSize=8.5, leading=12, textColor=muted)
    s_intro = style("intro", fontSize=10.5, leading=15.5, spaceAfter=5)
    s_h2 = style("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                 textColor=navy)
    s_h3 = style("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                 textColor=accent, spaceBefore=6, spaceAfter=2)
    s_para = style("para", spaceAfter=4)
    s_q = style("q", fontSize=10, leading=13.5)
    s_help = style("help", fontSize=8.5, leading=11.5, textColor=muted)
    s_tag = style("tag", fontName="Helvetica-Bold", fontSize=6.6, leading=9,
                  textColor=colors.white, alignment=1)
    s_bullet = style("bullet", fontSize=9.5, leading=13)

    story: List[Any] = []
    story.append(Paragraph("Onboarding checklist", s_title))
    if doc.client_name:
        story.append(Paragraph(_rich(doc.client_name), s_client))
    meta = " · ".join(p for p in (
        f"Reference {doc.case_ref}" if doc.case_ref else "",
        f"Prepared {generated_at}" if generated_at else "") if p)
    if meta:
        story.append(Paragraph(_rich(meta), s_meta))
    story.append(Spacer(1, 4 * mm))
    story.append(rl["HRFlowable"](width="100%", thickness=1.1, color=navy,
                                  spaceAfter=6))

    for block in doc.intro:
        story.append(Paragraph(_rich(block), s_intro))

    pending: List[Any] = []

    def flush_questions() -> None:
        """Emit the questions gathered so far as one checklist table."""
        if not pending:
            return
        rows: List[Any] = []
        for index, question in enumerate(pending):
            tag = "REQUIRED" if question.required else "OPTIONAL"
            tag_cell = Table(
                [[Paragraph(tag, s_tag)]], colWidths=[16 * mm],
                rowHeights=[BOX_SIZE * mm],
                style=rl["TableStyle"]([
                    ("BACKGROUND", (0, 0), (-1, -1),
                     accent if question.required
                     else colors.HexColor(SOFT_RULE)),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
            detail: List[Any] = [Paragraph(_rich(question.label), s_q)]
            if question.help:
                detail.append(Paragraph(_rich(question.help), s_help))
            if question.current:
                detail.append(Paragraph(
                    _rich(f"Currently: {question.current}"), s_help))
            # The answer box is a fixed square in its own nested table, NOT a
            # border on the outer cell. A bordered outer cell stretches to the
            # row, so a question carrying help text grew a box twice the height
            # of one that did not, and a column that should read as a tick-list
            # read as a ragged set of rectangles. Nested, it is the same size on
            # every row whatever the row contains.
            #
            # Drawn rather than typed, so it does not depend on a glyph the
            # standard PDF fonts may not carry.
            box = Table([[""]], colWidths=[BOX_SIZE * mm],
                        rowHeights=[BOX_SIZE * mm],
                        style=rl["TableStyle"]([
                            ("BOX", (0, 0), (-1, -1), 0.7, slate),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
            rows.append([box, tag_cell, detail])
        table = Table(rows, colWidths=[5 * mm, 18 * mm, None],
                      style=rl["TableStyle"]([
                          ("VALIGN", (0, 0), (-1, -1), "TOP"),
                          ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                          ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                          ("LEFTPADDING", (0, 0), (0, -1), 0),
                          ("RIGHTPADDING", (1, 0), (1, -1), 4),
                          ("LINEBELOW", (0, 0), (-1, -2), 0.3, rule),
                      ]))
        story.append(table)
        story.append(Spacer(1, 2.5 * mm))
        pending.clear()

    for node in doc.nodes:
        if node.kind == "question" and node.question is not None:
            pending.append(node.question)
            continue
        flush_questions()
        if node.kind == "heading":
            story.append(Spacer(1, 3.5 * mm))
            story.append(rl["KeepTogether"]([
                Paragraph(_rich(node.text), s_h2),
                rl["HRFlowable"](width="100%", thickness=0.6, color=rule,
                                 spaceBefore=3, spaceAfter=5)]))
        elif node.kind == "subheading":
            story.append(Paragraph(_rich(node.text), s_h3))
        elif node.kind == "bullet":
            text = f"<b>{_html.escape(node.text, quote=False)}</b>"
            if node.note:
                text += f" — {_html.escape(node.note, quote=False)}"
            story.append(Table(
                [["", Paragraph(text, s_bullet)]],
                colWidths=[3 * mm, None],
                style=rl["TableStyle"]([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0.6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("BACKGROUND", (0, 0), (0, 0), pale)])))
        else:
            story.append(Paragraph(_rich(node.text), s_para))
    flush_questions()

    trailer = " · ".join(p for p in (
        f"Reference {doc.case_ref}" if doc.case_ref else "",
        f"Communication {receipt_id}" if receipt_id else "",
        f"Content {content_hash}" if content_hash else "") if p)

    def furniture(canvas, document) -> None:
        """Wordmark, rule and traceable footer, on every page."""
        canvas.saveState()
        width, height = rl["A4"]
        canvas.setFillColor(navy)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawString(18 * mm, height - 13 * mm, WORDMARK)
        canvas.setStrokeColor(rule)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, height - 15.5 * mm, width - 18 * mm,
                    height - 15.5 * mm)
        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 7.2)
        canvas.drawString(18 * mm, 12 * mm, trailer)
        canvas.drawRightString(width - 18 * mm, 12 * mm,
                               f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    buffer = io.BytesIO()
    template = rl["SimpleDocTemplate"](
        buffer, pagesize=rl["A4"],
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=18 * mm,
        title=f"Trakt onboarding checklist — {doc.client_name or doc.case_ref}",
        author=SENDER_NAME, subject=f"Onboarding {doc.case_ref}".strip(),
        # Also in the document properties, not only printed on the page: a PDF
        # saved out of an email and opened years later still names the
        # communication it belongs to, without anyone having to read a footer.
        keywords=trailer)
    try:
        template.build(story, onFirstPage=furniture, onLaterPages=furniture)
    except Exception as exc:  # noqa: BLE001 — a layout failure, reported as one
        raise PresentationError(f"the checklist could not be laid out ({exc})") \
            from None
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# What the adapter asks for
# --------------------------------------------------------------------------- #

@dataclass
class ClientPresentation:
    """One approved communication, in the forms a client actually receives."""

    html_body: str
    text_body: str
    pdf: Optional[bytes] = None
    pdf_name: str = ""
    client_name: str = ""
    case_ref: str = ""

    @property
    def attachment_names(self) -> List[str]:
        return [self.pdf_name] if self.pdf else []


def build(covering: str, pack_markdown: str = "", *, receipt_id: str = "",
          content_hash: str = "", generated_at: str = "",
          fallback_case_ref: str = "") -> ClientPresentation:
    """Render one approved communication for a client.

    ``pack_markdown`` empty means there is no checklist to attach — the
    covering message is then the whole communication, which is a legitimate
    shape for a follow-up even though the initial pack always has one.
    """
    doc = parse_pack(pack_markdown) if pack_markdown.strip() else PackDocument()
    case_ref = doc.case_ref or fallback_case_ref
    doc.case_ref = case_ref

    pdf: Optional[bytes] = None
    pdf_name = ""
    if pack_markdown.strip():
        pdf_name = checklist_filename(doc.client_name, case_ref)
        pdf = render_pdf(doc, receipt_id=receipt_id,
                         content_hash=content_hash, generated_at=generated_at)

    return ClientPresentation(
        html_body=render_html(covering, case_ref=case_ref,
                              receipt_id=receipt_id, attachment_name=pdf_name),
        text_body=render_text(covering, case_ref=case_ref,
                              receipt_id=receipt_id, attachment_name=pdf_name),
        pdf=pdf, pdf_name=pdf_name, client_name=doc.client_name,
        case_ref=case_ref)
