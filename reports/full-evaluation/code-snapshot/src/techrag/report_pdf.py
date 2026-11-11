from __future__ import annotations

import os
import re
from datetime import datetime
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

_FONT_NAME = "InterviewReport"
_FONT_BOLD_NAME = "InterviewReportBold"


def export_report_pdf(markdown: str, output_path: str | Path) -> Path:
    """Render an interview Markdown report as a readable Chinese PDF."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    regular_font, bold_font = _register_fonts()
    styles = _styles(regular_font, bold_font)
    story: list[object] = []

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 2.5 * mm))
            continue
        if line.startswith("# "):
            if story:
                story.append(PageBreak())
            story.append(Paragraph(escape(line[2:]), styles["ReportTitle"]))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB")))
            story.append(Spacer(1, 5 * mm))
        elif line.startswith("## "):
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph(escape(line[3:]), styles["SectionHeading"]))
        elif line.startswith("### "):
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(escape(line[4:]), styles["QuestionHeading"]))
        elif line.startswith("- "):
            label, value = _split_label(line[2:])
            body = f"<b>{escape(label)}</b>{escape(value)}" if label else escape(value)
            story.append(Paragraph(body, styles["Bullet"], bulletText="•"))
        else:
            story.append(Paragraph(escape(line), styles["BodyTextChinese"]))

    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="模拟面试报告",
        author="InterviewPilot",
    )
    document.build(
        story,
        onFirstPage=lambda canvas, doc: _page_footer(canvas, doc, regular_font),
        onLaterPages=lambda canvas, doc: _page_footer(canvas, doc, regular_font),
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"PDF was not created: {destination}")
    return destination.resolve()


def _register_fonts() -> tuple[str, str]:
    if _FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return _FONT_NAME, _FONT_BOLD_NAME

    configured = os.getenv("INTERVIEW_PDF_FONT")
    regular_candidates = [
        configured,
        r"C:\Windows\Fonts\msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    bold_candidates = [
        os.getenv("INTERVIEW_PDF_BOLD_FONT"),
        r"C:\Windows\Fonts\msyhbd.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    regular = next(
        (Path(item) for item in regular_candidates if item and Path(item).is_file()),
        None,
    )
    bold = next((Path(item) for item in bold_candidates if item and Path(item).is_file()), regular)
    if regular:
        pdfmetrics.registerFont(TTFont(_FONT_NAME, str(regular)))
        pdfmetrics.registerFont(TTFont(_FONT_BOLD_NAME, str(bold)))
        return _FONT_NAME, _FONT_BOLD_NAME

    fallback = "STSong-Light"
    if fallback not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(fallback))
    return fallback, fallback


def _styles(regular_font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=3 * mm,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#1D4ED8"),
            keepWithNext=True,
        ),
        "QuestionHeading": ParagraphStyle(
            "QuestionHeading",
            parent=base["Heading3"],
            fontName=bold_font,
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#334155"),
            backColor=colors.HexColor("#EFF6FF"),
            borderPadding=(6, 7, 6, 7),
            keepWithNext=True,
        ),
        "BodyTextChinese": ParagraphStyle(
            "BodyTextChinese",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=9.5,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=9.5,
            leading=16,
            leftIndent=7 * mm,
            firstLineIndent=-3.5 * mm,
            bulletIndent=2 * mm,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=1.5 * mm,
        ),
    }


def _split_label(value: str) -> tuple[str, str]:
    match = re.match(r"^([^：:]{1,20}[：:])(.*)$", value)
    if not match:
        return "", value
    return match.group(1), match.group(2)


def _page_footer(canvas: object, document: object, font_name: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont(font_name, 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 8.5 * mm, f"InterviewPilot · {datetime.now():%Y-%m-%d}")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"第 {document.page} 页")
    canvas.restoreState()
