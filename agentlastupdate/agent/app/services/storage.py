"""
Purpose:
- Central file persistence utilities for raw data and reports (JSON/TXT/PDF).

Libraries used:
- pathlib/json/re: filesystem + serialization + simple text parsing.
- reportlab: PDF generation.
- arabic_reshaper + python-bidi: proper Arabic shaping/direction in PDFs.
"""

import json
import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

try:
    import arabic_reshaper  # type: ignore
    from bidi.algorithm import get_display  # type: ignore
except Exception:  # pragma: no cover
    arabic_reshaper = None
    get_display = None

SOURCE_FILES_DIR = Path("source-files")
RAW_DATA_DIR = Path("raw-data")
SINGLE_REPORTS_DIR = Path("single_reports")
FINAL_REPORT_PATH = Path("final_report.json")


def _is_arabic(language: str) -> bool:
    lang = (language or "").strip().lower()
    return lang in {"ar", "arabic", "العربية"}


def _font_for_language(language: str) -> tuple[str, str]:
    if not _is_arabic(language):
        return "Helvetica", "Helvetica-Bold"

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_name = f"CustomRTL-{font_path.stem}"
            try:
                pdfmetrics.getFont(font_name)
            except Exception:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            bold_candidate = font_path.with_name(font_path.stem + "bd" + font_path.suffix)
            bold_font_name = font_name
            if bold_candidate.exists():
                bold_font_name = f"CustomRTL-Bold-{bold_candidate.stem}"
                try:
                    pdfmetrics.getFont(bold_font_name)
                except Exception:
                    pdfmetrics.registerFont(TTFont(bold_font_name, str(bold_candidate)))

            return font_name, bold_font_name

    return "Helvetica", "Helvetica-Bold"


def _shape_if_arabic(text: str, language: str) -> str:
    if not _is_arabic(language):
        return text
    if arabic_reshaper and get_display:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text


def _wrap_text_lines(text: str, *, font_name: str, font_size: int, max_width: float) -> list[str]:
    wrapped: list[str] = []
    for paragraph in (text or "").splitlines():
        line = paragraph.strip()
        if not line:
            wrapped.append("")
            continue

        words = line.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            width = pdfmetrics.stringWidth(candidate, font_name, font_size)
            if width <= max_width:
                current = candidate
            else:
                if current:
                    wrapped.append(current)
                    current = word
                else:
                    wrapped.append(candidate)
                    current = ""
        if current:
            wrapped.append(current)
    return wrapped


def _tokenize_report_text(report_text: str) -> list[tuple[str, str]]:
    """
    Tokenize plain/markdown-like report text into: heading, bullet, paragraph, blank.
    """
    known_headings = {
        # English headings
        "executive summary",
        "key findings",
        "timeline",
        "people & entities",
        "risks",
        "recommended actions",
        "cross-file insights",
        "major risks",
        # Arabic headings
        "الملخص التنفيذي",
        "النتائج الرئيسية",
        "التسلسل الزمني",
        "الأشخاص والجهات",
        "المخاطر",
        "الإجراءات الموصى بها",
        "الرؤى عبر الملفات",
        "المخاطر الرئيسية",
    }

    tokens: list[tuple[str, str]] = []
    for raw_line in (report_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            tokens.append(("blank", ""))
            continue

        heading_match = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading_match:
            tokens.append(("heading", heading_match.group(1).strip()))
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", line)
        if bullet_match:
            tokens.append(("bullet", bullet_match.group(1).strip()))
            continue

        normalized = re.sub(r"[:：]\s*$", "", line).strip().lower()
        if normalized in known_headings:
            tokens.append(("heading", re.sub(r"[:：]\s*$", "", line).strip()))
            continue

        tokens.append(("paragraph", line))

    return tokens


def _draw_aligned_line(
    c: canvas.Canvas,
    *,
    line: str,
    language: str,
    page_width: float,
    margin: float,
    font_name: str,
    font_size: int,
    y: float,
    indent: float = 0,
) -> None:
    shaped = _shape_if_arabic(line, language)
    if _is_arabic(language):
        text_width = pdfmetrics.stringWidth(shaped, font_name, font_size)
        x = page_width - margin - text_width - indent
    else:
        x = margin + indent
    c.drawString(x, y, shaped)


def _save_report_pdf(output_path: Path, report_text: str, language: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = A4
    margin = 50
    body_font_size = 11
    heading_font_size = 14
    body_line_height = 16
    heading_line_height = 20
    usable_width = page_width - (2 * margin)

    body_font_name, heading_font_name = _font_for_language(language)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle(output_path.stem)
    c.setAuthor("Incident Analysis Agent")

    y = page_height - margin
    tokens = _tokenize_report_text(report_text)

    for token_type, text in tokens:
        if y < margin:
            c.showPage()
            y = page_height - margin

        if token_type == "blank":
            y -= body_line_height * 0.7
            continue

        if token_type == "heading":
            c.setFont(heading_font_name, heading_font_size)
            wrapped = _wrap_text_lines(text, font_name=heading_font_name, font_size=heading_font_size, max_width=usable_width)
            for line in wrapped:
                if y < margin:
                    c.showPage()
                    y = page_height - margin
                    c.setFont(heading_font_name, heading_font_size)
                _draw_aligned_line(
                    c,
                    line=line,
                    language=language,
                    page_width=page_width,
                    margin=margin,
                    font_name=heading_font_name,
                    font_size=heading_font_size,
                    y=y,
                )
                y -= heading_line_height
            y -= 4
            continue

        if token_type == "bullet":
            c.setFont(body_font_name, body_font_size)
            wrapped = _wrap_text_lines(text, font_name=body_font_name, font_size=body_font_size, max_width=usable_width - 18)
            for i, line in enumerate(wrapped):
                if y < margin:
                    c.showPage()
                    y = page_height - margin
                    c.setFont(body_font_name, body_font_size)
                prefix = "• " if i == 0 else "  "
                bullet_line = f"{prefix}{line}"
                _draw_aligned_line(
                    c,
                    line=bullet_line,
                    language=language,
                    page_width=page_width,
                    margin=margin,
                    font_name=body_font_name,
                    font_size=body_font_size,
                    y=y,
                    indent=8,
                )
                y -= body_line_height
            continue

        c.setFont(body_font_name, body_font_size)
        wrapped = _wrap_text_lines(text, font_name=body_font_name, font_size=body_font_size, max_width=usable_width)
        for line in wrapped:
            if y < margin:
                c.showPage()
                y = page_height - margin
                c.setFont(body_font_name, body_font_size)
            _draw_aligned_line(
                c,
                line=line,
                language=language,
                page_width=page_width,
                margin=margin,
                font_name=body_font_name,
                font_size=body_font_size,
                y=y,
            )
            y -= body_line_height

    c.save()
    return output_path

def ensure_base_directories() -> None:
    """Ensure all base folders exist."""
    SOURCE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_raw_data(filename: str, data: dict) -> Path:
    """
    Save extraction output to raw-data/<filename>.json.

    Example:
    input file:  source-files/invoice.pdf
    output file: raw-data/invoice.pdf.json
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"{filename}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return output_path


def save_single_report(filename: str, report: dict) -> Path:
    """
    Save one generated report to single_reports/<filename>.report.json
    """
    SINGLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SINGLE_REPORTS_DIR / f"{filename}.report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    return output_path


def save_single_report_by_language(filename: str, report: dict, language: str) -> Path:
    """
    Save one generated report to single_reports/<filename>.report.<lang>.json
    where lang is typically 'en' or 'ar'.
    """
    lang = (language or "en").strip().lower()
    SINGLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SINGLE_REPORTS_DIR / f"{filename}.report.{lang}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    return output_path


def save_single_report_text_by_language(filename: str, report_text: str, language: str) -> Path:
    """
    Save one generated report text to single_reports/<filename>.report.<lang>.txt
    where lang is typically 'en' or 'ar'.
    """
    lang = (language or "en").strip().lower()
    SINGLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SINGLE_REPORTS_DIR / f"{filename}.report.{lang}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write((report_text or "").strip() + "\n")
    return output_path


def save_single_report_pdf_by_language(filename: str, report_text: str, language: str) -> Path:
    """
    Save one generated report as PDF to single_reports/<filename>.report.<lang>.pdf
    with language-aware alignment.
    """
    lang = (language or "en").strip().lower()
    output_path = SINGLE_REPORTS_DIR / f"{filename}.report.{lang}.pdf"
    return _save_report_pdf(output_path, report_text, language=lang)


def save_final_report(report: dict) -> Path:
    """Save the aggregated final report to final_report.json."""
    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    return FINAL_REPORT_PATH


def save_final_report_by_language(report: dict, language: str) -> Path:
    """
    Save the aggregated final report to final_report.<lang>.json
    where lang is typically 'en' or 'ar'.
    """
    lang = (language or "en").strip().lower()
    output_path = Path(f"final_report.{lang}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    return output_path


def save_final_report_text_by_language(report_text: str, language: str) -> Path:
    """
    Save the aggregated final report text to final_report.<lang>.txt
    where lang is typically 'en' or 'ar'.
    """
    lang = (language or "en").strip().lower()
    output_path = Path(f"final_report.{lang}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write((report_text or "").strip() + "\n")
    return output_path


def save_final_report_pdf_by_language(report_text: str, language: str) -> Path:
    """
    Save the aggregated final report as PDF to final_report.<lang>.pdf
    with language-aware alignment.
    """
    lang = (language or "en").strip().lower()
    output_path = Path(f"final_report.{lang}.pdf")
    return _save_report_pdf(output_path, report_text, language=lang)


def save_final_report_text(report_text: str) -> Path:
    """Save the aggregated final report text to final_report.txt (default alias)."""
    output_path = Path("final_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write((report_text or "").strip() + "\n")
    return output_path


def save_final_report_pdf(report_text: str) -> Path:
    """Save the aggregated final report as final_report.pdf (default alias)."""
    output_path = Path("final_report.pdf")
    return _save_report_pdf(output_path, report_text, language="en")
