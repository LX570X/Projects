from pathlib import Path
import zipfile
import fitz
from docx import Document


def extract_text_from_document(file_path: Path) -> dict:
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            text = extract_pdf_text(file_path)
        elif ext == ".txt":
            text = extract_txt_text(file_path)
        elif ext == ".docx":
            text = extract_docx_text(file_path)
        elif ext == ".doc":
            return {
                "status": "not_supported",
                "summary": ".doc files are not supported yet. Please convert to .docx or PDF.",
                "text": ""
            }
        else:
            return {
                "status": "error",
                "summary": f"Unsupported document type: {ext}",
                "text": ""
            }

        return {
            "status": "processed",
            "summary": "Document text extracted successfully",
            "text": text
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": f"Failed to extract text: {str(e)}",
            "text": ""
        }


def extract_pdf_text(file_path: Path) -> str:
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text() + "\n"
    return text.strip()


def extract_txt_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_docx_text(file_path: Path) -> str:
    if not zipfile.is_zipfile(file_path):
        raise ValueError(
            "This file has a .docx extension but is not a valid DOCX file. "
            "It may actually be a .doc file, corrupted, or renamed incorrectly."
        )

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()