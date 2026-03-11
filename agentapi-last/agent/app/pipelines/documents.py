from pathlib import Path
import zipfile
import tempfile
import shutil

import fitz
from docx import Document

from app.pipelines.images import extract_text_and_description_from_image


MIN_TEXT_THRESHOLD = 40


def extract_text_from_document(file_path: Path) -> dict:
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            return process_pdf(file_path)

        if ext == ".txt":
            text = extract_txt_text(file_path)
            return {
                "status": "processed",
                "summary": "Text file processed successfully",
                "processing_mode": "text_only",
                "document_text": text,
                "images_found": 0,
                "image_analyses": [],
                "combined_text": text,
            }

        if ext == ".docx":
            return process_docx(file_path)

        if ext == ".doc":
            return {
                "status": "not_supported",
                "summary": ".doc files are not supported yet. Please convert to .docx or PDF.",
                "processing_mode": "unsupported",
                "document_text": "",
                "images_found": 0,
                "image_analyses": [],
                "combined_text": "",
            }

        return {
            "status": "error",
            "summary": f"Unsupported document type: {ext}",
            "processing_mode": "unsupported",
            "document_text": "",
            "images_found": 0,
            "image_analyses": [],
            "combined_text": "",
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": f"Failed to process document: {str(e)}",
            "processing_mode": "error",
            "document_text": "",
            "images_found": 0,
            "image_analyses": [],
            "combined_text": "",
        }


def process_pdf(file_path: Path) -> dict:
    document_text = extract_pdf_text(file_path)
    embedded_image_analyses = extract_and_analyze_pdf_images(file_path)

    document_text = safe_strip(document_text)
    embedded_image_analyses = embedded_image_analyses or []

    # If little or no text was extracted, treat it as scanned/image-based
    if len(document_text) < MIN_TEXT_THRESHOLD:
        page_image_analyses = render_pdf_pages_and_analyze(file_path)

        combined_text = merge_texts(
            [document_text] +
            [build_image_analysis_text(item) for item in page_image_analyses]
        )

        return {
            "status": "processed",
            "summary": "PDF processed as scanned/image-based document",
            "processing_mode": "scanned_document",
            "document_text": document_text,
            "images_found": len(page_image_analyses),
            "image_analyses": page_image_analyses,
            "combined_text": combined_text,
        }

    combined_text = merge_texts(
        [document_text] +
        [build_image_analysis_text(item) for item in embedded_image_analyses]
    )

    processing_mode = "mixed_text_and_images" if embedded_image_analyses else "text_only"

    return {
        "status": "processed",
        "summary": "PDF processed successfully",
        "processing_mode": processing_mode,
        "document_text": document_text,
        "images_found": len(embedded_image_analyses),
        "image_analyses": embedded_image_analyses,
        "combined_text": combined_text,
    }


def process_docx(file_path: Path) -> dict:
    document_text = extract_docx_text(file_path)
    image_analyses = extract_and_analyze_docx_images(file_path)

    document_text = safe_strip(document_text)
    image_analyses = image_analyses or []

    combined_text = merge_texts(
        [document_text] +
        [build_image_analysis_text(item) for item in image_analyses]
    )

    processing_mode = "mixed_text_and_images" if image_analyses else "text_only"

    return {
        "status": "processed",
        "summary": "DOCX processed successfully",
        "processing_mode": processing_mode,
        "document_text": document_text,
        "images_found": len(image_analyses),
        "image_analyses": image_analyses,
        "combined_text": combined_text,
    }


def extract_pdf_text(file_path: Path) -> str:
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
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


def extract_and_analyze_pdf_images(file_path: Path) -> list:
    analyses = []
    doc = fitz.open(file_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)

            for image_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")

                temp_image_path = temp_dir_path / f"pdf_page_{page_index + 1}_img_{image_index + 1}.{image_ext}"
                temp_image_path.write_bytes(image_bytes)

                result = extract_text_and_description_from_image(temp_image_path)

                analyses.append({
                    "source": "embedded_image",
                    "page": page_index + 1,
                    "image_index": image_index + 1,
                    "text": result.get("text", ""),
                    "description": result.get("description", ""),
                    "objects": result.get("objects", []),
                    "possible_type": result.get("possible_type", ""),
                    "status": result.get("status", ""),
                })

    doc.close()
    return analyses


def render_pdf_pages_and_analyze(file_path: Path) -> list:
    analyses = []
    doc = fitz.open(file_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            page_image_path = temp_dir_path / f"pdf_page_render_{page_index + 1}.png"
            pix.save(str(page_image_path))

            result = extract_text_and_description_from_image(page_image_path)

            analyses.append({
                "source": "rendered_page",
                "page": page_index + 1,
                "text": result.get("text", ""),
                "description": result.get("description", ""),
                "objects": result.get("objects", []),
                "possible_type": result.get("possible_type", ""),
                "status": result.get("status", ""),
            })

    doc.close()
    return analyses


def extract_and_analyze_docx_images(file_path: Path) -> list:
    analyses = []

    if not zipfile.is_zipfile(file_path):
        return analyses

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        with zipfile.ZipFile(file_path, "r") as docx_zip:
            media_files = [
                name for name in docx_zip.namelist()
                if name.startswith("word/media/")
            ]

            for idx, media_name in enumerate(media_files, start=1):
                raw_name = Path(media_name).name
                extracted_path = temp_dir_path / raw_name

                with docx_zip.open(media_name) as src, open(extracted_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                result = extract_text_and_description_from_image(extracted_path)

                analyses.append({
                    "source": "embedded_image",
                    "image_index": idx,
                    "text": result.get("text", ""),
                    "description": result.get("description", ""),
                    "objects": result.get("objects", []),
                    "possible_type": result.get("possible_type", ""),
                    "status": result.get("status", ""),
                })

    return analyses


def build_image_analysis_text(item: dict) -> str:
    parts = []

    source = item.get("source", "")
    page = item.get("page")
    image_index = item.get("image_index")

    header_bits = []
    if source:
        header_bits.append(source)
    if page is not None:
        header_bits.append(f"page {page}")
    if image_index is not None:
        header_bits.append(f"image {image_index}")

    if header_bits:
        parts.append(f"[{' | '.join(header_bits)}]")

    text = safe_strip(item.get("text", ""))
    description = safe_strip(item.get("description", ""))

    if text:
        parts.append(f"Image text: {text}")
    if description:
        parts.append(f"Image description: {description}")

    return "\n".join(parts).strip()


def merge_texts(parts: list) -> str:
    cleaned = [safe_strip(part) for part in parts if safe_strip(part)]
    return "\n\n".join(cleaned).strip()


def safe_strip(value: str) -> str:
    return str(value).strip() if value is not None else ""