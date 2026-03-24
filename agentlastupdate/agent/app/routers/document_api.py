"""
Purpose:
- FastAPI endpoint to upload documents and return extraction results.

Libraries used:
- fastapi: API routing, upload handling, and HTTP errors.
- pathlib/shutil: safe local file saving before processing.
- app.pipelines.documents: core document extraction logic.
"""

from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.pipelines.documents import extract_text_from_document

router = APIRouter(prefix="/api", tags=["document"])

DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/extract-document-text")
async def extract_document_text(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()

    if ext not in DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(DOCUMENT_EXTENSIONS)}"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = extract_text_from_document(file_path)

        return {
            "file_name": file.filename,
            "file_type": ext,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))