from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.pipelines.spreadsheet import extract_structured_data_from_spreadsheet

router = APIRouter(prefix="/api", tags=["spreadsheet"])

STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/extract-structured-data")
async def extract_structured_data(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()

    if ext not in STRUCTURED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(STRUCTURED_EXTENSIONS)}"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = extract_structured_data_from_spreadsheet(file_path)

        return {
            "file_name": file.filename,
            "file_type": ext,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))