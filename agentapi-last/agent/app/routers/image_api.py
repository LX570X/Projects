from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.pipelines.images import extract_text_and_description_from_image

router = APIRouter(prefix="/api", tags=["image"])

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/extract-image-text")
async def extract_image_text(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(IMAGE_EXTENSIONS)}"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = extract_text_and_description_from_image(file_path)

        return {
            "file_name": file.filename,
            "file_type": ext,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))