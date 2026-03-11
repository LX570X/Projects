from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.pipelines.audio import extract_text_from_audio

router = APIRouter(prefix="/api", tags=["audio"])

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/extract-audio-text")
async def extract_audio_text(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()

    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(AUDIO_EXTENSIONS)}"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = extract_text_from_audio(file_path)

        return {
            "file_name": file.filename,
            "file_type": ext,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))