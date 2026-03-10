from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.pipelines.video import analyze_video

router = APIRouter(prefix="/api", tags=["video"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/extract-video-info")
async def extract_video_info(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()

    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {ext}. Allowed: {sorted(VIDEO_EXTENSIONS)}"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyze_video(file_path)

        return {
            "file_name": file.filename,
            "file_type": ext,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))