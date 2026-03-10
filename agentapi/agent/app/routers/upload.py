from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.orchestrator import process_uploaded_file

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        result = await process_uploaded_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))