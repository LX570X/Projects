from fastapi import FastAPI

from app.routers.document_api import router as document_router
from app.routers.image_api import router as image_router
from app.routers.audio_api import router as audio_router
from app.routers.video_api import router as video_router
from app.routers.spreadsheet_api import router as spreadsheet_router

app = FastAPI(
    title="File Extraction API",
    description="APIs for extracting information from documents, images, audio, video, and spreadsheets",
    version="0.1.0",
)

app.include_router(document_router)
app.include_router(image_router)
app.include_router(audio_router)
app.include_router(video_router)
app.include_router(spreadsheet_router)


@app.get("/")
def root():
    return {"status": "ok", "message": "File Extraction API is running"}