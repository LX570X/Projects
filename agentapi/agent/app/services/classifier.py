from pathlib import Path

DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
STRUCTURED_EXTENSIONS = {".csv", ".xlsx"}


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in STRUCTURED_EXTENSIONS:
        return "structured"

    return "unknown"