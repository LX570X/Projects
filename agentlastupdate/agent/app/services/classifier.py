"""
Purpose:
- Maps file extensions to normalized file categories used by the pipeline router.

Libraries used:
- pathlib: clean extension parsing from file names.
"""

from pathlib import Path

def get_file_classification(filename: str) -> str:
    """
    Classify a file based on its extension into one of:
    - document
    - image
    - audio
    - video
    - spreadsheet
    - unknown
    """
    ext = Path(filename).suffix.lower()
    
    classification_map = {
        # Documents
        ".pdf": "document",
        ".docx": "document",
        ".doc": "document",
        ".txt": "document",
        # Images
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".webp": "image",
        ".bmp": "image",
        ".tiff": "image",
        # Audio
        ".mp3": "audio",
        ".wav": "audio",
        ".m4a": "audio",
        ".flac": "audio",
        # Video
        ".mp4": "video",
        ".avi": "video",
        ".mov": "video",
        ".mkv": "video",
        # Spreadsheets
        ".csv": "spreadsheet",
        ".xlsx": "spreadsheet",
        ".xls": "spreadsheet",
    }
    
    return classification_map.get(ext, "unknown")
