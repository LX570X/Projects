from pathlib import Path
from app.pipelines.documents import process_document


def run(file_path: Path) -> dict:
    return process_document(file_path)