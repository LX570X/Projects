from pathlib import Path
from app.pipelines.audio import process_audio


def run(file_path: Path) -> dict:
    return process_audio(file_path)