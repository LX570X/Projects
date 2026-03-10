from pathlib import Path
from app.pipelines.images import process_image


def run(file_path: Path) -> dict:
    return process_image(file_path)