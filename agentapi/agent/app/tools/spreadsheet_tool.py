from pathlib import Path


def run(file_path: Path) -> dict:
    return {
        "status": "not_implemented",
        "summary": f"Spreadsheet analysis is not implemented yet for {file_path.name}",
        "raw_text": "",
        "facts": []
    }