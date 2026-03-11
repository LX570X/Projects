import json
from pathlib import Path

SOURCE_FILES_DIR = Path("source-files")
RAW_DATA_DIR = Path("raw-data")
SINGLE_REPORTS_DIR = Path("single_reports")
FINAL_REPORT_PATH = Path("final_report.json")

def ensure_base_directories() -> None:
    """Ensure all base folders exist."""
    SOURCE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_raw_data(filename: str, data: dict) -> Path:
    """
    Save extraction output to raw-data/<filename>.json.

    Example:
    input file:  source-files/invoice.pdf
    output file: raw-data/invoice.pdf.json
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"{filename}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return output_path


def save_single_report(filename: str, report: dict) -> Path:
    """
    Save one generated report to single_reports/<filename>.report.json
    """
    SINGLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SINGLE_REPORTS_DIR / f"{filename}.report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    return output_path


def save_final_report(report: dict) -> Path:
    """Save the aggregated final report to final_report.json."""
    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    return FINAL_REPORT_PATH
