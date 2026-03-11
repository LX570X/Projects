import json
from pathlib import Path

from app.pipelines.audio import extract_text_from_audio
from app.pipelines.documents import extract_text_from_document
from app.pipelines.images import extract_text_and_description_from_image
from app.pipelines.spreadsheet import extract_structured_data_from_spreadsheet
from app.pipelines.video import analyze_video
from app.agents.report_agent import generate_final_report, generate_single_report
from app.services.classifier import get_file_classification
from app.services.storage import (
    SINGLE_REPORTS_DIR,
    SOURCE_FILES_DIR,
    ensure_base_directories,
    save_final_report,
    save_raw_data,
    save_single_report,
)


def _extract_by_type(file_path: Path, file_type: str) -> dict:
    if file_type == "image":
        return extract_text_and_description_from_image(file_path)
    if file_type == "audio":
        return extract_text_from_audio(file_path)
    if file_type == "document":
        return extract_text_from_document(file_path)
    if file_type == "spreadsheet":
        return extract_structured_data_from_spreadsheet(file_path)
    if file_type == "video":
        return analyze_video(file_path)

    return {"message": "No extraction available for this file type."}


def _single_report_path_for_file(file_name: str) -> Path:
    return SINGLE_REPORTS_DIR / f"{file_name}.report.json"


def _load_all_single_reports() -> list[dict]:
    reports: list[dict] = []
    for path in SINGLE_REPORTS_DIR.glob("*.report.json"):
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        except Exception:
            # Skip invalid/corrupted report files and continue.
            continue
    return reports


def _is_final_report_meaningful(report: dict) -> bool:
    summary_ok = bool(str(report.get("overall_summary", "")).strip())
    has_lists_content = any(
        bool(report.get(key))
        for key in [
            "key_people",
            "key_dates",
            "cross_file_insights",
            "recommended_next_steps",
        ]
    )
    return summary_ok and has_lists_content


def _build_fallback_final_report(single_reports: list[dict]) -> dict:
    all_people: set[str] = set()
    all_dates: set[str] = set()
    all_details: list[str] = []
    all_risks: list[str] = []

    for r in single_reports:
        all_people.update(r.get("people_involved", []) or [])
        all_dates.update(r.get("dates_mentioned", []) or [])
        all_details.extend(r.get("important_details", []) or [])
        all_risks.extend(r.get("risk_flags", []) or [])

    insights = []
    if all_details:
        insights.append(f"Compiled {len(all_details)} important detail points across files.")
    if all_risks:
        insights.append(f"Identified {len(all_risks)} risk-related observations.")
    if all_people:
        insights.append("Multiple references to involved people/entities across documents.")

    next_steps = [
        "Review high-priority risk flags and assign owners.",
        "Validate key facts (dates, names, and claims) against source evidence.",
        "Create a consolidated incident timeline from extracted dates and events.",
    ]

    return {
        "overall_summary": (
            f"Aggregated analysis from {len(single_reports)} single reports. "
            "This fallback summary was generated to avoid empty final-report output."
        ),
        "key_people": sorted(all_people),
        "key_dates": sorted(all_dates),
        "cross_file_insights": insights or ["Cross-file synthesis completed with limited explicit overlap."],
        "recommended_next_steps": next_steps,
    }


def process_source_files() -> dict:
    """
    Process files from source-files/ and save outputs to raw-data/.
    No API upload flow. Files are read directly from disk.
    """
    ensure_base_directories()

    results = []
    generated_single_reports: list[dict] = []
    skipped_files = 0
    for file_path in SOURCE_FILES_DIR.iterdir():
        if not file_path.is_file() or file_path.name == ".gitkeep":
            continue

        file_type = get_file_classification(file_path.name)
        error = None

        single_report_path = _single_report_path_for_file(file_path.name)
        if single_report_path.exists():
            skipped_files += 1
            results.append(
                {
                    "source_file": str(file_path),
                    "classification": file_type,
                    "status": "skipped",
                    "output_file": None,
                    "single_report_file": str(single_report_path),
                    "error": None,
                }
            )
            continue

        try:
            extracted = _extract_by_type(file_path, file_type)

            output_path = save_raw_data(file_path.name, extracted)

            single_report = generate_single_report(
                file_name=file_path.name,
                file_type=file_type,
                extraction_data=extracted,
            )
            single_report_path = save_single_report(file_path.name, single_report.model_dump())
            generated_single_reports.append(single_report.model_dump())

        except Exception as exc:
            output_path = None
            single_report_path = None
            error = str(exc)

        results.append(
            {
                "source_file": str(file_path),
                "classification": file_type,
                "status": "success" if error is None else "failed",
                "output_file": str(output_path) if output_path else None,
                "single_report_file": str(single_report_path) if single_report_path else None,
                "error": error,
            }
        )

    # Always regenerate final report using all available single reports
    # (old + newly generated in this run).
    final_report_path = None
    all_single_reports = _load_all_single_reports()
    if all_single_reports:
        final_report = generate_final_report(all_single_reports)
        final_report_data = final_report.model_dump()

        if not _is_final_report_meaningful(final_report_data):
            final_report_data = _build_fallback_final_report(all_single_reports)

        final_report_path = save_final_report(final_report_data)

    return {
        "processed": len(results),
        "skipped": skipped_files,
        "single_reports_generated": len(generated_single_reports),
        "single_reports_used_for_final": len(all_single_reports),
        "final_report_file": str(final_report_path) if final_report_path else None,
        "results": results,
    }


if __name__ == "__main__":
    summary = process_source_files()
    print(summary)
